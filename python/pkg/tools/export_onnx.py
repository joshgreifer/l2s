# pkg/tools/export_onnx.py
"""Export PCA and MLP stacks of :class:`GazePCAMLP` as separate ONNX graphs.

This script exports the PCA transform and the MLP stack independently:

* ``cache/checkpoints/pca.onnx`` expects landmark inputs of shape ``[1, 478, 3]``
  and outputs PCA features of shape ``[1, 32]``.
* ``cache/checkpoints/gaze_mlp.onnx`` expects PCA features of shape ``[1, 32]``
  and outputs gaze predictions of shape ``[1, 2]``.
"""

import argparse
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from pkg.config import Config
from pkg.model_pca_mlp import GazePCAMLP


def export_pca(pca, out_path: str, opset: int) -> None:
    """Export an sklearn PCA model to ONNX with a 3D landmark input."""

    n_features = pca.mean_.shape[0]

    # Convert the PCA as a standard sklearn model first (expects 2D input)
    pca_model = convert_sklearn(
        pca,
        initial_types=[("flat", FloatTensorType([1, n_features]))],
        target_opset=opset,
    )

    # Adjust graph to accept [1, 478, 3] input and flatten internally
    # Rename both the graph output and the producing node so the checker
    # sees a valid connection (the original skl2onnx export names the
    # final output something like "variable").
    pca_model.graph.output[0].name = "pca"
    if pca_model.graph.node:
        pca_model.graph.node[-1].output[0] = "pca"

    # Insert new 3D input
    landmarks = helper.make_tensor_value_info(
        "landmarks", TensorProto.FLOAT, [1, 478, 3]
    )
    pca_model.graph.input.insert(0, landmarks)

    # Remove old 2D input
    del pca_model.graph.input[1]

    # Reshape from [1, 478, 3] -> [1, n_features]
    shape_const = numpy_helper.from_array(
        np.array([1, n_features], dtype=np.int64), name="shape"
    )
    pca_model.graph.initializer.append(shape_const)
    reshape = helper.make_node("Reshape", ["landmarks", "shape"], ["flat"])
    pca_model.graph.node.insert(0, reshape)

    onnx.save(pca_model, out_path)
    onnx.checker.check_model(pca_model)
    print(f"✅ exported PCA: {out_path}")


class ForwardFeatures(torch.nn.Module):
    """Wrapper module to export ``GazePCAMLP.forward_features``."""

    def __init__(self, model: GazePCAMLP):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor):  # pragma: no cover - simple wrapper
        return self.model.forward_features(feats)


def export_mlp(model: GazePCAMLP, out_path: str, opset: int) -> None:
    """Export the MLP stack of ``GazePCAMLP`` to ONNX."""

    wrapper = ForwardFeatures(model).train()  # training graph export
    dummy = torch.randn(1, model.pca.n_components_, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["pca"],
        output_names=["gaze"],
        dynamic_axes={"pca": {0: "batch"}, "gaze": {0: "batch"}},
        export_params=True,
        do_constant_folding=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.TRAINING,
    )

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ exported MLP: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="cache/config.json")
    p.add_argument("--pca-out", default="cache/checkpoints/pca.onnx")
    p.add_argument("--mlp-out", default="cache/checkpoints/gaze_mlp.onnx")
    p.add_argument("--opset", type=int, default=19)
    args = p.parse_args()

    # Load config on CPU to avoid CUDA dependency during export
    cfg = Config(args.config)
    cfg.device = "cpu"

    model = GazePCAMLP(cfg).eval()  # loads checkpoint (or random weights)

    export_pca(model.pca, args.pca_out, args.opset)
    export_mlp(model, args.mlp_out, args.opset)


if __name__ == "__main__":
    main()

