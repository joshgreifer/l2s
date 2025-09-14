# pkg/tools/export_trainable_onnx.py
"""Export PCA and MLP ONNX graphs and generate ONNX Runtime training artifacts.

This combines the previous ``export_onnx.py`` and ``gen_ort_artifacts.py``
scripts. Given a PCA ``.joblib`` file and an MLP checkpoint ``.pt`` file,
this script emits trainable ONNX models for both stages and produces the
corresponding ONNX Runtime training artifacts.
"""

import argparse
import os
from types import SimpleNamespace
import joblib
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


from pkg.model_pca_mlp import GazePCAMLP


# ---------------------------------------------------------------------------
# ONNX export helpers (adapted from previous export_onnx.py)
# ---------------------------------------------------------------------------

def export_pca(pca, out_path: str, opset: int) -> None:
    """Export an sklearn PCA model to ONNX with a 3D landmark input."""

    n_features = pca.mean_.shape[0]

    # Convert the PCA as a standard sklearn model first (expects 2D input)
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    pca_model = convert_sklearn(
        pca,
        initial_types=[("flat", FloatTensorType([1, n_features]))],
        target_opset=opset,
    )

    # Adjust graph to accept [1, 478, 3] input and flatten internally
    pca_model.graph.output[0].name = "pca"
    if pca_model.graph.node:
        pca_model.graph.node[-1].output[0] = "pca"

    landmarks = helper.make_tensor_value_info("landmarks", TensorProto.FLOAT, [1, 478, 3])
    pca_model.graph.input.insert(0, landmarks)
    del pca_model.graph.input[1]

    shape_const = numpy_helper.from_array(
        np.array([1, n_features], dtype=np.int64), name="shape"
    )
    pca_model.graph.initializer.append(shape_const)
    reshape = helper.make_node("Reshape", ["landmarks", "shape"], ["flat"])
    pca_model.graph.node.insert(0, reshape)

    onnx.save(pca_model, out_path)
    onnx.checker.check_model(pca_model)
    print(f"✅ exported PCA ONNX: {out_path}")


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
    print(f"✅ exported MLP ONNX: {out_path}")




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_config(pca_model: str, mlp_checkpoint: str, state_dict: dict) -> SimpleNamespace:
    """Construct a minimal Config object required by ``GazePCAMLP``."""

    hidden = state_dict["mlp.0.weight"].shape[0]
    n_linear = len([k for k in state_dict if k.startswith("mlp.") and k.endswith("weight")])
    num_mlp_layers = n_linear - 1  # minus the input layer
    num_calib = len([k for k in state_dict if k.startswith("calibration_layers.") and k.endswith("weight")])

    pca = joblib.load(pca_model)

    cfg = SimpleNamespace(
        model_type="GazePCAMLP",
        device="cpu",
        checkpoint=mlp_checkpoint,
        pca_path=pca_model,
        dataset_path="",
        train=SimpleNamespace(dataset_capacity=0),
        model=SimpleNamespace(
            pca_num=pca.n_components_,
            hidden_channels=hidden,
            num_mlp_layers=num_mlp_layers,
            num_calibration_layers=num_calib,
        ),
    )
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pca_model", help="Path to PCA joblib file")
    p.add_argument("--mlp_checkpoint", help="Path to MLP .pt checkpoint")
    p.add_argument("--out", default="cache", help="Output directory (default: cache)")
    p.add_argument("--opset", type=int, default=19)
    args = p.parse_args()

    checkpoints = os.path.join(args.out, "checkpoints")
    ort_dir = os.path.join(args.out, "ort_artifacts")
    os.makedirs(checkpoints, exist_ok=True)
    os.makedirs(ort_dir, exist_ok=True)

    pca_out = os.path.join(checkpoints, "pca.onnx")
    mlp_out = os.path.join(checkpoints, "gaze_mlp.onnx")

    state_dict = torch.load(args.mlp_checkpoint, map_location="cpu", weights_only=False)
    cfg = build_config(args.pca_model, args.mlp_checkpoint, state_dict)
    model = GazePCAMLP(cfg).eval()

    # We already loaded PCA when building the config.
    pca = model.pca
    export_pca(pca, pca_out, args.opset)
    export_mlp(model, mlp_out, args.opset)

    # generate_ort_artifacts(pca_out, ort_dir, "pca_", "pca")
    # generate_ort_artifacts(mlp_out, ort_dir, "gaze_", "gaze")


if __name__ == "__main__":
    main()
