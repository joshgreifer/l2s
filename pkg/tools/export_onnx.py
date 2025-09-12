# pkg/tools/export_onnx.py
import argparse, torch, onnx
from pkg.config import Config
from pkg.model_pca_mlp import GazePCAMLP
from pkg.tools.bake_pca import bake_pca_into_first_linear

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="cache/config.json")
    p.add_argument("--out", default="cache/checkpoints/GazePCAMLP_baked_forward.onnx")
    p.add_argument("--opset", type=int, default=19)  # safe for ORT(Web)
    args = p.parse_args()

    # Load config on CPU to avoid CUDA dependency during export
    cfg = Config(args.config)
    cfg.device = "cpu"

    model = GazePCAMLP(cfg).eval()       # loads checkpoint
    model = bake_pca_into_first_linear(model).train()  # TRAINING graph export

    dummy = torch.randn(1, 478, 3, dtype=torch.float32)  # (B, N, C)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["landmarks"],
        output_names=["gaze"],
        dynamic_axes={"landmarks": {0: "batch"}, "gaze": {0: "batch"}},
        export_params=True,
        do_constant_folding=False,
        opset_version=args.opset,
        training=torch.onnx.TrainingMode.TRAINING,
    )

    onnx_model = onnx.load(args.out)
    onnx.checker.check_model(onnx_model)
    print(f"✅ exported: {args.out}")

if __name__ == "__main__":
    main()
