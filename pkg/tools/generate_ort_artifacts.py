# ---------------------------------------------------------------------------
# ORT artifact generation
# ---------------------------------------------------------------------------
import os
import onnx
import argparse

def generate_ort_artifacts(model_path: str, outdir: str, prefix: str, loss_input: str) -> None:
    """Generate ONNX Runtime training artifacts for ``model_path``."""
    # ``onnxruntime-training`` is only available on Linux/WSL.
    if os.name != "posix":
        raise RuntimeError(
            "onnxruntime-training is only available on Linux or WSL."
        )
    try:  # pragma: no cover - import side effects
        from onnxruntime.training import artifacts
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            "onnxruntime-training is required. Install with 'pip install onnxruntime-training'."
        ) from e

    os.makedirs(outdir, exist_ok=True)
    m = onnx.load(model_path)
    # param_names = [init.name for init in m.graph.initializer]

    # ``generate_artifacts`` treats every name in ``requires_grad`` as a trainable
    # parameter. Some ONNX models contain non-float initializers (e.g. ``shape``
    # inputs for ``Reshape``) which cannot participate in gradient computation.
    # Attempting to request gradients for such tensors results in errors like
    # ``failed to find NodeArg by name: <name>_grad`` during artifact generation.
    #
    # Restrict the list of parameters that require gradients to floating point
    # tensors only so that constant integer tensors are automatically excluded.
    float_types = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
    }
    param_names = [
        init.name for init in m.graph.initializer if init.data_type in float_types
    ]

    artifacts.generate_artifacts(
        model=m,
        requires_grad=param_names,
        frozen_params=None,
        loss=artifacts.LossType.MSELoss,
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=outdir,
        prefix=prefix,
        loss_input_names=[loss_input],
    )

    print(f"✅ wrote ORT artifacts for {prefix} to {outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to ONNX model file")
    p.add_argument("--outdir", default="cache/ort_artifacts", help="Output directory (default: cache/ort_artifacts)")
    p.add_argument("--prefix", default="gaze_", help="Prefix for output files (default: gaze_)")
    p.add_argument("--loss_input", default="gaze", help="Name of the model output to use for loss (default: gaze)")
    args = p.parse_args()
    generate_ort_artifacts(args.model_path, args.outdir, args.prefix, args.loss_input)
    print(f"✅ wrote ORT artifacts for {args.prefix} to {args.outdir}")
