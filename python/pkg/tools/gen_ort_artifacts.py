# pkg/tools/gen_ort_artifacts.py
import os, onnx
# NOTE: This only works under linux or WSL, as onnxruntime-training is not available on Windows.
# Check we're runnining under Linux/WSL
if os.name != 'posix':
    raise RuntimeError("This script only works under Linux or WSL, as onnxruntime-training is not available on Windows.")
else:
    try:
        from onnxruntime.training import artifacts
    except ImportError as e:
        raise ImportError("onnxruntime-training is not installed. Please install it via 'pip install onnxruntime-training'") from e


    MODEL = os.path.join("cache", "checkpoints", "gaze_mlp.onnx")
    OUTDIR = "cache/ort_artifacts"
    PREFIX = "gaze_"

    os.makedirs(OUTDIR, exist_ok=True)

    # Collect all parameter tensor names so they get gradients
    m = onnx.load(MODEL)
    param_names = [init.name for init in m.graph.initializer]  # all weights/biases

    artifacts.generate_artifacts(
        model=m,
        requires_grad=param_names,          # train all params
        frozen_params=None,                 # or provide a subset to freeze
        loss=artifacts.LossType.MSELoss,    # regression: (pred - target)^2
        optimizer=artifacts.OptimType.AdamW,
        artifact_directory=OUTDIR,
        prefix=PREFIX,
        # Tell ORT which model output feeds the loss; label input will be auto-added.
        loss_input_names=["gaze"],          # matches your export's output_names
    )

    print("Wrote:", os.listdir(OUTDIR))
