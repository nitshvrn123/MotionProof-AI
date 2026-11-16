"""Create a reproducibility manifest for frozen MotionProof AI artifacts."""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform

import numpy
import pandas
import sklearn
import torch


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    ROOT / "results/metrics/final_test/reproducibility_manifest.json"
)

CHECKPOINTS = {
    "cnn": "checkpoints/ppg_cnn_best.pt",
    "reliability_classifier":
        "checkpoints/reliability_classifier_nonlinear_full.joblib",
    "fitted_q_policy":
        "checkpoints/learned_policy_fitted_q_full.pt",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("=" * 75)
    print("MOTIONPROOF AI — STEP 38.3A")
    print("REPRODUCIBILITY MANIFEST")
    print("=" * 75)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {OUTPUT_PATH}"
        )

    manifest = {
        "project": "MotionProof AI",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "sklearn_version": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "test_subjects": [13, 14, 15],
        "test_windows": 13007,
        "random_seed": 42,
        "checkpoints": {},
    }

    for name, relative_path in CHECKPOINTS.items():
        path = ROOT / relative_path

        if not path.is_file():
            raise FileNotFoundError(path)

        manifest["checkpoints"][name] = {
            "path": relative_path,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

        print(f"\n{name}")
        print("Path:", relative_path)
        print("SHA-256:", manifest["checkpoints"][name]["sha256"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("\nEnvironment:")
    print("Python:", manifest["python_version"])
    print("PyTorch:", manifest["pytorch_version"])
    print("NumPy:", manifest["numpy_version"])
    print("Pandas:", manifest["pandas_version"])
    print("scikit-learn:", manifest["sklearn_version"])
    print("CUDA available:", manifest["cuda_available"])

    print("\nSaved:", OUTPUT_PATH)
    print("\nPASS: Step 38.3A completed.")


if __name__ == "__main__":
    main()
