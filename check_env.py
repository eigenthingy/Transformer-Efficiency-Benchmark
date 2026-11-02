"""Check that the environment can run the benchmark, and say how to size it.

Run this before the sweep. It reports which dependencies are present, what
hardware the timings will be measured on, and — because the absolute times in
this benchmark are hardware-specific — what settings are sensible here.

    python check_env.py
"""

from __future__ import annotations

import importlib.metadata as md
import sys

REQUIRED = ["torch", "transformers", "datasets", "accelerate", "numpy", "pandas", "matplotlib"]
OPTIONAL = {
    "evaluate": "metrics fall back to scikit-learn",
    "scikit-learn": "needed for the metrics fallback",
}


def _version(package: str) -> str | None:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


def main() -> int:
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}\n")

    missing = []
    print("Required")
    for package in REQUIRED:
        version = _version(package)
        print(f"  {package:<14} {version or 'MISSING'}")
        if version is None:
            missing.append(package)

    print("\nOptional")
    for package, consequence in OPTIONAL.items():
        version = _version(package)
        suffix = "" if version else f"  (absent -> {consequence})"
        print(f"  {package:<14} {version or '-'}{suffix}")

    # The hardware and sizing report is the reason to run this script *before*
    # installing anything, so it prints whenever torch is importable — the
    # missing-package verdict is deferred to the end.
    if _version("torch") is None:
        print("\nMissing: torch — install it first, matched to your platform:")
        print("  https://pytorch.org/get-started/locally/")
        return 1

    import common  # imported late: it needs torch

    common.enable_utf8_stdout()
    env = common.environment_info()
    print("\nHardware")
    print(f"  device        {env['device']}")
    if env["cuda_available"]:
        print(f"  gpu           {env.get('gpu')} ({env.get('gpu_memory_gb')} GB)")
    print(f"  cpu           {env['cpu']}")
    print(f"  logical cores {env['cpu_count_logical']}")
    print(f"  torch threads {env['torch_num_threads']} (intra-op)")

    print("\nSizing")
    if env["cuda_available"]:
        print("  GPU available. The defaults are conservative for a GPU; consider")
        print("  the full splits and more epochs:")
        print("      python run_benchmark.py --max-train-samples -1 --epochs 2 --batch-size 32")
        print("  Rough estimate for the 3-model default sweep: 10-25 min.")
    else:
        threads = env["torch_num_threads"]
        print("  CPU only — which is the setting this benchmark is most interesting in.")
        print("  The defaults are 4000 train / 2000 eval, max_length 128, 1 epoch.")
        print(f"  Pin the thread count so timings are comparable between models. torch")
        print(f"  already picked {threads}, which is usually the physical core count — pinning")
        print("  it explicitly costs no speed and removes a source of variance:")
        print(f"      python run_benchmark.py --num-threads {threads}")
        print()
        print("  Expect roughly 1 to 1.5 hours for the default 3-model sweep on a recent")
        print("  desktop CPU, and several hours on an older laptop. BERT-base alone is about")
        print("  half of it, so the cost is dominated by the baseline, not by the sweep.")
        print("  Add ~1 GB of model downloads on the first run.")
        print()
        print("  Measure your own machine first (~5 min, meaningless accuracy but real timings):")
        print("      python run_benchmark.py --smoke")
        print("  train.py prints samples/s per model; the full run scales linearly from there.")
        print()
        print("  To cut it down: --max-train-samples 2000 halves the training, --max-length 64")
        print("  cuts most of the rest, and `--models distilbert minilm-l12` drops the")
        print("  baseline entirely (at the cost of every relative column). train.py reuses")
        print("  a checkpoint trained with the same settings, so the sweep can be run in pieces.")

    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print("Install with:  pip install -r requirements.txt")
        print("           or:  .\\setup.ps1   (Windows)  /  ./setup.sh   (macOS, Linux)")
        return 1

    print("\nEnvironment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
