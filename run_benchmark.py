"""Run the whole benchmark end to end: train -> evaluate -> report.

A thin orchestrator so the sweep is one command and one set of flags. It calls
the same `main()` functions the individual scripts expose, so nothing is
duplicated and the intermediate artefacts are identical either way.

Examples
--------
    python run_benchmark.py --smoke               # ~5 min on CPU, proves the pipeline (writes to results/smoke/)
    python run_benchmark.py --num-threads 12      # the real run; check_env.py prints your value
    python run_benchmark.py --seeds 42 43 44      # three seeds, for variance
    python run_benchmark.py --dataset imdb --max-length 256
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time

from config import ROOT, DATASETS, DEFAULT_MODELS, MODELS, SWEEPS, BASELINE, BenchConfig


def _load_sibling(name: str):
    """Import a script from this repo by path.

    `evaluate.py` collides with the Hugging Face `evaluate` package, so which
    one a bare `import evaluate` returns depends on import order elsewhere in
    the process. Loading by file path removes the question. (`common.py` does
    the mirror-image trick to reach the library — see `load_hf_evaluate`.)
    """
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_script = _load_sibling("train")
evaluate_script = _load_sibling("evaluate")
report_script = _load_sibling("report")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None, choices=list(MODELS),
                   help=f"explicit model list (default: the '{'classic'}' sweep)")
    p.add_argument("--sweep", default=None, choices=list(SWEEPS),
                   help="a named model set from config.SWEEPS; each one is a question. "
                        + " · ".join(f"{k}: {len(v)} models" for k, v in SWEEPS.items()))
    p.add_argument("--dataset", default=BenchConfig.dataset, choices=list(DATASETS))
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=BenchConfig.max_train_samples)
    p.add_argument("--max-eval-samples", type=int, default=BenchConfig.max_eval_samples)
    p.add_argument("--epochs", type=float, default=BenchConfig.epochs)
    p.add_argument("--batch-size", type=int, default=BenchConfig.train_batch_size)
    p.add_argument("--eval-batch-size", type=int, default=BenchConfig.eval_batch_size)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--uniform-lr", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[BenchConfig.seed],
                   help="run the full sweep once per seed; the first drives the figures")
    p.add_argument("--num-threads", type=int, default=None)
    p.add_argument("--latency-rounds", type=int, default=BenchConfig.latency_rounds,
                   help="interleaved timing rounds (see evaluate.py)")
    p.add_argument("--latency-samples", type=int, default=BenchConfig.latency_samples)
    p.add_argument("--baseline", default=BASELINE, choices=list(MODELS))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-train", action="store_true", help="evaluate and report existing checkpoints")
    return p.parse_args(argv)


def _shared(args: argparse.Namespace, seed: int) -> list[str]:
    flags = ["--models", *args.models,
             "--dataset", args.dataset,
             "--seed", str(seed)]
    if args.max_length is not None:
        flags += ["--max-length", str(args.max_length)]
    if args.num_threads is not None:
        flags += ["--num-threads", str(args.num_threads)]
    if args.smoke:
        flags.append("--smoke")
    return flags


def resolve_models(args: argparse.Namespace) -> list[str]:
    """`--models` wins over `--sweep`; with neither, the classic three.

    Both are accepted rather than one, because `--sweep` names a question and
    `--models` answers an ad-hoc one. Passing both is a contradiction, so it is
    refused rather than silently resolved.
    """
    if args.models and args.sweep:
        raise SystemExit("pass --models or --sweep, not both")
    if args.models:
        return list(args.models)
    if args.sweep:
        return list(SWEEPS[args.sweep])
    return list(DEFAULT_MODELS)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.models = resolve_models(args)
    started = time.perf_counter()
    print(f"Models: {', '.join(args.models)}"
          + (f"   (sweep '{args.sweep}')" if args.sweep else ""))

    for seed in args.seeds:
        print(f"\n{'#' * 72}\n#  seed {seed}\n{'#' * 72}")

        if not args.skip_train:
            flags = _shared(args, seed) + [
                "--max-train-samples", str(args.max_train_samples),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
            ]
            if args.lr is not None:
                flags += ["--lr", str(args.lr)]
            if args.uniform_lr:
                flags.append("--uniform-lr")
            if args.overwrite:
                flags.append("--overwrite")
            if (code := train_script.main(flags)) != 0:
                return code

        flags = _shared(args, seed) + [
            "--max-eval-samples", str(args.max_eval_samples),
            "--batch-size", str(args.eval_batch_size),
            "--latency-rounds", str(args.latency_rounds),
            "--latency-samples", str(args.latency_samples),
        ]
        if (code := evaluate_script.main(flags)) != 0:
            return code

    report_flags = ["--dataset", args.dataset, "--seed", str(args.seeds[0]),
                    "--baseline", args.baseline]
    if len(args.seeds) > 1:
        report_flags += ["--seeds", *(str(s) for s in args.seeds)]
    if args.smoke:
        report_flags.append("--smoke")
    code = report_script.main(report_flags)

    print(f"\nTotal wall clock: {(time.perf_counter() - started) / 60:.1f} min")
    return code


if __name__ == "__main__":
    sys.exit(main())
