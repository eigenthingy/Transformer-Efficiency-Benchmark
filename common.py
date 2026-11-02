"""Shared plumbing: data, metrics, timing, model accounting, statistics.

Kept deliberately free of `Trainer` so that `evaluate.py` can be run against a
checkpoint produced by anything, and so the timing code has no framework
overhead in the measured region.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from config import DATASETS, MODELS, DatasetSpec, ModelSpec


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def enable_utf8_stdout() -> None:
    """Make stdout/stderr able to carry the report's symbols.

    The written report uses Δ, ×, → and the like, and the console echo reuses
    the same strings. A stock Windows console is cp1252, where printing them
    raises `UnicodeEncodeError` — so the run would die on its own summary after
    all the work was done. Files are always written with an explicit utf-8
    encoding; this fixes only the terminal.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if encoding.replace("-", "") not in {"utf8", "utf16", "utf32"}:
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):  # not a reconfigurable stream
                pass


def set_seed(seed: int) -> None:
    """Seed every RNG that touches training.

    Deterministic kernels are deliberately left off: they cost speed, and this
    benchmark measures speed. Seeds make runs repeatable on one machine, not
    bit-identical across machines.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # keep it fast; seeds are enough here
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Registry helpers
# --------------------------------------------------------------------------- #
def get_model_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. Known: {', '.join(MODELS)}")
    return MODELS[name]


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'. Known: {', '.join(DATASETS)}")
    return DATASETS[name]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_splits(
    spec: DatasetSpec,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    seed: int,
):
    """Load and optionally subsample the train/test splits.

    Subsampling is a *shuffled* selection with a fixed seed, so every model in
    the sweep trains and is evaluated on byte-identical subsets. That is what
    makes the paired statistics in `report.py` valid.
    """
    from datasets import load_dataset

    ds = load_dataset(spec.hf_name)
    train = ds[spec.train_split]
    test = ds[spec.test_split]

    if max_train_samples is not None and max_train_samples < len(train):
        train = train.shuffle(seed=seed).select(range(max_train_samples))
    if max_eval_samples is not None and max_eval_samples < len(test):
        test = test.shuffle(seed=seed).select(range(max_eval_samples))
    return train, test


def tokenize_splits(train, test, tokenizer, spec: DatasetSpec, max_length: int):
    """Tokenize to a fixed length.

    Fixed-length (not dynamic) padding is intentional: every model then
    processes tensors of identical shape, so the latency column compares
    architectures rather than incidental batch geometry.
    """

    def _tok(batch):
        return tokenizer(
            batch[spec.text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}

    def _prep(split):
        out = split.map(_tok, batched=True, desc="tokenizing")
        if spec.label_column != "labels":
            out = out.rename_column(spec.label_column, "labels")
        out = out.remove_columns([c for c in out.column_names if c not in keep])
        out.set_format("torch")
        return out

    return _prep(train), _prep(test)


def truncation_rate(split, tokenizer, spec: DatasetSpec, max_length: int) -> float:
    """Fraction of examples longer than `max_length` tokens.

    Reported alongside accuracy so a reader can tell whether the accuracy
    column is measuring the model or measuring the truncation.
    """
    lengths = [
        len(tokenizer(t, truncation=False)["input_ids"])
        for t in split[spec.text_column][: min(len(split), 512)]
    ]
    return sum(l > max_length for l in lengths) / len(lengths)


# --------------------------------------------------------------------------- #
# Model accounting
# --------------------------------------------------------------------------- #
def count_parameters(model) -> dict[str, int]:
    """Total / trainable / embedding / encoder parameter counts.

    The embedding split matters: BERT-base and DistilBERT share an identical
    ~23.8M of embeddings (23.4M of token table plus positions), so "half the parameters" is really
    "half the *encoder* parameters". MiniLM shrinks both.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    embedding = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Embedding":
            embedding += sum(p.numel() for p in module.parameters(recurse=False))

    return {
        "params_total": total,
        "params_trainable": trainable,
        "params_embedding": embedding,
        "params_non_embedding": total - embedding,
    }


def directory_size_mb(path: str | Path, patterns: Iterable[str] = ("*.safetensors", "*.bin")) -> float:
    """On-disk size of the weight files in a checkpoint directory, in MB."""
    path = Path(path)
    total = 0
    for pattern in patterns:
        total += sum(f.stat().st_size for f in path.glob(pattern))
    return total / (1024 * 1024)


def theoretical_size_mb(n_params: int, bytes_per_param: int = 4) -> float:
    return n_params * bytes_per_param / (1024 * 1024)


def architecture_summary(model) -> dict[str, Any]:
    cfg = model.config
    return {
        "num_layers": getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None) or getattr(cfg, "dim", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_heads", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None) or getattr(cfg, "hidden_dim", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def load_hf_evaluate():
    """Import Hugging Face's `evaluate` package despite this repo's `evaluate.py`.

    `evaluate.py` sits next to this file, and the script's own directory is
    first on `sys.path` — so a plain `import evaluate` resolves to the local
    module, not the library. Rather than rename the script, ask the path finder
    for the package while excluding this directory, and register it under a
    non-colliding name.

    Returns None if the library isn't installed.
    """
    if (cached := sys.modules.get("_hf_evaluate")) is not None:
        return cached

    here = Path(__file__).resolve().parent
    search = [p for p in sys.path if p and Path(p).resolve() != here]
    spec = importlib.machinery.PathFinder.find_spec("evaluate", path=search)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules["_hf_evaluate"] = module

    # The package resolves its own submodules by ABSOLUTE import through
    # sys.modules["evaluate"] — during exec, and again lazily at first attribute
    # access, because Hugging Face packages defer submodule loading. So the
    # binding has to persist after this function returns.
    #
    # An earlier version restored the previous binding in a `finally`, for
    # tidiness. That broke it: `evaluate.load(...)` then re-imported `evaluate`,
    # found this repo's `evaluate.py` first on sys.path, and raised
    # AttributeError — silently downgrading every run to the scikit-learn
    # fallback. The rollback below therefore happens only when exec fails.
    previous = sys.modules.get("evaluate")
    sys.modules["evaluate"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        if previous is not None:
            sys.modules["evaluate"] = previous
        else:
            sys.modules.pop("evaluate", None)
        sys.modules.pop("_hf_evaluate", None)
        print(f"[metrics] could not load the `evaluate` library: "
              f"{type(exc).__name__}: {exc}")
        return None

    if not callable(getattr(module, "load", None)):
        # Not the library: something else answered to the name `evaluate`.
        sys.modules.pop("_hf_evaluate", None)
        return None
    return module


_METRIC_CACHE: dict[str, Any] = {}


def _hf_metric(hf_evaluate, name: str):
    """`evaluate.load` fetches a script from the hub; do it once per process."""
    if name not in _METRIC_CACHE:
        _METRIC_CACHE[name] = hf_evaluate.load(name)
    return _METRIC_CACHE[name]


def prefetch_metrics() -> str:
    """Load the metric implementations before any timing begins.

    `evaluate.load` fetches from the hub the first time it is called. That
    network round trip landed inside the *first* model's evaluation, leaving the
    CPU idle — and therefore clocked down — immediately before its calibration
    probe. The probe read 276 GFLOP/s against ~550 for every later model, a 2x
    artefact that presented as "the first model is slow".

    Doing the fetch once, up front, moves the I/O out of every measured region.
    Returns the backend that will be used, so the caller can report it.
    """
    module = load_hf_evaluate()
    if module is None:
        return "sklearn"
    try:
        _hf_metric(module, "accuracy")
        _hf_metric(module, "f1")
        return "evaluate"
    except Exception as exc:
        print(f"[metrics] prefetch failed, will use scikit-learn "
              f"({type(exc).__name__}: {exc})")
        return "sklearn"


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Accuracy and macro-F1.

    Uses the `evaluate` library when it is importable and its metric scripts
    resolve, and falls back to scikit-learn otherwise (offline runs, or a hub
    fetch failure). Both paths compute the same quantities; the record notes
    which one ran.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    hf_evaluate = load_hf_evaluate()
    if hf_evaluate is not None:
        try:
            acc = _hf_metric(hf_evaluate, "accuracy").compute(
                predictions=preds, references=labels)["accuracy"]
            f1 = _hf_metric(hf_evaluate, "f1").compute(
                predictions=preds, references=labels, average="macro")["f1"]
            return {"accuracy": float(acc), "f1_macro": float(f1), "metric_backend": "evaluate"}
        except Exception as exc:
            # Print the message, not just the type: the two realistic causes —
            # no network for the hub fetch, and an `evaluate`/`datasets` version
            # mismatch — are only distinguishable from the message.
            print(f"[metrics] `evaluate` failed, using scikit-learn instead "
                  f"({type(exc).__name__}: {exc})")

    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "metric_backend": "sklearn",
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Preferred over the normal approximation: it stays inside [0, 1] and keeps
    nominal coverage at the accuracies (>0.9) and sample sizes (~10^3) this
    benchmark operates at.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired correctness vectors.

    The right test for "do these two classifiers differ on this test set":
    it conditions on the discordant pairs only, which is exactly the
    information a paired comparison contains. Examples both models get right
    (or both wrong) carry no evidence either way and are discarded by design.
    """
    correct_a = np.asarray(correct_a).astype(bool)
    correct_b = np.asarray(correct_b).astype(bool)
    b = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}

    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    p = min(1.0, 2.0 * tail)
    return {"b": b, "c": c, "n_discordant": n, "p_value": float(p)}


def paired_bootstrap_delta(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile CI for the accuracy difference (a - b), resampling *examples*.

    Paired: each bootstrap replicate draws the same example indices for both
    models, so the CI reflects the disagreement between them rather than the
    sum of two independent accuracy uncertainties. This is why the interval is
    much tighter than differencing two Wilson intervals would suggest.
    """
    correct_a = np.asarray(correct_a).astype(float)
    correct_b = np.asarray(correct_b).astype(float)
    n = len(correct_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = correct_a[idx].mean(axis=1) - correct_b[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta_mean": float(correct_a.mean() - correct_b.mean()),
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "n_boot": n_boot,
    }


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the non-dominated points for (maximize x0, minimize x1).

    Used for (accuracy, latency): a model is on the frontier when no other
    model is simultaneously at least as accurate and at least as fast.
    """
    front: list[int] = []
    for i, (acc_i, lat_i) in enumerate(points):
        dominated = any(
            (acc_j >= acc_i and lat_j <= lat_i) and (acc_j > acc_i or lat_j < lat_i)
            for j, (acc_j, lat_j) in enumerate(points)
            if j != i
        )
        if not dominated:
            front.append(i)
    return front


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def calibration_probe(size: int = 1024, iters: int = 8, warmup: int = 3) -> dict[str, float]:
    """Time a fixed matrix multiply, to record how fast this machine is *right now*.

    `drift_ratio` catches a machine that changes speed **during** one model's
    measurement. It cannot catch one that changes speed **between** two models —
    and that is the failure that actually matters here, because the benchmark's
    whole claim is that every model was measured under the same conditions.

    The probe is the missing control: an identical, model-independent workload
    run beside each model's timing. If two models report different probe
    figures, they were not measured on the same machine, whatever their own
    numbers look like. Contention from another process is the usual cause;
    thermal state and power profile do it too.

    `min` rather than mean: we want the best the machine could manage, which is
    the cleanest estimate of the capacity available at that moment.
    """
    import torch

    a = torch.randn(size, size)
    b = torch.randn(size, size)
    with torch.inference_mode():
        for _ in range(warmup):
            a @ b
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            a @ b
            times.append(time.perf_counter() - t0)

    best = min(times)
    return {
        "calibration_gflops": 2.0 * size**3 / best / 1e9,
        "calibration_ms_min": 1000.0 * best,
        "calibration_ms_median": 1000.0 * statistics.median(times),
        "calibration_size": float(size),
    }


def settle_machine(tolerance: float = 0.05, min_seconds: float = 3.0,
                   max_seconds: float = 45.0, compare_seconds: float = 1.5,
                   verbose: bool = True) -> dict[str, Any]:
    """Load the CPU until its speed stops changing, before anything is timed.

    A processor at idle is not at its sustained clock, and takes seconds of load
    to get there. So the first model measured in a process meets a slower
    machine than every model after it — measured here as 227 GFLOP/s for the
    first against 480-620 for the rest, a 2.7x spread that made the entire
    latency column meaningless. Per-model warmup cannot fix this: the bias lives
    *between* models, not within one.

    Convergence compares two windows **separated in time**, not consecutive
    probes. A first version compared three probes in a row: at ~65 ms per probe
    they span 200 ms, over which even a brisk ramp moves less than the
    tolerance, so a machine climbing steadily from 463 to 690 GFLOP/s was
    declared settled while still climbing. Comparing the last `compare_seconds`
    against the `compare_seconds` before it gives the ramp room to show.

    The figure returned is not comparable to the per-model probes: running back
    to back keeps the operand matrices in cache, which the per-model probe —
    fired after a model has filled the cache with its own tensors — does not
    enjoy. It is a stability signal, not a machine rating.
    """
    start = time.perf_counter()
    history: list[tuple[float, float]] = []  # (elapsed, gflops), in order
    converged = False

    def _median_between(lo: float, hi: float) -> float | None:
        values = [g for t, g in history if lo <= t < hi]
        return statistics.median(values) if len(values) >= 2 else None

    while True:
        gflops = calibration_probe()["calibration_gflops"]
        elapsed = time.perf_counter() - start
        history.append((elapsed, gflops))

        if elapsed >= max(min_seconds, 2 * compare_seconds):
            recent = _median_between(elapsed - compare_seconds, elapsed + 1.0)
            earlier = _median_between(elapsed - 2 * compare_seconds,
                                      elapsed - compare_seconds)
            if recent and earlier and abs(recent / earlier - 1.0) <= tolerance:
                converged = True
                break
        if elapsed >= max_seconds:
            break

    total = time.perf_counter() - start
    tail = [g for t, g in history if t >= total - compare_seconds]
    settled = statistics.median(tail) if tail else float("nan")
    first = history[0][1] if history else float("nan")

    if verbose:
        status = "settled" if converged else f"NOT settled after {max_seconds:.0f}s"
        print(f"  machine {status} at {settled:.0f} GFLOP/s "
              f"(from {first:.0f}, {len(history)} probes, {total:.1f}s)")
        if not converged:
            print("    [warn] the machine never held a steady speed. Either something "
                  "else is running, or it is thermally unstable — timing comparisons "
                  "will not be trustworthy either way.")

    return {
        "settled_gflops": settled,
        "settle_first_gflops": first,
        "settle_rise": settled / first if first else float("nan"),
        "settle_seconds": total,
        "settle_probes": len(history),
        "settle_converged": converged,
    }


class Stopwatch:
    """Wall-clock timer. `perf_counter` because we want elapsed time, not CPU time."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False


def drift_ratio(samples: Sequence[float]) -> float:
    """Median of the last third over the median of the first third, in order.

    A value near 1 means the measurement was stationary. Above ~1.15 it was
    not: on CPU the usual cause is clock frequency dropping under sustained
    vector load, which makes a mean over the whole run a blend of two different
    machines. Reporting it turns "these numbers look odd" into a quantity.
    """
    n = len(samples)
    if n < 6:
        return float("nan")
    third = max(1, n // 3)
    first = statistics.median(samples[:third])
    last = statistics.median(samples[-third:])
    return float(last / first) if first > 0 else float("nan")


def summarize_latencies(samples_s: Sequence[float]) -> dict[str, float]:
    """Order-sensitive statistics on per-call durations.

    `samples_s` must stay in measurement order — `drift_ratio` reads it — so the
    sorted copy is separate.
    """
    ms = [s * 1000.0 for s in samples_s]
    ordered = sorted(ms)
    n = len(ordered)
    return {
        "latency_ms_mean": float(statistics.fmean(ordered)),
        "latency_ms_p50": float(ordered[int(0.50 * (n - 1))]),
        "latency_ms_p95": float(ordered[int(0.95 * (n - 1))]),
        "latency_ms_min": float(ordered[0]),
        "latency_ms_std": float(statistics.pstdev(ordered)) if n > 1 else 0.0,
        "latency_drift": drift_ratio(ms),
        "latency_n": n,
    }


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #
def _cpu_name() -> str:
    name = platform.processor() or ""
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif platform.system() == "Windows":
        env = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if env:
            name = env
    elif platform.system() == "Darwin":
        try:
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return name or "unknown"


def environment_info() -> dict[str, Any]:
    """Everything a reader needs to know before comparing to their own numbers."""
    import torch

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count_logical": os.cpu_count(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    for pkg in ("transformers", "datasets", "evaluate", "accelerate", "numpy", "sklearn"):
        try:
            import importlib.metadata as md

            info[f"{pkg}_version"] = md.version("scikit-learn" if pkg == "sklearn" else pkg)
        except Exception:
            info[f"{pkg}_version"] = None
    return info


def resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def apply_thread_limit(num_threads: int | None) -> int:
    """Pin torch intra-op threads and return the effective value."""
    import torch

    if num_threads is not None:
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()


# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #
def _default(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON-serializable: {type(obj)}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_tag(slug: str, dataset: str, seed: int) -> str:
    """Filename stem that keeps runs across datasets and seeds from colliding."""
    return f"{slug}__{dataset}__seed{seed}"


# --------------------------------------------------------------------------- #
# transformers version compatibility
# --------------------------------------------------------------------------- #
def build_training_arguments(**kwargs):
    """Construct `TrainingArguments`, dropping keys this version doesn't accept.

    `evaluation_strategy` was renamed `eval_strategy` in transformers 4.46 and
    several logging/reporting flags have come and gone. Filtering against the
    live signature keeps the repo runnable across versions instead of pinning
    one exact release.
    """
    import inspect

    from transformers import TrainingArguments

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)

    if "eval_strategy" in kwargs and "eval_strategy" not in accepted:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in accepted:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    dropped = sorted(set(kwargs) - accepted)
    for key in dropped:
        kwargs.pop(key)
    if dropped:
        print(f"[compat] TrainingArguments: ignoring unsupported {dropped}")
    return TrainingArguments(**kwargs)
