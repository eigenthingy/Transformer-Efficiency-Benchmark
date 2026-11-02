"""Measure accuracy and inference cost for every fine-tuned checkpoint.

Writes one `results/eval_<tag>.json` per model plus the per-example predictions
in `results/predictions/<tag>.npz`, which `report.py` needs for the paired
significance tests.

Why this runs in two phases
---------------------------
Accuracy is deterministic: the same weights on the same examples give the same
answer whenever it is computed. Timings are not — a CPU's speed moves with its
clock state, thermal headroom and whatever else the machine is doing.

Measuring model A completely, then model B completely, therefore **confounds**
machine drift with model identity: whatever the machine was doing while A was
timed becomes part of A's number. Measured that way on a fluctuating desktop,
the same model gave 70, 92, 74 and 70 ms across four runs, and DistilBERT's
speedup ranged from 1.6x to 2.5x. The ordering held; the multipliers were noise.

So:

* **Phase 1** scores every model for accuracy, once. Order is irrelevant here.
* **Phase 2** times the models in interleaved rounds — A, B, C, D, E, then again
  — with the **starting position rotated each round**, so no model is always
  measured first. Drift then spreads across all models instead of attaching to
  one, and the spread across rounds becomes a measurement of the uncertainty
  rather than a hidden bias.
* **Phase 3** reports the median across rounds, plus the observed range.

Timing protocol
---------------
Three numbers per round, because they answer different questions:

* **latency at batch=1** — median and p95 ms. What one interactive request
  costs, and what the tail costs.
* **throughput at batch=B** — examples/second, for offline scoring.
* **calibration probe** — a fixed matmul timed beside the model, so the machine
  itself is on the record. See `common.calibration_probe`.

Every measurement is preceded by warmup with both an iteration and a duration
floor, runs under `torch.inference_mode()`, and uses fixed-length padding so all
models process identically shaped tensors.

Examples
--------
    python evaluate.py --smoke
    python evaluate.py --num-threads 12
    python evaluate.py --latency-rounds 5      # tighter timing, more wall clock
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

import common
from config import (
    CHECKPOINTS_DIR,
    DATASETS,
    DEFAULT_MODELS,
    MODELS,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    SMOKE_OVERRIDES,
    BenchConfig,
    smoke_dirs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
    p.add_argument("--dataset", default=BenchConfig.dataset, choices=list(DATASETS))
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=BenchConfig.max_eval_samples,
                   help="-1 for the full split")
    p.add_argument("--batch-size", type=int, default=BenchConfig.eval_batch_size,
                   help="batch size for the throughput measurement and the scoring pass")
    p.add_argument("--seed", type=int, default=BenchConfig.seed)
    p.add_argument("--num-threads", type=int, default=None,
                   help="pin torch intra-op threads; strongly recommended on CPU")
    p.add_argument("--latency-samples", type=int, default=BenchConfig.latency_samples)
    p.add_argument("--latency-rounds", type=int, default=BenchConfig.latency_rounds,
                   help="interleaved timing rounds; the report takes the median and shows "
                        "the range. 1 disables interleaving (fast, but drift becomes bias)")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> BenchConfig:
    cfg = BenchConfig(dataset=args.dataset, eval_batch_size=args.batch_size,
                      seed=args.seed, num_threads=args.num_threads,
                      latency_samples=args.latency_samples,
                      latency_rounds=args.latency_rounds)
    spec = common.get_dataset_spec(args.dataset)
    cfg.max_length = args.max_length if args.max_length is not None else spec.recommended_max_length
    cfg.max_eval_samples = None if args.max_eval_samples == -1 else args.max_eval_samples
    if args.smoke:
        for k, v in SMOKE_OVERRIDES.items():
            setattr(cfg, k, v)
    return cfg


def _features(batch: dict) -> dict:
    """Model inputs only — drop labels so the forward pass computes no loss."""
    return {k: v for k, v in batch.items() if k != "labels"}


def _load_model(ckpt_dir, device: str):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Phase 1 — accuracy
# --------------------------------------------------------------------------- #
def _score_pass(model, dataset, batch_size: int, device: str):
    """Score the whole eval set; return (preds, labels, elapsed_seconds)."""
    import torch

    preds, labels = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            inputs = {k: v.to(device) for k, v in _features(batch).items()}
            logits = model(**inputs).logits
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            labels.append(batch["labels"].numpy())
    elapsed = time.perf_counter() - start
    return np.concatenate(preds), np.concatenate(labels), elapsed


def _timing_inputs(test_ds, cfg: BenchConfig, device: str):
    """Pre-build the tensors both timers consume.

    Built once and kept, so no round pays for tensor construction or host-to-
    device transfer inside a measured region. A few megabytes per model.
    """
    n = min(cfg.latency_samples, len(test_ds))
    examples = [
        {k: v.unsqueeze(0).to(device) for k, v in _features(test_ds[i]).items()}
        for i in range(n)
    ]

    bs = cfg.eval_batch_size
    n_batches = min(cfg.throughput_batches, max(1, len(test_ds) // bs))
    batches = [
        {k: v.to(device) for k, v in _features(test_ds[i * bs : (i + 1) * bs]).items()}
        for i in range(n_batches)
    ]
    return examples, batches


def score_one(name: str, cfg: BenchConfig, device: str) -> dict:
    """Accuracy, parameter counts, and the tensors phase 2 will time."""
    from transformers import AutoTokenizer

    model_spec = common.get_model_spec(name)
    data_spec = common.get_dataset_spec(cfg.dataset)
    tag = common.run_tag(model_spec.slug, cfg.dataset, cfg.seed)
    ckpt_dir = CHECKPOINTS_DIR / tag

    if not (ckpt_dir / "config.json").exists():
        raise FileNotFoundError(f"no checkpoint for {name} at {ckpt_dir}. Run train.py first.")

    print(f"\n{'=' * 72}\n{model_spec.label}\n{'=' * 72}")

    common.set_seed(cfg.seed)
    threads = common.apply_thread_limit(cfg.num_threads)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = _load_model(ckpt_dir, device)

    _, test_raw = common.load_splits(data_spec, 1, cfg.max_eval_samples, cfg.seed)
    trunc = common.truncation_rate(test_raw, tokenizer, data_spec, cfg.max_length)
    _, test_ds = common.tokenize_splits(test_raw.select(range(1)), test_raw,
                                        tokenizer, data_spec, cfg.max_length)

    print(f"  eval examples={len(test_ds)} max_length={cfg.max_length} "
          f"truncated={trunc:.1%} device={device} threads={threads}")

    preds, labels, pass_s = _score_pass(model, test_ds, cfg.eval_batch_size, device)
    metrics = common.compute_metrics(preds, labels)

    correct = (preds == labels).astype(np.int8)
    n_correct = int(correct.sum())
    ci_low, ci_high = common.wilson_interval(n_correct, len(correct))
    print(f"  accuracy={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} "
          f"({metrics['metric_backend']})  95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDICTIONS_DIR / f"{tag}.npz",
                        preds=preds.astype(np.int16),
                        labels=labels.astype(np.int16),
                        correct=correct)

    examples, batches = _timing_inputs(test_ds, cfg, device)
    params = common.count_parameters(model)
    record = {
        "slug": model_spec.slug,
        "label": model_spec.label,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "eval_examples": len(test_ds),
        "eval_truncation_rate": trunc,
        "n_correct": n_correct,
        **metrics,
        "accuracy_ci95_low": ci_low,
        "accuracy_ci95_high": ci_high,
        "full_pass_s": pass_s,
        **params,
        "size_on_disk_mb": common.directory_size_mb(ckpt_dir),
        "architecture": common.architecture_summary(model),
        "max_length": cfg.max_length,
        "device": device,
        "torch_num_threads": threads,
    }
    del model  # reloaded per round in phase 2, so nothing large is held here
    return {"name": name, "tag": tag, "ckpt_dir": ckpt_dir, "record": record,
            "examples": examples, "batches": batches}


# --------------------------------------------------------------------------- #
# Phase 2 — interleaved timing
# --------------------------------------------------------------------------- #
def _warmup(model, inputs: list[dict], min_iters: int, min_seconds: float) -> int:
    """Untimed passes until BOTH a count and a duration floor are met.

    The duration floor matters: a fixed iteration count warms a fast model for
    less wall-clock time than a slow one, biasing the comparison toward whichever
    model warmed least.
    """
    start = time.perf_counter()
    done = 0
    while done < min_iters or (time.perf_counter() - start) < min_seconds:
        model(**inputs[done % len(inputs)])
        done += 1
        if done >= 10_000:  # pathological guard
            break
    return done


def _measure_latency(model, examples: list[dict], cfg: BenchConfig) -> dict:
    import torch

    with torch.inference_mode():
        warmed = _warmup(model, examples, cfg.latency_warmup, cfg.latency_warmup_s)
        samples = []
        for ex in examples:
            t0 = time.perf_counter()
            model(**ex)
            samples.append(time.perf_counter() - t0)
    return {**common.summarize_latencies(samples), "latency_warmup_iters": warmed}


def _measure_throughput(model, batches: list[dict], cfg: BenchConfig) -> dict:
    """Each batch timed individually, so the median is robust and drift visible."""
    import torch

    with torch.inference_mode():
        warmed = _warmup(model, batches, min(cfg.latency_warmup, len(batches)),
                         cfg.throughput_warmup_s)
        per_batch = []
        for b in batches:
            t0 = time.perf_counter()
            model(**b)
            per_batch.append(time.perf_counter() - t0)

    median_s = statistics.median(per_batch)
    bs = cfg.eval_batch_size
    return {
        "throughput_batch_size": bs,
        "throughput_examples_per_s": bs / median_s,
        "throughput_ms_per_batch_p50": 1000.0 * median_s,
        "throughput_drift": common.drift_ratio(per_batch),
        "throughput_batches_timed": len(per_batch),
        "throughput_warmup_iters": warmed,
    }


def _rotated(names: list[str], offset: int) -> list[str]:
    """Rotate the measurement order so no model is always first.

    Within a round the models are still measured one after another, so position
    still carries a small penalty. Rotating spreads that penalty evenly instead
    of concentrating it — with as many rounds as models, each model occupies each
    position exactly once.
    """
    k = offset % len(names)
    return names[k:] + names[:k]


def time_all(prepared: dict[str, dict], cfg: BenchConfig, device: str) -> dict[str, list[dict]]:
    names = list(prepared)
    rounds: dict[str, list[dict]] = {n: [] for n in names}

    for r in range(cfg.latency_rounds):
        order = _rotated(names, r)
        print(f"\n--- timing round {r + 1}/{cfg.latency_rounds} "
              f"({' -> '.join(prepared[n]['record']['label'] for n in order)})")

        for name in order:
            item = prepared[name]
            label = item["record"]["label"]
            model = _load_model(item["ckpt_dir"], device)

            # Reloading does disk I/O, which leaves the CPU idle and clocked
            # down. Settle before reading the probe, or the probe measures the
            # dip rather than the machine.
            calib = common.settle_machine(min_seconds=1.0, max_seconds=8.0,
                                          compare_seconds=0.5, verbose=False)
            latency = _measure_latency(model, item["examples"], cfg)
            throughput = _measure_throughput(model, item["batches"], cfg)

            rounds[name].append({
                **latency,
                **throughput,
                "calibration_gflops": calib["settled_gflops"],
                "calibration_converged": bool(calib["settle_converged"]),
            })
            print(f"    {label:18s} p50={latency['latency_ms_p50']:7.1f}ms  "
                  f"{throughput['throughput_examples_per_s']:6.1f} ex/s  "
                  f"probe {calib['settled_gflops']:.0f} GFLOP/s"
                  f"{'' if calib['settle_converged'] else '  (unsettled!)'}")
            del model

    return rounds


# --------------------------------------------------------------------------- #
# Phase 3 — aggregate across rounds
# --------------------------------------------------------------------------- #
def aggregate_timings(samples: list[dict]) -> dict:
    """Median across rounds, with the observed range.

    The median is the reported value; the range is what makes it honest. A model
    whose p50 ranged 20 to 26 ms across rounds does not have a 20 ms latency, and
    a single number would hide that.
    """
    def med(key: str) -> float:
        return float(statistics.median([s[key] for s in samples]))

    def spread(key: str) -> float:
        values = [s[key] for s in samples]
        return float(max(values) / min(values)) if min(values) > 0 else float("nan")

    p50s = [s["latency_ms_p50"] for s in samples]
    throughputs = [s["throughput_examples_per_s"] for s in samples]

    return {
        "latency_rounds": len(samples),
        "latency_ms_p50": med("latency_ms_p50"),
        "latency_ms_p50_min": float(min(p50s)),
        "latency_ms_p50_max": float(max(p50s)),
        "latency_p50_spread": spread("latency_ms_p50"),
        "latency_ms_p95": med("latency_ms_p95"),
        "latency_ms_mean": med("latency_ms_mean"),
        "latency_ms_min": med("latency_ms_min"),
        "latency_ms_std": med("latency_ms_std"),
        "latency_drift": med("latency_drift"),
        "latency_n": samples[0]["latency_n"],
        "latency_warmup_iters": med("latency_warmup_iters"),
        "throughput_batch_size": samples[0]["throughput_batch_size"],
        "throughput_examples_per_s": med("throughput_examples_per_s"),
        "throughput_min": float(min(throughputs)),
        "throughput_max": float(max(throughputs)),
        "throughput_spread": spread("throughput_examples_per_s"),
        "throughput_ms_per_batch_p50": med("throughput_ms_per_batch_p50"),
        "throughput_drift": med("throughput_drift"),
        "throughput_batches_timed": samples[0]["throughput_batches_timed"],
        "calibration_gflops": med("calibration_gflops"),
        "calibration_spread": spread("calibration_gflops"),
        "calibration_converged": all(s["calibration_converged"] for s in samples),
    }


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    common.enable_utf8_stdout()
    args = parse_args(argv)
    if args.smoke:
        globals().update(smoke_dirs())  # never overwrite a real run
    cfg = build_config(args)

    common.apply_thread_limit(cfg.num_threads)
    device = common.resolve_device()

    # Fetch the metric implementations before anything is timed — the first
    # `evaluate.load` is a ~10 s network round trip, and inside a measured region
    # it leaves the CPU idle long enough to halve the clock. See
    # `common.prefetch_metrics`.
    backend = common.prefetch_metrics()
    print(f"Metrics backend: {backend}")

    # --- phase 1: accuracy ------------------------------------------------ #
    prepared: dict[str, dict] = {}
    for name in args.models:
        try:
            prepared[name] = score_one(name, cfg, device)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")

    if not prepared:
        print("\nNothing evaluated. Train first:  python train.py")
        return 1

    # --- phase 2: interleaved timing -------------------------------------- #
    print(f"\n{'=' * 72}\nTiming — {cfg.latency_rounds} interleaved round(s), order rotated"
          f"\n{'=' * 72}")
    print("Preparing the machine for timing ...")
    settle = common.settle_machine()
    common.write_json(RESULTS_DIR / "environment.json",
                      {**common.environment_info(), **settle,
                       "latency_rounds": cfg.latency_rounds})

    rounds = time_all(prepared, cfg, device)

    # --- phase 3: aggregate and write ------------------------------------- #
    print(f"\n{'=' * 72}\nResults\n{'=' * 72}")
    for name, item in prepared.items():
        timings = aggregate_timings(rounds[name])
        record = {**item["record"], **timings}
        common.write_json(RESULTS_DIR / f"eval_{item['tag']}.json", record)

        spread_note = ""
        if timings["latency_rounds"] > 1:
            spread_note = (f"  (range {timings['latency_ms_p50_min']:.1f}-"
                           f"{timings['latency_ms_p50_max']:.1f}, "
                           f"spread {timings['latency_p50_spread']:.2f}x)")
        print(f"  {record['label']:18s} acc={100 * record['accuracy']:.2f}%  "
              f"p50={timings['latency_ms_p50']:6.1f}ms{spread_note}")

    print("\nNext: python report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
