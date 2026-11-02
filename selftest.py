"""Verify the analysis and plotting layers without training anything.

Runs on `numpy` / `pandas` / `matplotlib` alone — no `transformers`, no
`datasets`, no network — so it can check the parts of the pipeline that are
easy to get quietly wrong:

* the statistics, against values that can be computed by hand;
* the report assembly, on synthetic records;
* the figures, by rendering all three in both modes.

    python selftest.py                    # figures to a temp dir, then discarded
    python selftest.py --out /tmp/figs     # keep them somewhere to look at

Runs every check, then exits non-zero if any failed. This is a smoke test for the reporting
code, not a substitute for `run_benchmark.py --smoke`, which exercises the
Hugging Face path end to end.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import common
import plotting
import report

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}{'  — ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(name)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_statistics() -> None:
    print("statistics")

    lo, hi = common.wilson_interval(90, 100)
    check("Wilson(90/100) matches the published interval",
          close(lo, 0.8256, 5e-4) and close(hi, 0.9448, 5e-4),
          f"[{lo:.4f}, {hi:.4f}] vs [0.8256, 0.9448]")

    lo, hi = common.wilson_interval(50, 100)
    check("Wilson is centred at p when p = 0.5", close((lo + hi) / 2, 0.5, 1e-12))

    lo, hi = common.wilson_interval(100, 100)
    check("Wilson stays inside [0, 1] at p = 1", 0.0 <= lo <= hi <= 1.0, f"[{lo:.4f}, {hi:.4f}]")

    check("Wilson narrows as n grows",
          (common.wilson_interval(900, 1000)[1] - common.wilson_interval(900, 1000)[0])
          < (common.wilson_interval(90, 100)[1] - common.wilson_interval(90, 100)[0]))

    # b = 10, c = 1  ->  p = 2 * sum_{k<=1} C(11,k) / 2^11 = 2 * 12 / 2048
    base = np.array([1] * 10 + [0] * 1 + [1] * 50)
    other = np.array([0] * 10 + [1] * 1 + [1] * 50)
    result = common.mcnemar_exact(base, other)
    expected = 2 * (math.comb(11, 0) + math.comb(11, 1)) / 2**11
    check("McNemar exact matches the hand-computed binomial tail",
          close(result["p_value"], expected, 1e-12),
          f"{result['p_value']:.6f} vs {expected:.6f}")
    check("McNemar counts the discordant pairs correctly",
          result["b"] == 10 and result["c"] == 1 and result["n_discordant"] == 11)

    same = np.array([1, 0, 1, 1, 0, 1] * 20)
    check("McNemar on identical classifiers gives p = 1",
          close(common.mcnemar_exact(same, same)["p_value"], 1.0))
    check("McNemar is symmetric in its arguments",
          close(common.mcnemar_exact(base, other)["p_value"],
                common.mcnemar_exact(other, base)["p_value"]))

    boot = common.paired_bootstrap_delta(same, same, n_boot=500, seed=0)
    check("Paired bootstrap gives a zero-width interval for identical inputs",
          close(boot["delta_ci_low"], 0.0) and close(boot["delta_ci_high"], 0.0))

    rng = np.random.default_rng(7)
    strong = rng.random(4000) < 0.92
    weak = strong.copy()
    flip = rng.choice(np.flatnonzero(strong), size=200, replace=False)
    weak[flip] = False
    boot = common.paired_bootstrap_delta(weak, strong, n_boot=2000, seed=1)
    check("Paired bootstrap brackets a known 5 pp gap",
          boot["delta_ci_low"] < -0.04 < boot["delta_ci_high"] or boot["delta_ci_high"] < -0.04,
          f"delta {boot['delta_mean']:+.4f}, CI "
          f"[{boot['delta_ci_low']:+.4f}, {boot['delta_ci_high']:+.4f}]")
    check("A real 5 pp gap is detected by McNemar",
          common.mcnemar_exact(strong, weak)["p_value"] < 1e-6)

    front = common.pareto_front([(0.90, 100.0), (0.89, 50.0), (0.85, 60.0), (0.90, 120.0)])
    check("Pareto front keeps only the non-dominated points", front == [0, 1], str(front))
    check("Pareto front of a single point is that point",
          common.pareto_front([(0.9, 10.0)]) == [0])

    summary = common.summarize_latencies([0.010, 0.011, 0.012, 0.030])
    check("Latency summary reports the median, not the mean",
          close(summary["latency_ms_p50"], 11.0) and summary["latency_ms_mean"] > 15.0,
          f"p50 {summary['latency_ms_p50']:.1f} ms, mean {summary['latency_ms_mean']:.2f} ms")

    # Drift must read the series IN ORDER; sorting it first would always give 1.
    steady = [0.010] * 12
    slowing = [0.010] * 6 + [0.014] * 6
    check("Drift is ~1 on a stationary series",
          close(common.drift_ratio(steady), 1.0, 1e-9),
          f"{common.drift_ratio(steady):.3f}")
    check("Drift detects a 40% slowdown",
          close(common.drift_ratio(slowing), 1.4, 1e-9),
          f"{common.drift_ratio(slowing):.3f}")
    check("Drift is NaN when there are too few samples to judge",
          common.drift_ratio([0.01, 0.02]) != common.drift_ratio([0.01, 0.02]))
    ordered_input = [0.020] * 6 + [0.010] * 6
    check("Drift survives summarize_latencies without being sorted away",
          common.summarize_latencies(ordered_input)["latency_drift"] < 0.9,
          f"{common.summarize_latencies(ordered_input)['latency_drift']:.3f} (speeding up)")


# --------------------------------------------------------------------------- #
# Synthetic records
# --------------------------------------------------------------------------- #
def synthetic_rows() -> list[dict]:
    """Three plausible records, shaped exactly like `evaluate.py` output.

    Values are invented — this exercises the code paths, it does not stand in
    for a measurement.
    """
    specs = [
        ("bert-base", "BERT-base", 0.9410, 109_483_778, 23_440_896, 12, 768, 418.0, 512.0, 96.0, 148.0, 41.0),
        ("distilbert", "DistilBERT", 0.9375, 66_955_010, 23_440_896, 6, 768, 213.0, 261.0, 189.0, 76.0, 21.0),
        ("minilm-l12", "MiniLM-L12", 0.9330, 33_362_180, 11_720_448, 12, 384, 127.0, 156.0, 318.0, 43.0, 11.0),
    ]
    rows = []
    for (slug, label, acc, params, emb, layers, hidden,
         p50, p95, thr, size_mb, train_s) in specs:
        n = 2000
        n_correct = round(acc * n)
        lo, hi = common.wilson_interval(n_correct, n)
        rows.append({
            "slug": slug, "label": label, "family": "test", "checkpoint": f"test/{slug}",
            "dataset": "ag_news", "seed": 42, "tag": common.run_tag(slug, "ag_news", 42),
            "eval_examples": n, "n_correct": n_correct, "eval_truncation_rate": 0.004,
            "accuracy": acc, "f1_macro": acc - 0.002, "metric_backend": "synthetic",
            "accuracy_ci95_low": lo, "accuracy_ci95_high": hi,
            "latency_ms_p50": p50, "latency_ms_p95": p95, "latency_ms_mean": p50 * 1.05,
            "latency_ms_min": p50 * 0.95, "latency_ms_std": p50 * 0.08, "latency_n": 200,
            "throughput_batch_size": 32, "throughput_examples_per_s": thr,
            "throughput_batches_timed": 20, "throughput_ms_per_batch": 32_000 / thr,
            "full_pass_s": n / thr,
            "params_total": params, "params_trainable": params,
            "params_embedding": emb, "params_non_embedding": params - emb,
            "size_on_disk_mb": size_mb, "size_fp32_theoretical_mb": params * 4 / 1024**2,
            "architecture": {"num_layers": layers, "hidden_size": hidden,
                             "num_attention_heads": hidden // 64,
                             "intermediate_size": hidden * 4, "vocab_size": 30522},
            "max_length": 128, "device": "cpu", "torch_num_threads": 4,
            "train_time_s": train_s * 60, "train_samples_per_s": 4000 / (train_s * 60),
            "train_loss": 0.21, "train_examples": 4000, "learning_rate": 3e-5,
            "uniform_lr": False, "train_truncation_rate": 0.005,
            "note": "synthetic", "size_fp32_theoretical_mb_": None,
        })
    return rows


def _fake_correctness(rows: list[dict], out_dir: Path) -> None:
    """Write correctness vectors consistent with each row's accuracy.

    Nested so the smaller models' mistakes are a superset of the larger ones',
    which is roughly how distilled models behave and gives McNemar a
    well-defined direction to find.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = rows[0]["eval_examples"]
    order = rng.permutation(n)
    for row in rows:
        correct = np.ones(n, dtype=bool)
        n_wrong = n - row["n_correct"]
        correct[order[:n_wrong]] = False
        np.savez_compressed(out_dir / f"{row['tag']}.npz",
                            preds=np.zeros(n, dtype=np.int16),
                            labels=np.zeros(n, dtype=np.int16),
                            correct=correct.astype(np.int8))


def test_machine_probe() -> None:
    """The calibration probe and the settle phase must report a live machine.

    Not a timing assertion — those are unreliable in a test — but a contract
    check: the probe must return a plausible positive throughput, and the settle
    phase must terminate and report whether it converged rather than claiming
    success unconditionally.
    """
    print("\nmachine probe")
    try:
        probe = common.calibration_probe(size=256, iters=3, warmup=1)
    except ImportError:
        check("torch is absent — probe skipped", True)
        return

    check("The probe reports a positive throughput",
          probe["calibration_gflops"] > 0, f"{probe['calibration_gflops']:.0f} GFLOP/s")
    check("The probe reports min <= median",
          probe["calibration_ms_min"] <= probe["calibration_ms_median"] + 1e-9)

    settled = common.settle_machine(min_seconds=0.1, max_seconds=2.0,
                                    compare_seconds=0.2, verbose=False)
    check("The settle phase terminates within its budget",
          settled["settle_seconds"] <= 3.0, f"{settled['settle_seconds']:.2f}s")
    check("The settle phase reports convergence honestly",
          isinstance(settled["settle_converged"], bool),
          f"converged={settled['settle_converged']}, {settled['settle_probes']} probes")
    check("The settle phase records where it started and how far it rose",
          "settle_first_gflops" in settled and "settle_rise" in settled,
          f"rise x{settled['settle_rise']:.2f}")

    # The convergence test must compare windows separated in TIME. Consecutive
    # probes span ~200 ms, over which a steady ramp moves less than the
    # tolerance — which is how a machine climbing 463 -> 690 GFLOP/s was once
    # declared settled while still climbing.
    saved = common.calibration_probe
    try:
        start = time.perf_counter()

        def rising(**_):
            time.sleep(0.005)
            return {"calibration_gflops": 100.0 * (1.0 + 2.0 * (time.perf_counter() - start))}

        common.calibration_probe = rising
        result = common.settle_machine(min_seconds=0.0, max_seconds=0.6,
                                       compare_seconds=0.1, tolerance=0.05,
                                       verbose=False)
        check("A steadily rising machine is NOT declared settled",
              result["settle_converged"] is False,
              f"rose x{result['settle_rise']:.2f} without converging")

        def steady(**_):
            time.sleep(0.005)
            return {"calibration_gflops": 500.0}

        common.calibration_probe = steady
        result = common.settle_machine(min_seconds=0.0, max_seconds=2.0,
                                       compare_seconds=0.1, tolerance=0.05,
                                       verbose=False)
        check("A steady machine IS declared settled", result["settle_converged"] is True,
              f"{result['settle_probes']} probes in {result['settle_seconds']:.2f}s")
    finally:
        common.calibration_probe = saved


def test_evaluate_shim() -> None:
    """The `evaluate` import shim must return the library, not this repo's script.

    Regression guard for a bug that cost a whole run: the shim restored
    `sys.modules["evaluate"]` after loading, so the library's own lazy submodule
    imports resolved to `evaluate.py` in this directory and every metric silently
    fell back to scikit-learn. The numbers stayed correct; the claim in the README
    did not. No network is needed here — loading a metric would fetch from the
    hub, so this checks only that the module resolved is the right one.
    """
    print("\n`evaluate` import shim")
    module = common.load_hf_evaluate()

    if module is None:
        check("`evaluate` is not installed — shim correctly returns None", True,
              "install it to exercise this path")
        return

    path = getattr(module, "__file__", "") or ""
    # Compare against the exact file, not the directory: `.venv` lives inside the
    # repo, so a prefix test would reject the correctly-resolved library.
    local_script = Path(__file__).resolve().parent / "evaluate.py"
    check("The shim resolved the library, not the local evaluate.py",
          bool(path) and Path(path).resolve() != local_script,
          path or "<no __file__>")
    check("The resolved module exposes a callable `load`",
          callable(getattr(module, "load", None)))
    check("`sys.modules['evaluate']` stays bound to the library after loading",
          sys.modules.get("evaluate") is module,
          "popping it is what broke the lazy submodule imports")
    check("The shim is cached, not re-executed", common.load_hf_evaluate() is module)


def test_checkpoint_reuse() -> None:
    """A checkpoint may only be reused when it answers the same question.

    The concrete hazard: `--smoke` trains on 200 examples and writes to the same
    tag as the real run. Reusing it silently would report smoke accuracy as if it
    were the real thing.
    """
    print("\ncheckpoint reuse")
    import argparse as ap

    import train as train_script
    from config import BenchConfig, SMOKE_OVERRIDES

    def args(**kw):
        return ap.Namespace(**{"lr": None, "uniform_lr": False, **kw})

    real = BenchConfig()
    smoke = BenchConfig()
    for key, value in SMOKE_OVERRIDES.items():
        setattr(smoke, key, value)

    real_fp = train_script.planned_fingerprint(real, train_script.resolve_lr("bert-base", args()))
    smoke_fp = train_script.planned_fingerprint(smoke, train_script.resolve_lr("bert-base", args()))

    check("An identical configuration is reused",
          train_script.stale_reasons(real_fp, real_fp) == [])
    reasons = train_script.stale_reasons(real_fp, smoke_fp)
    check("A smoke checkpoint is NOT reused for a real run", reasons != [],
          "; ".join(reasons))
    check("The stale reason names max_train_samples",
          any("max_train_samples" in r for r in reasons))

    for field, value in (("epochs", 2.0), ("max_length", 64),
                         ("train_batch_size", 8), ("seed", 43)):
        changed = BenchConfig()
        setattr(changed, field, value)
        changed_fp = train_script.planned_fingerprint(
            changed, train_script.resolve_lr("bert-base", args()))
        check(f"Changing {field} forces a retrain",
              train_script.stale_reasons(changed_fp, real_fp) != [])

    check("--uniform-lr changes the fingerprint for a non-baseline model",
          train_script.resolve_lr("minilm-l12", args(uniform_lr=True))
          != train_script.resolve_lr("minilm-l12", args()))
    check("--lr overrides both",
          train_script.resolve_lr("minilm-l12", args(lr=7e-5, uniform_lr=True)) == 7e-5)


def test_analysis(tmp: Path) -> list[dict]:
    print("\nanalysis")
    rows = synthetic_rows()

    original = report.PREDICTIONS_DIR
    predictions = tmp / "predictions"
    _fake_correctness(rows, predictions)
    report.PREDICTIONS_DIR = predictions
    try:
        rows, summary = report.analyse(rows, "bert-base", seed=42)
    finally:
        report.PREDICTIONS_DIR = original

    base, distil, mini = rows
    check("The baseline is flagged and retains exactly 100%",
          base["is_baseline"] and close(base["accuracy_retention"], 1.0))
    check("Retention is below 1 for the smaller models",
          distil["accuracy_retention"] < 1.0 and mini["accuracy_retention"] < distil["accuracy_retention"])
    check("Speedup is baseline latency over model latency",
          close(distil["latency_speedup"], 418.0 / 213.0, 1e-9),
          f"{distil['latency_speedup']:.4f}")
    check("Latency reduction agrees with the speedup",
          close(distil["latency_reduction_pct"], 100 * (1 - 1 / distil["latency_speedup"]), 1e-9))
    check("Non-embedding parameter cut exceeds the total cut for DistilBERT",
          distil["param_reduction_non_embedding_pct"] > distil["param_reduction_pct"],
          f"{distil['param_reduction_non_embedding_pct']:.1f}% vs {distil['param_reduction_pct']:.1f}%")
    check("Paired tests ran and produced a p-value",
          "mcnemar_p" in distil and "delta_ci95_low_pp" in distil,
          f"p = {distil.get('mcnemar_p', float('nan')):.4f}")
    check("The delta CI brackets the point estimate",
          distil["delta_ci95_low_pp"] <= distil["delta_accuracy_pp"] <= distil["delta_ci95_high_pp"])
    check("The exchange rate is finite and non-negative",
          distil["accuracy_pp_per_extra_speedup"] >= 0
          and math.isfinite(distil["accuracy_pp_per_extra_speedup"]),
          f"{distil['accuracy_pp_per_extra_speedup']:.3f} pp per extra x")
    check("Every model here is on the frontier (accuracy and latency both ordered)",
          set(summary["pareto_frontier"]) == {"bert-base", "distilbert", "minilm-l12"},
          str(summary["pareto_frontier"]))
    check("A recommendation was produced", bool(summary.get("recommendation", {}).get("label")),
          summary.get("recommendation", {}).get("label", "none"))
    check("Frontier models carry no dominators",
          all(r["dominated_by"] == [] for r in rows))

    # Off the frontier, "beaten" must survive a test on the accuracy side.
    dominated = [
        {"label": "Noisy", "on_pareto_frontier": False,
         "dominated_by": [{"label": "Fast", "accuracy_mcnemar_p": 0.40}]},
        {"label": "Real", "on_pareto_frontier": False,
         "dominated_by": [{"label": "Fast", "accuracy_mcnemar_p": 0.01}]},
    ]
    lines = report.dominance_lines(dominated)
    check("A dominance inside the noise is reported as latency-only",
          len(lines) == 2 and "not statistically detectable" in lines[0])
    check("A significant dominance is reported as both axes",
          len(lines) == 2 and "both axes by Fast" in lines[1])

    import config
    dirs = config.smoke_dirs()
    check("--smoke writes nowhere a real run does",
          all(config.RESULTS_DIR / "smoke" in [p, *p.parents] for p in dirs.values())
          and set(dirs) == {"RESULTS_DIR", "PREDICTIONS_DIR", "CHECKPOINTS_DIR", "PLOTS_DIR"})

    # --- the timing-validity guards -------------------------------------- #
    # Which guard applies depends on the design that produced the data, so both
    # branches need covering.

    # Single-pass: drift and identity are confounded, so cross-model calibration
    # spread is fatal.
    single = [dict(r, latency_rounds=1, calibration_gflops=400.0) for r in rows]
    check("Single pass, equal calibration: no warning",
          report.timing_validity(single) == [])

    contended = [dict(r, latency_rounds=1, calibration_gflops=g)
                 for r, g in zip(rows, [40.0, 380.0, 400.0])]
    warns = report.timing_validity(contended)
    check("Single pass, 10x calibration spread: flagged", len(warns) >= 1)
    check("The warning names both ends of the spread",
          bool(warns) and "BERT-base" in warns[0] and "MiniLM-L12" in warns[0])
    check("The warning says accuracy is unaffected",
          bool(warns) and "deterministic" in warns[0])
    check("The warning points at the fix (interleaving)",
          bool(warns) and "latency-rounds" in warns[0])

    borderline = [dict(r, latency_rounds=1, calibration_gflops=g)
                  for r, g in zip(rows, [400.0, 420.0, 430.0])]
    check("Single pass, 7.5% jitter: under tolerance",
          report.timing_validity(borderline) == [],
          f"tolerance {100 * report.CALIBRATION_TOLERANCE:.0f}%")

    # Interleaved: cross-model calibration spread is expected and harmless; what
    # matters is each model's own spread across rounds.
    interleaved = [dict(r, latency_rounds=3, calibration_gflops=g,
                        latency_p50_spread=1.05,
                        latency_ms_p50_min=r["latency_ms_p50"] * 0.98,
                        latency_ms_p50_max=r["latency_ms_p50"] * 1.03)
                   for r, g in zip(rows, [40.0, 380.0, 400.0])]
    check("Interleaved: a big calibration spread is NOT fatal",
          report.timing_validity(interleaved) == [],
          "rotation removes the confound, so the probe spread is expected")

    noisy = [dict(r, latency_rounds=3, calibration_gflops=400.0,
                  latency_p50_spread=1.60,
                  latency_ms_p50_min=r["latency_ms_p50"] * 0.8,
                  latency_ms_p50_max=r["latency_ms_p50"] * 1.28)
             for r in rows]
    warns = report.timing_validity(noisy)
    check("Interleaved: a wide per-model round spread IS flagged", len(warns) >= 1)
    check("That warning keeps the ordering claim while rejecting the point estimate",
          bool(warns) and "ordering is" in warns[0] and "range" in warns[0])

    check("Results with no calibration data say so",
          any("No calibration data" in w
              for w in report.timing_validity([dict(r, latency_rounds=1) for r in rows])))

    markdown = report.build_markdown(rows, summary, {"device": "cpu", "cpu": "test"}, {}, [])
    for fragment in ("# Benchmark results", "## Headline", "## The answer",
                     "## Verdict", "## Relative to the baseline", "## Method notes"):
        check(f"Report contains '{fragment}'", fragment in markdown)
    check("Report renders a well-formed markdown table",
          markdown.count("|") > 40 and "| :" in markdown or "|:" in markdown)
    check("No unformatted placeholders leaked into the report",
          "{" not in markdown.replace("{}", ""), "")
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_figures(rows: list[dict], out_dir: Path) -> None:
    print("\nfigures")
    frontier = {r["slug"] for r in rows if r["on_pareto_frontier"]}
    latency_label = "Latency, batch = 1  (ms, median)"

    compressed = [r for r in rows if not r["is_baseline"]]
    specs = [
        ("accuracy_vs_latency",
         lambda th: plotting.fig_pareto(rows, frontier, th, cmap=plotting.color_map(rows, th),
                                        latency_key="latency_ms_p50", latency_label=latency_label)),
        ("retention_vs_speedup",
         lambda th: plotting.fig_tradeoff(compressed, th, cmap=plotting.color_map(rows, th),
                                          baseline_label="BERT-base")),
        ("cost_panels",
         lambda th: plotting.fig_cost_panels(rows, th, cmap=plotting.color_map(rows, th),
                                             latency_key="latency_ms_p50", latency_label=latency_label)),
    ]
    for stem, builder in specs:
        try:
            written = plotting.render(builder, out_dir, stem)
        except Exception as exc:  # noqa: BLE001 - the failure is the result
            check(f"{stem} renders", False, f"{type(exc).__name__}: {exc}")
            continue
        sizes = [p.stat().st_size for p in written]
        check(f"{stem} renders in light and dark",
              len(written) == 2 and all(s > 8_000 for s in sizes),
              " · ".join(f"{p.name} {s // 1024} KB" for p, s in zip(written, sizes)))

    # A fourth model must not silently seat a fourth categorical hue.
    theme = plotting.THEMES["light"]
    check("Palette holds three validated slots and folds past that",
          len(set(plotting.series_colors(3, theme))) == 3
          and len(set(plotting.series_colors(4, theme))) == 1)

    # Colour must follow the model, not its row index in a filtered panel.
    full = plotting.color_map(rows, theme)
    partial = plotting.color_map(compressed, theme)
    check("A filtered panel keeps each model's colour when the map is shared",
          plotting._colors_for(compressed, theme, full) == [full[r["slug"]] for r in compressed]
          and partial[compressed[0]["slug"]] != full[compressed[0]["slug"]],
          "shared map holds; an unshared map would recolour")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None,
                        help="keep the rendered figures here instead of a temp dir")
    args = parser.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="hf-bench-selftest-"))
    figures_dir = args.out or (tmp / "plots")
    try:
        test_statistics()
        test_machine_probe()
        test_evaluate_shim()
        test_checkpoint_reuse()
        rows = test_analysis(tmp)
        test_figures(rows, figures_dir)
    finally:
        if args.out is None:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            shutil.rmtree(tmp / "predictions", ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    if args.out:
        print(f"Figures kept in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
