"""Aggregate the per-model results into a table, figures and a written analysis.

Reads whatever `train.py` / `evaluate.py` left in `results/` and produces:

    results/benchmark.csv     one row per model, every raw measurement
    results/benchmark.md      the tables + the quantitative analysis
    results/analysis.json     the same analysis, machine-readable
    plots/*.png               three figures, light and dark

The analysis is the point of this file. A benchmark that only ranks models
answers "which is biggest"; the question is "what does the smaller one cost
you", which needs an effect size, an interval, and a test:

* **Accuracy retention** — the share of baseline accuracy that survives.
* **Paired bootstrap CI** on the accuracy difference. Paired because every
  model saw the identical eval subset, so we resample examples, not models.
* **Exact McNemar test** on the discordant pairs — the correct test for two
  classifiers on one test set. It answers whether a gap is distinguishable
  from sampling noise at this eval size.
* **Elasticity** — accuracy points surrendered per additional × of speed. The
  exchange rate, which is what an engineering decision actually turns on.

Usage
-----
    python report.py
    python report.py --dataset imdb
    python report.py --seeds 42 43 44        # also report seed-to-seed spread
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common
import plotting
from config import (
    BASELINE,
    DATASETS,
    MODELS,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    ROOT,
    BenchConfig,
    smoke_dirs,
)

LATENCY_KEY = "latency_ms_p50"
LATENCY_LABEL = "Latency, batch = 1  (ms, median)"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_rows(dataset: str, seed: int) -> list[dict]:
    """Join the train and eval records for every model that has both."""
    rows = []
    for name, spec in MODELS.items():
        tag = common.run_tag(spec.slug, dataset, seed)
        eval_path = RESULTS_DIR / f"eval_{tag}.json"
        train_path = RESULTS_DIR / f"train_{tag}.json"
        if not eval_path.exists():
            continue
        row = common.read_json(eval_path)
        if train_path.exists():
            train = common.read_json(train_path)
            for key in ("train_time_s", "train_samples_per_s", "train_loss",
                        "train_examples", "learning_rate", "uniform_lr",
                        "size_fp32_theoretical_mb", "family", "note",
                        "checkpoint", "train_truncation_rate"):
                row[key] = train.get(key)
        row.setdefault("train_time_s", float("nan"))
        row["tag"] = tag
        rows.append(row)

    order = list(MODELS)
    rows.sort(key=lambda r: order.index(r["slug"]) if r["slug"] in order else 99)
    return rows


def load_correctness(tag: str) -> np.ndarray | None:
    path = PREDICTIONS_DIR / f"{tag}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["correct"].astype(bool)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(rows: list[dict], baseline_slug: str, seed: int) -> tuple[list[dict], dict]:
    """Attach every relative and inferential quantity to each row."""
    base = next((r for r in rows if r["slug"] == baseline_slug), None)
    if base is None:
        base = rows[0]
        print(f"[warn] baseline '{baseline_slug}' not found; using {base['label']}")

    base_correct = load_correctness(base["tag"])

    for row in rows:
        row["is_baseline"] = row["slug"] == base["slug"]

        row["accuracy_retention"] = row["accuracy"] / base["accuracy"]
        row["f1_retention"] = row["f1_macro"] / base["f1_macro"]
        row["delta_accuracy_pp"] = 100.0 * (row["accuracy"] - base["accuracy"])

        row["latency_speedup"] = base[LATENCY_KEY] / row[LATENCY_KEY]
        row["latency_reduction_pct"] = 100.0 * (1.0 - row[LATENCY_KEY] / base[LATENCY_KEY])
        row["throughput_gain"] = (
            row["throughput_examples_per_s"] / base["throughput_examples_per_s"]
        )

        if base.get("train_time_s") and row.get("train_time_s") == row.get("train_time_s"):
            row["train_speedup"] = base["train_time_s"] / row["train_time_s"]
            row["train_time_reduction_pct"] = 100.0 * (1.0 - row["train_time_s"] / base["train_time_s"])
        else:
            row["train_speedup"] = float("nan")
            row["train_time_reduction_pct"] = float("nan")

        row["param_reduction_pct"] = 100.0 * (1.0 - row["params_total"] / base["params_total"])
        row["param_reduction_non_embedding_pct"] = 100.0 * (
            1.0 - row["params_non_embedding"] / base["params_non_embedding"]
        )
        row["size_reduction_pct"] = 100.0 * (1.0 - row["size_on_disk_mb"] / base["size_on_disk_mb"])

        # --- paired inference against the baseline ------------------------ #
        row_correct = load_correctness(row["tag"])
        if base_correct is not None and row_correct is not None and not row["is_baseline"]:
            if len(base_correct) != len(row_correct):
                print(f"[warn] {row['label']}: eval sets differ in size; skipping paired tests")
            else:
                boot = common.paired_bootstrap_delta(row_correct, base_correct, seed=seed)
                mcnemar = common.mcnemar_exact(base_correct, row_correct)
                row["delta_ci95_low_pp"] = 100.0 * boot["delta_ci_low"]
                row["delta_ci95_high_pp"] = 100.0 * boot["delta_ci_high"]
                row["mcnemar_p"] = mcnemar["p_value"]
                row["mcnemar_b_baseline_only_correct"] = mcnemar["b"]
                row["mcnemar_c_model_only_correct"] = mcnemar["c"]
                row["mcnemar_n_discordant"] = mcnemar["n_discordant"]
                row["significant_at_05"] = bool(mcnemar["p_value"] < 0.05)

        # --- the exchange rate -------------------------------------------- #
        loss_pp = max(0.0, -row["delta_accuracy_pp"])
        extra_speed = row["latency_speedup"] - 1.0
        row["accuracy_pp_per_extra_speedup"] = (
            loss_pp / extra_speed if extra_speed > 0.01 else float("nan")
        )
        row["accuracy_pp_per_10pct_params_cut"] = (
            loss_pp / (row["param_reduction_pct"] / 10.0)
            if row["param_reduction_pct"] > 0.5 else float("nan")
        )

    front_idx = common.pareto_front([(r["accuracy"], r[LATENCY_KEY]) for r in rows])
    frontier = {rows[i]["slug"] for i in front_idx}
    for row in rows:
        row["on_pareto_frontier"] = row["slug"] in frontier

    # The frontier is drawn on point estimates. Latency gaps are large and
    # stable; accuracy gaps often are not. So for every model off the frontier,
    # test its accuracy against each model that dominates it — otherwise
    # "beaten on both axes" would rank inside the noise.
    correctness = {r["slug"]: load_correctness(r["tag"]) for r in rows}
    for row in rows:
        row["dominated_by"] = []
        if row["on_pareto_frontier"]:
            continue
        for other in sorted(rows, key=lambda r: r[LATENCY_KEY]):
            if other is row or not (other["accuracy"] >= row["accuracy"]
                                    and other[LATENCY_KEY] <= row[LATENCY_KEY]):
                continue
            a, b = correctness[other["slug"]], correctness[row["slug"]]
            p = (common.mcnemar_exact(a, b)["p_value"]
                 if a is not None and b is not None and len(a) == len(b) else None)
            row["dominated_by"].append({"slug": other["slug"], "label": other["label"],
                                        "accuracy_mcnemar_p": p})

    summary = {
        "baseline": base["slug"],
        "baseline_label": base["label"],
        "dataset": rows[0]["dataset"],
        "eval_examples": rows[0]["eval_examples"],
        "pareto_frontier": sorted(frontier),
        "recommendation": recommend(rows),
    }
    return rows, summary


def recommend(rows: list[dict]) -> dict:
    """Pick the fastest model whose accuracy gap is not statistically detectable.

    "Not detectable at this eval size" is a weaker claim than "equal", and the
    recommendation says so. Falling back to the frontier point with the best
    retention keeps an answer available when every gap is significant.
    """
    candidates = [
        r for r in rows
        if not r["is_baseline"] and r.get("significant_at_05") is False
        and r["latency_speedup"] > 1.0
    ]
    if candidates:
        pick = max(candidates, key=lambda r: r["latency_speedup"])
        return {
            "slug": pick["slug"],
            "label": pick["label"],
            "basis": "fastest model with no statistically detectable accuracy gap (α = 0.05)",
            "speedup": pick["latency_speedup"],
            "retention": pick["accuracy_retention"],
            "mcnemar_p": pick.get("mcnemar_p"),
        }
    non_base = [r for r in rows if not r["is_baseline"]]
    if not non_base:
        return {}
    pick = max((r for r in non_base if r["on_pareto_frontier"]), key=lambda r: r["accuracy_retention"],
               default=max(non_base, key=lambda r: r["accuracy_retention"]))
    return {
        "slug": pick["slug"],
        "label": pick["label"],
        "basis": "every gap is statistically detectable; best retention on the frontier",
        "speedup": pick["latency_speedup"],
        "retention": pick["accuracy_retention"],
        "mcnemar_p": pick.get("mcnemar_p"),
    }


def seed_variation(dataset: str, seeds: list[int]) -> dict[str, dict]:
    """Mean and spread of accuracy across seeds, where more than one seed was run.

    Without this, a 0.4 pp gap between two models is uninterpretable — it may
    be smaller than the run-to-run variation of either one.
    """
    out: dict[str, dict] = {}
    for name, spec in MODELS.items():
        accs = []
        for s in seeds:
            path = RESULTS_DIR / f"eval_{common.run_tag(spec.slug, dataset, s)}.json"
            if path.exists():
                accs.append(common.read_json(path)["accuracy"])
        if len(accs) > 1:
            out[spec.slug] = {
                "label": spec.label,
                "n_seeds": len(accs),
                "accuracy_mean": statistics.fmean(accs),
                "accuracy_std": statistics.stdev(accs),
                "accuracy_min": min(accs),
                "accuracy_max": max(accs),
            }
    return out


# --------------------------------------------------------------------------- #
# Prose
# --------------------------------------------------------------------------- #
def _p_str(p: float | None) -> str:
    if p is None:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def headline_sentence(row: dict, baseline_label: str) -> str:
    """The one sentence this repo exists to produce, for a single model."""
    parts = [
        f"**{row['label']}** preserves **{100 * row['accuracy_retention']:.1f}%** "
        f"of {baseline_label}'s accuracy"
    ]
    if "delta_ci95_low_pp" in row:
        parts.append(
            f"(Δ = {row['delta_accuracy_pp']:+.2f} pp, 95% CI "
            f"[{row['delta_ci95_low_pp']:+.2f}, {row['delta_ci95_high_pp']:+.2f}] pp; "
            f"McNemar p {_p_str(row.get('mcnemar_p'))})"
        )
    else:
        parts.append(f"(Δ = {row['delta_accuracy_pp']:+.2f} pp)")

    gains = [f"cutting p50 inference latency by **{row['latency_reduction_pct']:.1f}%** "
             f"({row['latency_speedup']:.2f}× faster)"]
    if row["train_time_reduction_pct"] == row["train_time_reduction_pct"]:
        gains.append(f"training time by {row['train_time_reduction_pct']:.1f}%")
    gains.append(f"parameters by {row['param_reduction_pct']:.1f}%")
    gains.append(f"on-disk size by {row['size_reduction_pct']:.1f}%")

    return parts[0] + " " + parts[1] + " while " + ", ".join(gains) + "."


def dominance_lines(rows: list[dict]) -> list[str]:
    """One bullet per model off the frontier: who beats it, and whether on accuracy too."""
    lines = []
    for row in rows:
        doms = row.get("dominated_by") or []
        if row["on_pareto_frontier"] or not doms:
            continue
        who = ", ".join(f"{d['label']} (accuracy McNemar p {_p_str(d['accuracy_mcnemar_p'])})"
                        for d in doms)
        real = [d["label"] for d in doms
                if d["accuracy_mcnemar_p"] is not None and d["accuracy_mcnemar_p"] < 0.05]
        if real:
            verdict = f"beaten on both axes by {', '.join(real)}."
        else:
            verdict = ("beaten on latency; its accuracy deficit is not statistically "
                       "detectable, so on accuracy it is not shown to be worse.")
        lines.append(f"- **{row['label']}**, dominated by {who}: {verdict}")
    return lines


def verdict_sentence(row: dict, n_eval: int) -> str:
    sig = row.get("significant_at_05")
    if sig is None:
        return "No paired predictions on disk, so no significance test was run."
    if sig:
        return (
            f"The gap is statistically detectable at this eval size "
            f"(n = {n_eval:,}, McNemar p {_p_str(row.get('mcnemar_p'))}): "
            f"{row['mcnemar_b_baseline_only_correct']} examples only the baseline got right "
            f"against {row['mcnemar_c_model_only_correct']} only this model got right."
        )
    return (
        f"The gap is **not** statistically detectable at this eval size "
        f"(n = {n_eval:,}, McNemar p {_p_str(row.get('mcnemar_p'))}): "
        f"{row['mcnemar_b_baseline_only_correct']} vs "
        f"{row['mcnemar_c_model_only_correct']} discordant examples. "
        f"That is 'no evidence of a difference', not 'no difference' — a larger eval "
        f"set could still resolve one."
    )


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def md_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["left"] + ["right"] * (len(headers) - 1)
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]

    def fmt_row(cells: list[str]) -> str:
        out = []
        for cell, width, a in zip(cells, widths, align):
            out.append(cell.ljust(width) if a == "left" else cell.rjust(width))
        return "| " + " | ".join(out) + " |"

    sep = "| " + " | ".join(
        (":" + "-" * (w - 1)) if a == "left" else ("-" * (w - 1) + ":")
        for w, a in zip(widths, align)
    ) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])


def _duration(seconds: float) -> str:
    """Human-readable wall clock. Minutes and hours read faster than 4-digit seconds."""
    if seconds != seconds:  # NaN
        return "—"
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.0f} min" if minutes >= 10 else f"{minutes:.1f} min"
    total = int(round(seconds))
    return f"{total // 3600} h {(total % 3600) // 60:02d} min"


def headline_table(rows: list[dict]) -> str:
    body = []
    for r in rows:
        body.append([
            r["label"] + (" *(baseline)*" if r["is_baseline"] else ""),
            f"{100 * r['accuracy']:.2f}%",
            _duration(r["train_time_s"]),
            f"{r[LATENCY_KEY]:.1f} ms",
            f"{r['params_total'] / 1e6:.1f} M",
            f"{r['size_on_disk_mb']:.0f} MB",
        ])
    return md_table(["Model", "Accuracy", "Train time", "Inference (p50)", "Params", "Size"], body)


def detail_table(rows: list[dict]) -> str:
    body = []
    for r in rows:
        body.append([
            r["label"],
            f"{100 * r['accuracy']:.2f}",
            f"[{100 * r['accuracy_ci95_low']:.2f}, {100 * r['accuracy_ci95_high']:.2f}]",
            f"{100 * r['f1_macro']:.2f}",
            f"{r[LATENCY_KEY]:.1f}",
            f"{r['latency_ms_p95']:.1f}",
            (f"{r['latency_ms_p50_min']:.1f}–{r['latency_ms_p50_max']:.1f}"
             if r.get("latency_ms_p50_min") is not None and (r.get("latency_rounds") or 1) > 1
             else "—"),
            f"{r['throughput_examples_per_s']:.1f}",
            f"{r['params_non_embedding'] / 1e6:.1f}",
            "yes" if r["on_pareto_frontier"] else "no",
        ])
    return md_table(
        ["Model", "Acc (%)", "Acc 95% CI", "F1 macro (%)", "p50 (ms)", "p95 (ms)",
         "p50 range over rounds", "Throughput (ex/s)", "Non-emb. params (M)", "On frontier"],
        body,
    )


def relative_table(rows: list[dict], baseline_label: str) -> str:
    body = []
    for r in rows:
        if r["is_baseline"]:
            continue
        body.append([
            r["label"],
            f"{100 * r['accuracy_retention']:.2f}%",
            f"{r['delta_accuracy_pp']:+.2f}",
            (f"[{r['delta_ci95_low_pp']:+.2f}, {r['delta_ci95_high_pp']:+.2f}]"
             if "delta_ci95_low_pp" in r else "—"),
            _p_str(r.get("mcnemar_p")).replace("= ", ""),
            f"{r['latency_speedup']:.2f}×",
            f"{r['train_speedup']:.2f}×" if r["train_speedup"] == r["train_speedup"] else "—",
            f"{r['param_reduction_pct']:.1f}%",
            (f"{r['accuracy_pp_per_extra_speedup']:.3f}"
             if r["accuracy_pp_per_extra_speedup"] == r["accuracy_pp_per_extra_speedup"] else "—"),
        ])
    return md_table(
        ["Model", "Acc retained", "Δ acc (pp)", "Δ 95% CI (pp)", "McNemar p",
         "Speedup", "Train speedup", "Params cut", "pp / extra ×"],
        body,
    )


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
#: Relative spread in the calibration probe above which the models were not
#: measured under the same conditions. 10% is well beyond normal jitter on an
#: idle machine and well below the factor seen under real contention.
CALIBRATION_TOLERANCE = 0.10


#: A model's own p50 may vary this much across interleaved rounds before the
#: figure stops deserving the precision the tables print it with.
ROUND_SPREAD_TOLERANCE = 1.25


def timing_validity(rows: list[dict]) -> list[str]:
    """Warnings about whether the timing columns can be trusted, and how far.

    Which check applies depends on the experimental design that produced the
    data, so the two cases are handled separately rather than blended:

    * **Interleaved (`latency_rounds > 1`)** — every model was timed in every
      round with the order rotated, so machine drift no longer maps onto model
      identity. Cross-model calibration spread is then expected and harmless;
      what matters is how much each model's *own* figure moved between rounds,
      because that is the real uncertainty on it.
    * **Single pass (`latency_rounds <= 1`)** — models were measured one after
      another, so drift and identity are confounded. The calibration probe is
      the only guard available, and a spread across models invalidates every
      relative speed figure.

    Returned as blockquotes placed *above* the tables: a reader must meet the
    caveat before the number, not after.
    """
    warnings: list[str] = []
    rounds = max((r.get("latency_rounds") or 1) for r in rows) if rows else 1

    if rounds > 1:
        wide = [(r["label"], r["latency_p50_spread"], r.get("latency_ms_p50_min"),
                 r.get("latency_ms_p50_max")) for r in rows
                if isinstance(r.get("latency_p50_spread"), float)
                and r["latency_p50_spread"] == r["latency_p50_spread"]
                and r["latency_p50_spread"] > ROUND_SPREAD_TOLERANCE]
        if wide:
            warnings.append(
                f"> **Latency figures are noisy on this machine.** Across "
                f"{rounds} interleaved rounds, these models' median latency moved by more "
                f"than {100 * (ROUND_SPREAD_TOLERANCE - 1):.0f}%: "
                + ", ".join(f"{label} ({lo:.1f}–{hi:.1f} ms, {s:.2f}×)"
                            for label, s, lo, hi in wide)
                + ". The reported value is the median across rounds and the ordering is "
                "reliable, but quote the range rather than the point estimate, and prefer "
                "more rounds (`--latency-rounds 7`) on a quiet machine."
            )
    else:
        probes = [(r["label"], r["calibration_gflops"]) for r in rows
                  if isinstance(r.get("calibration_gflops"), (int, float))
                  and r["calibration_gflops"] == r["calibration_gflops"]]
        if len(probes) >= 2:
            lo_label, lo = min(probes, key=lambda p: p[1])
            hi_label, hi = max(probes, key=lambda p: p[1])
            if lo > 0 and (hi / lo - 1.0) > CALIBRATION_TOLERANCE:
                warnings.append(
                    f"> **The timing columns are not comparable.** Models were timed one "
                    f"after another, and an identical calibration workload run beside each "
                    f"returned {lo:.0f} GFLOP/s for {lo_label} against {hi:.0f} for "
                    f"{hi_label} — a {100 * (hi / lo - 1):.0f}% spread. Machine drift is "
                    f"therefore indistinguishable from architecture in the latency, "
                    f"speedup, exchange-rate and Pareto columns. Accuracy and every "
                    f"statistic derived from it are unaffected: predictions are "
                    f"deterministic. Re-run with `--latency-rounds 3` or more, which "
                    f"interleaves the models and removes the confound."
                )
        elif not probes:
            warnings.append(
                "> **No calibration data.** These results predate the calibration probe, so "
                "there is no evidence that the models were measured under the same machine "
                "conditions. Re-run `evaluate.py` to obtain it."
            )

    unstable = [(r["label"], r["throughput_drift"]) for r in rows
                if isinstance(r.get("throughput_drift"), float)
                and r["throughput_drift"] == r["throughput_drift"]
                and abs(r["throughput_drift"] - 1.0) > 0.15]
    if unstable:
        warnings.append(
            "> **Throughput stability.** Batch times changed measurably during the "
            "throughput measurement for "
            + ", ".join(f"{label} ({100 * (d - 1):+.0f}%)" for label, d in unstable)
            + ". On CPU this is usually clock frequency moving under sustained vector "
            "load. The batch-1 latency column is less exposed, being measured in short "
            "bursts."
        )
    return warnings


def build_markdown(rows: list[dict], summary: dict, env: dict,
                   variation: dict[str, dict], figures: list[str]) -> str:
    data_spec = common.get_dataset_spec(summary["dataset"])
    base_label = summary["baseline_label"]
    n_eval = summary["eval_examples"]
    ref = rows[0]

    device = env.get("device", "?")
    hw = env.get("gpu") or env.get("cpu", "?")

    # Each entry is one markdown block; they are joined with a blank line, so
    # consecutive paragraphs never collapse into one on render.
    blocks: list[str] = [
        "# Benchmark results",
        (f"Task: **{data_spec.hf_name}** ({data_spec.num_labels} classes) · "
         f"train {ref.get('train_examples', '?')} / eval {n_eval:,} examples · "
         f"max_length {ref['max_length']} · seed {ref['seed']}"),
        (f"Hardware: **{device}** — {hw} · {env.get('cpu_count_logical', '?')} logical cores · "
         f"torch {env.get('torch', '?')} with {ref.get('torch_num_threads', '?')} intra-op "
         f"threads · transformers {env.get('transformers_version', '?')}"),
        ("_Generated by `report.py`. Absolute times are hardware-specific; the relative "
         "columns are what transfer to another machine._"),
        # Caveats come before the numbers, never after.
        *timing_validity(rows),
        "## Headline",
        headline_table(rows),
        "## The answer",
    ]

    for row in rows:
        if row["is_baseline"]:
            continue
        blocks.append(f"### {row['label']}")
        blocks.append(headline_sentence(row, base_label))
        blocks.append(verdict_sentence(row, n_eval))
        if row["accuracy_pp_per_extra_speedup"] == row["accuracy_pp_per_extra_speedup"]:
            # The rate divides a point estimate of the accuracy loss; when that
            # loss is inside the noise, so is the rate, and ranking on it would
            # break the rule the rest of the report keeps.
            noise = (
                " The accuracy loss behind it is not statistically detectable, so neither "
                "is this rate: it is not evidence that this model is a worse deal than one "
                "with a lower rate."
                if row.get("significant_at_05") is False and row["delta_accuracy_pp"] < 0 else ""
            )
            blocks.append(
                f"Exchange rate: **{row['accuracy_pp_per_extra_speedup']:.3f} accuracy points "
                f"per additional × of inference speed** "
                f"(and {row['accuracy_pp_per_10pct_params_cut']:.3f} pp per 10% of "
                f"parameters removed).{noise}"
            )
        blocks.append(
            f"Parameter accounting: total -{row['param_reduction_pct']:.1f}%, but "
            f"-{row['param_reduction_non_embedding_pct']:.1f}% once the embedding table is "
            f"excluded. Embeddings are a lookup, not compute, so the non-embedding figure is "
            f"the one that tracks latency."
        )

    rec = summary.get("recommendation") or {}
    if rec:
        blocks.append("## Verdict")
        blocks.append(
            f"On this task and this hardware, **{rec['label']}** is the pick: "
            f"{100 * rec['retention']:.1f}% of {base_label}'s accuracy at "
            f"{rec['speedup']:.2f}× the inference speed. Basis: {rec['basis']}."
        )
        blocks.append(
            f"Pareto frontier (accuracy ↑, latency ↓): "
            f"**{', '.join(r['label'] for r in rows if r['on_pareto_frontier'])}**. "
            "The frontier is drawn on point estimates, so each model off it is tested "
            "against the models that dominate it:"
        )
        blocks.append("\n".join(dominance_lines(rows)))

    blocks += [
        "## Detail",
        detail_table(rows),
        "## Relative to the baseline",
        relative_table(rows, base_label),
        ("`pp / extra ×` is the exchange rate: accuracy points lost for each additional "
         "multiple of inference speed. Lower is a better deal; a blank means the model was "
         "not faster than the baseline. Compare rates only between models whose accuracy "
         "loss is significant: a rate built on a loss inside the noise is itself noise."),
        "## Seed-to-seed variation",
    ]

    if variation:
        body = [[
            v["label"], str(v["n_seeds"]), f"{100 * v['accuracy_mean']:.2f}",
            f"{100 * v['accuracy_std']:.2f}",
            f"{100 * v['accuracy_min']:.2f} – {100 * v['accuracy_max']:.2f}",
        ] for v in variation.values()]
        worst = max(v["accuracy_std"] for v in variation.values())
        blocks.append(md_table(["Model", "Seeds", "Mean acc (%)", "Std (pp)", "Range (%)"], body))
        blocks.append(
            f"Largest run-to-run standard deviation: **{100 * worst:.2f} pp**. Treat any "
            "between-model gap smaller than this as unresolved by these runs."
        )
    else:
        blocks.append(
            "Single seed. Fine-tuning variance on a run this size is typically a few tenths "
            "of a point, which is the same order as the smaller gaps above — so run "
            "`run_benchmark.py --seeds 42 43 44` before treating a sub-half-point difference "
            "as real."
        )

    if figures:
        blocks.append("## Figures")
        for fig in figures:
            blocks.append(f"![{Path(fig).stem}]({fig})")

    lr_note = ("Learning rate is held constant across models (`--uniform-lr`)."
               if ref.get("uniform_lr") else
               "Each model uses its own recommended learning rate (see `config.py`); "
               "pass `--uniform-lr` to hold it constant instead.")
    blocks.append("## Method notes")
    blocks.append("\n".join([
        "- Every model trains and is evaluated on byte-identical subsets (shuffle seeded, "
        "then `select`), which is what licenses the paired tests.",
        '- Training time excludes evaluation: `eval_strategy="no"` during `train.py`.',
        "- Latency is measured after warmup passes, under `torch.inference_mode()`, with "
        "fixed-length padding so every model sees identically shaped tensors.",
        "- Warmup has both a count and a duration floor. A fixed iteration count warms a "
        "fast model for less wall-clock time than a slow one, which on CPU biases the "
        "comparison toward whichever model warmed least.",
        f"- Timings come from {max((r.get('latency_rounds') or 1) for r in rows)} "
        "interleaved round(s): every round times every model, with the starting position "
        "rotated so no model is always measured first. The reported figure is the median "
        "across rounds; the detail table shows the range. Measuring one model to "
        "completion before starting the next would make machine drift indistinguishable "
        "from architecture.",
        "- Throughput times each batch individually and reports the median, plus a drift "
        "ratio (last third over first third). Drift near 1 means the machine held its "
        "speed within that measurement.",
        "- Before the first model is timed, `evaluate.py` loads the CPU until its speed "
        "stops changing. A processor at idle is well below its sustained clock, so "
        "without this the first model in the list is measured on a slower machine than "
        "the rest — a bias no per-model warmup can reach.",
        "- A fixed calibration matmul is then timed beside every model's measurement. "
        "Comparing it across models is the only way to know they met the same machine: "
        "drift cannot see between models, only within one.",
        f"- {lr_note}",
        f"- {data_spec.note}",
        f"- Truncation at max_length {ref['max_length']}: "
        f"{ref.get('eval_truncation_rate', float('nan')):.1%} of eval examples.",
    ]))

    return "\n\n".join(blocks) + "\n"


CSV_COLUMNS = [
    "slug", "label", "family", "checkpoint", "dataset", "seed", "is_baseline",
    "accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "f1_macro",
    "accuracy_retention", "delta_accuracy_pp", "delta_ci95_low_pp", "delta_ci95_high_pp",
    "mcnemar_p", "significant_at_05", "mcnemar_n_discordant",
    "train_time_s", "train_speedup", "train_samples_per_s", "train_loss",
    "latency_ms_p50", "latency_ms_p95", "latency_ms_mean", "latency_speedup",
    "latency_reduction_pct", "latency_drift", "latency_warmup_iters",
    "latency_rounds", "latency_ms_p50_min", "latency_ms_p50_max", "latency_p50_spread",
    "throughput_examples_per_s", "throughput_gain", "throughput_drift",
    "throughput_min", "throughput_max", "throughput_spread",
    "throughput_ms_per_batch_p50", "throughput_warmup_iters",
    "calibration_gflops", "calibration_spread", "calibration_converged",
    "metric_backend",
    "full_pass_s", "params_total", "params_non_embedding", "params_embedding",
    "param_reduction_pct", "param_reduction_non_embedding_pct",
    "size_on_disk_mb", "size_reduction_pct",
    "accuracy_pp_per_extra_speedup", "accuracy_pp_per_10pct_params_cut",
    "on_pareto_frontier", "eval_examples", "max_length", "learning_rate",
    "device", "torch_num_threads",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=BenchConfig.dataset, choices=list(DATASETS))
    p.add_argument("--seed", type=int, default=BenchConfig.seed,
                   help="the seed whose run drives the tables and figures")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="additional seeds to include in the variation section")
    p.add_argument("--baseline", default=BASELINE, choices=list(MODELS))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="report on the --smoke run, under results/smoke/")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    common.enable_utf8_stdout()
    args = parse_args(argv)
    if args.smoke:
        globals().update(smoke_dirs())

    rows = load_rows(args.dataset, args.seed)
    if not rows:
        print(f"No results for dataset '{args.dataset}' seed {args.seed}.\n"
              f"Run:  python train.py --dataset {args.dataset}  &&  "
              f"python evaluate.py --dataset {args.dataset}")
        return 1
    if len(rows) == 1:
        print(f"[warn] only {rows[0]['label']} has results — relative columns will be trivial.")

    rows, summary = analyse(rows, args.baseline, args.seed)
    seeds = sorted(set((args.seeds or []) + [args.seed]))
    variation = seed_variation(args.dataset, seeds)

    env_path = RESULTS_DIR / "environment.json"
    env = common.read_json(env_path) if env_path.exists() else {}

    # --- figures ---------------------------------------------------------- #
    figures: list[str] = []
    if not args.no_plots:
        frontier = set(summary["pareto_frontier"])
        compressed = [r for r in rows if not r["is_baseline"]] or rows
        # One slug -> colour map per theme, shared by all three figures, so a
        # model keeps its colour in the panel that drops the baseline.
        specs = [
            ("accuracy_vs_latency",
             lambda th: plotting.fig_pareto(rows, frontier, th, cmap=plotting.color_map(rows, th),
                                            latency_key=LATENCY_KEY, latency_label=LATENCY_LABEL)),
            ("retention_vs_speedup",
             lambda th: plotting.fig_tradeoff(compressed, th, cmap=plotting.color_map(rows, th),
                                             baseline_label=summary["baseline_label"])),
            ("cost_panels",
             lambda th: plotting.fig_cost_panels(rows, th, cmap=plotting.color_map(rows, th),
                                                latency_key=LATENCY_KEY, latency_label=LATENCY_LABEL)),
        ]
        for stem, builder in specs:
            for path in plotting.render(builder, PLOTS_DIR, stem):
                print(f"  wrote {path.relative_to(ROOT)}")
            # Relative to benchmark.md, which lives in RESULTS_DIR.
            figures.append(Path(os.path.relpath(PLOTS_DIR / f"{stem}.png", RESULTS_DIR)).as_posix())

    # --- tabular + prose outputs ------------------------------------------ #
    frame = pd.DataFrame(rows)
    for col in CSV_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    csv_path = RESULTS_DIR / f"benchmark_{args.dataset}.csv"
    frame[CSV_COLUMNS].to_csv(csv_path, index=False)
    frame[CSV_COLUMNS].to_csv(RESULTS_DIR / "benchmark.csv", index=False)

    markdown = build_markdown(rows, summary, env, variation, figures)
    (RESULTS_DIR / "benchmark.md").write_text(markdown, encoding="utf-8")
    common.write_json(RESULTS_DIR / "analysis.json",
                      {"summary": summary, "seed_variation": variation, "rows": rows})

    # --- console ---------------------------------------------------------- #
    print("\n" + headline_table(rows) + "\n")
    for row in rows:
        if not row["is_baseline"]:
            plain = headline_sentence(row, summary["baseline_label"]).replace("**", "")
            print("* " + plain)
    rec = summary.get("recommendation") or {}
    if rec:
        print(f"\nVerdict: {rec['label']} — {rec['basis']}.")
    print(f"\nWrote {csv_path.name}, benchmark.md, analysis.json to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
