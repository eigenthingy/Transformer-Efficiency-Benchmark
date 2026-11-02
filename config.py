"""Central configuration: model registry, dataset registry, benchmark defaults.

Everything the benchmark treats as a controlled variable lives here so that a
run is fully described by this file plus the CLI overrides recorded in
`results/environment.json`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
PLOTS_DIR = ROOT / "plots"


def smoke_dirs() -> dict[str, Path]:
    """Where a `--smoke` run writes, so it can never overwrite a real one.

    Smoke runs use the same run tags as real runs. Writing them to the same
    directories meant that checking the pipeline after a real sweep replaced
    its checkpoints, records, predictions and figures with 200-example
    versions. Each script rebinds its directory globals to these when
    `--smoke` is passed.
    """
    smoke = RESULTS_DIR / "smoke"
    return {
        "RESULTS_DIR": smoke,
        "PREDICTIONS_DIR": smoke / "predictions",
        "CHECKPOINTS_DIR": smoke / "checkpoints",
        "PLOTS_DIR": smoke / "plots",
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """One entry in the benchmark.

    Attributes
    ----------
    slug:
        Short filesystem-safe id used in filenames and plots.
    checkpoint:
        Hugging Face Hub id.
    label:
        Display name for tables and figures.
    lr:
        Recommended fine-tuning learning rate. Smaller/distilled encoders
        generally tolerate (and need) a larger LR than BERT-base; using each
        model's own recommended value measures "best achievable per model"
        rather than "best achievable at BERT-base's LR". Pass ``--uniform-lr``
        to hold LR constant instead and measure the other thing.
    family:
        Rough architecture family, for grouping in the report.
    note:
        One-line description used in the README/report.
    """

    slug: str
    checkpoint: str
    label: str
    lr: float
    family: str
    note: str


MODELS: dict[str, ModelSpec] = {
    "bert-base": ModelSpec(
        slug="bert-base",
        checkpoint="bert-base-uncased",
        label="BERT-base",
        lr=3e-5,
        family="BERT",
        note="12 layers, hidden 768. The reference point every number is relative to.",
    ),
    "distilbert": ModelSpec(
        slug="distilbert",
        checkpoint="distilbert-base-uncased",
        label="DistilBERT",
        lr=5e-5,
        family="BERT (distilled)",
        note="6 layers, hidden 768. Knowledge-distilled from BERT-base.",
    ),
    "minilm-l12": ModelSpec(
        slug="minilm-l12",
        checkpoint="microsoft/MiniLM-L12-H384-uncased",
        label="MiniLM-L12",
        lr=5e-5,
        family="MiniLM",
        note="12 layers, hidden 384. Deep-and-thin: self-attention distillation.",
    ),
    # ---- extended set: opt in with --models or --sweep (see SWEEPS below) ---- #
    "modernbert": ModelSpec(
        slug="modernbert",
        checkpoint="answerdotai/ModernBERT-base",
        label="ModernBERT-base",
        lr=5e-5,
        family="ModernBERT",
        note=(
            "22 layers, hidden 768 (2024). A modern reference point rather than a "
            "compressed one: it asks whether six years of architecture work beats "
            "distillation. Depth is misleading here — its GeGLU feed-forward is "
            "narrow (intermediate 1152, not 4x768), so per token it is about 1.3x "
            "BERT-base in compute, not 1.8x. Its native advantages (8192 context, "
            "alternating local/global attention, unpadding) do not show at 128 "
            "tokens under fixed-length padding, so this protocol measures it in its "
            "least favourable setting."
        ),
    ),
    "deberta-v3-xsmall": ModelSpec(
        slug="deberta-v3-xsmall",
        checkpoint="microsoft/deberta-v3-xsmall",
        label="DeBERTa-v3-xsmall",
        lr=5e-5,
        family="DeBERTa",
        note=(
            "12 layers, hidden 384, but a 128k-token vocabulary. Most of its "
            "parameters are the embedding table, which makes it the case that "
            "settles why this benchmark reports embedding and non-embedding counts "
            "separately: total parameters badly mispredict its latency."
        ),
    ),
    "modernbert-large": ModelSpec(
        slug="modernbert-large",
        checkpoint="answerdotai/ModernBERT-large",
        label="ModernBERT-large",
        lr=3e-5,
        family="ModernBERT",
        note=(
            "28 layers, hidden 1024 (2024). Extends the frontier upward: without a "
            "point above the baseline, 'what do you lose going smaller' has no idea "
            "what the ceiling costs. Roughly 4x BERT-base in compute — expensive on CPU."
        ),
    ),
    "deberta-v3-small": ModelSpec(
        slug="deberta-v3-small",
        checkpoint="microsoft/deberta-v3-small",
        label="DeBERTa-v3-small",
        lr=3e-5,
        family="DeBERTa",
        note=(
            "6 layers, hidden 768, 128k vocabulary. Paired with the xsmall variant it "
            "gives DeBERTa its own internal scaling curve, so family effects can be "
            "separated from size effects."
        ),
    ),
    "electra-small": ModelSpec(
        slug="electra-small",
        checkpoint="google/electra-small-discriminator",
        label="ELECTRA-small",
        lr=1e-4,
        family="ELECTRA",
        note=(
            "12 layers, hidden 256 (2020). Small by a different route: not compressed "
            "from a teacher but pretrained from scratch on replaced-token detection, a "
            "denser signal than masked-language modelling. The test of whether a better "
            "objective beats distillation at equal cost."
        ),
    ),
    "albert-base": ModelSpec(
        slug="albert-base",
        checkpoint="albert-base-v2",
        label="ALBERT-base",
        lr=3e-5,
        family="ALBERT",
        note=(
            "12 layers, hidden 768, but every layer shares one set of weights. The "
            "deliberate counter-example: ~12M parameters and the compute of a full "
            "BERT-base. It should land far off the latency frontier, which is exactly "
            "the point — weight sharing saves memory, not matrix multiplications."
        ),
    ),
    "roberta-base": ModelSpec(
        slug="roberta-base",
        checkpoint="roberta-base",
        label="RoBERTa-base",
        lr=2e-5,
        family="RoBERTa",
        note=(
            "12 layers, hidden 768 — architecturally BERT-base, trained longer on more "
            "data. Isolates training recipe from architecture: same compute, same shape, "
            "different accuracy."
        ),
    ),
    "distilroberta": ModelSpec(
        slug="distilroberta",
        checkpoint="distilroberta-base",
        label="DistilRoBERTa",
        lr=5e-5,
        family="RoBERTa (distilled)",
        note=(
            "6 layers, hidden 768. With RoBERTa-base as its teacher it forms a second, "
            "independent distillation pair — so 'what distillation costs' can be measured "
            "twice instead of asserted from one case."
        ),
    ),
    # Ettin: the ModernBERT architecture released as a full size ladder with an
    # open training recipe (17M to 400M). Verified from the checkpoints' configs:
    # `model_type` is `modernbert`, and ettin-encoder-150m / -400m carry exactly
    # the dimensions of ModernBERT-base / -large. That is what makes size-matched
    # old-vs-modern comparisons possible at all — before this, the only modern
    # encoders in the registry were bigger than the baseline.
    "ettin-17m": ModelSpec(
        slug="ettin-17m",
        checkpoint="jhu-clsp/ettin-encoder-17m",
        label="Ettin-17M",
        lr=1e-4,
        family="Ettin (ModernBERT arch.)",
        note=(
            "7 layers, hidden 256. The modern floor, to be read against BERT-mini "
            "(4x256) and ELECTRA-small (12x256): at the very bottom of the size range, "
            "does a 2025 recipe still buy anything?"
        ),
    ),
    "ettin-32m": ModelSpec(
        slug="ettin-32m",
        checkpoint="jhu-clsp/ettin-encoder-32m",
        label="Ettin-32M",
        lr=8e-5,
        family="Ettin (ModernBERT arch.)",
        note=(
            "10 layers, hidden 384 — the same width as MiniLM-L12, one layer band "
            "shallower. The cleanest head-to-head in the registry: 2020 self-attention "
            "distillation against 2025 pretraining, at matched geometry."
        ),
    ),
    "ettin-68m": ModelSpec(
        slug="ettin-68m",
        checkpoint="jhu-clsp/ettin-encoder-68m",
        label="Ettin-68M",
        lr=5e-5,
        family="Ettin (ModernBERT arch.)",
        note=(
            "19 layers, hidden 512 — the same parameter budget as DistilBERT but a very "
            "different shape (deep and narrow against shallow and wide). The question a "
            "deployment actually faces: at this budget, distil an old model or take a "
            "new small one?"
        ),
    ),
    "minilm-l6": ModelSpec(
        slug="minilm-l6",
        checkpoint="nreimers/MiniLM-L6-H384-uncased",
        label="MiniLM-L6",
        lr=5e-5,
        family="MiniLM",
        note="6 layers, hidden 384. Half the depth of MiniLM-L12.",
    ),
    "bert-mini": ModelSpec(
        slug="bert-mini",
        checkpoint="google/bert_uncased_L-4_H-256_A-4",
        label="BERT-mini",
        lr=1e-4,
        family="BERT (small)",
        note="4 layers, hidden 256. Pretrained small rather than compressed.",
    ),
    "bert-tiny": ModelSpec(
        slug="bert-tiny",
        checkpoint="google/bert_uncased_L-2_H-128_A-2",
        label="BERT-tiny",
        lr=3e-4,
        family="BERT (small)",
        note=(
            "2 layers, hidden 128. The floor of the sweep — there to show where the "
            "accuracy curve finally breaks, which is the only way to know the others "
            "were on its flat part."
        ),
    ),
}

#: Models run when ``--models`` is not given.
DEFAULT_MODELS: list[str] = ["bert-base", "distilbert", "minilm-l12"]

#: Named model sets, for ``run_benchmark.py --sweep NAME``.
#:
#: Each sweep is a question, not a size bracket. Every one keeps `bert-base` so
#: the relative columns stay comparable across sweeps, and every one is small
#: enough to finish in an evening on a CPU.
#:
#: Running them in sequence is cheap after the first: `train.py` reuses any
#: checkpoint whose configuration matches, so the shared models are trained once.
SWEEPS: dict[str, list[str]] = {
    # The cheapest sweep that answers the original question.
    "classic": DEFAULT_MODELS,

    # THE question, and the reason the Ettin models are here: at matched size, is
    # a 2019/2020 distilled model or a 2025 small one the better buy? Ettin-32M
    # sits opposite MiniLM-L12 at the same width; Ettin-68M opposite DistilBERT
    # at the same parameter budget.
    "modern": ["bert-base", "distilbert", "minilm-l12", "ettin-32m", "ettin-68m"],

    # Two scaling curves on one pair of axes: this one is the modern family, to
    # be overlaid on `scaling` below. If the curves are parallel, the recipe
    # bought a constant offset; if they diverge, it bought more at one end.
    "modern-scaling": ["bert-base", "ettin-17m", "ettin-32m", "ettin-68m", "modernbert"],

    # Four routes to a cheaper model: distillation, deep-and-thin distillation, a
    # better pretraining objective, and weight sharing.
    "mechanisms": ["bert-base", "distilbert", "minilm-l12", "electra-small", "albert-base"],

    # The 2018-2020 curve: enough points over an order of magnitude of encoder
    # size for the fit in notebooks/analysis.ipynb to mean something.
    "scaling": ["bert-base", "distilbert", "minilm-l12", "minilm-l6", "bert-mini", "bert-tiny"],

    # Does distillation cost the same on a better-trained backbone? Two
    # independent teacher/student pairs, measured the same way.
    "families": ["bert-base", "distilbert", "roberta-base", "distilroberta"],

    # Vocabulary size as a confound: DeBERTa keeps ~70% of its parameters in the
    # embedding table, so total parameters mispredict its cost badly.
    "vocabulary": ["bert-base", "minilm-l12", "deberta-v3-xsmall", "deberta-v3-small"],

    # The full frontier, from 4M to 395M parameters. Hours, not minutes.
    "all": list(MODELS),
}

#: Every relative number in the report is computed against this model.
BASELINE = "bert-base"


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    hf_name: str
    text_column: str
    label_column: str
    num_labels: int
    train_split: str
    test_split: str
    label_names: list[str] = field(default_factory=list)
    #: Sequence length at which the dataset is *not* materially truncated.
    recommended_max_length: int = 128
    note: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "ag_news": DatasetSpec(
        hf_name="ag_news",
        text_column="text",
        label_column="label",
        num_labels=4,
        train_split="train",
        test_split="test",
        label_names=["World", "Sports", "Business", "Sci/Tech"],
        recommended_max_length=128,
        note=(
            "News topic classification, 4 balanced classes. Headline + short "
            "description, so 128 tokens covers ~99% of examples: the "
            "accuracy comparison is not confounded by truncation."
        ),
    ),
    "imdb": DatasetSpec(
        hf_name="imdb",
        text_column="text",
        label_column="label",
        num_labels=2,
        train_split="train",
        test_split="test",
        label_names=["negative", "positive"],
        recommended_max_length=256,
        note=(
            "Binary sentiment on long movie reviews. Reviews far exceed 256 "
            "tokens, so every model sees the same truncated prefix; treat the "
            "accuracy column as 'accuracy at this budget', not ceiling accuracy."
        ),
    ),
}

DEFAULT_DATASET = "ag_news"


# --------------------------------------------------------------------------- #
# Benchmark defaults
# --------------------------------------------------------------------------- #
@dataclass
class BenchConfig:
    """Defaults sized so the whole sweep finishes on a laptop CPU."""

    dataset: str = DEFAULT_DATASET
    max_length: int = 128
    max_train_samples: int | None = 4000
    max_eval_samples: int | None = 2000
    epochs: float = 1.0
    train_batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42

    # --- inference-latency protocol -------------------------------------- #
    #: Untimed forward passes before measurement: at least this many AND at
    #: least `*_warmup_s` seconds, whichever takes longer.
    #:
    #: The duration floor is not redundant. A fixed iteration count warms a fast
    #: model for a fraction of the time it warms a slow one — 10 passes is 0.35 s
    #: for MiniLM and 0.95 s for BERT-base — so the small model gets measured on
    #: a colder machine. On CPU that matters, because sustained vector load pulls
    #: clock frequencies down: the bias runs in favour of whichever model warmed
    #: least, which is exactly the one the benchmark is trying to judge fairly.
    latency_warmup: int = 10
    latency_warmup_s: float = 1.0
    #: Timed single-example forward passes; we report median, p95 and drift.
    latency_samples: int = 200
    #: Interleaved timing rounds. Every round times every model, with the
    #: starting position rotated, and the report takes the median.
    #:
    #: This exists because measuring one model to completion before starting the
    #: next confounds machine drift with model identity. On a fluctuating desktop
    #: that produced 70, 92, 74 and 70 ms for the same model across four runs,
    #: and speedups that moved by a factor of 1.5. Interleaving spreads the drift
    #: over all models, and the round-to-round range becomes a stated
    #: uncertainty instead of a hidden bias. 1 disables it.
    latency_rounds: int = 3
    #: Timed batches for the throughput figure, each timed individually.
    throughput_batches: int = 20
    throughput_warmup_s: float = 3.0
    #: Torch intra-op threads. Pinned so timings are comparable across models
    #: and across machines that report the same value.
    num_threads: int | None = None  # None -> leave torch default, but record it


SMOKE_OVERRIDES = dict(
    max_train_samples=200,
    max_eval_samples=200,
    latency_warmup=3,
    latency_warmup_s=0.2,
    latency_samples=20,
    latency_rounds=1,
    throughput_batches=3,
    throughput_warmup_s=0.3,
)
