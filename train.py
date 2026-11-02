"""Fine-tune one or more encoders on a text-classification task and time it.

Each model gets its own checkpoint under `results/checkpoints/` plus a
`results/train_<tag>.json` recording wall-clock training time, parameter
counts, on-disk size and the exact hyper-parameters used.

Evaluation is deliberately *not* run during training (`eval_strategy="no"`) so
the reported training time contains training and nothing else.

Examples
--------
    python train.py --smoke                       # tiny subsets, a few minutes on CPU
    python train.py                               # default 3-model sweep
    python train.py --models bert-base distilbert minilm-l12 minilm-l6 bert-mini
    python train.py --dataset imdb --max-length 256
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys

import common
from config import (
    BASELINE,
    CHECKPOINTS_DIR,
    DATASETS,
    DEFAULT_MODELS,
    MODELS,
    RESULTS_DIR,
    ROOT,
    SMOKE_OVERRIDES,
    BenchConfig,
    smoke_dirs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS),
                   help="models to fine-tune (default: %(default)s)")
    p.add_argument("--dataset", default=BenchConfig.dataset, choices=list(DATASETS))
    p.add_argument("--max-length", type=int, default=None,
                   help="token budget; defaults to the dataset's recommended value")
    p.add_argument("--max-train-samples", type=int, default=BenchConfig.max_train_samples,
                   help="-1 for the full split")
    p.add_argument("--epochs", type=float, default=BenchConfig.epochs)
    p.add_argument("--batch-size", type=int, default=BenchConfig.train_batch_size)
    p.add_argument("--lr", type=float, default=None,
                   help="override the per-model recommended learning rate")
    p.add_argument("--uniform-lr", action="store_true",
                   help="use the baseline model's LR for every model, holding LR constant "
                        "as a controlled variable instead of tuning per architecture")
    p.add_argument("--seed", type=int, default=BenchConfig.seed)
    p.add_argument("--num-threads", type=int, default=None,
                   help="pin torch intra-op threads (recommended for stable CPU timings)")
    p.add_argument("--smoke", action="store_true", help="tiny subsets, just to prove the pipeline runs")
    p.add_argument("--overwrite", action="store_true", help="retrain even if a checkpoint exists")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> BenchConfig:
    cfg = BenchConfig(dataset=args.dataset, epochs=args.epochs,
                      train_batch_size=args.batch_size, seed=args.seed,
                      num_threads=args.num_threads)
    spec = common.get_dataset_spec(args.dataset)
    cfg.max_length = args.max_length if args.max_length is not None else spec.recommended_max_length
    cfg.max_train_samples = None if args.max_train_samples == -1 else args.max_train_samples
    if args.smoke:
        for k, v in SMOKE_OVERRIDES.items():
            setattr(cfg, k, v)
    return cfg


def resolve_lr(name: str, args: argparse.Namespace) -> float:
    """The learning rate this run will use for `name`.

    `--lr` wins over everything; otherwise `--uniform-lr` holds the baseline's
    rate constant across models, and the default is each model's own
    recommended rate from `config.py`.
    """
    if args.lr is not None:
        return args.lr
    return MODELS[BASELINE].lr if args.uniform_lr else MODELS[name].lr


#: The settings that determine what the trained weights *are*. If any of them
#: differs from the record stored beside a checkpoint, that checkpoint answers a
#: different question and must not be reused — see `stale_reasons`.
def planned_fingerprint(cfg: BenchConfig, lr: float) -> dict:
    return {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "max_length": cfg.max_length,
        "max_train_samples": cfg.max_train_samples,
        "epochs": cfg.epochs,
        "train_batch_size": cfg.train_batch_size,
        "learning_rate": lr,
    }


def stored_fingerprint(record: dict) -> dict:
    """The same fields, read back out of a `results/train_<tag>.json`."""
    stored_cfg = record.get("config") or {}
    return {
        "dataset": record.get("dataset"),
        "seed": record.get("seed"),
        "max_length": stored_cfg.get("max_length"),
        "max_train_samples": stored_cfg.get("max_train_samples"),
        "epochs": stored_cfg.get("epochs"),
        "train_batch_size": stored_cfg.get("train_batch_size"),
        "learning_rate": record.get("learning_rate"),
    }


def stale_reasons(planned: dict, stored: dict) -> list[str]:
    """Human-readable list of what changed. Empty means the checkpoint is reusable.

    This is what stops `--smoke` from poisoning a real run: the smoke sweep
    trains on 200 examples but writes to the same tag, so without this check the
    real run would find a checkpoint, skip training, and quietly report accuracy
    for a model trained on a twentieth of the data.
    """
    return [
        f"{key}: {stored.get(key)!r} -> {planned[key]!r}"
        for key in planned
        if planned[key] != stored.get(key)
    ]


def train_one(name: str, cfg: BenchConfig, args: argparse.Namespace) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

    model_spec = common.get_model_spec(name)
    data_spec = common.get_dataset_spec(cfg.dataset)
    tag = common.run_tag(model_spec.slug, cfg.dataset, cfg.seed)
    ckpt_dir = CHECKPOINTS_DIR / tag

    lr = resolve_lr(name, args)

    print(f"\n{'=' * 72}\n{model_spec.label}  ({model_spec.checkpoint})\n{'=' * 72}")
    print(f"  {model_spec.note}")

    common.set_seed(cfg.seed)
    threads = common.apply_thread_limit(cfg.num_threads)
    device = common.resolve_device()

    tokenizer = AutoTokenizer.from_pretrained(model_spec.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_spec.checkpoint,
        num_labels=data_spec.num_labels,
        id2label={i: n for i, n in enumerate(data_spec.label_names)} or None,
    )

    params = common.count_parameters(model)
    arch = common.architecture_summary(model)
    print(f"  layers={arch['num_layers']} hidden={arch['hidden_size']} "
          f"heads={arch['num_attention_heads']} | params={params['params_total'] / 1e6:.1f}M "
          f"(embeddings {params['params_embedding'] / 1e6:.1f}M)")

    train_raw, _ = common.load_splits(data_spec, cfg.max_train_samples, 1, cfg.seed)
    trunc = common.truncation_rate(train_raw, tokenizer, data_spec, cfg.max_length)
    train_ds, _ = common.tokenize_splits(train_raw, train_raw.select(range(1)),
                                         tokenizer, data_spec, cfg.max_length)

    print(f"  train examples={len(train_ds)} max_length={cfg.max_length} "
          f"truncated={trunc:.1%} lr={lr:g} epochs={cfg.epochs} bs={cfg.train_batch_size} "
          f"device={device} threads={threads}")

    training_args = common.build_training_arguments(
        output_dir=str(ckpt_dir / "_trainer"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        learning_rate=lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=25,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],
        disable_tqdm=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)

    with common.Stopwatch() as sw:
        train_result = trainer.train()
    wall = sw.elapsed

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    shutil.rmtree(ckpt_dir / "_trainer", ignore_errors=True)

    size_mb = common.directory_size_mb(ckpt_dir)
    n_examples = len(train_ds) * cfg.epochs

    record = {
        "slug": model_spec.slug,
        "label": model_spec.label,
        "checkpoint": model_spec.checkpoint,
        "family": model_spec.family,
        "note": model_spec.note,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "learning_rate": lr,
        "uniform_lr": bool(args.uniform_lr),
        "config": dataclasses.asdict(cfg),
        "architecture": arch,
        **params,
        "size_on_disk_mb": size_mb,
        "size_fp32_theoretical_mb": common.theoretical_size_mb(params["params_total"]),
        "train_examples": len(train_ds),
        "train_truncation_rate": trunc,
        "train_time_s": wall,
        "train_time_s_trainer": float(train_result.metrics.get("train_runtime", wall)),
        "train_samples_per_s": n_examples / wall if wall > 0 else 0.0,
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        # Relative to the repo: the record is committed, and an absolute path
        # would publish the local directory layout (user name included).
        "checkpoint_dir": ckpt_dir.relative_to(ROOT).as_posix(),
        "device": device,
        "torch_num_threads": threads,
    }

    common.write_json(RESULTS_DIR / f"train_{tag}.json", record)
    print(f"  -> train time {wall:6.1f}s | {record['train_samples_per_s']:.1f} samples/s "
          f"| loss {record['train_loss']:.4f} | {size_mb:.0f} MB on disk")
    return record


def main(argv: list[str] | None = None) -> int:
    common.enable_utf8_stdout()
    args = parse_args(argv)
    if args.smoke:
        globals().update(smoke_dirs())  # never overwrite a real run
    cfg = build_config(args)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.apply_thread_limit(cfg.num_threads)
    common.write_json(RESULTS_DIR / "environment.json", common.environment_info())

    for name in args.models:
        tag = common.run_tag(MODELS[name].slug, cfg.dataset, cfg.seed)
        train_json = RESULTS_DIR / f"train_{tag}.json"

        if not args.overwrite and train_json.exists() \
                and (CHECKPOINTS_DIR / tag / "config.json").exists():
            reasons = stale_reasons(
                planned_fingerprint(cfg, resolve_lr(name, args)),
                stored_fingerprint(common.read_json(train_json)),
            )
            if not reasons:
                print(f"[skip] {name}: checkpoint matches this configuration "
                      f"(--overwrite to retrain anyway)")
                continue
            print(f"[retrain] {name}: configuration changed — {'; '.join(reasons)}")

        train_one(name, cfg, args)

    print(f"\nCheckpoints in {CHECKPOINTS_DIR}\nNext: python evaluate.py"
          + (" --smoke" if args.smoke else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
