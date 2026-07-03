from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tathate.data.unified_dataset import UnifiedJsonlDataset
from tathate.models import TADIClassifier
from tathate.training import (
    average_metric,
    build_classification_dataloader,
    evaluate_tadi_model,
    unwrap_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--eval_datasets", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output_path", type=Path, default=None)
    return parser.parse_args()


def parse_dataset_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def build_target_category_id_map(
    dataset: UnifiedJsonlDataset,
    *,
    merge_tail: bool,
    min_freq: int,
) -> tuple[dict[int, int], int]:
    category_counts: dict[int, int] = {}
    for row in dataset.rows:
        for value in row.get("target_category_ids", []):
            key = int(value)
            category_counts[key] = category_counts.get(key, 0) + 1
    if not category_counts:
        return {}, 0

    if not merge_tail or int(min_freq) <= 1:
        sorted_ids = sorted(category_counts)
        return {category_id: index for index, category_id in enumerate(sorted_ids)}, len(sorted_ids)

    head_ids = sorted([category_id for category_id, count in category_counts.items() if int(count) >= int(min_freq)])
    tail_ids = sorted([category_id for category_id, count in category_counts.items() if int(count) < int(min_freq)])
    id_map: dict[int, int] = {}
    for index, category_id in enumerate(head_ids):
        id_map[int(category_id)] = int(index)
    if tail_ids:
        tail_bucket_id = len(head_ids)
        for category_id in tail_ids:
            id_map[int(category_id)] = int(tail_bucket_id)
    target_category_size = len(head_ids) + (1 if tail_ids else 0)
    return id_map, target_category_size


def resolve_checkpoint(run_dir: Path, checkpoint_arg: Path | None) -> Path:
    if checkpoint_arg is None:
        return run_dir / "best_model.pt"
    if checkpoint_arg.is_absolute():
        return checkpoint_arg
    return run_dir / checkpoint_arg


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    saved_args = load_json(run_dir / "args.json")
    metadata = load_json(resolve_repo_path(saved_args["metadata_path"]))

    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    label_to_id = {str(key): int(value) for key, value in checkpoint_payload["label_to_id"].items()}
    id_to_label = {int(value): str(key) for key, value in label_to_id.items()}

    train_datasets = parse_dataset_list(saved_args["train_datasets"])
    default_eval_datasets = parse_dataset_list(saved_args["eval_datasets"]) or list(train_datasets)
    eval_datasets = parse_dataset_list(args.eval_datasets) or default_eval_datasets

    train_dataset = UnifiedJsonlDataset(
        resolve_repo_path(saved_args["train_path"]),
        datasets=train_datasets,
        max_examples=None,
        seed=int(saved_args["seed"]),
    )
    target_category_id_map, target_category_size = build_target_category_id_map(
        train_dataset,
        merge_tail=bool(saved_args["target_category_merge_tail"]),
        min_freq=int(saved_args["target_category_min_freq"]),
    )

    split_path = resolve_repo_path(saved_args["val_path"] if args.split == "val" else saved_args["test_path"])
    eval_dataset = UnifiedJsonlDataset(
        split_path,
        datasets=eval_datasets,
        max_examples=args.max_eval_examples if args.max_eval_examples is not None else saved_args["max_eval_examples"],
        seed=int(saved_args["seed"]),
    )

    model_name_or_path = args.model_name_or_path or str(saved_args["model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    coarse_label_to_id = {
        str(label): int(index)
        for label, index in sorted(metadata["coarse_label_to_id"].items(), key=lambda item: int(item[1]))
    }
    dataloader = build_classification_dataloader(
        eval_dataset,
        tokenizer=tokenizer,
        max_length=int(saved_args["max_length"]),
        label_key=str(saved_args["label_mode"]) + "_label" if str(saved_args["label_mode"]) != "coarse" else "coarse_label",
        label_to_id=label_to_id,
        coarse_label_to_id=coarse_label_to_id,
        target_category_size=target_category_size,
        target_category_id_map=target_category_id_map,
        batch_size=args.eval_batch_size if args.eval_batch_size is not None else int(saved_args["eval_batch_size"]),
        num_workers=args.num_workers if args.num_workers is not None else int(saved_args["num_workers"]),
        shuffle=False,
        balance_datasets=False,
        dataset_sampling_power=float(saved_args["dataset_sampling_power"]),
        pseudo_span_fallback=bool(saved_args["pseudo_span_fallback"]),
        pseudo_span_scope=str(saved_args["pseudo_span_scope"]),
        pseudo_span_max_tokens=int(saved_args["pseudo_span_max_tokens"]),
        pseudo_span_min_char_len=int(saved_args["pseudo_span_min_char_len"]),
    )

    requested_device = args.device or str(saved_args["device"])
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    model = TADIClassifier(
        model_name_or_path=model_name_or_path,
        num_labels=len(label_to_id),
        num_target_categories=target_category_size,
        dropout_prob=float(saved_args["dropout"]),
        pooling_strategy=str(checkpoint_payload.get("pooling_strategy", saved_args["pooling_strategy"])),
        implicit_attention_temperature=float(saved_args["implicit_attention_temperature"]),
        implicit_injection_scale=float(saved_args["implicit_injection_scale"]),
        explicit_injection_scale=float(saved_args["explicit_injection_scale"]),
        implicit_relation_clamp=float(saved_args["implicit_relation_clamp"]),
        implicit_gate_mode=str(saved_args["implicit_gate_mode"]),
        explicit_gate_mode=str(saved_args["explicit_gate_mode"]),
        implicit_clamp_mode=str(saved_args["implicit_clamp_mode"]),
        implicit_gate_floor=float(saved_args["implicit_gate_floor"]),
        explicit_gate_floor=float(saved_args["explicit_gate_floor"]),
    ).to(device)
    missing, unexpected = model.load_state_dict(unwrap_state_dict(checkpoint_payload), strict=False)

    metrics = evaluate_tadi_model(
        model=model,
        dataloader=dataloader,
        device=device,
        id_to_label=id_to_label,
    )
    avg_macro_f1 = average_metric(metrics["main"], "macro_f1")
    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "eval_datasets": eval_datasets,
        "model_name_or_path": model_name_or_path,
        "device": str(device),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "avg_macro_f1": avg_macro_f1,
        "metrics": metrics["main"],
    }
    if "auxiliary" in metrics:
        result["auxiliary_metrics"] = metrics["auxiliary"]

    output_path = args.output_path
    if output_path is None:
        checkpoint_stem = checkpoint_path.stem
        output_path = run_dir / f"eval_{args.split}_{checkpoint_stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
