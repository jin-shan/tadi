from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tathate.models import TADIClassifier
from tathate.training import (
    average_metric,
    build_classification_dataloader,
    compute_tadi_losses,
    evaluate_tadi_model,
    load_split_datasets,
    prepare_label_metadata,
    set_seed,
    unwrap_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=Path, default=Path("data/unified/train.jsonl"))
    parser.add_argument("--val_path", type=Path, default=Path("data/unified/val.jsonl"))
    parser.add_argument("--test_path", type=Path, default=Path("data/unified/test.jsonl"))
    parser.add_argument("--metadata_path", type=Path, default=Path("data/unified/metadata.json"))
    parser.add_argument("--model_name_or_path", type=str, default=r"D:\models\roberta-base")
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_datasets", type=str, default="ethos,olid,ihc,toxicn")
    parser.add_argument("--eval_datasets", type=str, default="")
    parser.add_argument("--label_mode", choices=["coarse"], default="coarse")
    parser.add_argument("--pooling_strategy", choices=["cls", "mean"], default="cls")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--implicit_attention_temperature", type=float, default=1.0)
    parser.add_argument("--implicit_injection_scale", type=float, default=0.05)
    parser.add_argument("--explicit_injection_scale", type=float, default=0.05)
    parser.add_argument("--implicit_relation_clamp", type=float, default=12.0)
    parser.add_argument("--implicit_gate_mode", choices=["learned", "fixed_one"], default="learned")
    parser.add_argument("--explicit_gate_mode", choices=["learned", "fixed_one"], default="learned")
    parser.add_argument("--implicit_clamp_mode", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--implicit_gate_floor", type=float, default=0.0)
    parser.add_argument("--explicit_gate_floor", type=float, default=0.0)
    parser.add_argument("--target_supervision_weight", type=float, default=0.05)
    parser.add_argument("--target_supervision_scope", choices=["all", "positive_only"], default="positive_only")
    parser.add_argument("--target_supervision_pseudo_scale", type=float, default=0.1)
    parser.add_argument("--pseudo_span_fallback", action="store_true")
    parser.add_argument("--no_pseudo_span_fallback", action="store_false", dest="pseudo_span_fallback")
    parser.set_defaults(pseudo_span_fallback=True)
    parser.add_argument("--pseudo_span_scope", choices=["all", "positive_only"], default="positive_only")
    parser.add_argument("--pseudo_span_max_tokens", type=int, default=6)
    parser.add_argument("--pseudo_span_min_char_len", type=int, default=2)
    parser.add_argument("--target_category_weight", type=float, default=0.02)
    parser.add_argument("--target_category_scope", choices=["all", "positive_only"], default="positive_only")
    parser.add_argument("--target_category_merge_tail", action="store_true")
    parser.add_argument("--no_target_category_merge_tail", action="store_false", dest="target_category_merge_tail")
    parser.set_defaults(target_category_merge_tail=True)
    parser.add_argument("--target_category_min_freq", type=int, default=5)
    parser.add_argument("--target_category_loss_mode", choices=["bce", "weighted_bce"], default="weighted_bce")
    parser.add_argument("--target_category_pos_weight_cap", type=float, default=8.0)
    parser.add_argument("--target_category_pos_weight_smoothing", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.01)
    parser.add_argument("--contrastive_label_mode", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--contrastive_temperature", type=float, default=0.2)
    parser.add_argument("--hard_negative_k", type=int, default=4)
    parser.add_argument("--contrastive_negative_sampling_mode", choices=["hard", "mixed"], default="mixed")
    parser.add_argument("--contrastive_hard_negative_ratio", type=float, default=0.25)
    parser.add_argument("--explicit_gate_supervision_weight", type=float, default=0.005)
    parser.add_argument("--explicit_gate_supervision_scope", choices=["all", "positive_only"], default="positive_only")
    parser.add_argument("--aux_ramp_start_ratio", type=float, default=0.0)
    parser.add_argument("--aux_ramp_duration_ratio", type=float, default=0.5)
    parser.add_argument("--aux_balance_by_active_ratio", action="store_true")
    parser.add_argument("--no_aux_balance_by_active_ratio", action="store_false", dest="aux_balance_by_active_ratio")
    parser.set_defaults(aux_balance_by_active_ratio=True)
    parser.add_argument("--aux_balance_target_ratio", type=float, default=0.25)
    parser.add_argument("--aux_balance_ratio_floor", type=float, default=0.05)
    parser.add_argument("--aux_balance_min_scale", type=float, default=0.5)
    parser.add_argument("--aux_balance_max_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--balance_datasets", action="store_true")
    parser.add_argument("--dataset_sampling_power", type=float, default=1.0)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--selection_strategy", choices=["best", "last"], default="best")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def summarize_dataset_labels(dataset, *, label_key: str) -> dict[str, dict[str, dict[str, int]]]:
    summary: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {"main": Counter(), "coarse": Counter()})
    for row in dataset.rows:
        summary[row["dataset"]]["main"][row[label_key]] += 1
        summary[row["dataset"]]["coarse"][row["coarse_label"]] += 1
    return {
        dataset_name: {
            "main": dict(sorted(parts["main"].items())),
            "coarse": dict(sorted(parts["coarse"].items())),
        }
        for dataset_name, parts in sorted(summary.items())
    }


def build_target_category_id_map(
    dataset,
    *,
    merge_tail: bool,
    min_freq: int,
) -> tuple[dict[int, int], dict[str, int]]:
    category_counter: Counter[int] = Counter()
    for row in dataset.rows:
        for value in row.get("target_category_ids", []):
            category_counter[int(value)] += 1
    if not category_counter:
        return {}, {"original_size": 0, "mapped_size": 0, "tail_size": 0, "tail_bucket_id": -1}

    if not merge_tail or int(min_freq) <= 1:
        sorted_ids = sorted(category_counter)
        id_map = {category_id: index for index, category_id in enumerate(sorted_ids)}
        return id_map, {
            "original_size": len(sorted_ids),
            "mapped_size": len(sorted_ids),
            "tail_size": 0,
            "tail_bucket_id": -1,
        }

    head_ids = sorted([category_id for category_id, count in category_counter.items() if int(count) >= int(min_freq)])
    tail_ids = sorted([category_id for category_id, count in category_counter.items() if int(count) < int(min_freq)])
    id_map: dict[int, int] = {}
    for index, category_id in enumerate(head_ids):
        id_map[int(category_id)] = int(index)
    tail_bucket_id = -1
    if tail_ids:
        tail_bucket_id = len(head_ids)
        for category_id in tail_ids:
            id_map[int(category_id)] = int(tail_bucket_id)
    mapped_size = len(head_ids) + (1 if tail_ids else 0)
    return id_map, {
        "original_size": len(category_counter),
        "mapped_size": mapped_size,
        "tail_size": len(tail_ids),
        "tail_bucket_id": tail_bucket_id,
    }


def build_target_category_pos_weight(
    dataset,
    *,
    target_category_id_map: dict[int, int],
    target_category_size: int,
    target_category_scope: str,
    positive_label_name: str,
    smoothing: float,
    cap: float,
) -> torch.Tensor | None:
    if target_category_size <= 0:
        return None
    pos_counts = torch.zeros(target_category_size, dtype=torch.float32)
    active_rows = 0
    for row in dataset.rows:
        if target_category_scope == "positive_only" and str(row.get("coarse_label", "")) != str(positive_label_name):
            continue
        mapped_ids: set[int] = set()
        for value in row.get("target_category_ids", []):
            raw_id = int(value)
            if raw_id in target_category_id_map:
                mapped_ids.add(int(target_category_id_map[raw_id]))
        if not mapped_ids:
            continue
        active_rows += 1
        for mapped_id in mapped_ids:
            if 0 <= mapped_id < target_category_size:
                pos_counts[mapped_id] += 1.0
    if active_rows <= 0:
        return None
    safe_smoothing = max(1e-6, float(smoothing))
    safe_cap = max(1.0, float(cap))
    neg_counts = float(active_rows) - pos_counts
    pos_weight = (neg_counts + safe_smoothing) / (pos_counts + safe_smoothing)
    return pos_weight.clamp(min=1e-6, max=safe_cap)


def build_alignment_report(
    args: argparse.Namespace,
    *,
    label_to_id: dict[str, int],
) -> dict[str, object]:
    implemented_modules = [
        "shared_encoder",
        "single_task_branch",
        "single_binary_head",
        "binary_classification_objective",
        "explicit_target_heuristic_masking",
    ]
    implemented_modules.extend(
        [
            "implicit_target_extractor",
            "implicit_relation",
            "direct_relation_injection",
            "explicit_relation_injection",
        ]
    )
    if args.implicit_gate_mode == "learned":
        implemented_modules.append("samplewise_relation_gate")
    else:
        implemented_modules.append("fixed_relation_gate")
    if args.explicit_gate_mode == "learned":
        implemented_modules.append("samplewise_explicit_gate")
    else:
        implemented_modules.append("fixed_explicit_gate")
    if args.implicit_clamp_mode == "enabled":
        implemented_modules.append("relation_norm_clamp")
    if args.target_supervision_weight > 0:
        implemented_modules.append("target_rationale_span_supervision")
        if args.pseudo_span_fallback:
            implemented_modules.append("pseudo_target_span_fallback")
    if args.target_category_weight > 0:
        implemented_modules.append("target_category_supervision")
        if args.target_category_merge_tail and args.target_category_min_freq > 1:
            implemented_modules.append("target_category_tail_merge")
        if args.target_category_loss_mode == "weighted_bce":
            implemented_modules.append("target_category_weighted_bce")
    if args.contrastive_weight > 0:
        implemented_modules.append("label_aware_hard_negative_contrastive")
    if args.explicit_gate_supervision_weight > 0:
        implemented_modules.append("explicit_gate_supervision")

    deferred_modules = [
        "momentum_queue_hard_negative_mining",
        "cross_batch_memory_bank",
        "dataset_specific_target_annotators",
    ]
    return {
        "method": "TADI",
        "implemented_modules": implemented_modules,
        "deferred_modules": deferred_modules,
        "reference_alignment": {
            "lahn": "Retains implicit target relational modeling, but removes fine/coarse multi-task coupling.",
            "amplehate": "Keeps direct relation injection while simplifying to one binary decision head.",
            "target_aware_design": "Target-aware direct injection with optional explicit target regularization.",
        },
        "training_objective": {
            "classification_loss": 1.0,
            "target_supervision_weight": args.target_supervision_weight,
            "target_supervision_scope": args.target_supervision_scope,
            "target_supervision_pseudo_scale": args.target_supervision_pseudo_scale,
            "pseudo_span_fallback": args.pseudo_span_fallback,
            "pseudo_span_scope": args.pseudo_span_scope,
            "pseudo_span_max_tokens": args.pseudo_span_max_tokens,
            "pseudo_span_min_char_len": args.pseudo_span_min_char_len,
            "target_category_weight": args.target_category_weight,
            "target_category_scope": args.target_category_scope,
            "target_category_merge_tail": args.target_category_merge_tail,
            "target_category_min_freq": args.target_category_min_freq,
            "target_category_loss_mode": args.target_category_loss_mode,
            "target_category_pos_weight_cap": args.target_category_pos_weight_cap,
            "target_category_pos_weight_smoothing": args.target_category_pos_weight_smoothing,
            "contrastive_weight": args.contrastive_weight,
            "contrastive_label_mode": args.contrastive_label_mode,
            "contrastive_temperature": args.contrastive_temperature,
            "hard_negative_k": args.hard_negative_k,
            "contrastive_negative_sampling_mode": args.contrastive_negative_sampling_mode,
            "contrastive_hard_negative_ratio": args.contrastive_hard_negative_ratio,
            "explicit_gate_supervision_weight": args.explicit_gate_supervision_weight,
            "explicit_gate_supervision_scope": args.explicit_gate_supervision_scope,
            "aux_ramp_start_ratio": args.aux_ramp_start_ratio,
            "aux_ramp_duration_ratio": args.aux_ramp_duration_ratio,
            "aux_balance_by_active_ratio": args.aux_balance_by_active_ratio,
            "aux_balance_target_ratio": args.aux_balance_target_ratio,
            "aux_balance_ratio_floor": args.aux_balance_ratio_floor,
            "aux_balance_min_scale": args.aux_balance_min_scale,
            "aux_balance_max_scale": args.aux_balance_max_scale,
            "removed_losses": [
                "coarse_loss",
                "coarse_from_fine_loss",
                "coarse_consistency_loss",
                "branch_separation_loss",
            ],
            "implicit_attention_temperature": args.implicit_attention_temperature,
            "implicit_injection_scale": args.implicit_injection_scale,
            "explicit_injection_scale": args.explicit_injection_scale,
            "implicit_relation_clamp": args.implicit_relation_clamp,
            "implicit_gate_mode": args.implicit_gate_mode,
            "explicit_gate_mode": args.explicit_gate_mode,
            "implicit_clamp_mode": args.implicit_clamp_mode,
            "implicit_gate_floor": args.implicit_gate_floor,
            "explicit_gate_floor": args.explicit_gate_floor,
            "selection_metric": "avg_val_macro_f1",
            "selection_strategy": args.selection_strategy,
        },
        "label_mode": args.label_mode,
        "label_space": label_to_id,
    }


def average_optional_metric(metrics: dict[str, dict[str, float]], key: str) -> float | None:
    available = [float(values[key]) for values in metrics.values() if key in values]
    if not available:
        return None
    return sum(available) / len(available)


def snapshot_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    train_datasets = [item.strip() for item in args.train_datasets.split(",") if item.strip()]
    eval_datasets = [item.strip() for item in args.eval_datasets.split(",") if item.strip()] or list(train_datasets)
    label_space_datasets = list(dict.fromkeys(train_datasets + eval_datasets))
    label_metadata = prepare_label_metadata(metadata, args.label_mode, label_space_datasets)
    label_key = str(label_metadata["label_key"])
    label_to_id = dict(label_metadata["label_to_id"])
    id_to_label = dict(label_metadata["id_to_label"])
    coarse_label_to_id = dict(label_metadata["coarse_label_to_id"])
    positive_label_id = int(label_to_id.get("hate", max(label_to_id.values())))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    train_dataset, val_dataset, test_dataset = load_split_datasets(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        seed=args.seed,
    )
    target_category_id_map, target_category_mapping_meta = build_target_category_id_map(
        train_dataset,
        merge_tail=args.target_category_merge_tail,
        min_freq=args.target_category_min_freq,
    )
    target_category_size = int(target_category_mapping_meta["mapped_size"])
    target_category_pos_weight = build_target_category_pos_weight(
        train_dataset,
        target_category_id_map=target_category_id_map,
        target_category_size=target_category_size,
        target_category_scope=args.target_category_scope,
        positive_label_name="hate",
        smoothing=args.target_category_pos_weight_smoothing,
        cap=args.target_category_pos_weight_cap,
    )
    target_category_pos_weight_device = None

    train_loader = build_classification_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_key=label_key,
        label_to_id=label_to_id,
        coarse_label_to_id=coarse_label_to_id,
        target_category_size=target_category_size,
        target_category_id_map=target_category_id_map,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        balance_datasets=args.balance_datasets,
        dataset_sampling_power=args.dataset_sampling_power,
        pseudo_span_fallback=args.pseudo_span_fallback,
        pseudo_span_scope=args.pseudo_span_scope,
        pseudo_span_max_tokens=args.pseudo_span_max_tokens,
        pseudo_span_min_char_len=args.pseudo_span_min_char_len,
    )
    val_loader = build_classification_dataloader(
        val_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_key=label_key,
        label_to_id=label_to_id,
        coarse_label_to_id=coarse_label_to_id,
        target_category_size=target_category_size,
        target_category_id_map=target_category_id_map,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        balance_datasets=False,
        dataset_sampling_power=args.dataset_sampling_power,
        pseudo_span_fallback=args.pseudo_span_fallback,
        pseudo_span_scope=args.pseudo_span_scope,
        pseudo_span_max_tokens=args.pseudo_span_max_tokens,
        pseudo_span_min_char_len=args.pseudo_span_min_char_len,
    )
    test_loader = build_classification_dataloader(
        test_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_key=label_key,
        label_to_id=label_to_id,
        coarse_label_to_id=coarse_label_to_id,
        target_category_size=target_category_size,
        target_category_id_map=target_category_id_map,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        balance_datasets=False,
        dataset_sampling_power=args.dataset_sampling_power,
        pseudo_span_fallback=args.pseudo_span_fallback,
        pseudo_span_scope=args.pseudo_span_scope,
        pseudo_span_max_tokens=args.pseudo_span_max_tokens,
        pseudo_span_min_char_len=args.pseudo_span_min_char_len,
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if target_category_pos_weight is not None:
        target_category_pos_weight_device = target_category_pos_weight.to(device=device)
    model = TADIClassifier(
        model_name_or_path=args.model_name_or_path,
        num_labels=len(label_to_id),
        num_target_categories=target_category_size,
        dropout_prob=args.dropout,
        pooling_strategy=args.pooling_strategy,
        implicit_attention_temperature=args.implicit_attention_temperature,
        implicit_injection_scale=args.implicit_injection_scale,
        explicit_injection_scale=args.explicit_injection_scale,
        implicit_relation_clamp=args.implicit_relation_clamp,
        implicit_gate_mode=args.implicit_gate_mode,
        explicit_gate_mode=args.explicit_gate_mode,
        implicit_clamp_mode=args.implicit_clamp_mode,
        implicit_gate_floor=args.implicit_gate_floor,
        explicit_gate_floor=args.explicit_gate_floor,
    ).to(device)

    if args.init_checkpoint is not None:
        init_payload = torch.load(args.init_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(unwrap_state_dict(init_payload), strict=False)
        print(
            json.dumps(
                {
                    "init_checkpoint": str(args.init_checkpoint),
                    "missing_keys": list(missing),
                    "unexpected_keys": list(unexpected),
                },
                ensure_ascii=False,
            )
        )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [parameter for name, parameter in model.named_parameters() if not any(term in name for term in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [parameter for name, parameter in model.named_parameters() if any(term in name for term in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_optimizer_steps = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs)
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    classification_criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    best_val = -1.0
    best_state = None
    last_state = None
    optimizer_steps = 0

    train_dataset_counts = train_dataset.dataset_counts()
    train_label_summary = summarize_dataset_labels(train_dataset, label_key=label_key)
    val_dataset_counts = val_dataset.dataset_counts()
    test_dataset_counts = test_dataset.dataset_counts()
    alignment_report = build_alignment_report(args, label_to_id=label_to_id)
    alignment_report["training_objective"]["target_category_size"] = target_category_size
    alignment_report["training_objective"]["target_category_original_size"] = target_category_mapping_meta["original_size"]
    alignment_report["training_objective"]["target_category_tail_size"] = target_category_mapping_meta["tail_size"]
    alignment_report["training_objective"]["target_category_tail_bucket_id"] = target_category_mapping_meta["tail_bucket_id"]
    if target_category_pos_weight is not None:
        alignment_report["training_objective"]["target_category_pos_weight_mean"] = float(target_category_pos_weight.mean().item())
        alignment_report["training_objective"]["target_category_pos_weight_max"] = float(target_category_pos_weight.max().item())
        alignment_report["training_objective"]["target_category_pos_weight_min"] = float(target_category_pos_weight.min().item())
    (args.output_dir / "alignment_report.json").write_text(
        json.dumps(alignment_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total_loss = 0.0
        running_classification_loss = 0.0
        running_target_alignment_loss = 0.0
        running_target_supervision_ratio = 0.0
        running_target_supervision_pseudo_ratio = 0.0
        running_target_category_loss = 0.0
        running_target_category_supervision_ratio = 0.0
        running_target_category_pos_weight_mean = 0.0
        running_contrastive_loss = 0.0
        running_contrastive_anchor_ratio = 0.0
        running_explicit_gate_supervision_loss = 0.0
        running_explicit_gate_supervision_active_ratio = 0.0
        running_auxiliary_weight_scale = 0.0
        running_scaled_target_supervision_weight = 0.0
        running_scaled_target_category_weight = 0.0
        running_scaled_contrastive_weight = 0.0
        running_scaled_explicit_gate_supervision_weight = 0.0
        running_target_supervision_ratio_scale = 0.0
        running_target_category_ratio_scale = 0.0
        running_contrastive_ratio_scale = 0.0
        running_explicit_gate_supervision_ratio_scale = 0.0
        running_optional_metrics: defaultdict[str, float] = defaultdict(float)
        num_batches = 0
        optimizer.zero_grad()
        pending_steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            special_tokens_mask = batch["special_tokens_mask"].to(device)
            target_supervision_mask = batch["target_supervision_mask"].to(device)
            target_supervision_flag = batch["target_supervision_flag"].to(device)
            target_supervision_pseudo_flag = batch["target_supervision_pseudo_flag"].to(device)
            target_category_targets = batch["target_category_targets"].to(device)
            target_category_flag = batch["target_category_flag"].to(device)
            explicit_candidate_mask = batch["explicit_candidate_mask"].to(device)
            labels = batch["label_id"].to(device)
            contrastive_group_ids = labels
            if args.contrastive_label_mode == "fine":
                fine_group_names = batch["fine_group_name"]
                fine_group_to_id: dict[str, int] = {}
                local_group_ids: list[int] = []
                for name in fine_group_names:
                    key = str(name)
                    if key not in fine_group_to_id:
                        fine_group_to_id[key] = len(fine_group_to_id)
                    local_group_ids.append(fine_group_to_id[key])
                contrastive_group_ids = torch.tensor(local_group_ids, dtype=torch.long, device=device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                explicit_candidate_mask=explicit_candidate_mask,
            )
            if total_optimizer_steps <= 1:
                training_progress = 1.0
            else:
                training_progress = min(1.0, float(optimizer_steps) / float(total_optimizer_steps - 1))
            loss_dict = compute_tadi_losses(
                outputs=outputs,
                labels=labels,
                target_supervision_mask=target_supervision_mask,
                target_supervision_flag=target_supervision_flag,
                target_supervision_pseudo_flag=target_supervision_pseudo_flag,
                target_category_targets=target_category_targets,
                target_category_flag=target_category_flag,
                classification_criterion=classification_criterion,
                target_supervision_weight=args.target_supervision_weight,
                target_supervision_pseudo_scale=args.target_supervision_pseudo_scale,
                target_category_weight=args.target_category_weight,
                target_category_loss_mode=args.target_category_loss_mode,
                target_category_pos_weight=target_category_pos_weight_device,
                contrastive_weight=args.contrastive_weight,
                contrastive_temperature=args.contrastive_temperature,
                hard_negative_k=args.hard_negative_k,
                explicit_gate_supervision_weight=args.explicit_gate_supervision_weight,
                target_supervision_scope=args.target_supervision_scope,
                target_category_scope=args.target_category_scope,
                explicit_gate_supervision_scope=args.explicit_gate_supervision_scope,
                positive_label_id=positive_label_id,
                contrastive_group_ids=contrastive_group_ids,
                training_progress=training_progress,
                aux_ramp_start_ratio=args.aux_ramp_start_ratio,
                aux_ramp_duration_ratio=args.aux_ramp_duration_ratio,
                aux_balance_by_active_ratio=args.aux_balance_by_active_ratio,
                aux_balance_target_ratio=args.aux_balance_target_ratio,
                aux_balance_ratio_floor=args.aux_balance_ratio_floor,
                aux_balance_min_scale=args.aux_balance_min_scale,
                aux_balance_max_scale=args.aux_balance_max_scale,
                contrastive_negative_sampling_mode=args.contrastive_negative_sampling_mode,
                contrastive_hard_negative_ratio=args.contrastive_hard_negative_ratio,
            )
            total_loss = loss_dict["total_loss"] / args.gradient_accumulation_steps
            total_loss.backward()
            pending_steps += 1

            if pending_steps == args.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_steps += 1
                pending_steps = 0

            running_total_loss += float(loss_dict["total_loss"].item())
            running_classification_loss += float(loss_dict["classification_loss"].item())
            running_target_alignment_loss += float(loss_dict["target_alignment_loss"].item())
            running_target_supervision_ratio += float(loss_dict["target_supervision_ratio"].item())
            running_target_supervision_pseudo_ratio += float(loss_dict["target_supervision_pseudo_ratio"].item())
            running_target_category_loss += float(loss_dict["target_category_loss"].item())
            running_target_category_supervision_ratio += float(loss_dict["target_category_supervision_ratio"].item())
            running_target_category_pos_weight_mean += float(loss_dict["target_category_pos_weight_mean"].item())
            running_contrastive_loss += float(loss_dict["contrastive_loss"].item())
            running_contrastive_anchor_ratio += float(loss_dict["contrastive_anchor_ratio"].item())
            running_explicit_gate_supervision_loss += float(loss_dict["explicit_gate_supervision_loss"].item())
            running_explicit_gate_supervision_active_ratio += float(
                loss_dict["explicit_gate_supervision_active_ratio"].item()
            )
            running_auxiliary_weight_scale += float(loss_dict["auxiliary_weight_scale"].item())
            running_scaled_target_supervision_weight += float(loss_dict["scaled_target_supervision_weight"].item())
            running_scaled_target_category_weight += float(loss_dict["scaled_target_category_weight"].item())
            running_scaled_contrastive_weight += float(loss_dict["scaled_contrastive_weight"].item())
            running_scaled_explicit_gate_supervision_weight += float(
                loss_dict["scaled_explicit_gate_supervision_weight"].item()
            )
            running_target_supervision_ratio_scale += float(loss_dict["target_supervision_ratio_scale"].item())
            running_target_category_ratio_scale += float(loss_dict["target_category_ratio_scale"].item())
            running_contrastive_ratio_scale += float(loss_dict["contrastive_ratio_scale"].item())
            running_explicit_gate_supervision_ratio_scale += float(
                loss_dict["explicit_gate_supervision_ratio_scale"].item()
            )
            for metric_name, metric_value in loss_dict.items():
                if metric_name.startswith("mean_"):
                    running_optional_metrics[metric_name] += float(metric_value.item())
            num_batches += 1

        if pending_steps > 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_steps += 1

        val_metrics = evaluate_tadi_model(
            model=model,
            dataloader=val_loader,
            device=device,
            id_to_label=id_to_label,
        )
        avg_val_macro = average_metric(val_metrics["main"], "macro_f1")
        val_aux_metrics = val_metrics.get("auxiliary", {})
        epoch_record = {
            "epoch": epoch,
            "label_mode": args.label_mode,
            "train_total_loss": running_total_loss / max(1, num_batches),
            "train_classification_loss": running_classification_loss / max(1, num_batches),
            "train_target_alignment_loss": running_target_alignment_loss / max(1, num_batches),
            "train_target_supervision_ratio": running_target_supervision_ratio / max(1, num_batches),
            "train_target_supervision_pseudo_ratio": running_target_supervision_pseudo_ratio / max(1, num_batches),
            "train_target_category_loss": running_target_category_loss / max(1, num_batches),
            "train_target_category_supervision_ratio": running_target_category_supervision_ratio / max(1, num_batches),
            "train_target_category_pos_weight_mean": running_target_category_pos_weight_mean / max(1, num_batches),
            "train_contrastive_loss": running_contrastive_loss / max(1, num_batches),
            "train_contrastive_anchor_ratio": running_contrastive_anchor_ratio / max(1, num_batches),
            "train_explicit_gate_supervision_loss": running_explicit_gate_supervision_loss / max(1, num_batches),
            "train_explicit_gate_supervision_active_ratio": (
                running_explicit_gate_supervision_active_ratio / max(1, num_batches)
            ),
            "train_auxiliary_weight_scale": running_auxiliary_weight_scale / max(1, num_batches),
            "train_scaled_target_supervision_weight": running_scaled_target_supervision_weight / max(1, num_batches),
            "train_scaled_target_category_weight": running_scaled_target_category_weight / max(1, num_batches),
            "train_scaled_contrastive_weight": running_scaled_contrastive_weight / max(1, num_batches),
            "train_scaled_explicit_gate_supervision_weight": (
                running_scaled_explicit_gate_supervision_weight / max(1, num_batches)
            ),
            "train_target_supervision_ratio_scale": running_target_supervision_ratio_scale / max(1, num_batches),
            "train_target_category_ratio_scale": running_target_category_ratio_scale / max(1, num_batches),
            "train_contrastive_ratio_scale": running_contrastive_ratio_scale / max(1, num_batches),
            "train_explicit_gate_supervision_ratio_scale": (
                running_explicit_gate_supervision_ratio_scale / max(1, num_batches)
            ),
            "optimizer_steps": optimizer_steps,
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "current_lr": float(scheduler.get_last_lr()[0]),
            "train_dataset_counts": train_dataset_counts,
            "train_label_summary": train_label_summary,
            "val_dataset_counts": val_dataset_counts,
            "val_metrics": val_metrics["main"],
            "val_aux_metrics": val_aux_metrics,
            "avg_val_macro_f1": avg_val_macro,
            "avg_val_fine_macro_f1": avg_val_macro,
        }
        if True:
            epoch_record["implicit_attention_temperature"] = args.implicit_attention_temperature
            epoch_record["implicit_injection_scale"] = args.implicit_injection_scale
            epoch_record["explicit_injection_scale"] = args.explicit_injection_scale
            epoch_record["implicit_relation_clamp"] = args.implicit_relation_clamp
            epoch_record["implicit_gate_mode"] = args.implicit_gate_mode
            epoch_record["explicit_gate_mode"] = args.explicit_gate_mode
            epoch_record["implicit_clamp_mode"] = args.implicit_clamp_mode
            epoch_record["implicit_gate_floor"] = args.implicit_gate_floor
            epoch_record["explicit_gate_floor"] = args.explicit_gate_floor
            epoch_record["target_supervision_weight"] = args.target_supervision_weight
            epoch_record["target_supervision_scope"] = args.target_supervision_scope
            epoch_record["target_supervision_pseudo_scale"] = args.target_supervision_pseudo_scale
            epoch_record["pseudo_span_fallback"] = args.pseudo_span_fallback
            epoch_record["pseudo_span_scope"] = args.pseudo_span_scope
            epoch_record["pseudo_span_max_tokens"] = args.pseudo_span_max_tokens
            epoch_record["pseudo_span_min_char_len"] = args.pseudo_span_min_char_len
            epoch_record["target_category_weight"] = args.target_category_weight
            epoch_record["target_category_scope"] = args.target_category_scope
            epoch_record["target_category_merge_tail"] = args.target_category_merge_tail
            epoch_record["target_category_min_freq"] = args.target_category_min_freq
            epoch_record["target_category_loss_mode"] = args.target_category_loss_mode
            epoch_record["target_category_pos_weight_cap"] = args.target_category_pos_weight_cap
            epoch_record["target_category_pos_weight_smoothing"] = args.target_category_pos_weight_smoothing
            epoch_record["contrastive_weight"] = args.contrastive_weight
            epoch_record["contrastive_label_mode"] = args.contrastive_label_mode
            epoch_record["contrastive_temperature"] = args.contrastive_temperature
            epoch_record["hard_negative_k"] = args.hard_negative_k
            epoch_record["contrastive_negative_sampling_mode"] = args.contrastive_negative_sampling_mode
            epoch_record["contrastive_hard_negative_ratio"] = args.contrastive_hard_negative_ratio
            epoch_record["explicit_gate_supervision_weight"] = args.explicit_gate_supervision_weight
            epoch_record["explicit_gate_supervision_scope"] = args.explicit_gate_supervision_scope
            epoch_record["aux_ramp_start_ratio"] = args.aux_ramp_start_ratio
            epoch_record["aux_ramp_duration_ratio"] = args.aux_ramp_duration_ratio
            epoch_record["aux_balance_by_active_ratio"] = args.aux_balance_by_active_ratio
            epoch_record["aux_balance_target_ratio"] = args.aux_balance_target_ratio
            epoch_record["aux_balance_ratio_floor"] = args.aux_balance_ratio_floor
            epoch_record["aux_balance_min_scale"] = args.aux_balance_min_scale
            epoch_record["aux_balance_max_scale"] = args.aux_balance_max_scale
            for metric_name in sorted(running_optional_metrics):
                epoch_record[metric_name.replace("mean_", "train_mean_")] = (
                    running_optional_metrics[metric_name] / max(1, num_batches)
                )
            epoch_record["avg_val_target_entropy"] = average_optional_metric(val_aux_metrics, "target_entropy")
            epoch_record["avg_val_target_peak"] = average_optional_metric(val_aux_metrics, "target_peak")
            epoch_record["avg_val_raw_relation_norm"] = average_optional_metric(val_aux_metrics, "raw_relation_norm")
            epoch_record["avg_val_relation_norm"] = average_optional_metric(val_aux_metrics, "relation_norm")
            epoch_record["avg_val_injection_norm"] = average_optional_metric(val_aux_metrics, "injection_norm")
            epoch_record["avg_val_target_task_cosine"] = average_optional_metric(val_aux_metrics, "target_task_cosine")
            epoch_record["avg_val_gate_raw_value"] = average_optional_metric(val_aux_metrics, "gate_raw_value")
            epoch_record["avg_val_gate_value"] = average_optional_metric(val_aux_metrics, "gate_value")
            epoch_record["avg_val_clamp_ratio"] = average_optional_metric(val_aux_metrics, "clamp_ratio")
            epoch_record["avg_val_explicit_raw_relation_norm"] = average_optional_metric(
                val_aux_metrics, "explicit_raw_relation_norm"
            )
            epoch_record["avg_val_explicit_relation_norm"] = average_optional_metric(
                val_aux_metrics, "explicit_relation_norm"
            )
            epoch_record["avg_val_explicit_injection_norm"] = average_optional_metric(
                val_aux_metrics, "explicit_injection_norm"
            )
            epoch_record["avg_val_explicit_target_task_cosine"] = average_optional_metric(
                val_aux_metrics, "explicit_target_task_cosine"
            )
            epoch_record["avg_val_explicit_coverage_ratio"] = average_optional_metric(
                val_aux_metrics, "explicit_coverage_ratio"
            )
            epoch_record["avg_val_explicit_token_count"] = average_optional_metric(
                val_aux_metrics, "explicit_token_count"
            )
            epoch_record["avg_val_explicit_gate_raw_value"] = average_optional_metric(
                val_aux_metrics, "explicit_gate_raw_value"
            )
            epoch_record["avg_val_explicit_gate_value"] = average_optional_metric(
                val_aux_metrics, "explicit_gate_value"
            )
            epoch_record["avg_val_explicit_clamp_ratio"] = average_optional_metric(
                val_aux_metrics, "explicit_clamp_ratio"
            )
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        checkpoint = {
            "model": snapshot_state_dict(model),
            "epoch": epoch,
            "label_mode": args.label_mode,
            "avg_val_macro_f1": avg_val_macro,
            "avg_val_fine_macro_f1": avg_val_macro,
            "label_to_id": label_to_id,
            "pooling_strategy": args.pooling_strategy,
        }
        last_state = checkpoint
        torch.save(last_state, args.output_dir / "last_model.pt")

        if avg_val_macro >= best_val:
            best_val = avg_val_macro
            best_state = checkpoint
            torch.save(best_state, args.output_dir / "best_model.pt")

    selected_state = best_state if args.selection_strategy == "best" else last_state
    if selected_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(selected_state["model"])
    test_metrics = evaluate_tadi_model(
        model=model,
        dataloader=test_loader,
        device=device,
        id_to_label=id_to_label,
    )
    avg_test_macro = average_metric(test_metrics["main"], "macro_f1")
    test_aux_metrics = test_metrics.get("auxiliary", {})
    result = {
        "method": "TADI",
        "label_mode": args.label_mode,
        "selection_strategy": args.selection_strategy,
        "selected_epoch": selected_state["epoch"],
        "selected_val_macro_f1": selected_state["avg_val_macro_f1"],
        "selected_val_fine_macro_f1": selected_state["avg_val_fine_macro_f1"],
        "avg_test_macro_f1": avg_test_macro,
        "avg_test_fine_macro_f1": avg_test_macro,
        "pooling_strategy": args.pooling_strategy,
        "implicit_attention_temperature": args.implicit_attention_temperature,
        "implicit_injection_scale": args.implicit_injection_scale,
        "explicit_injection_scale": args.explicit_injection_scale,
        "implicit_relation_clamp": args.implicit_relation_clamp,
        "implicit_gate_mode": args.implicit_gate_mode,
        "explicit_gate_mode": args.explicit_gate_mode,
        "implicit_clamp_mode": args.implicit_clamp_mode,
        "implicit_gate_floor": args.implicit_gate_floor,
        "explicit_gate_floor": args.explicit_gate_floor,
        "target_supervision_weight": args.target_supervision_weight,
        "target_supervision_scope": args.target_supervision_scope,
        "target_supervision_pseudo_scale": args.target_supervision_pseudo_scale,
        "pseudo_span_fallback": args.pseudo_span_fallback,
        "pseudo_span_scope": args.pseudo_span_scope,
        "pseudo_span_max_tokens": args.pseudo_span_max_tokens,
        "pseudo_span_min_char_len": args.pseudo_span_min_char_len,
        "target_category_weight": args.target_category_weight,
        "target_category_scope": args.target_category_scope,
        "target_category_merge_tail": args.target_category_merge_tail,
        "target_category_min_freq": args.target_category_min_freq,
        "target_category_loss_mode": args.target_category_loss_mode,
        "target_category_pos_weight_cap": args.target_category_pos_weight_cap,
        "target_category_pos_weight_smoothing": args.target_category_pos_weight_smoothing,
        "contrastive_weight": args.contrastive_weight,
        "contrastive_label_mode": args.contrastive_label_mode,
        "contrastive_temperature": args.contrastive_temperature,
        "hard_negative_k": args.hard_negative_k,
        "contrastive_negative_sampling_mode": args.contrastive_negative_sampling_mode,
        "contrastive_hard_negative_ratio": args.contrastive_hard_negative_ratio,
        "explicit_gate_supervision_weight": args.explicit_gate_supervision_weight,
        "explicit_gate_supervision_scope": args.explicit_gate_supervision_scope,
        "aux_ramp_start_ratio": args.aux_ramp_start_ratio,
        "aux_ramp_duration_ratio": args.aux_ramp_duration_ratio,
        "aux_balance_by_active_ratio": args.aux_balance_by_active_ratio,
        "aux_balance_target_ratio": args.aux_balance_target_ratio,
        "aux_balance_ratio_floor": args.aux_balance_ratio_floor,
        "aux_balance_min_scale": args.aux_balance_min_scale,
        "aux_balance_max_scale": args.aux_balance_max_scale,
        "train_datasets": train_datasets,
        "eval_datasets": eval_datasets,
        "train_dataset_counts": train_dataset_counts,
        "test_dataset_counts": test_dataset_counts,
        "label_space": label_to_id,
        "coarse_label_space": coarse_label_to_id,
        "target_category_size": target_category_size,
        "target_category_original_size": target_category_mapping_meta["original_size"],
        "target_category_tail_size": target_category_mapping_meta["tail_size"],
        "target_category_tail_bucket_id": target_category_mapping_meta["tail_bucket_id"],
        "target_category_pos_weight_mean": (
            float(target_category_pos_weight.mean().item()) if target_category_pos_weight is not None else None
        ),
        "target_category_pos_weight_max": (
            float(target_category_pos_weight.max().item()) if target_category_pos_weight is not None else None
        ),
        "target_category_pos_weight_min": (
            float(target_category_pos_weight.min().item()) if target_category_pos_weight is not None else None
        ),
        "test_metrics": test_metrics["main"],
        "test_fine_metrics": test_metrics["main"],
        "test_aux_metrics": test_aux_metrics,
    }

    (args.output_dir / "args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (args.output_dir / "train_history.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in history) + ("\n" if history else ""),
        encoding="utf-8",
    )
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    train(parse_args())
