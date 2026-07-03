from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--model_name_or_path", type=str, default=r"D:\models\roberta-base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", type=str, default="ethos,olid,ihc,toxicn")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--runs_root", type=Path, default=Path("runs"))
    parser.add_argument("--run_name_prefix", type=str, default="tadi_seed0_u256_b32_e64_ep4")
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=Path("runs/tadi_seed0_u256_b32_e64_ep4_summary.json"),
    )
    parser.add_argument("--implicit_gate_mode", choices=["learned", "fixed_one"], default="learned")
    parser.add_argument("--explicit_gate_mode", choices=["learned", "fixed_one"], default="learned")
    parser.add_argument("--implicit_clamp_mode", choices=["enabled", "disabled"], default="enabled")
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
    parser.add_argument("--implicit_injection_scale", type=float, default=0.05)
    parser.add_argument("--explicit_injection_scale", type=float, default=0.05)
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
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with return code {completed.returncode}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    run_records: dict[str, dict[str, float | int]] = {}
    for dataset in datasets:
        out_dir = args.runs_root / f"{args.run_name_prefix}_{dataset}"
        command = [
            args.python,
            "scripts/train_tadi.py",
            "--train_path",
            "data/unified/train.jsonl",
            "--val_path",
            "data/unified/val.jsonl",
            "--test_path",
            "data/unified/test.jsonl",
            "--metadata_path",
            "data/unified/metadata.json",
            "--model_name_or_path",
            args.model_name_or_path,
            "--output_dir",
            str(out_dir),
            "--train_datasets",
            dataset,
            "--eval_datasets",
            dataset,
            "--label_mode",
            "coarse",
            "--pooling_strategy",
            "cls",
            "--max_length",
            str(args.max_length),
            "--train_batch_size",
            str(args.train_batch_size),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            "2e-5",
            "--weight_decay",
            "0.01",
            "--warmup_ratio",
            "0.1",
            "--dropout",
            "0.1",
            "--implicit_attention_temperature",
            "1.0",
            "--implicit_injection_scale",
            str(args.implicit_injection_scale),
            "--explicit_injection_scale",
            str(args.explicit_injection_scale),
            "--implicit_relation_clamp",
            "12.0",
            "--implicit_gate_mode",
            args.implicit_gate_mode,
            "--explicit_gate_mode",
            args.explicit_gate_mode,
            "--implicit_clamp_mode",
            args.implicit_clamp_mode,
            "--target_supervision_weight",
            str(args.target_supervision_weight),
            "--target_supervision_scope",
            args.target_supervision_scope,
            "--target_supervision_pseudo_scale",
            str(args.target_supervision_pseudo_scale),
            "--pseudo_span_scope",
            args.pseudo_span_scope,
            "--pseudo_span_max_tokens",
            str(args.pseudo_span_max_tokens),
            "--pseudo_span_min_char_len",
            str(args.pseudo_span_min_char_len),
            "--target_category_weight",
            str(args.target_category_weight),
            "--target_category_scope",
            args.target_category_scope,
            "--target_category_min_freq",
            str(args.target_category_min_freq),
            "--target_category_loss_mode",
            args.target_category_loss_mode,
            "--target_category_pos_weight_cap",
            str(args.target_category_pos_weight_cap),
            "--target_category_pos_weight_smoothing",
            str(args.target_category_pos_weight_smoothing),
            "--contrastive_weight",
            str(args.contrastive_weight),
            "--contrastive_label_mode",
            args.contrastive_label_mode,
            "--contrastive_temperature",
            str(args.contrastive_temperature),
            "--hard_negative_k",
            str(args.hard_negative_k),
            "--contrastive_negative_sampling_mode",
            args.contrastive_negative_sampling_mode,
            "--contrastive_hard_negative_ratio",
            str(args.contrastive_hard_negative_ratio),
            "--explicit_gate_supervision_weight",
            str(args.explicit_gate_supervision_weight),
            "--explicit_gate_supervision_scope",
            args.explicit_gate_supervision_scope,
            "--aux_ramp_start_ratio",
            str(args.aux_ramp_start_ratio),
            "--aux_ramp_duration_ratio",
            str(args.aux_ramp_duration_ratio),
            "--aux_balance_target_ratio",
            str(args.aux_balance_target_ratio),
            "--aux_balance_ratio_floor",
            str(args.aux_balance_ratio_floor),
            "--aux_balance_min_scale",
            str(args.aux_balance_min_scale),
            "--aux_balance_max_scale",
            str(args.aux_balance_max_scale),
            "--seed",
            str(args.seed),
            "--num_workers",
            "0",
            "--gradient_accumulation_steps",
            "1",
            "--selection_strategy",
            "best",
            "--device",
            "cuda",
        ]
        if args.pseudo_span_fallback:
            command.append("--pseudo_span_fallback")
        else:
            command.append("--no_pseudo_span_fallback")
        if args.target_category_merge_tail:
            command.append("--target_category_merge_tail")
        else:
            command.append("--no_target_category_merge_tail")
        if args.aux_balance_by_active_ratio:
            command.append("--aux_balance_by_active_ratio")
        else:
            command.append("--no_aux_balance_by_active_ratio")

        print(f"[tadi] {dataset} -> {out_dir}", flush=True)
        run_command(command)

        metrics_path = out_dir / "test_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics: {metrics_path}")
        metrics = load_json(metrics_path)
        run_records[dataset] = {
            "avg_test_macro_f1": float(metrics["avg_test_macro_f1"]),
            "selected_epoch": int(metrics["selected_epoch"]),
            "selected_val_macro_f1": float(metrics["selected_val_macro_f1"]),
        }

        for ckpt_name in ("best_model.pt", "last_model.pt"):
            ckpt_path = out_dir / ckpt_name
            if ckpt_path.exists():
                ckpt_path.unlink()

    average = sum(float(run_records[item]["avg_test_macro_f1"]) for item in datasets) / len(datasets)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "datasets": datasets,
                "uniform_config": {
                    "label_mode": "coarse",
                    "max_length": args.max_length,
                    "train_batch_size": args.train_batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "epochs": args.epochs,
                    "implicit_gate_mode": args.implicit_gate_mode,
                    "explicit_gate_mode": args.explicit_gate_mode,
                    "implicit_clamp_mode": args.implicit_clamp_mode,
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
                    "implicit_injection_scale": args.implicit_injection_scale,
                    "explicit_injection_scale": args.explicit_injection_scale,
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
                },
                "run_records": run_records,
                "avg_test_macro_f1": average,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[tadi] summary -> {args.summary_path}", flush=True)


if __name__ == "__main__":
    main()
