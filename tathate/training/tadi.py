from __future__ import annotations

from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def compute_auxiliary_weight_scale(
    *,
    training_progress: float,
    aux_ramp_start_ratio: float,
    aux_ramp_duration_ratio: float,
) -> float:
    start = max(0.0, min(1.0, float(aux_ramp_start_ratio)))
    duration = max(0.0, float(aux_ramp_duration_ratio))
    end = max(start, min(1.0, start + duration))
    progress = max(0.0, min(1.0, float(training_progress)))
    if progress <= start:
        return 0.0 if start > 0.0 else 1.0 if end <= 0.0 else 0.0
    if progress >= end:
        return 1.0
    if end <= start:
        return 1.0
    return (progress - start) / max(1e-8, end - start)


def compute_active_ratio_balance_scale(
    *,
    active_ratio: float,
    enabled: bool,
    target_ratio: float,
    ratio_floor: float,
    min_scale: float,
    max_scale: float,
) -> float:
    if not enabled:
        return 1.0
    safe_target_ratio = max(1e-6, float(target_ratio))
    safe_ratio_floor = max(1e-6, float(ratio_floor))
    safe_min_scale = max(0.0, float(min_scale))
    safe_max_scale = max(safe_min_scale, float(max_scale))
    safe_active_ratio = max(safe_ratio_floor, float(active_ratio))
    balance_scale = safe_target_ratio / safe_active_ratio
    return max(safe_min_scale, min(safe_max_scale, balance_scale))


def compute_hard_negative_contrastive_loss(
    *,
    representations: torch.Tensor,
    group_labels: torch.Tensor,
    temperature: float,
    hard_negative_k: int,
    negative_sampling_mode: str,
    hard_negative_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if negative_sampling_mode not in {"hard", "mixed"}:
        raise ValueError(f"Unsupported negative_sampling_mode: {negative_sampling_mode}")
    if representations.shape[0] <= 1:
        zero = torch.zeros((), dtype=representations.dtype, device=representations.device)
        return zero, zero
    norm_repr = F.normalize(representations, dim=-1)
    similarity = torch.matmul(norm_repr, norm_repr.T) / float(temperature)
    anchor_losses: list[torch.Tensor] = []
    active_anchor_count = 0
    for index in range(similarity.shape[0]):
        positive_mask = group_labels.eq(group_labels[index])
        positive_mask[index] = False
        negative_mask = ~group_labels.eq(group_labels[index])
        if int(positive_mask.sum().item()) == 0 or int(negative_mask.sum().item()) == 0:
            continue
        positive_logits = similarity[index][positive_mask]
        negative_logits = similarity[index][negative_mask]
        if hard_negative_k > 0 and negative_logits.shape[0] > hard_negative_k:
            if negative_sampling_mode == "hard":
                negative_logits, _ = torch.topk(negative_logits, k=hard_negative_k, dim=0)
            else:
                safe_ratio = max(0.0, min(1.0, float(hard_negative_ratio)))
                hard_keep = int(round(float(hard_negative_k) * safe_ratio))
                hard_keep = max(0, min(int(hard_negative_k), int(negative_logits.shape[0]), hard_keep))
                random_keep = max(0, int(hard_negative_k) - hard_keep)

                selected_parts: list[torch.Tensor] = []
                remaining_logits = negative_logits
                if hard_keep > 0:
                    hard_values, hard_indices = torch.topk(negative_logits, k=hard_keep, dim=0)
                    selected_parts.append(hard_values)
                    if random_keep > 0:
                        remaining_mask = torch.ones_like(negative_logits, dtype=torch.bool)
                        remaining_mask[hard_indices] = False
                        remaining_logits = negative_logits[remaining_mask]

                if random_keep > 0 and remaining_logits.numel() > 0:
                    random_take = min(random_keep, int(remaining_logits.numel()))
                    random_indices = torch.randperm(remaining_logits.numel(), device=remaining_logits.device)[:random_take]
                    selected_parts.append(remaining_logits.index_select(0, random_indices))

                if selected_parts:
                    negative_logits = torch.cat(selected_parts, dim=0)
                else:
                    negative_logits, _ = torch.topk(negative_logits, k=hard_negative_k, dim=0)
        numerator = torch.logsumexp(positive_logits, dim=0)
        denominator = torch.logsumexp(torch.cat([positive_logits, negative_logits], dim=0), dim=0)
        anchor_losses.append(-(numerator - denominator))
        active_anchor_count += 1
    if not anchor_losses:
        zero = torch.zeros((), dtype=representations.dtype, device=representations.device)
        return zero, zero
    loss = torch.stack(anchor_losses, dim=0).mean()
    active_ratio = torch.tensor(
        active_anchor_count / max(1, similarity.shape[0]),
        dtype=representations.dtype,
        device=representations.device,
    )
    return loss, active_ratio


def compute_tadi_losses(
    *,
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    target_supervision_mask: torch.Tensor | None,
    target_supervision_flag: torch.Tensor | None,
    target_supervision_pseudo_flag: torch.Tensor | None,
    target_category_targets: torch.Tensor | None,
    target_category_flag: torch.Tensor | None,
    classification_criterion,
    target_supervision_weight: float,
    target_supervision_pseudo_scale: float,
    target_category_weight: float,
    target_category_loss_mode: str,
    target_category_pos_weight: torch.Tensor | None,
    contrastive_weight: float,
    contrastive_temperature: float,
    hard_negative_k: int,
    explicit_gate_supervision_weight: float,
    target_supervision_scope: str,
    target_category_scope: str,
    explicit_gate_supervision_scope: str,
    positive_label_id: int,
    contrastive_group_ids: torch.Tensor | None,
    training_progress: float,
    aux_ramp_start_ratio: float,
    aux_ramp_duration_ratio: float,
    aux_balance_by_active_ratio: bool,
    aux_balance_target_ratio: float,
    aux_balance_ratio_floor: float,
    aux_balance_min_scale: float,
    aux_balance_max_scale: float,
    contrastive_negative_sampling_mode: str,
    contrastive_hard_negative_ratio: float,
) -> dict[str, torch.Tensor]:
    if target_supervision_scope not in {"all", "positive_only"}:
        raise ValueError(f"Unsupported target_supervision_scope: {target_supervision_scope}")
    if target_category_scope not in {"all", "positive_only"}:
        raise ValueError(f"Unsupported target_category_scope: {target_category_scope}")
    if target_category_loss_mode not in {"bce", "weighted_bce"}:
        raise ValueError(f"Unsupported target_category_loss_mode: {target_category_loss_mode}")
    if explicit_gate_supervision_scope not in {"all", "positive_only"}:
        raise ValueError(f"Unsupported explicit_gate_supervision_scope: {explicit_gate_supervision_scope}")
    auxiliary_weight_scale = compute_auxiliary_weight_scale(
        training_progress=training_progress,
        aux_ramp_start_ratio=aux_ramp_start_ratio,
        aux_ramp_duration_ratio=aux_ramp_duration_ratio,
    )
    scaled_target_supervision_weight = float(target_supervision_weight) * auxiliary_weight_scale
    scaled_target_category_weight = float(target_category_weight) * auxiliary_weight_scale
    scaled_contrastive_weight = float(contrastive_weight) * auxiliary_weight_scale
    scaled_explicit_gate_supervision_weight = float(explicit_gate_supervision_weight) * auxiliary_weight_scale

    classification_loss = classification_criterion(outputs["logits"], labels)
    total_loss = classification_loss

    target_alignment_loss = torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device)
    target_supervision_ratio = torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device)
    target_supervision_pseudo_ratio = torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device)
    target_supervision_ratio_scale = torch.ones((), dtype=classification_loss.dtype, device=classification_loss.device)
    if (
        "implicit_target_attention" in outputs
        and target_supervision_mask is not None
        and target_supervision_flag is not None
    ):
        supervision_mask = target_supervision_mask.float()
        supervision_flag = target_supervision_flag.float() > 0.5
        supervision_sum = supervision_mask.sum(dim=-1)
        valid_supervision = supervision_flag & (supervision_sum > 0.0)
        if target_supervision_scope == "positive_only":
            valid_supervision = valid_supervision & labels.eq(int(positive_label_id))
        target_supervision_ratio = valid_supervision.float().mean()
        target_supervision_ratio_scale = torch.tensor(
            compute_active_ratio_balance_scale(
                active_ratio=float(target_supervision_ratio.item()),
                enabled=aux_balance_by_active_ratio,
                target_ratio=aux_balance_target_ratio,
                ratio_floor=aux_balance_ratio_floor,
                min_scale=aux_balance_min_scale,
                max_scale=aux_balance_max_scale,
            ),
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        )
        effective_target_supervision_weight = scaled_target_supervision_weight * float(target_supervision_ratio_scale.item())
        if bool(valid_supervision.any()):
            target_distribution = supervision_mask / supervision_sum.unsqueeze(-1).clamp_min(1e-8)
            attention_distribution = outputs["implicit_target_attention"].clamp_min(1e-8)
            per_sample_target_loss = -(
                target_distribution[valid_supervision] * torch.log(attention_distribution[valid_supervision])
            ).sum(dim=-1)
            pseudo_flags = torch.zeros_like(valid_supervision, dtype=torch.bool)
            if target_supervision_pseudo_flag is not None:
                pseudo_flags = target_supervision_pseudo_flag.float() > 0.5
            pseudo_valid = pseudo_flags[valid_supervision]
            target_supervision_pseudo_ratio = pseudo_valid.float().mean()
            pseudo_scale = max(0.0, float(target_supervision_pseudo_scale))
            per_sample_weight = torch.ones_like(per_sample_target_loss)
            if pseudo_scale != 1.0:
                per_sample_weight = torch.where(
                    pseudo_valid,
                    torch.full_like(per_sample_target_loss, pseudo_scale),
                    per_sample_weight,
                )
            target_alignment_loss = (per_sample_target_loss * per_sample_weight).sum() / per_sample_weight.sum().clamp_min(1e-8)
            total_loss = total_loss + effective_target_supervision_weight * target_alignment_loss

    target_category_loss = torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device)
    target_category_supervision_ratio = torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device)
    target_category_pos_weight_mean = torch.ones((), dtype=classification_loss.dtype, device=classification_loss.device)
    target_category_ratio_scale = torch.ones((), dtype=classification_loss.dtype, device=classification_loss.device)
    if (
        scaled_target_category_weight > 0
        and "target_category_logits" in outputs
        and target_category_targets is not None
        and target_category_flag is not None
        and target_category_targets.shape[-1] > 0
    ):
        category_flag = target_category_flag.float() > 0.5
        if target_category_scope == "positive_only":
            category_flag = category_flag & labels.eq(int(positive_label_id))
        target_category_supervision_ratio = category_flag.float().mean()
        target_category_ratio_scale = torch.tensor(
            compute_active_ratio_balance_scale(
                active_ratio=float(target_category_supervision_ratio.item()),
                enabled=aux_balance_by_active_ratio,
                target_ratio=aux_balance_target_ratio,
                ratio_floor=aux_balance_ratio_floor,
                min_scale=aux_balance_min_scale,
                max_scale=aux_balance_max_scale,
            ),
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        )
        effective_target_category_weight = scaled_target_category_weight * float(target_category_ratio_scale.item())
        if bool(category_flag.any()):
            category_logits = outputs["target_category_logits"][category_flag]
            category_targets = target_category_targets.float()[category_flag]
            category_pos_weight = None
            if (
                target_category_loss_mode == "weighted_bce"
                and target_category_pos_weight is not None
                and target_category_pos_weight.numel() == category_targets.shape[-1]
            ):
                category_pos_weight = target_category_pos_weight.to(
                    device=category_logits.device,
                    dtype=category_logits.dtype,
                )
                target_category_pos_weight_mean = category_pos_weight.mean()
            target_category_loss = F.binary_cross_entropy_with_logits(
                category_logits,
                category_targets,
                pos_weight=category_pos_weight,
            )
            total_loss = total_loss + effective_target_category_weight * target_category_loss

    contrastive_loss = torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device)
    contrastive_anchor_ratio = torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device)
    contrastive_ratio_scale = torch.ones((), dtype=classification_loss.dtype, device=classification_loss.device)
    if scaled_contrastive_weight > 0 and "task_repr" in outputs:
        group_ids = contrastive_group_ids if contrastive_group_ids is not None else labels
        contrastive_loss, contrastive_anchor_ratio = compute_hard_negative_contrastive_loss(
            representations=outputs["task_repr"],
            group_labels=group_ids,
            temperature=max(1e-5, float(contrastive_temperature)),
            hard_negative_k=max(1, int(hard_negative_k)),
            negative_sampling_mode=contrastive_negative_sampling_mode,
            hard_negative_ratio=contrastive_hard_negative_ratio,
        )
        contrastive_ratio_scale = torch.tensor(
            compute_active_ratio_balance_scale(
                active_ratio=float(contrastive_anchor_ratio.item()),
                enabled=aux_balance_by_active_ratio,
                target_ratio=aux_balance_target_ratio,
                ratio_floor=aux_balance_ratio_floor,
                min_scale=aux_balance_min_scale,
                max_scale=aux_balance_max_scale,
            ),
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        )
        effective_contrastive_weight = scaled_contrastive_weight * float(contrastive_ratio_scale.item())
        total_loss = total_loss + effective_contrastive_weight * contrastive_loss

    explicit_gate_supervision_loss = torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device)
    explicit_gate_supervision_active_ratio = torch.zeros((), dtype=classification_loss.dtype, device=classification_loss.device)
    explicit_gate_supervision_ratio_scale = torch.ones((), dtype=classification_loss.dtype, device=classification_loss.device)
    if (
        scaled_explicit_gate_supervision_weight > 0
        and "explicit_candidate_flag" in outputs
        and "explicit_gate_value" in outputs
    ):
        explicit_targets = outputs["explicit_candidate_flag"].float()
        if explicit_gate_supervision_scope == "positive_only":
            explicit_targets = explicit_targets * labels.eq(int(positive_label_id)).float()
        explicit_gate_supervision_active_ratio = explicit_targets.mean()
        explicit_gate_supervision_ratio_scale = torch.tensor(
            compute_active_ratio_balance_scale(
                active_ratio=float(explicit_gate_supervision_active_ratio.item()),
                enabled=aux_balance_by_active_ratio,
                target_ratio=aux_balance_target_ratio,
                ratio_floor=aux_balance_ratio_floor,
                min_scale=aux_balance_min_scale,
                max_scale=aux_balance_max_scale,
            ),
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        )
        effective_explicit_gate_supervision_weight = scaled_explicit_gate_supervision_weight * float(
            explicit_gate_supervision_ratio_scale.item()
        )
        explicit_gate = outputs["explicit_gate_value"].float().clamp(min=1e-6, max=1.0 - 1e-6)
        explicit_gate_supervision_loss = -(
            explicit_targets * torch.log(explicit_gate) + (1.0 - explicit_targets) * torch.log(1.0 - explicit_gate)
        ).mean()
        total_loss = total_loss + effective_explicit_gate_supervision_weight * explicit_gate_supervision_loss

    effective_target_supervision_weight = scaled_target_supervision_weight * float(target_supervision_ratio_scale.item())
    effective_target_category_weight = scaled_target_category_weight * float(target_category_ratio_scale.item())
    effective_contrastive_weight = scaled_contrastive_weight * float(contrastive_ratio_scale.item())
    effective_explicit_gate_supervision_weight = scaled_explicit_gate_supervision_weight * float(
        explicit_gate_supervision_ratio_scale.item()
    )

    loss_dict = {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "target_alignment_loss": target_alignment_loss,
        "target_supervision_ratio": target_supervision_ratio,
        "target_supervision_pseudo_ratio": target_supervision_pseudo_ratio,
        "target_category_loss": target_category_loss,
        "target_category_supervision_ratio": target_category_supervision_ratio,
        "target_category_pos_weight_mean": target_category_pos_weight_mean,
        "contrastive_loss": contrastive_loss,
        "contrastive_anchor_ratio": contrastive_anchor_ratio,
        "explicit_gate_supervision_loss": explicit_gate_supervision_loss,
        "explicit_gate_supervision_active_ratio": explicit_gate_supervision_active_ratio,
        "auxiliary_weight_scale": torch.tensor(
            auxiliary_weight_scale,
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        ),
        "scaled_target_supervision_weight": torch.tensor(
            effective_target_supervision_weight,
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        ),
        "scaled_target_category_weight": torch.tensor(
            effective_target_category_weight,
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        ),
        "scaled_contrastive_weight": torch.tensor(
            effective_contrastive_weight,
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        ),
        "scaled_explicit_gate_supervision_weight": torch.tensor(
            effective_explicit_gate_supervision_weight,
            dtype=classification_loss.dtype,
            device=classification_loss.device,
        ),
        "target_supervision_ratio_scale": target_supervision_ratio_scale,
        "target_category_ratio_scale": target_category_ratio_scale,
        "contrastive_ratio_scale": contrastive_ratio_scale,
        "explicit_gate_supervision_ratio_scale": explicit_gate_supervision_ratio_scale,
    }
    optional_keys = [
        "implicit_target_entropy",
        "implicit_target_peak",
        "implicit_raw_relation_norm",
        "implicit_relation_norm",
        "implicit_injection_norm",
        "implicit_target_task_cosine",
        "implicit_target_fine_cosine",
        "implicit_gate_raw_value",
        "implicit_gate_value",
        "implicit_relation_clamp_ratio",
        "explicit_raw_relation_norm",
        "explicit_relation_norm",
        "explicit_injection_norm",
        "explicit_target_task_cosine",
        "explicit_coverage_ratio",
        "explicit_token_count",
        "explicit_gate_raw_value",
        "explicit_gate_value",
        "explicit_relation_clamp_ratio",
        "target_category_top_prob",
    ]
    for key in optional_keys:
        if key in outputs:
            loss_dict[f"mean_{key}"] = outputs[key].mean()
    return loss_dict


def _compute_label_metrics(
    *,
    outputs: dict[str, dict[str, list[int]]],
    id_to_label: dict[int, str],
) -> dict[str, dict[str, float | int | list[int] | list[str] | dict[str, int]]]:
    metrics: dict[str, dict[str, float | int | list[int] | list[str] | dict[str, int]]] = {}
    for dataset_name, values in outputs.items():
        active_label_ids = sorted(set(values["y_true"]) | set(values["y_pred"]))
        true_counts = Counter(values["y_true"])
        pred_counts = Counter(values["y_pred"])
        metrics[dataset_name] = {
            "num_examples": len(values["y_true"]),
            "macro_f1": f1_score(values["y_true"], values["y_pred"], average="macro", zero_division=0),
            "accuracy": accuracy_score(values["y_true"], values["y_pred"]),
            "active_label_ids": active_label_ids,
            "active_label_names": [id_to_label[index] for index in active_label_ids],
            "true_label_counts": {id_to_label[index]: int(true_counts[index]) for index in sorted(true_counts)},
            "pred_label_counts": {id_to_label[index]: int(pred_counts[index]) for index in sorted(pred_counts)},
        }
    return metrics


def evaluate_tadi_model(
    *,
    model,
    dataloader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
) -> dict[str, dict[str, dict[str, float | int | list[int] | list[str] | dict[str, int]]]]:
    model.eval()
    label_outputs: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    auxiliary_sums: dict[str, Counter[str]] = defaultdict(Counter)
    auxiliary_counts: Counter[str] = Counter()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            special_tokens_mask = batch.get("special_tokens_mask")
            if isinstance(special_tokens_mask, torch.Tensor):
                special_tokens_mask = special_tokens_mask.to(device)
            explicit_candidate_mask = batch.get("explicit_candidate_mask")
            if isinstance(explicit_candidate_mask, torch.Tensor):
                explicit_candidate_mask = explicit_candidate_mask.to(device)
            labels = batch["label_id"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                explicit_candidate_mask=explicit_candidate_mask,
            )
            predictions = torch.argmax(outputs["logits"], dim=-1)

            for dataset_name in sorted(set(batch["dataset"])):
                indices = [index for index, name in enumerate(batch["dataset"]) if name == dataset_name]
                if not indices:
                    continue
                index_tensor = torch.tensor(indices, device=device, dtype=torch.long)

                label_outputs[dataset_name]["y_true"].extend(labels.index_select(0, index_tensor).detach().cpu().tolist())
                label_outputs[dataset_name]["y_pred"].extend(
                    predictions.index_select(0, index_tensor).detach().cpu().tolist()
                )

                auxiliary_counts[dataset_name] += len(indices)
                optional_keys = {
                    "implicit_target_entropy": "target_entropy",
                    "implicit_target_peak": "target_peak",
                    "implicit_raw_relation_norm": "raw_relation_norm",
                    "implicit_relation_norm": "relation_norm",
                    "implicit_injection_norm": "injection_norm",
                    "implicit_target_task_cosine": "target_task_cosine",
                    "implicit_target_fine_cosine": "target_task_cosine",
                    "implicit_gate_raw_value": "gate_raw_value",
                    "implicit_gate_value": "gate_value",
                    "implicit_relation_clamp_ratio": "clamp_ratio",
                    "explicit_raw_relation_norm": "explicit_raw_relation_norm",
                    "explicit_relation_norm": "explicit_relation_norm",
                    "explicit_injection_norm": "explicit_injection_norm",
                    "explicit_target_task_cosine": "explicit_target_task_cosine",
                    "explicit_coverage_ratio": "explicit_coverage_ratio",
                    "explicit_token_count": "explicit_token_count",
                    "explicit_gate_raw_value": "explicit_gate_raw_value",
                    "explicit_gate_value": "explicit_gate_value",
                    "explicit_relation_clamp_ratio": "explicit_clamp_ratio",
                }
                for source_key, metric_key in optional_keys.items():
                    if source_key not in outputs:
                        continue
                    auxiliary_sums[dataset_name][metric_key] += float(
                        outputs[source_key].index_select(0, index_tensor).sum().item()
                    )

    auxiliary_metrics = {}
    for dataset_name, totals in auxiliary_sums.items():
        count = max(1, auxiliary_counts[dataset_name])
        auxiliary_metrics[dataset_name] = {
            metric_name: float(metric_value) / float(count)
            for metric_name, metric_value in sorted(totals.items())
        }

    result = {
        "main": _compute_label_metrics(outputs=label_outputs, id_to_label=id_to_label),
    }
    if auxiliary_metrics:
        result["auxiliary"] = auxiliary_metrics
    return result
