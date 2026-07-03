from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from tathate.data import build_label_space, invert_label_space
from tathate.data.unified_dataset import UnifiedJsonlDataset


def unwrap_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Unsupported checkpoint payload.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_label_metadata(metadata: dict, label_mode: str, datasets: list[str]) -> dict[str, object]:
    label_key = "coarse_label" if label_mode == "coarse" else "fine_label"
    label_to_id = build_label_space(metadata, datasets, label_mode)
    id_to_label = invert_label_space(label_to_id)
    coarse_label_to_id = {
        label: index
        for label, index in sorted(metadata["coarse_label_to_id"].items(), key=lambda item: item[1])
    }
    coarse_id_to_label = invert_label_space(coarse_label_to_id)
    return {
        "label_key": label_key,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "coarse_label_to_id": coarse_label_to_id,
        "coarse_id_to_label": coarse_id_to_label,
    }


def build_classification_collate_fn(
    tokenizer,
    max_length: int,
    *,
    label_key: str,
    label_to_id: dict[str, int],
    coarse_label_to_id: dict[str, int],
    target_category_size: int = 0,
    target_category_id_map: dict[int, int] | None = None,
    pseudo_span_fallback: bool = False,
    pseudo_span_scope: str = "positive_only",
    pseudo_span_max_tokens: int = 8,
    pseudo_span_min_char_len: int = 2,
):
    if pseudo_span_scope not in {"all", "positive_only"}:
        raise ValueError(f"Unsupported pseudo_span_scope: {pseudo_span_scope}")

    english_stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }

    def find_target_spans_from_categories(text: str, categories: list[str]) -> list[list[int]]:
        lowered_text = str(text).lower()
        spans: list[list[int]] = []
        seen: set[tuple[int, int]] = set()
        def add_phrase_matches(phrase: str) -> None:
            nonlocal spans, seen
            lowered_phrase = phrase.lower()
            cursor = 0
            while True:
                index = lowered_text.find(lowered_phrase, cursor)
                if index < 0:
                    break
                end = index + len(lowered_phrase)
                left_ok = index == 0 or not lowered_text[index - 1].isalnum()
                right_ok = end == len(lowered_text) or not lowered_text[end].isalnum()
                if left_ok and right_ok:
                    pair = (index, end)
                    if pair not in seen:
                        seen.add(pair)
                        spans.append([index, end])
                cursor = index + 1

        for category in categories:
            phrase = str(category).strip()
            if len(phrase) < 3:
                continue
            add_phrase_matches(phrase)
            if len(spans) > 0:
                continue
            terms = [item for item in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", phrase) if item]
            for term in terms:
                if len(term) >= 3:
                    add_phrase_matches(term)
        return spans

    def build_supervision_mask(offset_mapping: torch.Tensor, spans: list[list[int]]) -> torch.Tensor:
        token_mask = torch.zeros(offset_mapping.shape[0], dtype=torch.float32)
        if not spans:
            return token_mask
        for index in range(offset_mapping.shape[0]):
            start = int(offset_mapping[index, 0].item())
            end = int(offset_mapping[index, 1].item())
            if end <= start:
                continue
            for span_start, span_end in spans:
                if int(span_end) <= int(span_start):
                    continue
                if end > int(span_start) and start < int(span_end):
                    token_mask[index] = 1.0
                    break
        return token_mask

    def build_pseudo_mask(
        offset_mapping: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        text: str,
    ) -> torch.Tensor:
        token_mask = torch.zeros(offset_mapping.shape[0], dtype=torch.float32)
        candidates: list[tuple[int, int]] = []
        fallback_indices: list[int] = []
        min_char_len = max(1, int(pseudo_span_min_char_len))
        max_tokens = max(1, int(pseudo_span_max_tokens))
        for index in range(offset_mapping.shape[0]):
            start = int(offset_mapping[index, 0].item())
            end = int(offset_mapping[index, 1].item())
            if end <= start:
                continue
            if bool(special_tokens_mask[index].item()):
                continue
            token_text = str(text[start:end]).strip().lower()
            if not token_text:
                continue
            fallback_indices.append(index)
            normalized = "".join(char for char in token_text if char.isalnum())
            if not normalized:
                continue
            if len(normalized) < min_char_len:
                continue
            if normalized in english_stopwords:
                continue
            candidates.append((index, len(normalized)))
        if not candidates and fallback_indices:
            candidates = [(index, 1) for index in fallback_indices]
        if not candidates:
            return token_mask
        selected = sorted(candidates, key=lambda item: (-item[1], item[0]))[:max_tokens]
        for token_index, _ in selected:
            token_mask[token_index] = 1.0
        return token_mask

    def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        texts = [item["text"] for item in batch]
        encoding = tokenizer.batch_encode_plus(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        target_supervision_masks = []
        target_supervision_flags = []
        target_supervision_pseudo_flags = []
        explicit_candidate_masks = []
        explicit_candidate_flags = []
        target_category_targets = []
        target_category_flags = []
        for row_index, item in enumerate(batch):
            rationale_spans = item.get("rationale_spans", [])
            target_spans = item.get("target_spans", [])
            explicit_spans = list(target_spans)
            if not explicit_spans:
                explicit_spans.extend(find_target_spans_from_categories(item["text"], item.get("target_categories", [])))
            rationale_mask = build_supervision_mask(encoding["offset_mapping"][row_index], list(rationale_spans))
            explicit_mask = build_supervision_mask(encoding["offset_mapping"][row_index], explicit_spans)
            pseudo_flag = 0.0
            if (
                pseudo_span_fallback
                and float(explicit_mask.sum().item()) <= 0.0
                and float(rationale_mask.sum().item()) <= 0.0
            ):
                allow_pseudo = pseudo_span_scope == "all" or str(item.get("coarse_label", "")) == "hate"
                if allow_pseudo:
                    pseudo_mask = build_pseudo_mask(
                        encoding["offset_mapping"][row_index],
                        encoding["special_tokens_mask"][row_index],
                        item["text"],
                    )
                    if float(pseudo_mask.sum().item()) > 0.0:
                        explicit_mask = pseudo_mask
                        pseudo_flag = 1.0
            mask = torch.maximum(rationale_mask, explicit_mask)
            has_supervision = bool(mask.sum().item() > 0)
            target_supervision_masks.append(mask)
            target_supervision_flags.append(1.0 if has_supervision else 0.0)
            target_supervision_pseudo_flags.append(pseudo_flag if has_supervision else 0.0)
            has_explicit = bool(explicit_mask.sum().item() > 0)
            explicit_candidate_masks.append(explicit_mask)
            explicit_candidate_flags.append(1.0 if has_explicit else 0.0)
            raw_target_ids = [int(value) for value in item.get("target_category_ids", [])]
            if target_category_id_map:
                target_ids = [target_category_id_map[item_id] for item_id in raw_target_ids if item_id in target_category_id_map]
            else:
                target_ids = raw_target_ids
            target_ids = sorted(set(target_ids))
            if target_category_size > 0:
                target_vector = torch.zeros(target_category_size, dtype=torch.float32)
                for target_id in target_ids:
                    if 0 <= target_id < target_category_size:
                        target_vector[target_id] = 1.0
            else:
                target_vector = torch.empty(0, dtype=torch.float32)
            target_category_targets.append(target_vector)
            target_category_flags.append(1.0 if target_ids else 0.0)
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "special_tokens_mask": encoding["special_tokens_mask"],
            "target_supervision_mask": torch.stack(target_supervision_masks, dim=0),
            "target_supervision_flag": torch.tensor(target_supervision_flags, dtype=torch.float32),
            "target_supervision_pseudo_flag": torch.tensor(target_supervision_pseudo_flags, dtype=torch.float32),
            "explicit_candidate_mask": torch.stack(explicit_candidate_masks, dim=0),
            "explicit_candidate_flag": torch.tensor(explicit_candidate_flags, dtype=torch.float32),
            "target_category_targets": torch.stack(target_category_targets, dim=0),
            "target_category_flag": torch.tensor(target_category_flags, dtype=torch.float32),
            "label_id": torch.tensor([label_to_id[item[label_key]] for item in batch], dtype=torch.long),
            "coarse_label_id": torch.tensor([coarse_label_to_id[item["coarse_label"]] for item in batch], dtype=torch.long),
            "dataset": [item["dataset"] for item in batch],
            "example_id": [item["example_id"] for item in batch],
            "label_name": [item[label_key] for item in batch],
            "coarse_label_name": [item["coarse_label"] for item in batch],
            "fine_label_name": [item["fine_label"] for item in batch],
            "fine_group_name": [f"{item['dataset']}::{item['fine_label']}" for item in batch],
        }

    return collate


def build_classification_dataloader(
    dataset: UnifiedJsonlDataset,
    *,
    tokenizer,
    max_length: int,
    label_key: str,
    label_to_id: dict[str, int],
    coarse_label_to_id: dict[str, int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    balance_datasets: bool,
    dataset_sampling_power: float,
    target_category_size: int = 0,
    target_category_id_map: dict[int, int] | None = None,
    pseudo_span_fallback: bool = False,
    pseudo_span_scope: str = "positive_only",
    pseudo_span_max_tokens: int = 8,
    pseudo_span_min_char_len: int = 2,
) -> DataLoader:
    collate_fn = build_classification_collate_fn(
        tokenizer,
        max_length,
        label_key=label_key,
        label_to_id=label_to_id,
        coarse_label_to_id=coarse_label_to_id,
        target_category_size=target_category_size,
        target_category_id_map=target_category_id_map,
        pseudo_span_fallback=pseudo_span_fallback,
        pseudo_span_scope=pseudo_span_scope,
        pseudo_span_max_tokens=pseudo_span_max_tokens,
        pseudo_span_min_char_len=pseudo_span_min_char_len,
    )
    sampler = None
    if shuffle and balance_datasets:
        weights = dataset.build_sample_weights(power=dataset_sampling_power)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    elif shuffle:
        sampler = RandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

def average_metric(metrics: dict[str, dict[str, float]], key: str) -> float:
    if not metrics:
        return 0.0
    values = [float(item[key]) for item in metrics.values()]
    return sum(values) / len(values)


def load_split_datasets(
    *,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    train_datasets: list[str],
    eval_datasets: list[str],
    max_train_examples: int | None,
    max_eval_examples: int | None,
    seed: int,
) -> tuple[UnifiedJsonlDataset, UnifiedJsonlDataset, UnifiedJsonlDataset]:
    train_dataset = UnifiedJsonlDataset(train_path, datasets=train_datasets, max_examples=max_train_examples, seed=seed)
    val_dataset = UnifiedJsonlDataset(val_path, datasets=eval_datasets, max_examples=max_eval_examples, seed=seed)
    test_dataset = UnifiedJsonlDataset(test_path, datasets=eval_datasets, max_examples=max_eval_examples, seed=seed)
    return train_dataset, val_dataset, test_dataset
