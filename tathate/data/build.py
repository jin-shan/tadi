from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split

from .schema import (
    COARSE_LABEL_TO_ID,
    DATASET_IDS,
    FINE_LABEL_TO_ID,
    FINE_TO_COARSE,
    UnifiedExample,
)


TARGET_IGNORE = {"none", "other", "null", "nan", ""}
SPACE_RE = re.compile(r"\s+")
TOXICN_FINE_LABELS = {
    0: "non-toxic",
    1: "offensive",
    2: "hate speech",
}
TOXICN_EXPRESSION_LABELS = {
    0: "non-hate",
    1: "explicit",
    2: "implicit",
    3: "reporting",
}
TOXICN_TARGET_NAMES = ["LGBTQ", "Region", "Sexism", "Racism", "Others", "non-hate"]
IHC_STAGE1_LABELS = {
    "not_hate": "not hate",
    "implicit_hate": "implicit hate",
    "explicit_hate": "explicit hate",
}


def normalize_text(text: str) -> str:
    value = html.unescape(str(text))
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u200b", " ").replace("\ufeff", " ")
    value = SPACE_RE.sub(" ", value).strip()
    return value


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_delimited(path: Path, *, delimiter: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def load_label_pairs(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return {normalize_text(row[0]): normalize_text(row[1]) for row in reader if row}


def parse_index_vector(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list payload, got: {value}")
        return [int(item) for item in parsed]
    raise TypeError(f"Unsupported vector payload type: {type(value)!r}")


def majority_vote(values: list[str], priority: dict[str, int] | None = None) -> tuple[str, dict[str, int], bool]:
    counts = Counter(values)
    max_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == max_count]
    if priority is None:
        label = sorted(candidates)[0]
    else:
        label = sorted(candidates, key=lambda item: (priority.get(item, 10**6), item))[0]
    return label, dict(sorted(counts.items())), len(candidates) > 1


def join_tokens(tokens: list[str]) -> tuple[str, list[list[int]]]:
    parts: list[str] = []
    spans: list[list[int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        spans.append([start, cursor])
    text = "".join(parts)
    return text, spans


def aggregate_rationales(rationales: list[list[int]], expected_length: int) -> tuple[list[int], list[float]]:
    if not rationales:
        return [], []
    vote_counts = [0] * expected_length
    for rationale in rationales:
        for index in range(expected_length):
            value = rationale[index] if index < len(rationale) else 0
            vote_counts[index] += int(value)
    threshold = math.ceil(len(rationales) / 2)
    mask = [1 if value >= threshold else 0 for value in vote_counts]
    fractions = [value / len(rationales) for value in vote_counts]
    return mask, fractions


def spans_from_mask(mask: list[int], token_spans: list[list[int]]) -> list[list[int]]:
    spans: list[list[int]] = []
    start: int | None = None
    end: int | None = None
    for index, flag in enumerate(mask):
        if flag:
            token_start, token_end = token_spans[index]
            if start is None:
                start = token_start
            end = token_end
        elif start is not None:
            spans.append([start, end if end is not None else start])
            start = None
            end = None
    if start is not None:
        spans.append([start, end if end is not None else start])
    return spans


def normalize_target_name(value: str) -> str:
    return normalize_text(value).strip()


def collect_target_counts(target_lists: list[list[str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    canonical: dict[str, str] = {}
    for target_list in target_lists:
        for target in target_list:
            normalized = normalize_target_name(target)
            key = normalized.casefold()
            if key in TARGET_IGNORE:
                continue
            canonical.setdefault(key, normalized)
            counts[key] += 1
    return {canonical[key]: counts[key] for key in sorted(counts.keys())}


def find_exact_target_spans(tokens: list[str], token_spans: list[list[int]], target_categories: list[str]) -> list[list[int]]:
    normalized_tokens = [normalize_target_name(token).casefold() for token in tokens]
    spans: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for target in target_categories:
        parts = [part for part in normalize_target_name(target).casefold().split(" ") if part]
        if not parts:
            continue
        width = len(parts)
        for start_index in range(0, len(normalized_tokens) - width + 1):
            if normalized_tokens[start_index : start_index + width] == parts:
                char_start = token_spans[start_index][0]
                char_end = token_spans[start_index + width - 1][1]
                pair = (char_start, char_end)
                if pair not in seen:
                    seen.add(pair)
                    spans.append([char_start, char_end])
    return spans


def build_ethos_examples(data_root: Path) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for split in ["train", "val", "test"]:
        rows = read_json(data_root / "ethos" / f"{split}.json")
        examples: list[dict] = []
        for index, row in enumerate(rows):
            fine_label = normalize_text(row["ans_text"]).lower()
            coarse_label = FINE_TO_COARSE["ethos"][fine_label]
            text = normalize_text(row["text"])
            example = UnifiedExample(
                example_id=f"ethos-{split}-{index}",
                dataset="ethos",
                dataset_id=DATASET_IDS["ethos"],
                split=split,
                text=text,
                raw_text=row["text"],
                coarse_label=coarse_label,
                coarse_label_id=COARSE_LABEL_TO_ID[coarse_label],
                fine_label=fine_label,
                fine_label_id=FINE_LABEL_TO_ID["ethos"][fine_label],
                label_vote_counts={fine_label: 1},
                meta={"raw_index": index},
            )
            examples.append(example.to_dict())
        output[split] = examples
    return output


def normalize_olid_target(label: str | None) -> list[str]:
    mapping = {
        "IND": "Individual",
        "GRP": "Group",
        "OTH": "Other",
    }
    value = normalize_text(label or "").upper()
    if value in {"", "NULL"}:
        return []
    if value not in mapping:
        raise ValueError(f"Unexpected OLID subtask_c label: {label}")
    return [mapping[value]]


def parse_olid_row(row: dict[str, str], *, split: str) -> dict:
    subtask_a = normalize_text(row.get("subtask_a", "")).upper()
    subtask_b = normalize_text(row.get("subtask_b", "")).upper()
    subtask_c = normalize_text(row.get("subtask_c", "")).upper()
    if subtask_a == "NOT":
        fine_label = "not offensive"
    elif subtask_a == "OFF" and subtask_b == "UNT":
        fine_label = "untargeted offense"
    elif subtask_a == "OFF" and subtask_b == "TIN":
        fine_label = "targeted offense"
    else:
        raise ValueError(f"Unexpected OLID label combination: A={subtask_a} B={subtask_b} C={subtask_c}")

    coarse_label = FINE_TO_COARSE["olid"][fine_label]
    text = normalize_text(row["tweet"])
    target_categories = normalize_olid_target(subtask_c)
    example = UnifiedExample(
        example_id=f"olid-{split}-{normalize_text(row['id'])}",
        dataset="olid",
        dataset_id=DATASET_IDS["olid"],
        split=split,
        text=text,
        raw_text=row["tweet"],
        coarse_label=coarse_label,
        coarse_label_id=COARSE_LABEL_TO_ID[coarse_label],
        fine_label=fine_label,
        fine_label_id=FINE_LABEL_TO_ID["olid"][fine_label],
        label_vote_counts={fine_label: 1},
        target_categories=target_categories,
        target_vote_counts={item: 1 for item in target_categories},
        has_target_supervision=bool(target_categories),
        has_rationale_supervision=False,
        target_supervision_type="categories" if target_categories else "none",
        meta={
            "raw_id": normalize_text(row["id"]),
            "subtask_a": subtask_a,
            "subtask_b": subtask_b or "NULL",
            "subtask_c": subtask_c or "NULL",
        },
    )
    return example.to_dict()


def build_olid_examples(data_root: Path) -> dict[str, list[dict]]:
    train_rows = load_delimited(data_root / "olid" / "olid-training-v1.0.tsv", delimiter="\t")
    stratify_keys: list[str] = []
    for row in train_rows:
        subtask_a = normalize_text(row.get("subtask_a", "")).upper()
        subtask_b = normalize_text(row.get("subtask_b", "")).upper()
        subtask_c = normalize_text(row.get("subtask_c", "")).upper()
        if subtask_a == "NOT":
            stratify_keys.append("NOT")
        elif subtask_b == "UNT":
            stratify_keys.append("OFF:UNT")
        else:
            stratify_keys.append(f"OFF:TIN:{subtask_c}")

    train_split, val_split = train_test_split(
        train_rows,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=stratify_keys,
    )

    test_rows = load_delimited(data_root / "olid" / "testset-levela.tsv", delimiter="\t")
    label_a = {key: value.upper() for key, value in load_label_pairs(data_root / "olid" / "labels-levela.csv").items()}
    label_b = {key: value.upper() for key, value in load_label_pairs(data_root / "olid" / "labels-levelb.csv").items()}
    label_c = {key: value.upper() for key, value in load_label_pairs(data_root / "olid" / "labels-levelc.csv").items()}
    merged_test_rows: list[dict[str, str]] = []
    for row in test_rows:
        row_id = normalize_text(row["id"])
        merged_test_rows.append(
            {
                "id": row_id,
                "tweet": row["tweet"],
                "subtask_a": label_a[row_id],
                "subtask_b": label_b.get(row_id, "NULL"),
                "subtask_c": label_c.get(row_id, "NULL"),
            }
        )

    return {
        "train": [parse_olid_row(row, split="train") for row in train_split],
        "val": [parse_olid_row(row, split="val") for row in val_split],
        "test": [parse_olid_row(row, split="test") for row in merged_test_rows],
    }


def load_toxicn_csv_lookup(path: Path) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            content = normalize_text(row["content"])
            if content in lookup:
                raise ValueError(f"Duplicate ToxiCN content detected in CSV: {content}")
            lookup[content] = {
                "platform": normalize_text(row["platform"]),
                "topic": normalize_text(row["topic"]),
                "content": row["content"],
                "toxic": int(row["toxic"]),
                "toxic_one_hot": parse_index_vector(row["toxic_one_hot"]),
                "toxic_type": int(row["toxic_type"]),
                "toxic_type_one_hot": parse_index_vector(row["toxic_type_one_hot"]),
                "expression": int(row["expression"]),
                "expression_one_hot": parse_index_vector(row["expression_one_hot"]),
                "target": parse_index_vector(row["target"]),
                "length": int(row["length"]),
                "num_attacked": int(row.get("num_attacked", 0)),
            }
    return lookup


def toxicn_target_categories(target_vector: list[int]) -> list[str]:
    categories: list[str] = []
    for index, flag in enumerate(target_vector[:5]):
        if int(flag) == 1:
            categories.append(TOXICN_TARGET_NAMES[index])
    return categories


def parse_toxicn_row(
    row: dict[str, object],
    *,
    split: str,
    index: int,
    csv_lookup: dict[str, dict[str, object]],
) -> dict:
    raw_text = str(row["content"])
    text = normalize_text(raw_text)
    csv_row = csv_lookup.get(text)
    if csv_row is None:
        raise KeyError(f"Unable to locate ToxiCN CSV row for content: {raw_text}")

    toxic_type = int(csv_row["toxic_type"])
    if toxic_type not in TOXICN_FINE_LABELS:
        raise ValueError(f"Unexpected ToxiCN toxic_type value: {toxic_type}")
    fine_label = TOXICN_FINE_LABELS[toxic_type]
    coarse_label = FINE_TO_COARSE["toxicn"][fine_label]
    target_vector = list(csv_row["target"])
    target_categories = toxicn_target_categories(target_vector)
    expression_id = int(csv_row["expression"])
    expression_label = TOXICN_EXPRESSION_LABELS[expression_id]

    example = UnifiedExample(
        example_id=f"toxicn-{split}-{index}",
        dataset="toxicn",
        dataset_id=DATASET_IDS["toxicn"],
        split=split,
        text=text,
        raw_text=raw_text,
        coarse_label=coarse_label,
        coarse_label_id=COARSE_LABEL_TO_ID[coarse_label],
        fine_label=fine_label,
        fine_label_id=FINE_LABEL_TO_ID["toxicn"][fine_label],
        label_vote_counts={fine_label: 1},
        target_categories=target_categories,
        target_vote_counts={item: 1 for item in target_categories},
        has_target_supervision=bool(target_categories),
        has_rationale_supervision=False,
        target_supervision_type="categories" if target_categories else "none",
        meta={
            "platform": csv_row["platform"],
            "topic": csv_row["topic"],
            "toxic": int(csv_row["toxic"]),
            "toxic_one_hot": list(csv_row["toxic_one_hot"]),
            "toxic_type": toxic_type,
            "toxic_type_one_hot": list(csv_row["toxic_type_one_hot"]),
            "expression": expression_id,
            "expression_label": expression_label,
            "expression_one_hot": list(csv_row["expression_one_hot"]),
            "target_vector": target_vector,
            "length": int(csv_row["length"]),
            "num_attacked": int(csv_row["num_attacked"]),
            "split_source": "json+csv",
        },
    )
    return example.to_dict()


def build_toxicn_examples(data_root: Path) -> dict[str, list[dict]]:
    dataset_root = data_root / "toxicn"
    csv_lookup = load_toxicn_csv_lookup(dataset_root / "ToxiCN_1.0.csv")
    official_train_rows = read_json(dataset_root / "train.json")
    official_test_rows = read_json(dataset_root / "test.json")

    stratify_labels = [int(row["toxic_type"]) for row in official_train_rows]
    train_split, val_split = train_test_split(
        official_train_rows,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=stratify_labels,
    )

    return {
        "train": [
            parse_toxicn_row(row, split="train", index=index, csv_lookup=csv_lookup)
            for index, row in enumerate(train_split)
        ],
        "val": [
            parse_toxicn_row(row, split="val", index=index, csv_lookup=csv_lookup)
            for index, row in enumerate(val_split)
        ],
        "test": [
            parse_toxicn_row(row, split="test", index=index, csv_lookup=csv_lookup)
            for index, row in enumerate(official_test_rows)
        ],
    }


def build_ihc_examples(data_root: Path) -> dict[str, list[dict]]:
    dataset_root = data_root / "implicit-hate-corpus"
    stage1_rows = load_delimited(dataset_root / "implicit_hate_v1_stg1.tsv", delimiter="\t")
    stage1_post_rows = load_delimited(dataset_root / "implicit_hate_v1_stg1_posts.tsv", delimiter="\t")
    if len(stage1_rows) != len(stage1_post_rows):
        raise ValueError("IHC stage1 label file and posts file have mismatched lengths.")

    stage2_lookup: dict[str, dict[str, str]] = {}
    for row in load_delimited(dataset_root / "implicit_hate_v1_stg2.tsv", delimiter="\t"):
        raw_id = normalize_text(row["ID"])
        stage2_lookup[raw_id] = {
            "implicit_class": normalize_text(row["implicit_class"]).replace("_", " "),
            "extra_implicit_class": normalize_text(row.get("extra_implicit_class", "")).replace("_", " "),
        }

    stage3_lookup: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in load_delimited(dataset_root / "implicit_hate_v1_stg3.tsv", delimiter="\t"):
        raw_id = normalize_text(row["ID"])
        stage3_lookup[raw_id].append(
            {
                "target": normalize_text(row.get("target", "")),
                "implied_statement": normalize_text(row.get("implied_statement", "")),
            }
        )

    rows: list[dict] = []
    for index, (stage1_row, post_row) in enumerate(zip(stage1_rows, stage1_post_rows)):
        raw_id = normalize_text(stage1_row["ID"])
        raw_label = normalize_text(stage1_row["class"]).lower()
        post_label = normalize_text(post_row["class"]).lower()
        if raw_label != post_label:
            raise ValueError(f"IHC stage1 label mismatch at row {index}: {raw_label} != {post_label}")
        if raw_label not in IHC_STAGE1_LABELS:
            raise ValueError(f"Unexpected IHC stage1 label: {raw_label}")

        fine_label = IHC_STAGE1_LABELS[raw_label]
        coarse_label = FINE_TO_COARSE["ihc"][fine_label]
        text = normalize_text(post_row["post"])
        stage2_meta = stage2_lookup.get(raw_id, {})
        stage3_rows = stage3_lookup.get(raw_id, [])
        target_vote_counts = collect_target_counts([[item["target"]] for item in stage3_rows if item.get("target")])
        target_categories = list(target_vote_counts.keys())
        implied_counter = Counter(
            item["implied_statement"]
            for item in stage3_rows
            if item.get("implied_statement") and normalize_text(item["implied_statement"]).casefold() not in TARGET_IGNORE
        )

        example = UnifiedExample(
            example_id=f"ihc-{raw_id}",
            dataset="ihc",
            dataset_id=DATASET_IDS["ihc"],
            split="train",
            text=text,
            raw_text=post_row["post"],
            coarse_label=coarse_label,
            coarse_label_id=COARSE_LABEL_TO_ID[coarse_label],
            fine_label=fine_label,
            fine_label_id=FINE_LABEL_TO_ID["ihc"][fine_label],
            label_vote_counts={fine_label: 1},
            target_categories=target_categories,
            target_vote_counts=target_vote_counts,
            has_target_supervision=bool(target_categories),
            has_rationale_supervision=False,
            target_supervision_type="categories" if target_categories else "none",
            meta={
                "raw_id": raw_id,
                "source_label": raw_label,
                "stage2_available": bool(stage2_meta),
                "implicit_class": stage2_meta.get("implicit_class", ""),
                "extra_implicit_class": stage2_meta.get("extra_implicit_class", ""),
                "stage3_available": bool(stage3_rows),
                "implied_statement_counts": dict(sorted(implied_counter.items())),
            },
        )
        rows.append(example.to_dict())

    labels = [row["fine_label"] for row in rows]
    train_val_rows, test_rows = train_test_split(
        rows,
        test_size=0.1,
        random_state=42,
        shuffle=True,
        stratify=labels,
    )
    train_val_labels = [row["fine_label"] for row in train_val_rows]
    train_rows, val_rows = train_test_split(
        train_val_rows,
        test_size=1.0 / 9.0,
        random_state=42,
        shuffle=True,
        stratify=train_val_labels,
    )

    def assign_split(split: str, examples: list[dict]) -> list[dict]:
        output: list[dict] = []
        for row in examples:
            item = dict(row)
            item["split"] = split
            output.append(item)
        return output

    return {
        "train": assign_split("train", train_rows),
        "val": assign_split("val", val_rows),
        "test": assign_split("test", test_rows),
    }


def build_metadata(grouped: dict[str, dict[str, list[dict]]]) -> dict:
    split_stats: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    merged_counts: dict[str, int] = defaultdict(int)
    for dataset, dataset_splits in grouped.items():
        for split, rows in dataset_splits.items():
            fine_counts = Counter(row["fine_label"] for row in rows)
            coarse_counts = Counter(row["coarse_label"] for row in rows)
            split_stats[dataset][split] = {
                "num_examples": len(rows),
                "fine_label_counts": dict(sorted(fine_counts.items())),
                "coarse_label_counts": dict(sorted(coarse_counts.items())),
            }
            merged_counts[split] += len(rows)

    return {
        "dataset_ids": DATASET_IDS,
        "coarse_label_to_id": COARSE_LABEL_TO_ID,
        "fine_label_to_id": FINE_LABEL_TO_ID,
        "fine_to_coarse": FINE_TO_COARSE,
        "target_category_to_id": build_target_category_vocab(grouped),
        "split_stats": split_stats,
        "merged_split_sizes": dict(sorted(merged_counts.items())),
    }


def build_target_category_vocab(grouped: dict[str, dict[str, list[dict]]]) -> dict[str, int]:
    categories: set[str] = set()
    for dataset_splits in grouped.values():
        for rows in dataset_splits.values():
            for row in rows:
                categories.update(row["target_categories"])
    return {category: index for index, category in enumerate(sorted(categories))}


def build_unified_corpus(data_root: Path, out_root: Path) -> dict:
    builders = {
        "ethos": ("ethos", build_ethos_examples),
        "olid": ("olid", build_olid_examples),
        "toxicn": ("toxicn", build_toxicn_examples),
        "ihc": ("implicit-hate-corpus", build_ihc_examples),
    }
    grouped: dict[str, dict[str, list[dict]]] = {}
    for dataset_name, (folder_name, builder) in builders.items():
        if (data_root / folder_name).exists():
            grouped[dataset_name] = builder(data_root)
    if not grouped:
        raise FileNotFoundError(f"No supported datasets found under {data_root}")
    target_category_to_id = build_target_category_vocab(grouped)

    for dataset_splits in grouped.values():
        for rows in dataset_splits.values():
            for row in rows:
                row["target_category_ids"] = [target_category_to_id[item] for item in row["target_categories"] if item in target_category_to_id]

    out_root.mkdir(parents=True, exist_ok=True)
    merged: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for dataset, dataset_splits in grouped.items():
        dataset_root = out_root / dataset
        for split, rows in dataset_splits.items():
            write_jsonl(dataset_root / f"{split}.jsonl", rows)
            merged[split].extend(rows)

    for split, rows in merged.items():
        rows.sort(key=lambda row: (row["dataset_id"], row["example_id"]))
        write_jsonl(out_root / f"{split}.jsonl", rows)

    metadata = build_metadata(grouped)
    write_json(out_root / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--out_root", type=Path, default=Path("data") / "unified")
    args = parser.parse_args()
    metadata = build_unified_corpus(args.data_root, args.out_root)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
