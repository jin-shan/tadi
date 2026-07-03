from __future__ import annotations

from collections.abc import Iterable


def build_label_space(metadata: dict, datasets: Iterable[str], label_mode: str) -> dict[str, int]:
    if label_mode == "coarse":
        items = sorted(metadata["coarse_label_to_id"].items(), key=lambda item: item[1])
        return {label: index for index, (label, _) in enumerate(items)}

    fine_maps: dict[str, dict[str, int]] = metadata.get("fine_label_to_id", {})
    ordered_labels: list[str] = []
    seen: set[str] = set()
    for dataset in datasets:
        if dataset not in fine_maps:
            raise ValueError(f"Missing fine label metadata for dataset '{dataset}'.")
        for label, _ in sorted(fine_maps[dataset].items(), key=lambda item: item[1]):
            if label in seen:
                continue
            seen.add(label)
            ordered_labels.append(label)

    if not ordered_labels:
        raise ValueError("Unable to build a fine label space from the selected datasets.")
    return {label: index for index, label in enumerate(ordered_labels)}


def invert_label_space(label_to_id: dict[str, int]) -> dict[int, str]:
    return {index: label for label, index in label_to_id.items()}
