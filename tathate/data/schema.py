from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DATASET_IDS = {
    "ethos": 0,
    "olid": 1,
    "toxicn": 2,
    "ihc": 3,
}

COARSE_LABEL_TO_ID = {
    "non-hate": 0,
    "hate": 1,
}

FINE_LABEL_TO_ID = {
    "ethos": {
        "no hate": 0,
        "hate": 1,
    },
    "olid": {
        "not offensive": 0,
        "untargeted offense": 1,
        "targeted offense": 2,
    },
    "toxicn": {
        "non-toxic": 0,
        "offensive": 1,
        "hate speech": 2,
    },
    "ihc": {
        "not hate": 0,
        "implicit hate": 1,
        "explicit hate": 2,
    },
}

FINE_TO_COARSE = {
    "ethos": {
        "no hate": "non-hate",
        "hate": "hate",
    },
    "olid": {
        "not offensive": "non-hate",
        "untargeted offense": "hate",
        "targeted offense": "hate",
    },
    "toxicn": {
        "non-toxic": "non-hate",
        "offensive": "hate",
        "hate speech": "hate",
    },
    "ihc": {
        "not hate": "non-hate",
        "implicit hate": "hate",
        "explicit hate": "hate",
    },
}


@dataclass
class UnifiedExample:
    example_id: str
    dataset: str
    dataset_id: int
    split: str
    text: str
    raw_text: str
    coarse_label: str
    coarse_label_id: int
    fine_label: str
    fine_label_id: int
    label_vote_counts: dict[str, int]
    tokens: list[str] = field(default_factory=list)
    token_char_spans: list[list[int]] = field(default_factory=list)
    rationale_mask: list[int] = field(default_factory=list)
    rationale_spans: list[list[int]] = field(default_factory=list)
    target_categories: list[str] = field(default_factory=list)
    target_category_ids: list[int] = field(default_factory=list)
    target_vote_counts: dict[str, int] = field(default_factory=dict)
    target_spans: list[list[int]] = field(default_factory=list)
    has_target_supervision: bool = False
    has_rationale_supervision: bool = False
    target_supervision_type: str = "none"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
