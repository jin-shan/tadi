from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset


class UnifiedJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        datasets: list[str] | None = None,
        max_examples: int | None = None,
        seed: int = 0,
    ) -> None:
        self.path = Path(path)
        self.rows: list[dict] = []
        allowed = set(datasets) if datasets else None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if allowed and row["dataset"] not in allowed:
                    continue
                self.rows.append(row)
        if max_examples is not None and len(self.rows) > int(max_examples):
            rng = random.Random(int(seed))
            rng.shuffle(self.rows)
            self.rows = self.rows[: int(max_examples)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]

    def dataset_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(row["dataset"] for row in self.rows).items()))

    def build_sample_weights(self, power: float = 1.0) -> torch.Tensor:
        counts = Counter(row["dataset"] for row in self.rows)
        weights = []
        for row in self.rows:
            weights.append((1.0 / counts[row["dataset"]]) ** float(power))
        return torch.tensor(weights, dtype=torch.float)
