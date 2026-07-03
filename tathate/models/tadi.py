from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TADIClassifier(nn.Module):
    def __init__(
        self,
        *,
        model_name_or_path: str,
        num_labels: int,
        num_target_categories: int = 0,
        dropout_prob: float = 0.1,
        pooling_strategy: str = "cls",
        implicit_attention_temperature: float = 1.0,
        implicit_injection_scale: float = 0.1,
        explicit_injection_scale: float = 0.05,
        implicit_relation_clamp: float = 12.0,
        implicit_gate_mode: str = "learned",
        explicit_gate_mode: str = "learned",
        implicit_clamp_mode: str = "enabled",
        implicit_gate_floor: float = 0.0,
        explicit_gate_floor: float = 0.0,
    ) -> None:
        super().__init__()
        if pooling_strategy not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
        if implicit_attention_temperature <= 0:
            raise ValueError("implicit_attention_temperature must be positive.")
        if implicit_relation_clamp <= 0:
            raise ValueError("implicit_relation_clamp must be positive.")
        if implicit_gate_mode not in {"learned", "fixed_one"}:
            raise ValueError(f"Unsupported implicit_gate_mode: {implicit_gate_mode}")
        if explicit_gate_mode not in {"learned", "fixed_one"}:
            raise ValueError(f"Unsupported explicit_gate_mode: {explicit_gate_mode}")
        if implicit_clamp_mode not in {"enabled", "disabled"}:
            raise ValueError(f"Unsupported implicit_clamp_mode: {implicit_clamp_mode}")
        if not (0.0 <= implicit_gate_floor <= 1.0):
            raise ValueError("implicit_gate_floor must be within [0, 1].")
        if not (0.0 <= explicit_gate_floor <= 1.0):
            raise ValueError("explicit_gate_floor must be within [0, 1].")

        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Unable to infer hidden size from encoder config.")

        self.pooling_strategy = pooling_strategy
        self.hidden_size = int(hidden_size)
        self.num_target_categories = max(0, int(num_target_categories))
        self.implicit_attention_temperature = float(implicit_attention_temperature)
        self.implicit_injection_scale = float(implicit_injection_scale)
        self.explicit_injection_scale = float(explicit_injection_scale)
        self.implicit_relation_clamp = float(implicit_relation_clamp)
        self.implicit_gate_mode = implicit_gate_mode
        self.explicit_gate_mode = explicit_gate_mode
        self.implicit_clamp_mode = implicit_clamp_mode
        self.implicit_gate_floor = float(implicit_gate_floor)
        self.explicit_gate_floor = float(explicit_gate_floor)

        self.shared_norm = nn.LayerNorm(self.hidden_size)
        self.task_branch = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )
        self.implicit_target_query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.implicit_target_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.implicit_token_projector = nn.Linear(self.hidden_size, self.hidden_size)

        self.implicit_relation = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
        )
        self.explicit_relation = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.GELU(),
        )

        if self.implicit_gate_mode == "learned":
            self.implicit_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.hidden_size // 2, 1),
            )
        if self.explicit_gate_mode == "learned":
            self.explicit_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.hidden_size // 2, 1),
            )
        if self.num_target_categories > 0:
            self.target_category_classifier = nn.Linear(self.hidden_size, self.num_target_categories)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, int(num_labels))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            token_repr = outputs[0]
        else:
            token_repr = outputs.last_hidden_state

        cls_repr = token_repr[:, 0, :]
        if self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (token_repr * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)
        else:
            pooled = cls_repr

        pooled = self.shared_norm(pooled)
        return {
            "token_repr": token_repr,
            "cls_repr": cls_repr,
            "shared_repr": pooled,
        }

    def build_task_representation(self, shared_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"task_base_repr": self.task_branch(shared_repr)}

    def build_content_mask(
        self,
        *,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        content_mask = attention_mask.bool()
        if special_tokens_mask is not None:
            content_mask = content_mask & ~special_tokens_mask.bool()
        if content_mask.shape[1] == 0:
            return content_mask
        has_content = content_mask.any(dim=1)
        if not torch.all(has_content):
            repaired_mask = content_mask.clone()
            repaired_mask[~has_content] = attention_mask.bool()[~has_content]
            content_mask = repaired_mask
        return content_mask

    def summarize_masked_tokens(self, token_repr: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        weights = token_mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (token_repr * weights.unsqueeze(-1)).sum(dim=1) / denom

    def compute_relation_and_gate(
        self,
        *,
        task_base_repr: torch.Tensor,
        target_repr: torch.Tensor,
        relation_mlp: nn.Module,
        gate_mlp: nn.Module | None,
        gate_mode: str,
        gate_floor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_input = torch.cat(
            [
                task_base_repr,
                target_repr,
                task_base_repr * target_repr,
                task_base_repr - target_repr,
            ],
            dim=-1,
        )
        relation_raw = relation_mlp(relation_input)
        if gate_mode == "learned":
            if gate_mlp is None:
                raise RuntimeError("Gate MLP is required when gate_mode='learned'.")
            gate_raw = torch.sigmoid(gate_mlp(relation_input))
            gate = gate_floor + (1.0 - gate_floor) * gate_raw
        else:
            gate_raw = torch.ones(
                relation_raw.shape[0],
                1,
                dtype=relation_raw.dtype,
                device=relation_raw.device,
            )
            gate = torch.ones(
                relation_raw.shape[0],
                1,
                dtype=relation_raw.dtype,
                device=relation_raw.device,
            )
        return relation_input, relation_raw, gate_raw, gate

    def maybe_clamp_relation(self, relation_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_norm = relation_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.implicit_clamp_mode == "enabled":
            clamp_tensor = torch.full_like(raw_norm, self.implicit_relation_clamp)
            clamp_ratio = torch.minimum(torch.ones_like(raw_norm), clamp_tensor / raw_norm)
        else:
            clamp_ratio = torch.ones_like(raw_norm)
        relation_repr = relation_raw * clamp_ratio
        return relation_repr, clamp_ratio

    def build_target_injection(
        self,
        *,
        token_repr: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None,
        explicit_candidate_mask: torch.Tensor | None,
        task_base_repr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        content_mask = self.build_content_mask(attention_mask=attention_mask, special_tokens_mask=special_tokens_mask)
        token_task_repr = self.implicit_token_projector(token_repr)

        target_query = self.implicit_target_query(task_base_repr).unsqueeze(1)
        target_keys = self.implicit_target_key(token_task_repr)
        attention_scores = (target_query * target_keys).sum(dim=-1) / math.sqrt(self.hidden_size)
        attention_scores = attention_scores / self.implicit_attention_temperature
        attention_scores = attention_scores.masked_fill(~content_mask, -1e4)
        target_attention = torch.softmax(attention_scores, dim=-1)
        target_attention = target_attention * content_mask.float()
        target_attention = target_attention / target_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        implicit_target_repr = torch.bmm(target_attention.unsqueeze(1), token_task_repr).squeeze(1)
        target_category_logits = None
        if hasattr(self, "target_category_classifier"):
            target_category_logits = self.target_category_classifier(self.dropout(implicit_target_repr))

        _, implicit_relation_raw, implicit_gate_raw, implicit_gate = self.compute_relation_and_gate(
            task_base_repr=task_base_repr,
            target_repr=implicit_target_repr,
            relation_mlp=self.implicit_relation,
            gate_mlp=getattr(self, "implicit_gate", None),
            gate_mode=self.implicit_gate_mode,
            gate_floor=self.implicit_gate_floor,
        )
        implicit_relation_repr, implicit_clamp_ratio = self.maybe_clamp_relation(implicit_relation_raw)
        implicit_injected = self.implicit_injection_scale * implicit_gate * implicit_relation_repr

        if explicit_candidate_mask is None:
            explicit_mask = torch.zeros_like(content_mask)
        else:
            explicit_mask = explicit_candidate_mask.bool() & content_mask
        explicit_candidate_flag = explicit_mask.any(dim=-1).float()
        explicit_target_repr = self.summarize_masked_tokens(token_task_repr, explicit_mask)

        _, explicit_relation_raw, explicit_gate_raw, explicit_gate = self.compute_relation_and_gate(
            task_base_repr=task_base_repr,
            target_repr=explicit_target_repr,
            relation_mlp=self.explicit_relation,
            gate_mlp=getattr(self, "explicit_gate", None),
            gate_mode=self.explicit_gate_mode,
            gate_floor=self.explicit_gate_floor,
        )
        explicit_relation_repr, explicit_clamp_ratio = self.maybe_clamp_relation(explicit_relation_raw)
        explicit_gate = explicit_gate * explicit_candidate_flag.unsqueeze(-1)
        explicit_injected = self.explicit_injection_scale * explicit_gate * explicit_relation_repr

        injected_task_repr = task_base_repr + implicit_injected + explicit_injected

        valid_token_count = content_mask.sum(dim=-1).clamp_min(2).float()
        target_entropy = -(target_attention * torch.log(target_attention.clamp_min(1e-8))).sum(dim=-1)
        normalized_target_entropy = target_entropy / torch.log(valid_token_count)
        explicit_token_count = explicit_mask.sum(dim=-1).float()
        return {
            "content_mask": content_mask,
            "implicit_target_attention": target_attention,
            "implicit_target_repr": implicit_target_repr,
            **({"target_category_logits": target_category_logits} if target_category_logits is not None else {}),
            "explicit_target_repr": explicit_target_repr,
            "explicit_candidate_mask": explicit_mask.float(),
            "explicit_candidate_flag": explicit_candidate_flag,
            "implicit_token_task_repr": token_task_repr,
            "implicit_relation_raw": implicit_relation_raw,
            "implicit_relation_repr": implicit_relation_repr,
            "explicit_relation_raw": explicit_relation_raw,
            "explicit_relation_repr": explicit_relation_repr,
            "implicit_gate_raw_value": implicit_gate_raw.squeeze(-1),
            "implicit_gate_value": implicit_gate.squeeze(-1),
            "explicit_gate_raw_value": explicit_gate_raw.squeeze(-1),
            "explicit_gate_value": explicit_gate.squeeze(-1),
            "implicit_relation_clamp_ratio": implicit_clamp_ratio.squeeze(-1),
            "explicit_relation_clamp_ratio": explicit_clamp_ratio.squeeze(-1),
            "task_repr": injected_task_repr,
            "implicit_target_entropy": normalized_target_entropy,
            "implicit_target_peak": target_attention.max(dim=-1).values,
            **(
                {
                    "target_category_top_prob": torch.sigmoid(target_category_logits).max(dim=-1).values
                }
                if target_category_logits is not None
                else {}
            ),
            "implicit_raw_relation_norm": implicit_relation_raw.norm(dim=-1),
            "implicit_relation_norm": implicit_relation_repr.norm(dim=-1),
            "implicit_injection_norm": implicit_injected.norm(dim=-1),
            "explicit_raw_relation_norm": explicit_relation_raw.norm(dim=-1),
            "explicit_relation_norm": explicit_relation_repr.norm(dim=-1),
            "explicit_injection_norm": explicit_injected.norm(dim=-1),
            "explicit_coverage_ratio": explicit_candidate_flag,
            "explicit_token_count": explicit_token_count,
            "implicit_target_task_cosine": F.cosine_similarity(task_base_repr, implicit_target_repr, dim=-1),
            "implicit_target_fine_cosine": F.cosine_similarity(task_base_repr, implicit_target_repr, dim=-1),
            "explicit_target_task_cosine": F.cosine_similarity(task_base_repr, explicit_target_repr, dim=-1),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor | None = None,
        explicit_candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        task_outputs = self.build_task_representation(encoded["shared_repr"])
        injection_outputs = self.build_target_injection(
            token_repr=encoded["token_repr"],
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            explicit_candidate_mask=explicit_candidate_mask,
            task_base_repr=task_outputs["task_base_repr"],
        )
        task_repr = injection_outputs["task_repr"]

        logits = self.classifier(self.dropout(task_repr))
        return {
            "token_repr": encoded["token_repr"],
            "cls_repr": encoded["cls_repr"],
            "shared_repr": encoded["shared_repr"],
            "task_base_repr": task_outputs["task_base_repr"],
            "task_repr": task_repr,
            "logits": logits,
            "fine_base_repr": task_outputs["task_base_repr"],
            "fine_repr": task_repr,
            "fine_logits": logits,
            "coarse_logits": logits,
            **injection_outputs,
        }
