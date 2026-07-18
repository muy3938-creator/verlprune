from dataclasses import dataclass
from typing import Any

SELECTION_PROTOCOL_VERSION = 1


def compute_keep_count(token_count: int, keep_ratio: float) -> int:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    return max(1, min(token_count, int(round(token_count * keep_ratio))))


@dataclass(frozen=True)
class VisionTokenSelection:
    """Exact layer-0 selection produced by rollout and replayed by the actor."""

    keep_ratio: float
    original_visual_token_count: int
    kept_visual_indices: tuple[int, ...]
    version: int = SELECTION_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.version != SELECTION_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported visual-token selection protocol version {self.version}; "
                f"expected {SELECTION_PROTOCOL_VERSION}"
            )
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("selection keep_ratio must be in (0, 1]")
        if self.original_visual_token_count <= 0:
            raise ValueError("original_visual_token_count must be positive")
        expected_count = compute_keep_count(self.original_visual_token_count, self.keep_ratio)
        if len(self.kept_visual_indices) != expected_count:
            raise ValueError(
                f"kept_visual_indices has length {len(self.kept_visual_indices)}; expected {expected_count}"
            )
        if tuple(sorted(set(self.kept_visual_indices))) != self.kept_visual_indices:
            raise ValueError("kept_visual_indices must be sorted and unique")
        if self.kept_visual_indices[0] < 0 or self.kept_visual_indices[-1] >= self.original_visual_token_count:
            raise ValueError("kept_visual_indices contains an out of range index")
        if self.kept_visual_indices[-1] != self.original_visual_token_count - 1:
            raise ValueError("the final visual token must be retained as the MRoPE anchor")

    def to_wire(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "keep_ratio": self.keep_ratio,
            "original_visual_token_count": self.original_visual_token_count,
            "kept_visual_indices": list(self.kept_visual_indices),
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> "VisionTokenSelection":
        required = {"version", "keep_ratio", "original_visual_token_count", "kept_visual_indices"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"visual-token selection is missing fields: {sorted(missing)}")
        return cls(
            version=int(value["version"]),
            keep_ratio=float(value["keep_ratio"]),
            original_visual_token_count=int(value["original_visual_token_count"]),
            kept_visual_indices=tuple(int(index) for index in value["kept_visual_indices"]),
        )


def decode_rollout_selection(
    routed_experts: Any,
    *,
    keep_ratio: float,
    original_visual_token_count: int,
) -> VisionTokenSelection:
    """Decode positive 1-based indices carried by vLLM's replay channel."""

    values = routed_experts.tolist() if hasattr(routed_experts, "tolist") else routed_experts
    if values is None:
        raise ValueError("vLLM visual-token pruning plugin did not return selection metadata")
    try:
        encoded = [int(token_layers[0][0]) for token_layers in values]
    except (IndexError, TypeError) as exc:
        raise ValueError("invalid routed_experts shape for visual-token selection") from exc

    kept_indices = tuple(sorted({value - 1 for value in encoded if value > 0}))
    return VisionTokenSelection(
        keep_ratio=keep_ratio,
        original_visual_token_count=original_visual_token_count,
        kept_visual_indices=kept_indices,
    )
