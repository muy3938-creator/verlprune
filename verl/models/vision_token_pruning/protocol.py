from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .config import compute_selector_fingerprint

SELECTION_PROTOCOL_VERSION = 3


def compute_keep_count(token_count: int, keep_ratio: float) -> int:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    return max(1, min(token_count, int(round(token_count * keep_ratio))))


@dataclass(frozen=True)
class VisionTokenSelection:
    """Exact rollout selection replayed by the backend-neutral actor."""

    keep_ratio: float
    original_visual_token_count: int
    kept_visual_indices: tuple[int, ...]
    selector: str = "random"
    selector_fingerprint: str | None = field(default=None)
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
        if not self.selector or not self.selector.strip():
            raise ValueError("selection selector must be a non-empty name")
        object.__setattr__(self, "selector", self.selector.strip())
        if self.selector_fingerprint is None:
            object.__setattr__(
                self,
                "selector_fingerprint",
                compute_selector_fingerprint(self.selector, {}),
            )
        assert self.selector_fingerprint is not None
        if len(self.selector_fingerprint) != 64:
            raise ValueError("selection selector_fingerprint must be a SHA-256 hex digest")
        try:
            int(self.selector_fingerprint, 16)
        except ValueError as exc:
            raise ValueError("selection selector_fingerprint must be a SHA-256 hex digest") from exc
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
            "selector": self.selector,
            "selector_fingerprint": self.selector_fingerprint,
            "original_visual_token_count": self.original_visual_token_count,
            "kept_visual_indices": list(self.kept_visual_indices),
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> "VisionTokenSelection":
        required = {
            "version",
            "keep_ratio",
            "selector",
            "selector_fingerprint",
            "original_visual_token_count",
            "kept_visual_indices",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"visual-token selection is missing fields: {sorted(missing)}")
        return cls(
            version=int(value["version"]),
            keep_ratio=float(value["keep_ratio"]),
            selector=str(value["selector"]),
            selector_fingerprint=str(value["selector_fingerprint"]),
            original_visual_token_count=int(value["original_visual_token_count"]),
            kept_visual_indices=tuple(int(index) for index in value["kept_visual_indices"]),
        )


def decode_rollout_selection(
    routed_experts: Any,
    *,
    keep_ratio: float,
    original_visual_token_count: int,
    selector: str = "random",
    selector_kwargs: Mapping[str, Any] | None = None,
) -> VisionTokenSelection:
    """Compatibility wrapper for the isolated vLLM capture transport."""

    from .transport import decode_vllm_selection_capture

    return decode_vllm_selection_capture(
        routed_experts,
        keep_ratio=keep_ratio,
        selector=selector,
        selector_kwargs=selector_kwargs,
        original_visual_token_count=original_visual_token_count,
    )
