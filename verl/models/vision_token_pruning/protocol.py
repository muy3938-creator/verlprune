from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .config import compute_selector_fingerprint

SELECTION_PROTOCOL_VERSION = 3
DYNAMIC_SELECTION_PROTOCOL_VERSION = 1
DYNAMIC_SELECTION_KIND = "decode_dynamic"
TWO_STAGE_SELECTION_PROTOCOL_VERSION = 1
TWO_STAGE_SELECTION_KIND = "prefill_physical_decode_dynamic"


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


@dataclass(frozen=True)
class DynamicVisionTokenSelection:
    """Per-query visual KV selections captured during autoregressive decode."""

    nominal_keep_ratio: float
    original_visual_token_count: int
    query_kept_visual_indices: tuple[tuple[int, ...], ...]
    selector: str = "vision_pulse"
    selector_fingerprint: str | None = field(default=None)
    kind: str = DYNAMIC_SELECTION_KIND
    version: int = DYNAMIC_SELECTION_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.kind != DYNAMIC_SELECTION_KIND:
            raise ValueError(f"unsupported dynamic selection kind {self.kind!r}")
        if self.version != DYNAMIC_SELECTION_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported dynamic visual selection protocol version {self.version}; "
                f"expected {DYNAMIC_SELECTION_PROTOCOL_VERSION}"
            )
        if not 0.0 < self.nominal_keep_ratio <= 1.0:
            raise ValueError("dynamic selection nominal_keep_ratio must be in (0, 1]")
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

        normalized_rows = []
        for query_index, indices in enumerate(self.query_kept_visual_indices):
            normalized = tuple(int(index) for index in indices)
            if normalized and tuple(sorted(set(normalized))) != normalized:
                raise ValueError(
                    f"dynamic selection query {query_index} indices must be sorted and unique"
                )
            if normalized and (
                normalized[0] < 0 or normalized[-1] >= self.original_visual_token_count
            ):
                raise ValueError(
                    f"dynamic selection query {query_index} contains an out of range index"
                )
            normalized_rows.append(normalized)
        object.__setattr__(self, "query_kept_visual_indices", tuple(normalized_rows))

    def to_wire(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "nominal_keep_ratio": self.nominal_keep_ratio,
            "selector": self.selector,
            "selector_fingerprint": self.selector_fingerprint,
            "original_visual_token_count": self.original_visual_token_count,
            "query_kept_visual_indices": [
                list(indices) for indices in self.query_kept_visual_indices
            ],
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> "DynamicVisionTokenSelection":
        required = {
            "kind",
            "version",
            "nominal_keep_ratio",
            "selector",
            "selector_fingerprint",
            "original_visual_token_count",
            "query_kept_visual_indices",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"dynamic visual selection is missing fields: {sorted(missing)}")
        return cls(
            kind=str(value["kind"]),
            version=int(value["version"]),
            nominal_keep_ratio=float(value["nominal_keep_ratio"]),
            selector=str(value["selector"]),
            selector_fingerprint=str(value["selector_fingerprint"]),
            original_visual_token_count=int(value["original_visual_token_count"]),
            query_kept_visual_indices=tuple(
                tuple(int(index) for index in row)
                for row in value["query_kept_visual_indices"]
            ),
        )


@dataclass(frozen=True)
class TwoStageVisionTokenSelection:
    """Physical prefill selection plus per-query routing within that subset."""

    prefill: VisionTokenSelection
    decode: DynamicVisionTokenSelection
    kind: str = TWO_STAGE_SELECTION_KIND
    version: int = TWO_STAGE_SELECTION_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.kind != TWO_STAGE_SELECTION_KIND:
            raise ValueError(f"unsupported two-stage selection kind {self.kind!r}")
        if self.version != TWO_STAGE_SELECTION_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported two-stage selection version {self.version}; "
                f"expected {TWO_STAGE_SELECTION_PROTOCOL_VERSION}"
            )
        if self.decode.original_visual_token_count != len(self.prefill.kept_visual_indices):
            raise ValueError(
                "two-stage decode indices must be relative to the physically retained prefill subset"
            )

    @property
    def effective_nominal_keep_ratio(self) -> float:
        return self.prefill.keep_ratio * self.decode.nominal_keep_ratio

    def to_wire(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "prefill": self.prefill.to_wire(),
            "decode": self.decode.to_wire(),
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> "TwoStageVisionTokenSelection":
        required = {"kind", "version", "prefill", "decode"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"two-stage visual selection is missing fields: {sorted(missing)}")
        return cls(
            kind=str(value["kind"]),
            version=int(value["version"]),
            prefill=VisionTokenSelection.from_wire(dict(value["prefill"])),
            decode=DynamicVisionTokenSelection.from_wire(dict(value["decode"])),
        )


def selection_from_wire(
    value: dict[str, Any],
) -> VisionTokenSelection | DynamicVisionTokenSelection | TwoStageVisionTokenSelection:
    if value.get("kind") == TWO_STAGE_SELECTION_KIND:
        return TwoStageVisionTokenSelection.from_wire(value)
    if value.get("kind") == DYNAMIC_SELECTION_KIND:
        return DynamicVisionTokenSelection.from_wire(value)
    return VisionTokenSelection.from_wire(value)


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
