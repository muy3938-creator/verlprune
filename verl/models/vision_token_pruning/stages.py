"""Canonical three-stage experiment model for visual-token pruning.

Algorithms declare a policy. Experiments declare one or more stages.
Runtimes only observe tensors, invoke the policy, and apply the decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .config import VisionTokenPruningConfig, compute_selector_fingerprint


class StageKind(str, Enum):
    """The only three selection semantics the platform supports."""

    PHYSICAL_PRE_DECODER = "physical_pre_decoder"
    BOUNDARY_ONCE = "boundary_once"
    DECODE_QUERY = "decode_query"


class InputKind(str, Enum):
    """Tensors a policy is allowed to require."""

    VISION_EMBEDDING = "vision_embedding"
    BOUNDARY_QKV = "boundary_qkv"
    DECODE_QK = "decode_qk"


class RuntimeKind(str, Enum):
    PHYSICAL_FIXED = "physical_fixed"
    FLEX_MASK = "flex_mask"


_LEGACY_SELECTOR_INPUT_TO_INPUT_KIND = {
    "vision_embedding": InputKind.VISION_EMBEDDING,
    "decoder_key": InputKind.BOUNDARY_QKV,
    "decode_query": InputKind.DECODE_QK,
}

_INPUT_KIND_ALLOWED_BY_STAGE = {
    StageKind.PHYSICAL_PRE_DECODER: {InputKind.VISION_EMBEDDING},
    StageKind.BOUNDARY_ONCE: {InputKind.VISION_EMBEDDING, InputKind.BOUNDARY_QKV},
    StageKind.DECODE_QUERY: {InputKind.DECODE_QK},
}


@dataclass(frozen=True)
class StageSpec:
    """One observe → decide → apply stage."""

    kind: StageKind
    policy: str
    keep_ratio: float
    input_kind: InputKind
    policy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    observe_layer: int | None = None
    apply_from_layer: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, StageKind):
            object.__setattr__(self, "kind", StageKind(self.kind))
        if not isinstance(self.input_kind, InputKind):
            object.__setattr__(self, "input_kind", InputKind(self.input_kind))
        policy = self.policy.strip()
        if not policy:
            raise ValueError("stage policy must be a non-empty name")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "policy_kwargs", dict(self.policy_kwargs))
        if not 0.0 < self.keep_ratio < 1.0:
            raise ValueError(f"stage keep_ratio must be in (0, 1), got {self.keep_ratio}")
        if self.input_kind not in _INPUT_KIND_ALLOWED_BY_STAGE[self.kind]:
            raise ValueError(
                f"stage kind {self.kind.value!r} cannot use input_kind={self.input_kind.value!r}"
            )
        if self.kind is StageKind.PHYSICAL_PRE_DECODER:
            if self.observe_layer is not None or self.apply_from_layer is not None:
                raise ValueError(
                    "physical_pre_decoder forbids observe_layer/apply_from_layer; "
                    "selection happens before decoder layer 0"
                )
            return
        if self.observe_layer is None or self.apply_from_layer is None:
            raise ValueError(
                f"{self.kind.value} requires explicit observe_layer and apply_from_layer"
            )
        if self.observe_layer < 0:
            raise ValueError(f"{self.kind.value} observe_layer must be >= 0")
        if self.apply_from_layer != self.observe_layer + 1:
            raise ValueError(
                f"{self.kind.value} requires apply_from_layer == observe_layer + 1 "
                f"(got observe={self.observe_layer}, apply_from={self.apply_from_layer})"
            )

    @property
    def policy_fingerprint(self) -> str:
        return compute_selector_fingerprint(self.policy, self.policy_kwargs)


@dataclass(frozen=True)
class PruningSpec:
    """Immutable experiment plan resolved once at worker init."""

    enabled: bool
    stages: tuple[StageSpec, ...]
    runtime: RuntimeKind
    keep_ratio_schedule: Mapping[str, Any] = field(default_factory=dict)
    layerwise_backend: Literal["flex", "compact_flash"] = "flex"
    pre_pruning_backend: Literal["flex", "flash"] = "flex"

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RuntimeKind):
            object.__setattr__(self, "runtime", RuntimeKind(self.runtime))
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "keep_ratio_schedule", dict(self.keep_ratio_schedule or {}))
        if not self.enabled:
            if self.stages:
                raise ValueError("disabled PruningSpec must not declare stages")
            return
        if not self.stages:
            raise ValueError("enabled PruningSpec requires at least one stage")
        if len(self.stages) > 2:
            raise ValueError("at most two stages are supported (static/boundary + decode_query)")
        kinds = [stage.kind for stage in self.stages]
        if kinds.count(StageKind.DECODE_QUERY) > 1:
            raise ValueError("at most one decode_query stage is allowed")
        if StageKind.DECODE_QUERY in kinds and kinds[-1] is not StageKind.DECODE_QUERY:
            raise ValueError("decode_query must be the final stage when present")
        if kinds.count(StageKind.PHYSICAL_PRE_DECODER) > 1:
            raise ValueError("at most one physical_pre_decoder stage is allowed")
        if kinds.count(StageKind.BOUNDARY_ONCE) > 1:
            raise ValueError("at most one boundary_once stage is allowed")
        if (
            StageKind.PHYSICAL_PRE_DECODER in kinds
            and StageKind.BOUNDARY_ONCE in kinds
            and len(kinds) == 2
            and StageKind.DECODE_QUERY not in kinds
        ):
            raise ValueError(
                "physical_pre_decoder + boundary_once without decode_query is unsupported"
            )
        if self.runtime is RuntimeKind.PHYSICAL_FIXED:
            if any(stage.kind is not StageKind.PHYSICAL_PRE_DECODER for stage in self.stages):
                raise ValueError("runtime=physical_fixed only supports physical_pre_decoder stages")
            if self.keep_ratio_schedule:
                raise ValueError(
                    "runtime=physical_fixed rejects keep_ratio_schedule; "
                    "physical layout is fixed at engine launch"
                )
            if self.pre_pruning_backend != "flex":
                raise ValueError("runtime=physical_fixed rejects pre_pruning_backend overrides")
        if self.runtime is RuntimeKind.FLEX_MASK:
            if any(stage.kind is StageKind.PHYSICAL_PRE_DECODER for stage in self.stages) and not (
                len(self.stages) == 2 and self.stages[0].kind is StageKind.PHYSICAL_PRE_DECODER
            ):
                # physical prefill + flex decode is the only mixed case
                if not (
                    len(self.stages) == 2
                    and self.stages[0].kind is StageKind.PHYSICAL_PRE_DECODER
                    and self.stages[1].kind is StageKind.DECODE_QUERY
                ):
                    raise ValueError(
                        "flex_mask may combine physical_pre_decoder only as stage-0 of two-stage "
                        "physical-prefill + decode_query"
                    )
            if self.layerwise_backend not in {"flex", "compact_flash"}:
                raise ValueError("layerwise_backend must be 'flex' or 'compact_flash'")
            if self.pre_pruning_backend not in {"flex", "flash"}:
                raise ValueError("pre_pruning_backend must be 'flex' or 'flash'")
            if self.pre_pruning_backend == "flash" and self.layerwise_backend != "flex":
                raise ValueError("pre_pruning_backend='flash' requires layerwise_backend='flex'")
            if any(stage.kind is StageKind.DECODE_QUERY for stage in self.stages):
                if self.layerwise_backend != "flex":
                    raise ValueError("decode_query requires layerwise_backend='flex'")
            if any(stage.input_kind is InputKind.BOUNDARY_QKV for stage in self.stages):
                if self.layerwise_backend != "flex":
                    raise ValueError("boundary_qkv requires layerwise_backend='flex'")
        if self.keep_ratio_schedule and self.runtime is not RuntimeKind.FLEX_MASK:
            raise ValueError("keep_ratio_schedule requires runtime=flex_mask")
        if self.keep_ratio_schedule and self.layerwise_backend != "flex":
            raise ValueError("keep_ratio_schedule requires layerwise_backend='flex'")

    @property
    def primary_stage(self) -> StageSpec:
        if not self.enabled or not self.stages:
            raise ValueError("disabled PruningSpec has no primary stage")
        if self.stages[-1].kind is StageKind.DECODE_QUERY and len(self.stages) > 1:
            return self.stages[-1]
        return self.stages[0]

    @property
    def uses_physical_pre_decoder(self) -> bool:
        return any(stage.kind is StageKind.PHYSICAL_PRE_DECODER for stage in self.stages)

    @property
    def uses_boundary_once(self) -> bool:
        return any(stage.kind is StageKind.BOUNDARY_ONCE for stage in self.stages)

    @property
    def uses_decode_query(self) -> bool:
        return any(stage.kind is StageKind.DECODE_QUERY for stage in self.stages)

    @property
    def uses_two_stage(self) -> bool:
        return len(self.stages) == 2

    @property
    def decode_stage(self) -> StageSpec | None:
        for stage in self.stages:
            if stage.kind is StageKind.DECODE_QUERY:
                return stage
        return None

    @property
    def prefill_stage(self) -> StageSpec | None:
        if not self.uses_two_stage:
            return None
        return self.stages[0]


def pruning_spec_from_legacy_config(config: VisionTokenPruningConfig) -> PruningSpec:
    """Translate the historical flat config into explicit stages.

    Ambiguous or unsupported combinations raise. Nothing is inferred beyond the
    documented meaning of the legacy fields.
    """

    if not config.enabled:
        return PruningSpec(enabled=False, stages=(), runtime=RuntimeKind.PHYSICAL_FIXED)

    if config.prefill_keep_ratio is not None:
        return _two_stage_from_legacy(config)

    if config.prune_after_layer < 0:
        if config.selector_input != "vision_embedding":
            raise ValueError(
                "legacy physical pruning (prune_after_layer=-1) requires "
                "selector_input='vision_embedding'"
            )
        if config.selector == "vision_pulse":
            raise ValueError("vision_pulse cannot run as physical_pre_decoder")
        if config.keep_ratio_schedule:
            raise ValueError(
                "physical_pre_decoder rejects keep_ratio_schedule; "
                "physical layout is fixed at engine launch (use boundary_once/flex_mask)"
            )
        stage = StageSpec(
            kind=StageKind.PHYSICAL_PRE_DECODER,
            policy=config.selector,
            keep_ratio=config.keep_ratio,
            input_kind=InputKind.VISION_EMBEDDING,
            policy_kwargs=config.selector_kwargs,
            name="physical_pre_decoder",
        )
        return PruningSpec(
            enabled=True,
            stages=(stage,),
            runtime=RuntimeKind.PHYSICAL_FIXED,
        )

    input_kind = _LEGACY_SELECTOR_INPUT_TO_INPUT_KIND.get(config.selector_input)
    if input_kind is None:
        raise ValueError(f"unsupported legacy selector_input={config.selector_input!r}")

    if config.selector_input == "decode_query":
        stage = StageSpec(
            kind=StageKind.DECODE_QUERY,
            policy=config.selector,
            keep_ratio=config.keep_ratio,
            input_kind=InputKind.DECODE_QK,
            policy_kwargs=config.selector_kwargs,
            observe_layer=config.prune_after_layer,
            apply_from_layer=config.prune_after_layer + 1,
            name="decode_query",
        )
    else:
        stage = StageSpec(
            kind=StageKind.BOUNDARY_ONCE,
            policy=config.selector,
            keep_ratio=config.keep_ratio,
            input_kind=input_kind,
            policy_kwargs=config.selector_kwargs,
            observe_layer=config.prune_after_layer,
            apply_from_layer=config.prune_after_layer + 1,
            name="boundary_once",
        )
    return PruningSpec(
        enabled=True,
        stages=(stage,),
        runtime=RuntimeKind.FLEX_MASK,
        keep_ratio_schedule=config.keep_ratio_schedule,
        layerwise_backend=config.layerwise_backend,  # type: ignore[arg-type]
        pre_pruning_backend=config.pre_pruning_backend,  # type: ignore[arg-type]
    )


def _two_stage_from_legacy(config: VisionTokenPruningConfig) -> PruningSpec:
    if config.prefill_keep_ratio is None:
        raise ValueError("two-stage translation requires prefill_keep_ratio")
    if config.prune_after_layer < 0:
        raise ValueError("two-stage pruning requires prune_after_layer >= 0")
    if config.selector_input != "decode_query" or config.selector != "vision_pulse":
        raise ValueError(
            "legacy two-stage pruning requires selector_input='decode_query' "
            "and selector='vision_pulse'"
        )
    if config.layerwise_backend != "flex":
        raise ValueError("legacy two-stage pruning requires layerwise_backend='flex'")

    if config.prefill_prune_after_layer < 0:
        prefill = StageSpec(
            kind=StageKind.PHYSICAL_PRE_DECODER,
            policy=config.prefill_selector,
            keep_ratio=config.prefill_keep_ratio,
            input_kind=InputKind.VISION_EMBEDDING,
            policy_kwargs=config.prefill_selector_kwargs,
            name="prefill_physical",
        )
    else:
        if config.prefill_prune_after_layer > config.prune_after_layer:
            raise ValueError("prefill_prune_after_layer must be <= prune_after_layer")
        prefill = StageSpec(
            kind=StageKind.BOUNDARY_ONCE,
            policy=config.prefill_selector,
            keep_ratio=config.prefill_keep_ratio,
            input_kind=InputKind.VISION_EMBEDDING,
            policy_kwargs=config.prefill_selector_kwargs,
            observe_layer=config.prefill_prune_after_layer,
            apply_from_layer=config.prefill_prune_after_layer + 1,
            name="prefill_boundary",
        )

    decode = StageSpec(
        kind=StageKind.DECODE_QUERY,
        policy=config.selector,
        keep_ratio=config.keep_ratio,
        input_kind=InputKind.DECODE_QK,
        policy_kwargs=config.selector_kwargs,
        observe_layer=config.prune_after_layer,
        apply_from_layer=config.prune_after_layer + 1,
        name="decode_query",
    )
    return PruningSpec(
        enabled=True,
        stages=(prefill, decode),
        runtime=RuntimeKind.FLEX_MASK,
        keep_ratio_schedule=config.keep_ratio_schedule,
        layerwise_backend=config.layerwise_backend,  # type: ignore[arg-type]
        pre_pruning_backend=config.pre_pruning_backend,  # type: ignore[arg-type]
    )


def stage_specs_from_mapping(stages: Sequence[Mapping[str, Any]]) -> tuple[StageSpec, ...]:
    """Build stages from a YAML/dict list. Every required field must be present."""

    if not stages:
        raise ValueError("stages must be a non-empty list")
    built: list[StageSpec] = []
    for index, raw in enumerate(stages):
        if not isinstance(raw, Mapping):
            raise ValueError(f"stages[{index}] must be a mapping")
        required = {"kind", "policy", "keep_ratio", "input_kind"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"stages[{index}] missing fields: {sorted(missing)}")
        built.append(
            StageSpec(
                kind=StageKind(str(raw["kind"])),
                policy=str(raw["policy"]),
                keep_ratio=float(raw["keep_ratio"]),
                input_kind=InputKind(str(raw["input_kind"])),
                policy_kwargs=dict(raw.get("policy_kwargs") or {}),
                observe_layer=None if raw.get("observe_layer") is None else int(raw["observe_layer"]),
                apply_from_layer=(
                    None if raw.get("apply_from_layer") is None else int(raw["apply_from_layer"])
                ),
                name=None if raw.get("name") is None else str(raw["name"]),
            )
        )
    return tuple(built)
