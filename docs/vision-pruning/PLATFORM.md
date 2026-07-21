# Visual-token pruning platform

One platform. Three semantics. One execution plan.

## Mental model

```text
Policy (algorithm)  →  PruningSpec (stages)  →  Runtime apply  →  Trace  →  Actor replay
                                                              ↘ Teacher: strip, full image
```

YAML / Hydra is only a way to **construct** `PruningSpec`. It is not a second logic layer.

## Three stage kinds (only)

| Kind | Observe | Apply |
|---|---|---|
| `physical_pre_decoder` | vision embeddings | shorten sequence + shared KV |
| `boundary_once` | embeddings or Q/K/V after layer L | static keep mask from L+1 |
| `decode_query` | decode query × visual K | per-query visual KV mask |

Two-stage experiments = **two stages in order**, not a fourth kind.

## Single source of truth

```python
spec = config.to_pruning_spec()   # == config.spec
# runtime / rollout / validation all reason about spec.stages
```

`VisionTokenPruningConfig` keeps flat Hydra fields for existing launchers. On init it **builds and freezes** `PruningSpec`. Mode flags (`uses_*`) are derived from the spec, not re-implemented.

## Algorithm surface

```python
def my_policy(request: VisionTokenSelectionRequest) -> Tensor:
    # fail fast if required tensors missing
    ...
    return sorted unique keep indices  # length == request.keep_count
```

Register or pass `module:function`. Do not edit plugins, actor, or teacher for algorithm work.

## Invariants

1. Rollout selects once; actor replays exact indices.
2. Teacher always dense (metadata stripped).
3. Fail fast: missing fields / bad stage combo / schedule without step → error.
4. Numerical oracle: Transformers attention-mask path.
