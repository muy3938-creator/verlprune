# Algorithm Developer Guide

Write policies. Do not edit vLLM plugins, actor replay, or OPD loss.

## Three stage kinds

| Kind | When to use | Required `input_kind` |
|---|---|---|
| `physical_pre_decoder` | Cut visual tokens before decoder; real sequence/KV savings | `vision_embedding` |
| `boundary_once` | Observe after decoder layer L, apply from L+1 once | `vision_embedding` or `boundary_qkv` |
| `decode_query` | Per decode step, route visual KV by query | `decode_qk` |

## Add a policy

```python
# my_pkg/my_policy.py
import torch
from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest

def my_policy(request: VisionTokenSelectionRequest) -> torch.Tensor:
    if request.features is None:
        raise ValueError("my_policy requires vision features")  # fail fast
    scores = request.features.float().norm(dim=-1)
    return scores.topk(request.keep_count, sorted=False).indices.sort().values
```

Configure:

```yaml
vision_token_pruning:
  enabled: true
  keep_ratio: 0.5
  prune_after_layer: -1          # physical_pre_decoder
  selector: my_pkg.my_policy:my_policy
  selector_kwargs: {}
```

Or register:

```python
from verl.models.vision_token_pruning.strategy import register_vision_token_strategy
register_vision_token_strategy("my_policy", my_policy)
```

## Output contract (enforced)

- rank-1 int tensor on `request.device`
- length == `request.keep_count`
- sorted, unique, in `[0, token_count)`
- no forced anchor token

## Prefer StageSpec for new experiments

```python
from verl.models.vision_token_pruning.stages import (
    InputKind, PruningSpec, RuntimeKind, StageKind, StageSpec,
)

spec = PruningSpec(
    enabled=True,
    runtime=RuntimeKind.PHYSICAL_FIXED,
    stages=(
        StageSpec(
            kind=StageKind.PHYSICAL_PRE_DECODER,
            policy="my_policy",
            keep_ratio=0.5,
            input_kind=InputKind.VISION_EMBEDDING,
        ),
    ),
)
```

Legacy flat YAML still works via `VisionTokenPruningConfig.to_pruning_spec()`.
Ambiguous combinations raise; nothing is silently repaired.

## What you must not touch

- `verl/vllm_plugins/*` (unless adding a new observe tensor source)
- `training.py` actor replay
- teacher stripping / OPD loss
- paged KV / block tables

## Numerical oracle

Transformers attention-mask replay is the correctness standard.
vLLM Flex masks must match that oracle within floating tolerance.
