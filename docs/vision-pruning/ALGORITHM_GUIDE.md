# Algorithm guide (platform)

See also `PLATFORM.md`.

## You only write a policy

```python
from verl.models.vision_token_pruning.request import VisionTokenSelectionRequest

def my_policy(request: VisionTokenSelectionRequest):
    if request.features is None:
        raise ValueError("my_policy requires features")
    scores = request.features.float().norm(dim=-1)
    return scores.topk(request.keep_count, sorted=False).indices.sort().values
```

Configure (Hydra flat fields → `PruningSpec` on init):

```yaml
vision_token_pruning:
  enabled: true
  keep_ratio: 0.5
  prune_after_layer: -1          # physical_pre_decoder
  selector: my_pkg.mod:my_policy
```

Or:

```python
from verl.models.vision_token_pruning import register_vision_token_strategy
register_vision_token_strategy("my_policy", my_policy)
```

## Inspect the plan

```python
config = VisionTokenPruningConfig(enabled=True, keep_ratio=0.5)
assert config.spec.stages[0].kind.value == "physical_pre_decoder"
```

## Do not edit

- vLLM plugins (unless new observe tensor)
- `training.py` replay / teacher strip
- OPD loss

## Contract

- return rank-1 int indices, length `keep_count`, sorted unique, in range
- fail fast if required tensors missing
