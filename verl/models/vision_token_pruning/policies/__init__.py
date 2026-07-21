"""Built-in visual-token selection policies.

Each policy is a pure function of ``VisionTokenSelectionRequest`` and returns
sorted unique keep indices. Runtime code must not reimplement algorithm logic.
"""

from __future__ import annotations

from .dart import dart_policy
from .divprune import divprune_policy
from .embedding_norm import embedding_norm_policy
from .greedy_prune import greedy_prune_policy
from .key_norm import key_norm_policy
from .random import random_policy
from .uniform import uniform_policy

__all__ = [
    "dart_policy",
    "divprune_policy",
    "embedding_norm_policy",
    "greedy_prune_policy",
    "key_norm_policy",
    "random_policy",
    "uniform_policy",
]
