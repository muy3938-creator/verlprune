"""Pre-LLM vision token selection algorithms.

All algorithms accept flattened image embeddings of shape ``(N, D)`` and
return a sorted 1-D index tensor of the *kept* token positions.
"""

from __future__ import annotations

from typing import Any

import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_vision_tokens(
    image_embeds: torch.Tensor,
    *,
    keep_ratio: float | None = None,
    keep_count: int | None = None,
    text_embeds: torch.Tensor | None = None,
    method: str = "random",
    image_grid_thw: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    **method_kwargs: Any,
) -> torch.Tensor:
    """Select which visual tokens to keep before entering the LLM.

    Args:
        image_embeds: Visual features, shape ``(N, D)``.
        keep_ratio: Fraction of tokens to retain (0, 1). Mutually exclusive
            with *keep_count*.
        keep_count: Absolute number of tokens to retain.  Mutually exclusive
            with *keep_ratio*.
        text_embeds: Optional text/prompt embeddings for saliency-based
            scoring, shape ``(T, D)``.
        method: Selection algorithm name.  Built-in options:
            ``"random"``, ``"uniform"``, ``"text_saliency"``,
            ``"greedy_prune"``.
        image_grid_thw: Optional spatial layout ``(t, h, w)`` for
            structure-aware strategies.
        generator: Optional :class:`torch.Generator` for reproducibility.
        **method_kwargs: Forwarded to the selected algorithm.

    Returns:
        A **sorted** 1-D :class:`torch.Tensor` of kept token indices,
        length ``keep_count``.
    """
    if image_embeds.ndim != 2:
        raise ValueError(f"image_embeds must be rank-2, got shape {tuple(image_embeds.shape)}")

    n_tokens = image_embeds.shape[0]
    k = _resolve_keep_count(n_tokens, keep_ratio=keep_ratio, keep_count=keep_count)

    if k >= n_tokens:
        return torch.arange(n_tokens, device=image_embeds.device)
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=image_embeds.device)

    selector = _METHODS.get(method)
    if selector is None:
        raise ValueError(
            f"Unknown selection method {method!r}; "
            f"available: {sorted(_METHODS)}"
        )
    indices = selector(
        image_embeds,
        k=k,
        text_embeds=text_embeds,
        image_grid_thw=image_grid_thw,
        generator=generator,
        **method_kwargs,
    )
    return indices.sort().values


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _resolve_keep_count(
    total: int,
    *,
    keep_ratio: float | None,
    keep_count: int | None,
) -> int:
    if keep_ratio is not None and keep_count is not None:
        raise ValueError("Specify either keep_ratio or keep_count, not both.")
    if keep_ratio is not None:
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        return max(1, int(total * keep_ratio))
    if keep_count is not None:
        if keep_count < 0:
            raise ValueError(f"keep_count must be >= 0, got {keep_count}")
        return min(keep_count, total)
    raise ValueError("Must specify either keep_ratio or keep_count.")


# ---------------------------------------------------------------------------
# Built-in selection methods
# ---------------------------------------------------------------------------

def _random_select(
    image_embeds: torch.Tensor,
    *,
    k: int,
    generator: torch.Generator | None = None,
    **_kwargs: Any,
) -> torch.Tensor:
    """Uniformly random subset of *k* tokens."""
    n = image_embeds.shape[0]
    perm = torch.randperm(n, device=image_embeds.device, generator=generator)
    return perm[:k]


def _uniform_select(
    image_embeds: torch.Tensor,
    *,
    k: int,
    **_kwargs: Any,
) -> torch.Tensor:
    """Deterministic uniform stride over the token sequence."""
    n = image_embeds.shape[0]
    return torch.linspace(0, n - 1, steps=k, device=image_embeds.device).long()


def _text_saliency_select(
    image_embeds: torch.Tensor,
    *,
    k: int,
    text_embeds: torch.Tensor | None = None,
    **_kwargs: Any,
) -> torch.Tensor:
    """Keep the *k* visual tokens most similar to the mean text embedding."""
    if text_embeds is None or text_embeds.numel() == 0:
        # Fall back to random if no text context is available.
        return _random_select(image_embeds, k=k)
    img_norm = torch.nn.functional.normalize(image_embeds.float(), dim=-1)
    txt_mean = torch.nn.functional.normalize(
        text_embeds.float().mean(dim=0, keepdim=True), dim=-1
    )
    scores = (img_norm @ txt_mean.T).squeeze(-1)  # (N,)
    _, topk_indices = scores.topk(k, sorted=False)
    return topk_indices


def _greedy_prune_select(
    image_embeds: torch.Tensor,
    *,
    k: int,
    text_embeds: torch.Tensor | None = None,
    similarity_threshold: float = 0.9,
    **_kwargs: Any,
) -> torch.Tensor:
    """Greedy saliency + redundancy pruning (GPU-CPU hybrid optimized)."""
    device = image_embeds.device
    visual = torch.nn.functional.normalize(image_embeds.float(), dim=-1)

    # 1. Compute saliency on GPU
    if text_embeds is not None and text_embeds.numel() > 0:
        last_text = torch.nn.functional.normalize(
            text_embeds.float().mean(dim=0), dim=0
        )
        saliency = visual @ last_text
    else:
        saliency = visual.norm(dim=-1)

    # 2. Compute full similarity matrix on GPU (one-shot batched computation)
    sim_matrix = visual @ visual.T  # (N, N)

    # 3. Move selection tensors to CPU to avoid host-device sync stalls in the loop
    saliency_cpu = saliency.cpu()
    sim_matrix_cpu = sim_matrix.cpu()

    order_cpu = saliency_cpu.argsort(descending=True, stable=True)
    n_tokens = len(image_embeds)
    active = torch.ones(n_tokens, dtype=torch.bool, device="cpu")
    selected: list[int] = []

    # 4. Perform sequential greedy selection on CPU
    for pivot in order_cpu.tolist():
        if not active[pivot].item():
            continue
        selected.append(pivot)
        if len(selected) == k:
            break
        redundant = sim_matrix_cpu[pivot] > similarity_threshold
        active &= ~redundant

    # Fill remaining on CPU if greedy loop fell short
    if len(selected) < k:
        chosen = torch.zeros(n_tokens, dtype=torch.bool, device="cpu")
        if selected:
            chosen[torch.tensor(selected, dtype=torch.long)] = True
        for idx in order_cpu.tolist():
            if not chosen[idx].item():
                selected.append(idx)
                chosen[idx] = True
                if len(selected) == k:
                    break

    # 5. Return a GPU tensor
    return torch.tensor(selected, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

_METHODS: dict[str, Any] = {
    "random": _random_select,
    "uniform": _uniform_select,
    "text_saliency": _text_saliency_select,
    "greedy_prune": _greedy_prune_select,
}
