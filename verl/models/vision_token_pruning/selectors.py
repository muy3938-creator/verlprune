from __future__ import annotations

import torch


def select_random_visual_tokens(
    token_count: int,
    keep_count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly retain tokens while preserving the final MRoPE anchor token."""

    if not 0 < keep_count <= token_count:
        raise ValueError(f"keep_count must be in [1, {token_count}], got {keep_count}")
    if keep_count == token_count:
        return torch.arange(token_count, device=device)
    if keep_count == 1:
        return torch.tensor([token_count - 1], device=device)

    random_indices = torch.randperm(token_count - 1, device=device, generator=generator)[: keep_count - 1]
    return torch.cat((random_indices, random_indices.new_tensor([token_count - 1]))).sort().values
