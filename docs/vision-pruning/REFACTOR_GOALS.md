# Vision Token Pruning — Refactor Goals

Branch: `refactor/three-stage-pruning-api`  
Deadline: 2026-07-22 08:00 CST  
Principle: **fail fast, no silent fallbacks, algorithms only touch policies**

## Target semantics (exactly three)

| Kind | When | Observe | Apply | Real KV/seq savings? |
|---|---|---|---|---|
| `physical_pre_decoder` | Before decoder | vision embeddings | Physically shorten prompt + shared KV | Yes (fixed budget) |
| `boundary_once` | After decoder layer L | embeddings or Q/K/V | Static keep mask from L+1 | Flex: no; physical later: yes |
| `decode_query` | Each decode step | query × visual K | Per-query visual-KV mask | No (logical mask) |

Two-stage experiments = **ordered list of stages**, not a third protocol type.

## Invariants (do not break)

1. Selector runs **once** in rollout; actor **replays exact indices**.
2. OPD teacher strips all pruning metadata; full image always.
3. Policy is a pure function of `(SelectionContext, Budget)` → indices.
4. Policy never touches paged-KV / block tables.
5. Numerical oracle = Transformers attention mask path.

## Maintainability targets

1. Public experiment surface = `PruningSpec` + `StageSpec[]` + `Policy`.
2. Legacy YAML fields map through an **explicit translator** that raises on ambiguity.
3. Main runtimes: `physical_fixed` + `flex_mask` only in default path.
4. Flex plugin split: observe / apply / capture / model adapter.
5. No dual public APIs (`strategy` is canonical; `selectors`/`runtime` are thin or removed).
6. No silent defaults for boundary, budget, or stage kind.

## Fail-fast rules

- Missing required field → `ValueError` immediately.
- Schedule configured but `global_step` missing → error (no fallback ratio).
- `decode_query` without flex runtime → error.
- `decoder_key` / `boundary_qkv` with physical-only runtime → error.
- Unknown policy kwargs → error (no ignore).
- Protocol fingerprint mismatch on actor → error.

## Acceptance

- Existing unit tests still pass (legacy fields via translator).
- New unit tests cover StageSpec validation and translator.
- Optional H20: physical + flex smoke vs Transformers mask parity.

## Progress log

| Time (CST) | Item | Status |
|---|---|---|
| 2026-07-21 23:42 | Branch `refactor/three-stage-pruning-api` | done |
| 2026-07-21 23:45 | `stages.py` StageKind×3 + PruningSpec + legacy translator | done |
| 2026-07-21 23:50 | `policies/*` extracted from strategy.py | done |
| 2026-07-21 23:55 | `request.py` split; strategy becomes thin registry | done |
| 2026-07-22 00:00 | Unit suite on H20: **1039 passed** | done |
| 2026-07-22 00:01 | Fail-fast: schedule without global_step | done |
| 2026-07-22 00:02 | Flex `plan.py` + `observe.py` extracted | done |
| 2026-07-22 00:05 | GPU logit parity blocked: workspace reclaim + entry_point install | in progress |
