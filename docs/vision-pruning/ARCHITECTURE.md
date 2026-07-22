# Vision Token Pruning — Codebase Architecture

Branch: `refactor/three-stage-pruning-api`  
Principle: **one plan (`PruningSpec`), three stage kinds, exact rollout→actor replay, unpruned teacher**

---

## 1. One-sentence summary

```text
Rollout runs the policy once → wire indices travel with the sample →
actor replays the same indices → OPD teacher always sees the full image.
```

YAML/Hydra only **constructs** `PruningSpec`. Algorithms only implement **policies**.

---

## 2. End-to-end training flow

```mermaid
flowchart TB
    subgraph Config["① Config"]
        YAML["Hydra / YAML flat fields"]
        CFG["VisionTokenPruningConfig"]
        SPEC["PruningSpec + StageSpec[]\n(single source of truth)"]
        YAML --> CFG
        CFG -->|"__post_init__ builds"| SPEC
    end

    subgraph Rollout["② Rollout (vLLM worker)"]
        SRV["vllm_async_server\nVisionTokenPruningRollout"]
        BACK["backends.py\nresolve architecture + launch args"]
        PLUG["vLLM OOT plugin\nphysical | flex | compact_flash"]
        POL["Policy\npolicies/* + strategy engine"]
        CAP["Capture channel\nrouted_experts + embedding meta"]
        WIRE["Wire dict\nStatic | Dynamic | TwoStage"]

        SRV --> BACK --> PLUG
        PLUG -->|"observe tensors"| POL
        POL -->|"kept indices"| CAP
        CAP --> WIRE
        SRV -->|"decode_selection"| WIRE
    end

    subgraph Glue["③ Batch glue"]
        AG["agent_loop\nattach_selection_to_multi_modal_inputs"]
        MM["sample.multi_modal_inputs\nKEEP_MASK + SELECTION_WIRE"]
        WIRE --> AG --> MM
    end

    subgraph Train["④ Training"]
        ACT["dp_actor\nprepare_actor_pruning_inputs"]
        REPLAY["training.py\nexact index replay → masks"]
        QWEN["qwen2_vl / qwen3_vl\nembed prune or attn mask"]
        TEA["Teacher forward\napply_pruning=False\nstrip_pruning_metadata"]
        OPD["OPD loss / optimizer\nunchanged"]

        MM --> ACT
        ACT -->|"student"| REPLAY --> QWEN --> OPD
        ACT -->|"teacher"| TEA --> OPD
        SPEC -.->|"mode flags"| ACT
    end

    Config --> Rollout
```

### Step notes

| Step | Owner | Does | Does **not** |
|---|---|---|---|
| ① Config | `config.py` + `stages.py` | Build frozen `PruningSpec` | Run algorithms |
| ② Rollout | plugin + `strategy` | Select once, capture indices | Train / loss |
| ③ Glue | `agent_loop` | Attach wire to sample | Re-select |
| ④ Student | `training` + qwen patches | Replay exact indices | Call policy |
| ④ Teacher | same `prepare(..., apply=False)` | Full image, no meta | Any pruning |

---

## 3. Three stage semantics

```mermaid
flowchart LR
    subgraph S1["physical_pre_decoder"]
        E1["Observe: vision embeddings"]
        A1["Apply: physically shorten prompt\n+ shared reduced KV"]
        E1 --> A1
    end

    subgraph S2["boundary_once"]
        E2["Observe: emb or Q/K/V\nafter decoder layer L"]
        A2["Apply: static keep mask\nfrom layer L+1"]
        E2 --> A2
    end

    subgraph S3["decode_query"]
        E3["Observe: decode Q × visual K"]
        A3["Apply: per-query visual KV mask"]
        E3 --> A3
    end

    S1 -.->|"real seq/KV savings\nfixed budget"| R1["runtime: physical_fixed"]
    S2 -.->|"logical mask\nresearch default"| R2["runtime: flex_mask"]
    S3 -.->|"logical mask\nper step"| R2
```

| Kind | Config signal (legacy flat) | Runtime | Real memory save? |
|---|---|---|---|
| `physical_pre_decoder` | `prune_after_layer=-1` | physical plugin | **Yes** (fixed layout) |
| `boundary_once` | `prune_after_layer=L≥0`, input emb/key | flex (or compact_flash) | No (mask only) |
| `decode_query` | `selector_input=decode_query` | flex + VisionPulse path | No (mask only) |

**Two-stage** = ordered list of two stages (e.g. physical prefill + decode_query), not a fourth kind. Wire type `TwoStageVisionTokenSelection` is the transport form of that list.

---

## 4. Package map (what each file owns)

```mermaid
flowchart TB
    subgraph Core["verl/models/vision_token_pruning/"]
        stages["stages.py\nStageKind / StageSpec / PruningSpec"]
        config["config.py\nHydra fields → config.spec"]
        request["request.py\nVisionTokenSelectionRequest"]
        strategy["strategy.py\nregistry + SelectionEngine"]
        policies["policies/*\npure index algorithms"]
        protocol["protocol.py\nwire dataclasses"]
        transport["transport.py\nvLLM capture codec"]
        rollout["rollout.py\nlaunch / decode façade"]
        backends["backends.py\narchitecture profiles"]
        training["training.py\nactor replay + teacher strip"]
        embeddings["embeddings.py\nphysical gather"]
        curriculum["curriculum.py\nstep → keep_ratio"]
        dynamic["dynamic.py\ndecode_query math"]
        sampler["transformers_sampler.py\nbenchmark only"]
    end

    config --> stages
    strategy --> policies
    strategy --> request
    rollout --> backends
    rollout --> transport
    training --> protocol
    training --> embeddings

    subgraph Plugins["verl/vllm_plugins/"]
        reg["register.py\nentry-point models"]
        phys["vision_token_pruning.py\nphysical layer-0"]
        flex["layerwise_flex_*.py\nflex mask + dynamic"]
        compact["layerwise_vision_token_pruning.py\ncompact flash research"]
        flexpkg["flex/plan.py + observe.py"]
    end

    reg --> phys
    reg --> flex
    reg --> compact
    flex --> flexpkg
    phys --> strategy
    flex --> strategy
    flex --> dynamic

    subgraph Integration["Framework hooks only"]
        async["vllm_async_server.py"]
        agent["agent_loop.py"]
        actor["dp_actor.py"]
        qwen["qwen2_vl.py / qwen3_vl.py"]
    end

    async --> rollout
    agent --> training
    actor --> training
    qwen --> embeddings
```

### Role table

| Layer | Files | Responsibility |
|---|---|---|
| **Plan** | `stages`, `config` | What experiment means |
| **Algorithm** | `policies/*`, `strategy`, `request`, `dynamic` | Which tokens to keep |
| **Wire** | `protocol`, `transport` | How indices leave vLLM |
| **Rollout I/O** | `rollout`, `backends`, plugins | Launch engine, select, capture |
| **Train apply** | `training`, `embeddings`, qwen patches | Replay / strip / compact |
| **Glue** | async_server, agent_loop, dp_actor | Thin call sites (~3 hooks) |
| **Non-training** | `transformers_sampler`, compact-flash | Bench / research side path |

---

## 5. Data path detail (one sample)

```mermaid
sequenceDiagram
    participant H as Hydra config
    participant S as PruningSpec
    participant R as VisionTokenPruningRollout
    participant V as vLLM plugin
    participant P as Policy
    participant A as AgentLoop
    participant D as dp_actor student
    participant T as Teacher

    H->>S: build frozen plan
    R->>V: launch with backend arch + payload
    Note over V,P: generate one multimodal request
    V->>P: SelectionRequest (emb or QKV or Q×K)
    P-->>V: sorted unique keep indices
    V-->>R: routed_experts capture
    R-->>A: wire dict on TokenOutput
    A->>A: attach KEEP_MASK + SELECTION_WIRE
    A->>D: batch with multi_modal_inputs
    D->>D: prepare_actor_pruning_inputs(apply=True)
    D->>D: replay indices → mask / compact embeds
    D->>T: prepare(..., apply=False) strip meta
    T-->>D: dense logits
    D->>D: OPD loss (unchanged)
```

### Wire shapes (current)

| Wire class | When | Payload |
|---|---|---|
| `VisionTokenSelection` | physical / boundary static | `kept_visual_indices`, fingerprint |
| `DynamicVisionTokenSelection` | decode_query only | per-query index rows |
| `TwoStageVisionTokenSelection` | prefill + decode | `prefill` + `decode` nested |

Fingerprint = SHA-256 of `(policy name, policy kwargs)`. Actor rejects mismatch (never re-runs policy).

---

## 6. Runtime backends

```mermaid
flowchart TD
    SPEC{PruningSpec.runtime}
    SPEC -->|physical_fixed| PHYS["PHYSICAL_BACKEND\nVerlPrunedQwen*"]
    SPEC -->|flex_mask + flex| FLEX["LAYERWISE_FLEX\nVerlLayerwiseFlexPrunedQwen*"]
    SPEC -->|flex_mask + compact_flash| COMP["LAYERWISE_COMPACT_FLASH\nQwen2.5 only, research"]

    PHYS --> P1["Shorten placeholders\nselect on embeddings\nshared short KV"]
    FLEX --> F1["Full prompt length\nscore_mod mask after boundary\noptional decode_query"]
    COMP --> C1["Physical QKV compact after layer\neager, experimental"]
```

Constraints (fail-fast):

- Pruning capture needs **eager** vLLM (compiled graph skips Python hooks).
- Flex path disables chunked prefill / prefix cache.
- Exactly **one image**, no video.
- Cannot share `routed_experts` with routing-replay.

---

## 7. Where to change what

```mermaid
flowchart LR
    Q1["New scoring algorithm?"] --> A1["Add policies/foo.py\n+ register or module:fn"]
    Q2["Change observe layer / budget?"] --> A2["Hydra fields → rebuilds Spec"]
    Q3["New tensor input for policy?"] --> A3["request.py + plugin observe"]
    Q4["OPD loss / teacher?"] --> A4["Do not touch for pruning algos"]
    Q5["Paged KV / block table?"] --> A5["Never in policy code"]
```

| Goal | Edit | Avoid |
|---|---|---|
| New algorithm | `policies/` or external `module:fn` | plugins, actor, teacher |
| Layer / ratio / curriculum | YAML → `VisionTokenPruningConfig` | hardcoding in plugins |
| Transport bug | `transport.py` only | leaking vLLM types into actor |
| Numerical check | Transformers mask path = oracle | claiming Flex curriculum = physical speedup |

---

## 8. Module size (orientation)

| Area | Approx LOC | Notes |
|---|---:|---|
| Core plan + algo + wire + train | ~2.5k | main product surface |
| Physical plugin | ~280 | stable default |
| Flex plugin | ~830 + flex/* | research hot path |
| Compact flash | ~520 | optional research |
| Transformers sampler | ~450 | offline bench only |
| Framework hooks | small | async_server / agent_loop / dp_actor / qwen |

---

## 9. Invariants (do not break)

1. **Select once in rollout**; actor only validates + replays.
2. **Teacher always dense** (`apply_pruning=False` + strip metadata).
3. **Policy is pure** over `VisionTokenSelectionRequest` → indices.
4. **Fail fast** on bad stage combo, missing schedule step, fingerprint mismatch.
5. **Numerical oracle** = Transformers attention-mask implementation.
6. **`config.spec`** is the execution plan; `uses_*` flags derive from it.

---

## 10. Related docs

| Doc | Content |
|---|---|
| `PLATFORM.md` | Short mental model |
| `ALGORITHM_GUIDE.md` | How to write a policy |
| `REFACTOR_GOALS.md` | Refactor targets + progress |
| `../vision-pruning-refactor-proposal.md` | Longer historical proposal |
