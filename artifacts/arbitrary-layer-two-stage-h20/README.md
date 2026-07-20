# Arbitrary-layer two-stage H20 artifacts

These files are the raw rollout, actor-replay, and short-training records for
the arbitrary first-stage boundary implementation. The complete interpretation
and the Qwen2.5 results recovered from terminal output are in
`docs/arbitrary-layer-two-stage-validation.md`.

The Qwen3 directories contain:

- all five method families at layers 0, 7, 15, and 27;
- delayed VisionPulse `7 -> 15`, `50% -> 50%` rollout/training/parity;
- hybrid Flash/Flex `15 -> 27`, `10% -> 10%` rollout/training.

Some early Qwen2.5 output files were lost when short-lived CNB workspaces were
reclaimed. Their terminal measurements are retained verbatim in the report.
