# Backend-only no-prune H20 artifacts

These artifacts back `docs/backend-only-no-prune-parity.md`.

The experiment contains 23 cases and 2,528 sampled tokens. Every plugin case
records `plugin_no_prune: true` and a keep ratio of `0.999999999`; each rollout
asserted that all 64 visual-token indices were present before accepting the
result.

`backend-only-no-prune-matrix/summary.json` contains:

- native unpruned, all-Flex no-prune, and hybrid no-prune pooled groups;
- per-case rollout-versus-actor metrics;
- nine direct all-Flex-versus-hybrid pairs;
- common-prefix and identical-sequence rollout differences;
- the exact-zero actor replay comparison for identical inputs.

Each case directory contains `rollout.json`, `actor.json`, and
`comparison.json`. Generated images and verbose engine logs are omitted because
they are deterministic and contain no additional numerical results.
