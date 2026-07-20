# 128-row ChartQA matrix

The `*-128.json` benchmark files use the same validation prefix, layer 0,
512-token response cap, and 5%-50% dynamic clamp. The `trained-*` files use adapters from
the named training runs. `two-stage-*` is retained only as a diagnostic because
the Transformers sampler does not physically compact prefill tokens.

`default1024-smoke.json` is separate: it is a one-row H20 run using the new
1024-token default and expanded reasoning prompt.

The three `*cot1024-16.json` files compare dense, untrained top-p 0.80, and the
new 10-step adapter under that expanded prompt. `cot1024-training/` contains
the ten rollout records and raw training/benchmark logs. This small matrix is
a prompt/length diagnostic and must not be mixed into the historical 128-row
accuracy table.

The pure low-learning-rate run has ten rollout JSONL files and is the current
semantically aligned learned-budget result.
