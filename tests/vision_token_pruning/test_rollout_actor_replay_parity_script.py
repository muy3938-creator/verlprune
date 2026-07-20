import importlib.util
from pathlib import Path

import pytest

from verl.models.vision_token_pruning.protocol import compute_keep_count


def _module():
    path = Path(__file__).parents[2] / "scripts" / "rollout_actor_replay_parity.py"
    spec = importlib.util.spec_from_file_location("rollout_actor_replay_parity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parity_metrics_reports_token_and_sequence_drift():
    metrics = _module().parity_metrics(
        [[-1.0, -2.0], [-0.5, -1.5]],
        [[-0.9, -2.2], [-0.5, -1.4]],
    )

    assert metrics["token_count"] == 4
    assert metrics["sampled_logprob_abs_diff_mean"] == pytest.approx(0.1)
    assert metrics["sampled_logprob_abs_diff_max"] == pytest.approx(0.2)
    assert metrics["prefill_token_abs_diff_mean"] == pytest.approx(0.05)
    assert metrics["decode_token_abs_diff_mean"] == pytest.approx(0.15)
    assert metrics["sequence_log_ratio_abs_max"] == pytest.approx(0.1)
    assert 0.0 <= metrics["fraction_ratio_outside_0_9_1_1"] <= 1.0
    assert 0.0 <= metrics["fraction_ratio_outside_0_8_1_2"] <= 1.0
    assert len(metrics["per_position_abs_diff_mean"]) == 2


def test_parity_metrics_rejects_mismatched_batches():
    with pytest.raises(ValueError, match="equal size"):
        _module().parity_metrics([[-1.0]], [])


def test_parity_metrics_can_pool_variable_response_lengths():
    metrics = _module().parity_metrics(
        [[-1.0], [-0.5, -1.5]],
        [[-0.9], [-0.5, -1.4]],
    )

    assert metrics["token_count"] == 3
    assert metrics["response_length"] is None
    assert metrics["response_length_min"] == 1
    assert metrics["response_length_max"] == 2


def test_absolute_difference_stats_handles_empty_and_tail_values():
    module = _module()

    assert module._absolute_difference_stats([])["count"] == 0
    metrics = module._absolute_difference_stats([0.1, 0.2, 0.3])
    assert metrics["mean"] == pytest.approx(0.2)
    assert metrics["p99"] == pytest.approx(0.3)
    assert metrics["max"] == pytest.approx(0.3)


def test_plugin_no_prune_diagnostic_ratio_keeps_every_visual_token():
    assert compute_keep_count(64, 1.0 - 1e-9) == 64
