"""Bound linear decay without changing the in-horizon schedule or group scaling."""
from src.utils import schedulers as target
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('final', [0.0, 0.1, 1.5])
def test_linear_schedule_stays_at_terminal_value(final):
    cls = target.LinearDecaySchedule
    optimizer = SimpleNamespace(param_groups=[{'lr': 1.0}, {'lr': 1.0}])
    schedule = cls(optimizer, ref_lr=1.0, T_max=4, final_lr=final)
    values = [schedule.step() for _ in range(8)]
    assert values[:4] == pytest.approx([1 + (final - 1) * s / 4 for s in range(1, 5)])
    assert values[4:] == pytest.approx([final] * 4)
    assert all((group['lr'] == pytest.approx(final) for group in optimizer.param_groups))

@pytest.mark.parametrize('final', [0.0, 0.1, 1.5])
def test_wsd_schedule_stays_at_terminal_value_and_preserves_scale(final):
    cls = target.WSDSchedule
    optimizer = SimpleNamespace(param_groups=[{'lr': 0.0}, {'lr': 0.0, 'lr_scale': 0.2}])
    schedule = cls(optimizer, warmup_steps=2, anneal_steps=3, T_max=8, start_lr=0.0, ref_lr=1.0, final_lr=final)
    values = [schedule.step() for _ in range(12)]
    expected = [0.5, 1.0, 1.0, 1.0, 1.0, 1 + (final - 1) / 3, 1 + 2 * (final - 1) / 3, final]
    assert values[:8] == pytest.approx(expected)
    assert values[8:] == pytest.approx([final] * 4)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(final)
    assert optimizer.param_groups[1]['lr'] == pytest.approx(final * 0.2)

def test_linear_in_horizon_values():
    cls = target.LinearDecaySchedule
    optimizer = SimpleNamespace(param_groups=[{'lr': 1.0}])
    schedule = cls(optimizer, ref_lr=1.0, T_max=5, final_lr=0.2)
    assert [schedule.step() for _ in range(5)] == pytest.approx([0.84, 0.68, 0.52, 0.36, 0.2])
