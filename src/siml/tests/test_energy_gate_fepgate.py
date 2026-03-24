import unittest
from siml.energy_gate import EnergyWithGate

class FakeSimlCtx:
    def __init__(self, surprise_value: float, ready: bool = True):
        self._surprise_value = float(surprise_value)
        self._ready = bool(ready)
    def ready(self) -> bool:
        return self._ready
    def surprise(self, a_seq, s_k, extras=None) -> float:
        return self._surprise_value

def _fake_base(a_seq, z_k, s_k, z_g):
    return 2.5

class TestFEPGate(unittest.TestCase):
    def test_soft_gate_ok_bit_1_no_penalty(self):
        ctx = FakeSimlCtx(surprise_value=0.40)
        e = EnergyWithGate(base_predictor=_fake_base, mode="fepgate",
                           lambda_penalty=0.7, hard_gate=False,
                           siml_ctx=ctx, tau=0.5)
        self.assertEqual(e.score(None, None, None, None), 2.5)

    def test_soft_gate_ok_bit_0_adds_lambda(self):
        ctx = FakeSimlCtx(surprise_value=0.60)
        e = EnergyWithGate(base_predictor=_fake_base, mode="fepgate",
                           lambda_penalty=0.7, hard_gate=False,
                           siml_ctx=ctx, tau=0.5)
        self.assertEqual(e.score(None, None, None, None), 2.5 + 0.7)

    def test_hard_gate_adds_large_constant(self):
        ctx = FakeSimlCtx(surprise_value=0.60)
        e = EnergyWithGate(base_predictor=_fake_base, mode="fepgate",
                           lambda_penalty=0.7, hard_gate=True,
                           siml_ctx=ctx, tau=0.5)
        val = e.score(None, None, None, None)
        self.assertGreaterEqual(val, 2.5 + 1e8)  # generous bound

    def test_lambda_monotonicity_when_rejected(self):
        ctx = FakeSimlCtx(surprise_value=0.60)
        e1 = EnergyWithGate(base_predictor=_fake_base, mode="fepgate",
                            lambda_penalty=0.1, hard_gate=False,
                            siml_ctx=ctx, tau=0.5)
        e2 = EnergyWithGate(base_predictor=_fake_base, mode="fepgate",
                            lambda_penalty=0.9, hard_gate=False,
                            siml_ctx=ctx, tau=0.5)
        v1 = e1.score(None, None, None, None)
        v2 = e2.score(None, None, None, None)
        self.assertGreater(v2, v1)

if __name__ == "__main__":
    unittest.main()
