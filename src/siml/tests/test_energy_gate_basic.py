import unittest
from siml.energy_gate import EnergyWithGate

def _fake_base(a_seq, z_k, s_k, z_g):
    # deterministic placeholder for JEPA latent L1
    return 3.0

class TestEnergyGateBasic(unittest.TestCase):
    def test_mode_off_returns_base(self):
        e = EnergyWithGate(base_predictor=_fake_base, mode="off")
        self.assertEqual(e.score(None, None, None, None), 3.0)

    def test_mode_grip_returns_base(self):
        e = EnergyWithGate(base_predictor=_fake_base, mode="grip")
        self.assertEqual(e.score(None, None, None, None), 3.0)

    def test_fepgate_without_ctx_degrades_to_base(self):
        e = EnergyWithGate(base_predictor=_fake_base, mode="fepgate",
                           lambda_penalty=0.5, hard_gate=False,
                           siml_ctx=None, tau=None)
        self.assertEqual(e.score(None, None, None, None), 3.0)

if __name__ == "__main__":
    unittest.main()
