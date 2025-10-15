import unittest

class TestImports(unittest.TestCase):
    def test_imports(self):
        from siml.energy_gate import EnergyWithGate  # noqa
        from siml.siml_agent_io import SimlContext   # noqa
        from siml.siml_gate import fep_ok, grip_bit  # noqa

    def test_grip_bit_defaults(self):
        from siml.siml_gate import grip_bit
        self.assertEqual(grip_bit({"grip": 1}), 1)
        self.assertEqual(grip_bit({"grip": 0}), 0)
        self.assertEqual(grip_bit({}), 0)

if __name__ == "__main__":
    unittest.main()
