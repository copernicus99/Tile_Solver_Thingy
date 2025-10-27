import unittest

from config import SETTINGS


class PhaseRotationSettingsTest(unittest.TestCase):
    def test_all_phases_allow_rotation(self):
        phases = [
            SETTINGS.PHASE_A,
            SETTINGS.PHASE_B,
            SETTINGS.PHASE_C,
            SETTINGS.PHASE_D,
        ]
        for phase in phases:
            with self.subTest(phase=phase.name):
                self.assertTrue(phase.allow_rotation, f"{phase.name} should allow rotation")


if __name__ == "__main__":
    unittest.main()
