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


class PhasePopOutSettingsTest(unittest.TestCase):
    def test_pop_outs_mirror_discard_settings(self):
        expectations = {
            SETTINGS.PHASE_A.name: False,
            SETTINGS.PHASE_B.name: True,
            SETTINGS.PHASE_C.name: False,
            SETTINGS.PHASE_D.name: True,
        }
        for phase in (
            SETTINGS.PHASE_A,
            SETTINGS.PHASE_B,
            SETTINGS.PHASE_C,
            SETTINGS.PHASE_D,
        ):
            with self.subTest(phase=phase.name):
                expected = expectations[phase.name]
                self.assertEqual(
                    expected,
                    phase.allow_pop_outs,
                    f"{phase.name} pop-out setting should be {expected}",
                )


if __name__ == "__main__":
    unittest.main()
