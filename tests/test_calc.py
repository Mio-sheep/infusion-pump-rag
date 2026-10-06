"""剂量换算。这一页的数字要经得起人手算核对，所以用例都是整的。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infusion_rag.calc import (
    CalcError,
    dose_to_rate,
    parse_concentration,
    parse_dose,
    rate_to_dose,
)


class TestParseDose(unittest.TestCase):
    def test_common_formats(self):
        spec = parse_dose("0.1ug/kg/min")
        self.assertEqual(
            (spec.value, spec.substance, spec.per_kg, spec.per_minute), (0.1, "ug", True, True)
        )
        self.assertEqual(parse_dose("5mg/h").per_kg, False)
        self.assertEqual(parse_dose("12U/kg/h").substance, "U")
        self.assertEqual(parse_dose("0.5μg/kg/min").value, 0.5)  # 希腊字母 μ
        self.assertEqual(parse_dose("0.5mcg/kg/min").value, 0.5)

    def test_rejects_nonsense(self):
        for bad in ("", "abc", "5", "5L/h", "5mg/kg/sec"):
            with self.assertRaises(CalcError, msg=bad):
                parse_dose(bad)


class TestParseConcentration(unittest.TestCase):
    def test_total_amount_form(self):
        conc = parse_concentration("4mg/50mL")
        self.assertAlmostEqual(conc.per_ml, 80.0)  # 4000 ug / 50 mL

    def test_direct_form(self):
        conc = parse_concentration("80ug/mL")
        self.assertAlmostEqual(conc.per_ml, 80.0)

    def test_units(self):
        conc = parse_concentration("40U/40mL")
        self.assertAlmostEqual(conc.per_ml, 1.0)
        self.assertEqual(conc.substance, "U")

    def test_rejects_bad_volume(self):
        with self.assertRaises(CalcError):
            parse_concentration("4mg/50L")


class TestDoseToRate(unittest.TestCase):
    def test_readme_example(self):
        """0.1 μg/kg/min × 60 kg × 60 min = 360 μg/h；4mg/50mL = 80 μg/mL；360/80 = 4.5。"""
        result = dose_to_rate("0.1ug/kg/min", 60, "4mg/50mL")
        self.assertAlmostEqual(result.rate_ml_h, 4.5)
        self.assertAlmostEqual(result.dose_per_hour, 360.0)

    def test_not_weight_based(self):
        self.assertAlmostEqual(dose_to_rate("5mg/h", None, "100mg/100mL").rate_ml_h, 5.0)

    def test_insulin_units(self):
        self.assertAlmostEqual(dose_to_rate("6U/h", None, "40U/40mL").rate_ml_h, 6.0)

    def test_vtbi_gives_duration(self):
        result = dose_to_rate("0.1ug/kg/min", 60, "4mg/50mL", vtbi_ml=50)
        self.assertIn("小时", "\n".join(result.steps))

    def test_weight_required_when_per_kg(self):
        with self.assertRaises(CalcError) as ctx:
            dose_to_rate("0.1ug/kg/min", None, "4mg/50mL")
        self.assertIn("weight", str(ctx.exception))

    def test_weight_rejected_when_not_per_kg(self):
        """剂量率里没有 /kg 却给了体重，多半是用户没发现自己输错了单位，要拦下来。"""
        with self.assertRaises(CalcError):
            dose_to_rate("5mg/h", 70, "100mg/100mL")

    def test_mass_vs_units_cannot_mix(self):
        with self.assertRaises(CalcError) as ctx:
            dose_to_rate("5mg/h", None, "40U/40mL")
        self.assertIn("量纲", str(ctx.exception))

    def test_steps_are_human_readable(self):
        text = dose_to_rate("0.1ug/kg/min", 60, "4mg/50mL").render()
        self.assertIn("4.5", text)
        self.assertIn("80", text)  # 浓度出现在推导里
        self.assertIn("核对", text)


class TestRateToDose(unittest.TestCase):
    def test_round_trip(self):
        forward = dose_to_rate("0.1ug/kg/min", 60, "4mg/50mL")
        back = rate_to_dose(f"{forward.rate_ml_h}mL/h", 60, "4mg/50mL", as_unit="ug/kg/min")
        self.assertAlmostEqual(back.dose_per_hour, 360.0)

    def test_no_weight_unit(self):
        """内部量纲统一为 ug：5 mL/h × 1000 ug/mL = 5000 ug/h，即 5 mg/h。"""
        result = rate_to_dose("5mL/h", None, "100mg/100mL", as_unit="mg/h")
        self.assertAlmostEqual(result.dose_per_hour, 5000.0)
        self.assertEqual(result.dose_per_hour_unit, "ug/h")
        self.assertIn("5000", "\n".join(result.steps))

    def test_weight_needed_for_per_kg_target(self):
        with self.assertRaises(CalcError):
            rate_to_dose("5mL/h", None, "4mg/50mL", as_unit="ug/kg/min")


if __name__ == "__main__":
    unittest.main(verbosity=2)
