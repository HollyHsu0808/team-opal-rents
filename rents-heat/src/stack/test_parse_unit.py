"""Run: python -m unittest rents-heat/src/stack/test_parse_unit.py   (or python test_parse_unit.py in this folder)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_unit import parse_unit  # noqa: E402


class TestParseUnit(unittest.TestCase):
    def check(self, addr, unit, floor, suffix, conf, max_floor=None):
        p = parse_unit(addr, max_floor)
        self.assertEqual((p.unit_raw, p.floor_guess, p.suffix, p.confidence), (unit, floor, suffix, conf), msg=f"{addr}: {p}")

    def test_unit_word_full_address(self):
        self.check("UNIT 506, 20 DUNKERLEY PLACE WATERLOO 2017", "506", 5, "06", "high")

    def test_slash(self):
        self.check("506/20 Dunkerley Pl", "506", 5, "06", "high")

    def test_ground(self):
        self.check("Unit G03", "G03", 0, "03", "high")
        self.check("G03/5 Smith St", "G03", 0, "03", "high")

    def test_penthouse(self):
        p = parse_unit("PH1")
        self.assertEqual((p.unit_raw, p.floor_guess, p.confidence, p.reason), ("PH1", None, "low", "penthouse"))

    def test_letter_suffix(self):
        self.check("1203A", "1203A", 12, "03A", "medium")
        self.check("Apt 1203A, 1 Bligh St Sydney", "1203A", 12, "03A", "medium")

    def test_lot(self):
        p = parse_unit("LOT 5")
        self.assertEqual((p.unit_raw, p.floor_guess, p.suffix, p.confidence), ("5", None, None, "low"))

    def test_missing(self):
        for a in (None, "", "   ", float("nan")):
            self.assertEqual(parse_unit(a).confidence, "none")
        self.assertEqual(parse_unit("20 Dunkerley Pl").reason, "no unit")

    def test_sequential_short(self):
        self.check("7/20 Smith St", "7", None, None, "low")

    def test_max_floor_rejects(self):
        self.check("1506/20 Smith St", "1506", None, None, "low", max_floor=8)
        self.check("1506/20 Smith St", "1506", 15, "06", "high", max_floor=40)

    def test_four_digit(self):
        self.check("Unit 2104, 1 Bligh St", "2104", 21, "04", "high")

    def test_lowercase_and_spaces(self):
        self.check("unit 506 , 20 x st", "506", 5, "06", "high")
        self.check("506 / 20 x st", "506", 5, "06", "high")

    def test_suffix_00(self):
        self.assertEqual(parse_unit("500/2 x st").confidence, "low")


if __name__ == "__main__":
    unittest.main()
