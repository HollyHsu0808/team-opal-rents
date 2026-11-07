"""Parse a flat's unit number from an address string into (unit_raw, floor_guess, suffix, confidence, reason).

"Stack" = flats sharing the same suffix (e.g. 206, 306, 406, 506 -> suffix "06") are usually vertically aligned
with the same layout. The floor is only *guessed* from the unit number; the convention "floor + 2-digit suffix"
is common in Sydney towers but not universal (sequential 1..N numbering also exists), so confidence is reported.

Examples
  "UNIT 506, 20 DUNKERLEY PLACE WATERLOO 2017" -> ("506", 5, "06", "high", "floor+2digit")
  "506/20 Dunkerley Pl"                        -> ("506", 5, "06", "high", "floor+2digit")
  "Unit G03"                                   -> ("G03", 0, "03", "high", "ground letter")
  "PH1"                                        -> ("PH1", None, None, "low", "penthouse")
  "1203A"                                      -> ("1203A", 12, "03A", "medium", "floor+2digit+letter")
  "LOT 5"                                      -> ("5", None, None, "low", "lot number, not a unit")
  "20 Dunkerley Pl"                            -> (None, None, None, "none", "no unit")
"""
import re
from typing import NamedTuple, Optional


class UnitParse(NamedTuple):
    unit_raw: Optional[str]
    floor_guess: Optional[int]
    suffix: Optional[str]
    confidence: str  # high | medium | low | none
    reason: str


_UNIT_WORDS = r"(?:UNIT|APARTMENT|APT|FLAT|SUITE|STUDIO|U)"
# "UNIT 506, 20 ..." / "Apt 1203A" / "Unit G03"
_RE_WORD = re.compile(rf"^\s*{_UNIT_WORDS}\.?\s*#?\s*([A-Z]{{0,2}}\d{{1,5}}[A-Z]?|[A-Z]{{1,2}}\d{{0,3}})\b", re.I)
# "506/20 Dunkerley" ; "G03/5 x" ; "1203A/88 x"
_RE_SLASH = re.compile(r"^\s*([A-Z]{0,2}\d{1,5}[A-Z]?|[A-Z]{1,2})\s*/\s*\d", re.I)
_RE_LOT = re.compile(r"^\s*LOT\s+(\w+)", re.I)
_RE_PH = re.compile(r"^\s*(?:UNIT\s+)?(PH\s*\d*|PENTHOUSE\s*\d*)\b", re.I)
_RE_BARE = re.compile(r"^\s*([A-Z]?\d{1,5}[A-Z]?)\s*$", re.I)


def _split(token: str, max_floor: Optional[int] = None) -> UnitParse:
    t = token.upper().replace(" ", "")
    # ground / lower-ground / basement prefixes
    m = re.fullmatch(r"(G|LG|B|UG)(\d{1,3})([A-Z]?)", t)
    if m:
        pre, num, let = m.groups()
        floor = {"G": 0, "UG": 0, "LG": -1, "B": -1}[pre]
        suf = num.zfill(2)[-2:] + let
        return UnitParse(t, floor, suf, "high", "ground letter" if pre == "G" else f"{pre} prefix")
    m = re.fullmatch(r"(\d{1,5})([A-Z]?)", t)
    if m:
        num, let = m.groups()
        n = len(num)
        tag = "+letter" if let else ""
        if n <= 2:
            # 1..99: sequential numbering or ground-floor unit, floor not identifiable
            return UnitParse(t, None, None, "low", "short number (sequential?)")
        floor, suf = int(num[:-2]), num[-2:] + let
        if max_floor is not None and floor > max_floor:
            return UnitParse(t, None, None, "low", "floor guess exceeds building height (sequential?)")
        if suf[:2] == "00":
            return UnitParse(t, floor, suf, "low", "suffix 00 (unusual)")
        conf = "high" if (n in (3, 4) and not let) else "medium"
        return UnitParse(t, floor, suf, conf, "floor+2digit" + tag)
    if re.fullmatch(r"[A-Z]{1,2}", t):
        return UnitParse(t, None, None, "low", "letter-only unit")
    m = re.fullmatch(r"([A-Z])(\d{1,4})", t)  # e.g. "C506" building-wing letter + number
    if m:
        wing, num = m.groups()
        if len(num) >= 3:
            return UnitParse(t, int(num[:-2]), wing + num[-2:], "medium", "wing letter+floor+2digit")
        return UnitParse(t, None, None, "low", "wing letter + short number")
    return UnitParse(t, None, None, "low", "unrecognised unit token")


def parse_unit(address: Optional[str], max_floor: Optional[int] = None) -> UnitParse:
    """Parse the unit part of an address. max_floor (if known, e.g. OSM building:levels) rejects
    impossible floor guesses, which flags sequential numbering."""
    if address is None or (isinstance(address, float)) or not str(address).strip():
        return UnitParse(None, None, None, "none", "missing")
    s = str(address).strip()
    m = _RE_PH.match(s)
    if m:
        return UnitParse(m.group(1).upper().replace(" ", ""), None, None, "low", "penthouse")
    m = _RE_LOT.match(s)
    if m:
        return UnitParse(m.group(1).upper(), None, None, "low", "lot number, not a unit")
    m = _RE_WORD.match(s) or _RE_SLASH.match(s)
    if m:
        return _split(m.group(1), max_floor)
    m = _RE_BARE.match(s)
    if m:
        return _split(m.group(1), max_floor)
    return UnitParse(None, None, None, "none", "no unit")


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:] or ["UNIT 506, 20 DUNKERLEY PLACE WATERLOO 2017", "506/20 Dunkerley Pl", "Unit G03", "PH1",
                              "1203A", "LOT 5", "20 Dunkerley Pl"]:
        print(repr(a), "->", parse_unit(a))
