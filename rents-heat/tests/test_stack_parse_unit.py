"""Re-collect the existing src/stack unit-parser tests so `pytest tests` covers them."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "stack"))
from test_parse_unit import TestParseUnit  # noqa: E402,F401
