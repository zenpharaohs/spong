"""Bit parity: the potential-rate Python loops vs the C segment entry point.

Replays every entry of tests/corpus/potential_rate.json through the Python
oracle (`charts._potential_rate_*_python`) and through the production
dispatch wrapper with the native engine forced, and demands that both
reproduce the recorded answer to the last bit — vertices, endpoint, term,
captured target, and every counter.  This is the migration document's
definition of done for spong_potential_rate_segment; the corpus is recorded
by scripts/potential_corpus.py during ordinary zoo portraits.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import potential_corpus as pc                                # noqa: E402

CORPUS = pc.path_for()

if CORPUS.exists():
    ENTRIES = json.loads(CORPUS.read_text())
    IDS = [f"{e['case']}-{e['kind']}-{e['index']}" for e in ENTRIES]
else:
    ENTRIES, IDS = [], []


_MISMATCH = pc.platform_mismatch(CORPUS) if CORPUS.exists() else None
pytestmark = [
    pytest.mark.skipif(
        not CORPUS.exists(),
        reason="no potential-rate corpus — run scripts/potential_corpus.py"
               " record"),
    pytest.mark.skipif(_MISMATCH is not None, reason=_MISMATCH or ""),
]


@pytest.fixture(scope="module")
def native_ready():
    if not pc.native_present():
        pytest.skip("native potential_rate_segment not built")
    return True


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_oracle_matches_recording(entry):
    m, _e, _z = pc.context(entry["case"])
    got = pc.output_of(*pc.run_oracle(entry["kind"], m, entry["input"]))
    assert pc.compare(entry["output"], got) == []


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_native_matches_recording(entry, native_ready):
    m, _e, _z = pc.context(entry["case"])
    got = pc.output_of(*pc.run_native(entry["kind"], m, entry["input"]))
    assert pc.compare(entry["output"], got) == []
