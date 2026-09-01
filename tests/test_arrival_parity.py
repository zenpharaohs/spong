"""Bit parity: the centered raw arrival Python loop vs the C entry point.

Replays every entry of tests/corpus/centered_arrival.json through the
Python oracle (`charts._centered_raw_arrival_python`) and through the
production dispatch wrapper with the native engine forced, and demands that
both reproduce the recorded answer to the last bit — vertices, term and
every counter.  This is the definition of done for spong_centered_arrival;
the corpus is recorded by scripts/arrival_corpus.py during ordinary zoo
portraits.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import arrival_corpus as ac                                  # noqa: E402
import potential_corpus as pc                                # noqa: E402

CORPUS = ac.path_for()

if CORPUS.exists():
    ENTRIES = json.loads(CORPUS.read_text())
    IDS = [f"{e['case']}-arrival-{e['index']}" for e in ENTRIES]
else:
    ENTRIES, IDS = [], []


pytestmark = pytest.mark.skipif(
    not CORPUS.exists(),
    reason="no centered-arrival corpus — run scripts/arrival_corpus.py record")


@pytest.fixture(scope="module")
def native_ready():
    if not ac.native_present():
        pytest.skip("native centered_arrival not built")
    return True


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_oracle_matches_recording(entry):
    m, e, _z = pc.context(entry["case"])
    got = pc.output_of(*ac.run_oracle(m, e, entry["input"]))
    assert pc.compare(entry["output"], got) == []


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_native_matches_recording(entry, native_ready):
    m, e, _z = pc.context(entry["case"])
    got = pc.output_of(*ac.run_native(m, e, entry["input"]))
    assert pc.compare(entry["output"], got) == []
