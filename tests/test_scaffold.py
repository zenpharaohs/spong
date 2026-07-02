"""Phase 0: the package imports and the founding documents exist."""

import pathlib

import spong


def test_version():
    assert spong.__version__


def test_modules_import():
    import spong.atlas
    import spong.charts
    import spong.gauss
    import spong.model
    import spong.portrait
    import spong.render
    import spong.sturm  # noqa: F401


def test_founding_documents_present():
    root = pathlib.Path(__file__).resolve().parents[1]
    assert (root / "SPONG_FOUNDING.md").is_file()
    assert (root / "docs" / "theorems.md").is_file()
