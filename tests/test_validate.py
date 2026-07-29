import json

from spong import validate, zoo


def test_install_validation_smoke_writes_reproducible_report(tmp_path):
    output = tmp_path / "validation.json"
    status = validate.main([
        "--mundane", "4",
        "--targeted-per-exponent", "4",
        "--exponents", "8",
        "--no-zoo",
        "--jobs", "1",
        "--seed", "12345",
        "--output", str(output),
    ])
    assert status == 0
    payload = json.loads(output.read_text())
    assert payload["format"] == "spong-native-validation-v1"
    assert payload["summary"]["ok"]
    assert payload["summary"]["cases"] == 8
    assert payload["summary"]["failed"] == 0
    assert {x.get("family") for x in payload["results"]
            if x["kind"] == "targeted"} == {
                "close", "far", "repeated-real", "repeated-complex"}


def test_validation_specification_includes_complete_zoo():
    specs = validate._specifications(7, 0, 0, [8], True)
    assert {x["name"] for x in specs} == set(zoo.names())


def test_validation_returns_failure_status(monkeypatch):
    monkeypatch.setattr(validate, "_run", lambda spec: {
        **spec, "ok": False, "exception": "InjectedFailure",
        "message": "qualification failure", "elapsed_sec": 0.0,
    })
    assert validate.main([
        "--mundane", "1", "--targeted-per-exponent", "0",
        "--no-zoo", "--jobs", "1"]) == 1
