"""The CLI must use the same defaults as the dashboard, or the two report different numbers."""

from slmbench import cli, storage
from slmbench.schemas import DEFAULT_SYSTEM_PROMPT


def config_for(*argv):
    return cli.build_config(cli.build_parser().parse_args(["run", *argv]))


def test_defaults_match_dashboard(monkeypatch):
    monkeypatch.setenv("SLMBENCH_DEVICE_WATTS", "35")
    monkeypatch.delenv("SLMBENCH_USD_PER_KWH", raising=False)
    config = config_for("--demo")
    assert config.params.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert config.params.temperature == 0.0
    assert config.energy.watts == 35
    assert config.energy.usd_per_kwh == 0.17


def test_flags_override_defaults(monkeypatch):
    monkeypatch.setenv("SLMBENCH_DEVICE_WATTS", "35")
    config = config_for("--demo", "--system", "", "--watts", "100", "--kwh-price", "0.3")
    assert config.params.system_prompt is None
    assert (config.energy.watts, config.energy.usd_per_kwh) == (100, 0.3)


def test_demo_run_prints_summary_and_saves_hardware(capsys):
    assert cli.main(["run", "--demo"]) == 0
    out = capsys.readouterr().out
    assert "demo-api-frontier" in out and "accuracy" in out
    (entry,) = storage.list_runs()
    assert storage.load_run(entry.path).config["system"]["os"]  # same hardware snapshot the dashboard saves


def test_run_without_models_fails(capsys):
    assert cli.main(["run", "--no-save"]) == 2
    assert "--model" in capsys.readouterr().err
