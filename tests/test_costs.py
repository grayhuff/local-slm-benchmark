import pytest

from slmbench.costs import EnergyConfig, PricingTable, price_request


@pytest.fixture
def table():
    return PricingTable.from_records(
        [
            {"provider": "openai", "model": "gpt-4o", "input_per_mtok": 2.5, "output_per_mtok": 10},
            {"provider": "openai", "model": "gpt-4o-mini", "input_per_mtok": 0.15, "output_per_mtok": 0.6},
            {"provider": "openai", "model": "gpt-5", "input_per_mtok": 1.25, "output_per_mtok": 10},
        ]
    )


def test_exact_cost(table):
    # 1,000 in + 500 out on gpt-4o-mini = 0.00015 + 0.0003
    assert table.request_cost("openai", "gpt-4o-mini", 1000, 500) == pytest.approx(0.00045)


def test_dated_snapshot_inherits_base_price(table):
    assert table.lookup("openai", "gpt-4o-mini-2024-07-18").model == "gpt-4o-mini"
    assert table.lookup("openai", "gpt-4o-2024-08-06").model == "gpt-4o"


def test_different_model_does_not_inherit_price(table):
    assert table.lookup("openai", "gpt-5.3") is None
    assert table.lookup("openai", "gpt-4o-audio") is None
    assert table.request_cost("anthropic", "gpt-4o", 10, 10) is None


def test_local_requests_pay_only_electricity(table):
    energy = EnergyConfig(watts=60, usd_per_kwh=0.20)
    costs = price_request(
        provider="ollama",
        model="llama3.2:3b",
        local=True,
        input_tokens=100,
        output_tokens=100,
        latency_s=3600,
        pricing=table,
        energy=energy,
    )
    assert costs["api_cost_usd"] == 0.0
    assert costs["energy_wh"] == pytest.approx(60)
    assert costs["energy_cost_usd"] == pytest.approx(0.012)


def test_energy_can_be_disabled():
    assert EnergyConfig(enabled=False).energy(100) == (0.0, 0.0)


def test_shipped_pricing_file_loads():
    table = PricingTable.from_file()
    assert table.lookup("openai", "gpt-4o-mini") is not None
    assert all(p.input_per_mtok >= 0 and p.output_per_mtok >= 0 for p in table.prices)


def test_round_trip_save(tmp_path, table):
    path = tmp_path / "pricing.json"
    table.save(path)
    assert PricingTable.from_file(path).to_records() == table.to_records()
