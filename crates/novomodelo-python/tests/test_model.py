"""Integration tests for cobre.model Python wrapper classes.

These tests verify that the PyO3 wrapper classes for the Cobre data model
types are correctly exposed with the expected properties and behaviour.

Run with (from the repo root):
    pytest crates/cobre-python/tests/
"""

import pytest


def test_system_entity_counts() -> None:
    """Load the 1dtoy case and verify entity count properties on System."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case("examples/1dtoy")
    assert isinstance(system.n_buses, int)
    assert isinstance(system.n_hydros, int)
    assert isinstance(system.n_thermals, int)
    assert isinstance(system.n_lines, int)
    assert isinstance(system.n_stages, int)
    assert system.n_buses > 0


def test_bus_properties() -> None:
    """Load the 1dtoy case and verify Bus properties."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case("examples/1dtoy")
    buses = system.buses
    assert len(buses) > 0
    bus = buses[0]
    assert isinstance(bus.id, int)
    assert isinstance(bus.name, str)
    assert isinstance(bus.excess_cost, float)


def test_hydro_properties() -> None:
    """Load the 1dtoy case and verify Hydro properties."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case("examples/1dtoy")
    hydros = system.hydros
    assert len(hydros) > 0
    hydro = hydros[0]
    assert isinstance(hydro.id, int)
    assert isinstance(hydro.name, str)


def test_thermal_properties() -> None:
    """Load the 1dtoy case and verify Thermal properties."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case("examples/1dtoy")
    thermals = system.thermals
    assert len(thermals) > 0
    thermal = thermals[0]
    assert isinstance(thermal.id, int)
    assert isinstance(thermal.name, str)
    assert isinstance(thermal.bus_id, int)


def test_line_properties() -> None:
    """1dtoy has no lines — verify the accessor returns an empty list."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case("examples/1dtoy")
    lines = system.lines
    assert isinstance(lines, list)
    assert len(lines) == 0


def test_bus_repr() -> None:
    """Verify repr(bus) contains 'Bus(id=' and the bus name."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case("examples/1dtoy")
    bus = system.buses[0]
    r = repr(bus)
    assert "Bus(id=" in r
    assert bus.name in r


def test_system_not_constructable() -> None:
    """cobre.model.System() must raise TypeError (no Python constructor)."""
    import cobre.model  # noqa: PLC0415

    with pytest.raises(TypeError):
        cobre.model.System()  # type: ignore[call-arg]


def test_energy_contract_properties() -> None:
    """Load the D41 case and verify EnergyContract exposes its full field set."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case("examples/deterministic/d41-energy-contracts")
    contracts = system.contracts
    assert len(contracts) > 0
    contract = contracts[0]
    assert isinstance(contract.id, int)
    assert isinstance(contract.name, str)
    assert isinstance(contract.operational_start_date, str)
    assert isinstance(contract.bus_id, int)
    assert contract.contract_type in ("import", "export")
    assert contract.entry_stage_id is None or isinstance(contract.entry_stage_id, int)
    assert contract.exit_stage_id is None or isinstance(contract.exit_stage_id, int)
    assert isinstance(contract.price_per_mwh, float)
    assert isinstance(contract.min_mw, float)
    assert isinstance(contract.max_mw, float)


def test_pumping_station_properties() -> None:
    """Load the pumping-transfer fixture and verify PumpingStation's full field set."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case("crates/cobre-sddp/tests/fixtures/pumping_transfer")
    stations = system.pumping_stations
    assert len(stations) > 0
    station = stations[0]
    assert isinstance(station.id, int)
    assert isinstance(station.name, str)
    assert isinstance(station.operational_start_date, str)
    assert isinstance(station.bus_id, int)
    assert isinstance(station.source_hydro_id, int)
    assert isinstance(station.destination_hydro_id, int)
    assert station.entry_stage_id is None
    assert station.exit_stage_id is None
    assert isinstance(station.consumption_mw_per_m3s, float)
    assert isinstance(station.min_flow_m3s, float)
    assert isinstance(station.max_flow_m3s, float)


def test_non_controllable_source_properties() -> None:
    """Load the D15 case and verify NonControllableSource's full field set."""
    import cobre.io  # noqa: PLC0415

    system = cobre.io.load_case("examples/deterministic/d15-non-controllable-source")
    sources = system.non_controllable_sources
    assert len(sources) > 0
    source = sources[0]
    assert isinstance(source.id, int)
    assert isinstance(source.name, str)
    assert isinstance(source.operational_start_date, str)
    assert isinstance(source.bus_id, int)
    assert source.entry_stage_id is None
    assert source.exit_stage_id is None
    assert isinstance(source.max_generation_mw, float)
    assert isinstance(source.allow_curtailment, bool)
    assert isinstance(source.curtailment_cost, float)


def test_model_classes_importable() -> None:
    """All expected classes must be present in cobre.model."""
    import cobre.model  # noqa: PLC0415

    for name in (
        "System",
        "Bus",
        "Line",
        "Thermal",
        "Hydro",
        "EnergyContract",
        "PumpingStation",
        "NonControllableSource",
    ):
        assert hasattr(cobre.model, name), f"cobre.model.{name} not found"
