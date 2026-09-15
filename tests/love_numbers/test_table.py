"""The LoveNumbers table: constructors, truncation, the file with its
named columns and the legacy layout, and the Green's functions in any
units."""

import numpy as np
import pytest
from planetmodel import PREM
from planetmodel.units import Dimensions, Scales

from pyslfp.love_numbers import (
    LEGACY_COLUMNS,
    NAMES,
    LoveNumbers,
    love_numbers,
    read_love_numbers,
)


@pytest.fixture(scope="module")
def love():
    return love_numbers(PREM(ocean=False), 8)


def test_from_model_is_love_numbers(love):
    other = LoveNumbers.from_model(PREM(ocean=False), 8)
    for name in NAMES:
        assert np.array_equal(getattr(other, name), getattr(love, name))
    assert other.radius == love.radius and other.scales == love.scales


def test_file_round_trip_with_named_columns(tmp_path, love):
    path = tmp_path / "love.dat"
    love.write(path)
    assert "# columns: l " + " ".join(NAMES) in path.read_text()
    data = np.loadtxt(path)
    assert data.shape == (9, 1 + len(NAMES))
    assert np.array_equal(data[:, 0], np.arange(9))
    back = read_love_numbers(path)
    for j, name in enumerate(NAMES):
        assert np.allclose(data[:, j + 1], getattr(love, name), rtol=1e-12)
        assert np.allclose(getattr(back, name), getattr(love, name), rtol=1e-12)
    for attr in ("radius", "surface_gravity", "G"):
        assert np.isclose(getattr(back, attr), getattr(love, attr), rtol=1e-14)
    assert back.scales == Scales.SI and back.lmax == 8
    assert LoveNumbers.from_file(path, lmax=2).lmax == 2


def test_written_in_other_units_reads_the_same(tmp_path, love):
    nd = love.converted(PREM().nondimensionalised().scales)
    love.write(tmp_path / "si.dat")
    nd.write(tmp_path / "nd.dat")
    assert np.allclose(
        np.loadtxt(tmp_path / "nd.dat"), np.loadtxt(tmp_path / "si.dat"), rtol=1e-12
    )


def test_columns_may_be_reordered_or_missing(tmp_path, love):
    path = tmp_path / "partial.dat"
    np.savetxt(
        path,
        np.column_stack([love.degree, love.k_t, love.h_u]),
        header="columns: l k_t h_u\nanother comment line",
    )
    back = LoveNumbers.from_file(path)
    assert np.allclose(back.k_t, love.k_t) and np.allclose(back.h_u, love.h_u)
    assert np.all(np.isnan(back.k_u)) and np.isnan(back.radius)
    bad = tmp_path / "bad.dat"
    np.savetxt(bad, np.column_stack([love.degree, love.k_t]), header="columns: l")
    with pytest.raises(ValueError, match="names 1 columns"):
        LoveNumbers.from_file(bad)
    np.savetxt(bad, np.column_stack([love.degree, love.k_t]), header="columns: a b")
    with pytest.raises(ValueError, match="'l'"):
        LoveNumbers.from_file(bad)


def test_legacy_seven_column_file(tmp_path, love):
    path = tmp_path / "legacy.dat"
    cols = np.column_stack(
        [love.degree] + [getattr(love, n) for n in LEGACY_COLUMNS[1:]]
    )
    np.savetxt(path, cols)
    back = LoveNumbers.from_file(path)
    for name in LEGACY_COLUMNS[1:]:
        assert np.allclose(getattr(back, name), getattr(love, name), rtol=1e-12)
    assert np.all(np.isnan(back.l_u)) and np.isnan(back.radius)
    given = LoveNumbers.from_file(path, radius=1.0, surface_gravity=2.0, G=3.0)
    assert (given.radius, given.surface_gravity, given.G) == (1.0, 2.0, 3.0)
    bad = tmp_path / "bad.dat"
    np.savetxt(bad, cols[:, :5])
    with pytest.raises(ValueError, match="7 columns"):
        LoveNumbers.from_file(bad)


def test_write_refusals(tmp_path, love):
    partial = love.truncated(3)
    partial = LoveNumbers(
        partial.degree[1:],
        **{n: getattr(partial, n)[1:] for n in NAMES},
        radius=1.0,
        surface_gravity=1.0,
        G=1.0,
    )
    with pytest.raises(ValueError, match="from 0"):
        partial.write(tmp_path / "x.dat")


def test_truncated(love):
    t = love.truncated(3)
    assert t.lmax == 3 and np.array_equal(t.degree, np.arange(4))
    for name in NAMES:
        assert np.array_equal(getattr(t, name), getattr(love, name)[:4])
    assert t.radius == love.radius and t.scales == love.scales
    assert np.array_equal(t.h, love.h[:4])
    with pytest.raises(ValueError, match="exceeds"):
        love.truncated(9)


def test_greens_functions_in_any_units(love):
    nd = love.converted(PREM().nondimensionalised().scales)
    per_mass_length = Dimensions(mass=-1, length=1)
    per_mass_potential = Dimensions(mass=-1, length=2, time=-2)
    for angle in (0.0, 0.1, 1.0):
        a = love.displacement_greens_function(angle)
        b = nd.displacement_greens_function(angle)
        assert np.isclose(a, b * nd.scales.factor(per_mass_length), rtol=1e-10)
        a = love.potential_greens_function(angle)
        b = nd.potential_greens_function(angle)
        assert np.isclose(a, b * nd.scales.factor(per_mass_potential), rtol=1e-10)
    assert love.displacement_greens_function(0.0, lmax=4) != a


def test_default_reads_from_the_data_folder(tmp_path, love, monkeypatch):
    """`default` reads PREM_4096.dat from the folder `ensure_data` returns,
    passing `refresh` through."""
    from pyslfp.love_numbers import table

    folder = tmp_path / "pyslfp_love_numbers"
    folder.mkdir()
    love.write(folder / "PREM_4096.dat")
    calls = []

    def fake_ensure_data(key, *, refresh=False):
        calls.append((key, refresh))
        return folder

    monkeypatch.setattr("pyslfp.data.ensure_data", fake_ensure_data)
    got = table.LoveNumbers.default(lmax=4, refresh=True)
    assert calls == [("LOVE_NUMBERS", True)]
    assert got.G == love.G and got.lmax == 4
    assert np.allclose(got.l_u, love.l_u[:5])
