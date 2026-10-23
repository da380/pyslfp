"""
Test suite for the ICE-NG surface-file exporter.
"""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

surfacefile = pytest.importorskip(
    "planetmodel.surfacefile", reason="needs planetmodel >= 1.3.1"
)
pytest.importorskip("netCDF4", reason="needs planetmodel's netcdf extra")

from pyslfp.data import DATADIR  # noqa: E402
from pyslfp.ice import export_ice_history, standard_dates  # noqa: E402
from pyslfp.viscoelastic.evolution import KYR  # noqa: E402


def _fake_slice(path, *, ice, topo, lat, lon, names, attrs):
    """A tiny file shaped like an ICE-NG time slice."""
    ice_name, topo_name, lon_name = names
    ds = xr.Dataset(
        {
            ice_name: (["lat", lon_name], np.asarray(ice, dtype=np.float32)),
            topo_name: (["lat", lon_name], np.asarray(topo, dtype=np.float32)),
        },
        coords={
            "lat": np.asarray(lat, dtype=float),
            lon_name: np.asarray(lon, dtype=float),
        },
        attrs=attrs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


@patch("pyslfp.ice.export.ensure_data")
def test_exports_fake_ice7g_slices(mock_ensure, tmp_path, monkeypatch):
    """Two fake ICE-7G slices export to one history: the grid recognised
    as the shipped equiangular family, latitudes flipped, values and the
    linear interpolation exact, and the provenance carried."""
    monkeypatch.setattr("pyslfp.ice.ice_ng.DATADIR", tmp_path)
    lat = [-60.0, 0.0, 60.0]  # increasing: the exporter must flip
    lon = [0.0, 120.0, 240.0]
    base = np.arange(9.0).reshape(3, 3)
    for date, offset in ((0, 0.0), (1, 100.0)):
        _fake_slice(
            tmp_path / "ice7g" / f"I7G_NA.VM7_1deg.{date}.nc",
            ice=base + offset,
            topo=-base - offset,
            lat=lat,
            lon=lon,
            names=("stgit", "Topo", "lon"),
            attrs={"Model": "fake ICE-7G", "PAPER1": "A paper."},
        )

    out = export_ice_history(tmp_path / "history.nc", dates=[0.0, 1.0])
    history = surfacefile.read_surface(out)
    file = history.file
    assert file.grid.kind == "equiangular" and file.grid.weights is not None
    assert np.array_equal(file.time, [-1.0, 0.0])
    assert file.scales.time == KYR and file.scales.length == 1.0
    assert file.citation == ("A paper.",)
    assert file.meta["dataset"] == "fake ICE-7G"
    assert file.meta["ice_model"] == "ICE7G"
    assert "pyslfp" in file.source and file.created
    # flipped to latitude decreasing: the last file row comes first
    assert np.array_equal(history.values("ice_thickness", date=0.0), base[::-1])
    assert np.array_equal(
        history.values("topography", date=-1.0), (-base - 100.0)[::-1]
    )
    assert np.allclose(history.values("ice_thickness", date=-0.5), base[::-1] + 50.0)
    assert history.support("ice_thickness") == "node"


@patch("pyslfp.ice.export.ensure_data")
def test_exports_fake_ice5g_with_wrapped_longitudes(mock_ensure, tmp_path, monkeypatch):
    """ICE-5G's variable names, latitudes already decreasing, and
    longitudes west of Greenwich wrapped into one increasing run."""
    monkeypatch.setattr("pyslfp.ice.ice_ng.DATADIR", tmp_path)
    lat = [60.0, 0.0, -60.0]  # decreasing: no flip
    lon = [-120.0, 0.0, 120.0]  # wraps to [240, 0, 120]: rolled by one
    base = np.arange(9.0).reshape(3, 3)
    for date in ("00.0", "01.0"):
        _fake_slice(
            tmp_path / "ice5g" / f"ice5g_v1.2_{date}k_1deg.nc",
            ice=base,
            topo=-base,
            lat=lat,
            lon=lon,
            names=("sftgit", "orog", "long"),
            attrs={},
        )

    out = export_ice_history(tmp_path / "history.nc", version="ICE5G", dates=[0.0, 1.0])
    history = surfacefile.read_surface(out)
    # columns rolled so the longitudes start at 0: [0, 120, 240]
    want = base[:, [1, 2, 0]]
    assert np.array_equal(history.values("ice_thickness", date=0.0), want)
    assert np.allclose(np.rad2deg(history.file.grid.longitudes), [0.0, 120.0, 240.0])
    assert history.file.citation == ()


@patch("pyslfp.ice.export.ensure_data")
def test_refuses_dates_that_are_not_stored_slices(mock_ensure, tmp_path, monkeypatch):
    monkeypatch.setattr("pyslfp.ice.ice_ng.DATADIR", tmp_path)
    with pytest.raises(ValueError, match="stored slices"):
        export_ice_history(tmp_path / "history.nc", dates=[0.25])


@pytest.mark.skipif(
    not (DATADIR / "ice7g").exists(), reason="the ICE-7G dataset is not downloaded"
)
def test_exports_real_ice7g_slices(tmp_path):
    """Two real slices: the 1-degree grid is the shipped equiangular
    family, and the stored values are the files' own."""
    out = export_ice_history(tmp_path / "ice7g.nc", dates=[0.0, 26.0])
    history = surfacefile.read_surface(out)
    file = history.file
    assert file.grid.kind == "equiangular"
    assert (file.grid.ntheta, file.grid.nphi) == (180, 360)
    assert np.array_equal(file.time, [-26.0, 0.0])
    assert len(file.citation) >= 1
    raw = xr.open_dataset(DATADIR / "ice7g" / "I7G_NA.VM7_1deg.0.nc")
    assert np.array_equal(
        history.values("ice_thickness", date=0.0),
        np.asarray(raw["stgit"].values, dtype=float)[::-1],
    )


def test_standard_dates_cover_the_models():
    assert standard_dates("ICE7G").size == 48
    assert standard_dates("ICE5G").size == 39
    assert np.all(np.diff(standard_dates("ICE6G")) > 0)
