"""
Export of the ICE-NG deglaciation histories as planetmodel surface files.

The exporter reads a model's stored time slices from the pyslfp data
directory and writes one `planetmodel.surface/1` file holding the
`ice_thickness` and `topography` histories on the dataset's own grid, in
metres, at the stored dates alone — nothing is interpolated or
regridded.  Time is stored in kiloyears relative to the present,
negative in the past, so that the axis increases toward the present;
the dates in kyr before present are the negatives of the stored times.
The file carries the dataset's own citation attributes, and consumers
(sl3dpp, AdGIA) read it through planetmodel's surface-history readers.

Writing needs planetmodel >= 1.3.1 (the `netcdf` extra installed); the
data is downloaded first where it is missing, as for `IceNG`.
"""

from __future__ import annotations

import os
from importlib.metadata import version as _package_version
from pathlib import Path

import numpy as np
import xarray as xr

from pyslfp.data import ensure_data

from .ice_ng import IceNGVersion, slice_file, standard_dates

__all__ = ["export_ice_history"]

# The ice-thickness, topography and longitude names in a version's files.
_VARIABLES = {
    "ICE5G": ("sftgit", "orog", "long"),
    "ICE6G": ("stgit", "Topo", "lon"),
    "ICE7G": ("stgit", "Topo", "lon"),
}


def _surfacefile():
    """planetmodel's surface-file module, or a helpful refusal."""
    try:
        from planetmodel import surfacefile
    except ImportError as exc:
        raise ImportError(
            "exporting an ice history needs planetmodel >= 1.3.1 with its "
            "netcdf extra: pip install 'planetmodel[netcdf]>=1.3.1'"
        ) from exc
    return surfacefile


def _oriented(lat: np.ndarray, lon: np.ndarray) -> tuple[bool, int]:
    """How the file's axes map onto a grid's: whether latitude must be
    flipped to decrease, and the roll that makes the wrapped longitudes
    increase."""
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
        raise ValueError("the file's lat and lon are not one-dimensional axes")
    flip = bool(lat[0] < lat[-1])
    wrapped = np.mod(lon, 360.0)
    roll = int(np.argmin(wrapped))
    rolled = np.roll(wrapped, -roll)
    if not np.all(np.diff(rolled) > 0.0):
        raise ValueError("the file's longitudes do not wrap to one increasing run")
    return flip, roll


def _grid(lat: np.ndarray, lon: np.ndarray, flip: bool, roll: int):
    """The dataset's grid: the matching shipped equiangular family where
    the nodes are its, so that the grid carries weights and a band, and a
    custom grid otherwise."""
    from planetmodel import AngularGrid, equiangular

    ordered = lat[::-1] if flip else lat
    theta = np.deg2rad(90.0 - ordered)
    phi = np.deg2rad(np.roll(np.mod(lon, 360.0), -roll))
    shipped = equiangular(lat.size, lon.size, phi0=float(phi[0]))
    if np.allclose(shipped.colatitudes, theta, rtol=0.0, atol=1e-12) and np.allclose(
        shipped.longitudes, phi, rtol=0.0, atol=1e-12
    ):
        return shipped
    return AngularGrid(theta, phi)


def export_ice_history(
    path: str | os.PathLike[str],
    /,
    *,
    version: IceNGVersion = "ICE7G",
    dates: np.ndarray | None = None,
) -> Path:
    """
    Writes an ICE-NG model's stored slices as a planetmodel surface file.

    Args:
        path (str | os.PathLike): The file to write.
        version (IceNGVersion): The ice model version. Defaults to "ICE7G".
        dates (np.ndarray | None): The dates to export, in kyr before
            present, each one of the model's stored slices
            (`standard_dates(version)`); all of them by default.

    Returns:
        Path: The file written.

    Raises:
        ImportError: If planetmodel's surface file is unavailable.
        ValueError: If a date is not a stored slice, or the data files
            disagree about their grid.
    """
    surfacefile = _surfacefile()
    from planetmodel.units import LENGTH, Scales

    from pyslfp.viscoelastic.evolution import KYR

    ensure_data(version)
    stored = standard_dates(version)
    wanted = stored if dates is None else np.unique(np.asarray(dates, dtype=float))
    if wanted.size == 0 or not np.isin(wanted, stored).all():
        missing = wanted[~np.isin(wanted, stored)]
        raise ValueError(
            f"dates must be stored slices of {version}; "
            f"{missing.tolist() if wanted.size else 'none'} are not"
        )
    # oldest first, so the stored time (minus the date) increases
    wanted = wanted[::-1]

    ice_name, topo_name, lon_name = _VARIABLES[version]
    thickness, topography = [], []
    lat = lon = None
    attributes: dict[str, str] = {}
    for date in wanted:
        with xr.open_dataset(slice_file(version, float(date))) as data:
            if lat is None:
                lat = np.asarray(data["lat"].values, dtype=float)
                lon = np.asarray(data[lon_name].values, dtype=float)
                flip, roll = _oriented(lat, lon)
                attributes = {str(k): str(v) for k, v in data.attrs.items()}
            else:
                if not (
                    np.array_equal(data["lat"].values, lat)
                    and np.array_equal(data[lon_name].values, lon)
                ):
                    raise ValueError(
                        f"the {version} files do not share one grid: "
                        f"{slice_file(version, float(date))} differs"
                    )
            for held, name in ((thickness, ice_name), (topography, topo_name)):
                values = np.asarray(data[name].values, dtype=float)
                if flip:
                    values = values[::-1]
                held.append(np.roll(values, -roll, axis=1))

    grid = _grid(lat, lon, flip, roll)
    citation = tuple(
        attributes[key] for key in sorted(attributes) if key.upper().startswith("PAPER")
    )
    meta = {
        "ice_model": version,
        "time": "kiloyears relative to the present, negative in the past",
    }
    if "Model" in attributes:
        meta["dataset"] = attributes["Model"]
    fields = {
        "ice_thickness": surfacefile.SurfaceVariable(
            np.stack(thickness),
            support="node",
            time_interpolation="linear",
            dimensions=LENGTH,
            meaning=f"{version} ice thickness",
        ),
        "topography": surfacefile.SurfaceVariable(
            np.stack(topography),
            support="node",
            time_interpolation="linear",
            dimensions=LENGTH,
            meaning=f"{version} topography (bedrock where ice-covered)",
        ),
    }
    return surfacefile.write_surface(
        Path(path),
        grid,
        fields=fields,
        source=f"pyslfp {_package_version('pyslfp')} ICE-NG export",
        time=0.0 - wanted,  # 0.0 - 0.0 is +0.0; -0.0 would print oddly
        scales=Scales(time=KYR),
        title=f"{version} deglaciation history",
        meta=meta,
        citation=citation,
    )
