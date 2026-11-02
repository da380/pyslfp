"""Builds `pyslfp/data/landlocked.npz`, the present-day landlocked depressions.

The ETOPO 2022 bedrock and ice thickness at two arc-minutes, as pyslfp ships
them, are labelled by `pyslfp.core.water_bodies.label_water_bodies`: water is the
ice-free bedrock below sea level (under grounded ice flotation decides), the
Bosporus and the Dardanelles are opened, and bodies within 15 km of the open
ocean are joined to it, since near a coast the raster cannot tell a narrow
tidal channel from a dyke. What is left is landlocked: the cells of every
depression, with the larger true depressions named. Run it with

    poetry run python tools/build_landlocked.py

after `pyslfp.data.ensure_data("ETOPO")`.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from pyslfp.data import ensure_data
from pyslfp.core.water_bodies import STATED_CHANNELS, label_water_bodies

CONNECT_WITHIN_KM = 15.0

# A point inside each named depression (latitude, longitude, degrees).
NAMED = {
    "Caspian Sea": (42.0, 50.5),
    "Dead Sea": (31.5, 35.5),
    "Qattara Depression": (29.6, 27.2),
    "Lake Eyre": (-28.6, 137.3),
    "Danakil Depression": (13.9, 40.55),
    "Turpan Depression": (42.73, 89.3),
    "Salton Trough": (33.14, -115.71),
    "Chott Melrhir": (34.2, 6.44),
    "Fayum Depression": (29.45, 30.71),
}

OUT = Path(__file__).resolve().parents[1] / "pyslfp" / "data" / "landlocked.npz"


def main() -> None:
    folder = ensure_data("ETOPO")
    with xr.open_dataset(folder / "ETOPO_2022_bedrock_2.nc") as ds:
        elevation = ds["elevation"].values
        latitudes = ds["lat"].values
        longitudes = ds["lon"].values
    with xr.open_dataset(folder / "ETOPO_2022_ice_thickness_2.nc") as ds:
        ice = ds[list(ds.data_vars)[0]].values

    bodies = label_water_bodies(
        elevation,
        latitudes,
        longitudes,
        level=0.0,
        dry=ice > 0.0,
        channels=STATED_CHANNELS,
        connect_within=CONNECT_WITHIN_KM,
    )
    rows, columns = np.nonzero(bodies.landlocked)
    labels = bodies.labels[rows, columns]

    names = [""]
    body = np.zeros(rows.size, dtype=np.int16)
    for name, (lat, lon) in NAMED.items():
        i = int(np.argmin(np.abs(latitudes - lat)))
        j = int(np.argmin(np.abs(np.mod(longitudes, 360.0) - np.mod(lon, 360.0))))
        if not bodies.landlocked[i, j]:
            raise RuntimeError(f"the point given for {name} is not landlocked")
        names.append(name)
        body[labels == bodies.labels[i, j]] = len(names) - 1

    step = float(latitudes[1] - latitudes[0])
    np.savez_compressed(
        OUT,
        rows=rows.astype(np.int16),
        columns=columns.astype(np.int16),
        body=body,
        names=np.array(names),
        lat0=float(latitudes[0]),
        lon0=float(longitudes[0]),
        step=step,
        n_rows=elevation.shape[0],
        n_columns=elevation.shape[1],
        connect_within_km=CONNECT_WITHIN_KM,
        channels=np.array([channel.name for channel in STATED_CHANNELS]),
        source=np.array("ETOPO 2022 bedrock and ice thickness, 2 arc-minutes"),
    )
    area = (
        np.cos(np.radians(latitudes[rows])) * np.radians(step) ** 2 * 6371.0**2
    ).sum()
    print(f"wrote {OUT}: {rows.size} cells, {area:,.0f} km2, {len(names) - 1} named")


if __name__ == "__main__":
    main()
