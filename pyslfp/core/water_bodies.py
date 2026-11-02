"""
Water bodies on a topography: which water below a level reaches the open
ocean, and which is landlocked.

The ocean function of an `EarthState` is decided by flotation, so every
depression below sea level counts as ocean once the grid resolves it. What
separates the ocean from a landlocked basin is connection, and that is
decided here on a regular latitude-longitude raster of high resolution: the
straits that join the seas to the ocean (Gibraltar, Bab-el-Mandeb, Hormuz,
the Danish straits) close on a model grid, even at degree 256.

`label_water_bodies` labels the connected bodies of water below a level on
any raster, with stated channels opened where the raster is too coarse to
resolve them, and with bodies close to the open ocean counted as joined to
it, since near a coast a raster cannot tell a narrow tidal channel from a
dyke. It is the general tool, for other epochs and imagined topographies.

The present-day landlocked depressions -- the Caspian, the Dead Sea,
Qattara, the beds of the Great Lakes and Baikal below sea level, and the
rest -- are built once by it from the ETOPO 2022 bedrock at two arc-minutes
(`tools/build_landlocked.py`) and shipped as `landlocked.npz`.
`landlocked_fraction` carries them onto a model grid, as the fraction of
each node's cell they cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "Channel",
    "STATED_CHANNELS",
    "WaterBodies",
    "label_water_bodies",
    "landlocked_fraction",
    "landlocked_mask",
    "landlocked_names",
]

# The radius used for distances on the raster, km.
_RADIUS_KM = 6371.0

_PRODUCT = Path(__file__).resolve().parents[1] / "data" / "landlocked.npz"


@dataclass(frozen=True)
class Channel:
    """A water connection too narrow for the raster, opened along the great
    circle between two points (latitude, longitude in degrees) on either
    side of it."""

    name: str
    start: tuple[float, float]
    end: tuple[float, float]


#: The channels a two-arc-minute raster cannot resolve and on which a large
#: sea depends: the Bosporus (about 0.7 km at its narrowest) joining the
#: Black Sea to the Sea of Marmara, and the Dardanelles (about 1.2 km)
#: joining Marmara to the Aegean.
STATED_CHANNELS = (
    Channel("Bosporus", (41.23, 29.13), (41.00, 28.98)),
    Channel("Dardanelles", (40.42, 26.68), (40.03, 26.20)),
)


@dataclass(frozen=True)
class WaterBodies:
    """The water bodies of a raster.

    Attributes:
        labels: One label per raster cell, zero for dry cells.
        ocean: The label of the open ocean, the largest body.
        landlocked: True for the cells of water not joined to the ocean.
    """

    labels: np.ndarray
    ocean: int
    landlocked: np.ndarray


def label_water_bodies(
    elevation: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    /,
    *,
    level: float = 0.0,
    dry: Optional[np.ndarray] = None,
    channels: Sequence[Channel] = (),
    connect_within: float = 0.0,
) -> WaterBodies:
    """
    Labels the connected bodies of water below a level on a global raster.

    Water is every cell whose elevation is below `level` and that `dry` does
    not exclude. Cells touching along an edge or a corner are connected, and
    so are the first and last columns, the raster being global in longitude.
    The open ocean is the largest body.

    Args:
        elevation: The elevation, shape (latitudes, longitudes), in the units
            of `level`.
        latitudes: The cell-centre latitudes in degrees, equally spaced.
        longitudes: The cell-centre longitudes in degrees, equally spaced,
            covering the globe once.
        level: The water level.
        dry: Cells that hold no water whatever their elevation, such as those
            under grounded ice, where flotation decides; none if None.
        channels: Connections to open, each set to water along its great
            circle before labelling.
        connect_within: A distance in km: bodies with a cell this close to
            the open ocean are joined to it.

    Returns:
        The labels, the ocean's label and the landlocked cells.
    """
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    water = np.asarray(elevation) < level
    if dry is not None:
        water &= ~np.asarray(dry, dtype=bool)
    for channel in channels:
        rows, columns = _channel_cells(channel, latitudes, longitudes)
        water[rows, columns] = True

    wrapped = np.hstack([water, water[:, :1]])
    labels, _ = ndimage.label(wrapped, structure=np.ones((3, 3), bool))
    for first, last in zip(labels[:, 0], labels[:, -1]):
        if first and last and first != last:
            labels[labels == last] = first
    labels = labels[:, :-1]

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    ocean = int(np.argmax(sizes))
    connected = labels == ocean
    if connect_within > 0.0:
        near = _dilate(connected, latitudes, longitudes, connect_within)
        joined = np.unique(labels[near & (labels > 0)])
        connected = np.isin(labels, joined)
    return WaterBodies(labels=labels, ocean=ocean, landlocked=water & ~connected)


def landlocked_names() -> tuple[str, ...]:
    """The named present-day landlocked depressions, in the product's order."""
    return tuple(str(name) for name in _product()["names"] if name)


def landlocked_fraction(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    /,
    *,
    names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    The fraction of each node's cell covered by present-day landlocked
    depressions below sea level.

    Each two-arc-minute cell of the product belongs to the node nearest it,
    so the fractions are those of the nodes' own cells.

    Args:
        latitudes: The grid's latitudes in degrees, one per row.
        longitudes: The grid's distinct longitudes in degrees, equally
            spaced from zero, one per column.
        names: The named depressions to include (`landlocked_names`); all
            depressions, named or not, if None.

    Returns:
        The fractions, shape (rows, columns).

    Raises:
        ValueError: If a name is not that of a landlocked depression.
    """
    product = _product()
    rows, columns, body = product["rows"], product["columns"], product["body"]
    if names is not None:
        known = list(product["names"])
        unknown = [name for name in names if name not in known or not name]
        if unknown:
            raise ValueError(
                f"{unknown} are not named landlocked depressions; the names are "
                f"{landlocked_names()}."
            )
        keep = np.isin(body, [known.index(name) for name in names])
        rows, columns = rows[keep], columns[keep]

    latitudes = np.asarray(latitudes, dtype=float)
    n_lon = len(longitudes)
    cell_lat = product["lat0"] + product["step"] * np.arange(product["n_rows"])
    cell_lon = product["lon0"] + product["step"] * np.arange(product["n_columns"])
    node_row = np.argmin(np.abs(cell_lat[:, None] - latitudes[None, :]), axis=1)
    node_column = np.rint(np.mod(cell_lon, 360.0) * n_lon / 360.0).astype(int) % n_lon
    weight = np.cos(np.radians(cell_lat))

    row_area = np.bincount(node_row, weights=weight, minlength=len(latitudes))
    column_count = np.bincount(node_column, minlength=n_lon).astype(float)
    total = row_area[:, None] * column_count[None, :]
    covered = np.bincount(
        node_row[rows] * n_lon + node_column[columns],
        weights=weight[rows],
        minlength=len(latitudes) * n_lon,
    ).reshape(len(latitudes), n_lon)
    return np.divide(covered, total, out=np.zeros_like(covered), where=total > 0)


def landlocked_mask(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    /,
    *,
    extend: bool,
    names: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    True at the nodes of a grid whose cells the present-day landlocked
    depressions cover by more than `threshold`.

    Args:
        latitudes: The grid's latitudes in degrees, one per row.
        longitudes: The grid's longitudes in degrees, one per column.
        extend: Whether the last column repeats the first, as on an extended
            grid.
        names: The named depressions to include; all if None.
        threshold: The covered fraction above which a node is landlocked.
    """
    longitudes = np.asarray(longitudes, dtype=float)
    distinct = longitudes[:-1] if extend else longitudes
    fraction = landlocked_fraction(latitudes, distinct, names=names)
    if extend:
        fraction = np.hstack([fraction, fraction[:, :1]])
    return fraction > threshold


# ================================================================ #
#                         Private helpers                          #
# ================================================================ #


@lru_cache(maxsize=1)
def _product() -> dict:
    """The present-day landlocked depressions, as shipped."""
    with np.load(_PRODUCT, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _channel_cells(
    channel: Channel, latitudes: np.ndarray, longitudes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The raster cells along a channel's great circle, sampled finer than
    the raster so that the cells form an unbroken path."""
    start, end = (
        np.array(
            [
                np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                np.sin(np.radians(lat)),
            ]
        )
        for lat, lon in (channel.start, channel.end)
    )
    angle = np.arccos(np.clip(start @ end, -1.0, 1.0))
    step = np.radians(abs(latitudes[1] - latitudes[0])) / 4.0
    t = np.linspace(0.0, 1.0, max(2, int(np.ceil(angle / step)) + 1))
    if angle > 0.0:
        points = (
            np.sin((1.0 - t) * angle)[:, None] * start
            + np.sin(t * angle)[:, None] * end
        ) / np.sin(angle)
    else:
        points = np.repeat(start[None, :], t.size, axis=0)
    lat = np.degrees(np.arcsin(np.clip(points[:, 2], -1.0, 1.0)))
    lon = np.mod(np.degrees(np.arctan2(points[:, 1], points[:, 0])), 360.0)
    rows = np.argmin(np.abs(lat[:, None] - latitudes[None, :]), axis=1)
    lon_step = abs(longitudes[1] - longitudes[0])
    columns = np.rint((lon - np.mod(longitudes[0], 360.0)) / lon_step).astype(
        int
    ) % len(longitudes)
    return rows, columns


def _dilate(
    mask: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray, distance: float
) -> np.ndarray:
    """The mask grown by a distance in km: a box of that half-width, its
    east-west extent widening towards the poles as the cells narrow."""
    cell = np.radians(abs(latitudes[1] - latitudes[0])) * _RADIUS_KM
    k_lat = int(np.ceil(distance / cell))
    out = ndimage.maximum_filter1d(mask, size=2 * k_lat + 1, axis=0, mode="nearest")
    n_lon = mask.shape[1]
    lon_cell = np.radians(abs(longitudes[1] - longitudes[0])) * _RADIUS_KM
    for i, lat in enumerate(latitudes):
        width = lon_cell * max(np.cos(np.radians(lat)), 1e-6)
        k_lon = min(int(np.ceil(distance / width)), n_lon // 2)
        if k_lon > 0 and out[i].any():
            out[i] = ndimage.maximum_filter1d(out[i], size=2 * k_lon + 1, mode="wrap")
    return out
