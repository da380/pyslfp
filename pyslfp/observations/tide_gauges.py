"""
Tide gauges: the relative sea level change at stations.

`read_gloss_tide_gauge_data` reads the GLOSS network, `tide_gauge_operator`
samples a response's relative sea level change at points, and
`TideGaugeObservationModel` samples an observable, the relative sea level
change from a model space with any direct terms included.
"""

from __future__ import annotations
import csv
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
from pyshtools import SHGrid


from pygeoinf import (
    LinearOperator,
)

from pyslfp.data.downloader import ensure_data
from pyslfp.core.response import RELATIVE_SEA_LEVEL_CHANGE, GRDResponse, GRDSpace
from pyslfp.core.spaces import check_load_space, check_response_space
from pyslfp.elastic.operator import FingerPrintOperator
from pyslfp.observations.base import Noise, ObservationModel


# ================================================================ #
#                       Data Loaders                               #
# ================================================================ #


def read_gloss_tide_gauge_data(
    *,
    filter_func: Optional[Callable[[str, float, float], bool]] = None,
) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    Reads and parses the GLOSS tide gauge network coordinates and names.

    Automatically fetches the GLOSS dataset from Zenodo if not present locally.

    Args:
        filter_func: An optional callable that takes (name, lat, lon) and
            returns True if the station should be included, False otherwise.

    Returns:
        Two lists containing the station names and their corresponding
        (latitude, longitude) tuples.
    """
    data_dir = ensure_data("TIDE_GAUGE")
    file_path = data_dir / "gloss_full.txt"

    names: List[str] = []
    points: List[Tuple[float, float]] = []

    # Using encoding="utf-8" to handle special characters in international names.
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip the header row

        for row in reader:
            if len(row) >= 5:
                name = row[0].strip()
                lat = float(row[3])
                lon = float(row[4])

                if filter_func is None or filter_func(name, lat, lon):
                    names.append(name)
                    points.append((lat, lon))

    return names, points


# ================================================================ #
#                       Raw Math Operators                         #
# ================================================================ #


def tide_gauge_operator(
    response_space: GRDSpace, points: List[Tuple[float, float]], /
) -> LinearOperator[GRDResponse, np.ndarray]:
    """
    Maps the response to the sea level change at discrete locations: tide
    gauges measure the relative sea level change.

    Args:
        response_space: The response space, its fields of Sobolev order above 1.
        points: A list of (latitude, longitude) observation coordinates in degrees.

    Returns:
        A mapping from the response space to an N-dimensional Euclidean observation space.
    """
    check_response_space(response_space, point_values=True)

    point_evaluation_operator = response_space.field_space.point_evaluation_operator(
        points, route="dense"
    )

    return point_evaluation_operator @ response_space.field_operator(
        RELATIVE_SEA_LEVEL_CHANGE
    )


# ================================================================ #
#                     Observation Models                           #
# ================================================================ #


class TideGaugeObservationModel(ObservationModel):
    """
    Tide gauges: the relative sea level change, a field of a Sobolev space of
    order above 1, sampled at the stations.
    """

    def __init__(
        self,
        observable: LinearOperator[Any, SHGrid],
        points: List[Tuple[float, float]],
        /,
        *,
        names: Optional[List[str]] = None,
        noise: Noise = None,
    ):
        """
        Args:
            observable: The relative sea level change, from the model space
                into a field space, such as a set-up's
                `relative_sea_level_change(fingerprint)`.
            points: The stations, (latitude, longitude) in degrees.
            names: The stations' names; numbered by default.
            noise: As for `ObservationModel`.

        Raises:
            ValueError: If the field space does not allow point evaluation.
        """
        field_space = observable.codomain
        check_load_space(field_space, point_values=True)
        self._points = list(points)
        self._names = (
            list(names)
            if names is not None
            else [f"Station_{i}" for i in range(len(self._points))]
        )
        sampling = field_space.point_evaluation_operator(self._points, route="dense")
        super().__init__(observable, sampling, noise=noise)

    @classmethod
    def from_fingerprint(
        cls,
        fingerprint: FingerPrintOperator,
        points: List[Tuple[float, float]],
        /,
        *,
        names: Optional[List[str]] = None,
        noise: Noise = None,
    ) -> TideGaugeObservationModel:
        """Tide gauges observing the response of a fingerprint operator to its
        load, with no direct terms."""
        observable = (
            fingerprint.codomain.field_operator(RELATIVE_SEA_LEVEL_CHANGE) @ fingerprint
        )
        return cls(observable, points, names=names, noise=noise)

    @classmethod
    def from_gloss_network(
        cls,
        observable: LinearOperator[Any, SHGrid],
        /,
        *,
        filter_func: Optional[Callable[[str, float, float], bool]] = None,
        noise: Noise = None,
    ) -> TideGaugeObservationModel:
        """
        Tide gauges at the stations of the GLOSS network.

        Args:
            observable: The relative sea level change, as for the constructor.
            filter_func: An optional callable to filter stations by (name,
                lat, lon).
            noise: As for `ObservationModel`.
        """
        names, points = read_gloss_tide_gauge_data(filter_func=filter_func)
        return cls(observable, points, names=names, noise=noise)

    @property
    def points(self) -> List[Tuple[float, float]]:
        """The stations, (latitude, longitude) in degrees."""
        return list(self._points)

    @property
    def names(self) -> List[str]:
        """The stations' names."""
        return list(self._names)
