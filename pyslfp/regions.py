"""
Module defining the Regions mixin class for spatial masking and dataset handling.
"""

from __future__ import annotations
import warnings
from typing import Optional, List, Union, Any

import numpy as np
import geopandas as gpd
import regionmask
from cartopy import crs as ccrs
from pyshtools import SHGrid

from pyslfp.data import DATADIR, ensure_data


class Regions:
    """
    Mixin class providing spatial region masks and boundary plotting.

    This class provides a suite of methods to project physical fields onto
    specific geographic regions (e.g., Antarctica, Greenland, river basins,
    and ocean basins). It handles the lazy loading of shapefile datasets
    and uses the `regionmask` library to create binary masks on spherical
    harmonic grids.

    Note:
        This class is intended to be used as a mixin for the `FingerPrint` class.
        It expects the child class to provide `lats()`, `lons()`, `lmax`,
        and `grid` properties.
    """

    def __init__(self) -> None:
        """Initializes the Regions mixin with built-in AR6 region definitions."""
        self._ar6_regions = regionmask.defined_regions.ar6.all

        # Placeholders for custom datasets (lazy loaded via properties)
        self._imbie_ant_gdf: Optional[gpd.GeoDataFrame] = None
        self._imbie_ant_regions: Optional[regionmask.Regions] = None
        self._mouginot_grl_gdf: Optional[gpd.GeoDataFrame] = None
        self._mouginot_grl_regions: Optional[regionmask.Regions] = None
        self._hydrobasins_gdf: Optional[gpd.GeoDataFrame] = None
        self._hydrobasins_regions: Optional[regionmask.Regions] = None
        self._iho_seas_gdf: Optional[gpd.GeoDataFrame] = None
        self._iho_seas_regions: Optional[regionmask.Regions] = None
        self._ne_ocean_regions: Optional[regionmask.Regions] = None

    # ---------------------------------------------------------#
    #             Lazy Loading Properties (Type Hinted)        #
    # ---------------------------------------------------------#

    @property
    def imbie_ant_regions(self) -> regionmask.Regions:
        """IMBIE2 Antarctic drainage basins (Rignot et al., 2011)."""
        if self._imbie_ant_regions is None:

            ensure_data("IMBIE_ANT")

            path = DATADIR / "ANT_Basins_IMBIE2" / "ANT_Basins_IMBIE2_v1.6.shp"
            self._imbie_ant_gdf = gpd.read_file(path).to_crs(epsg=4326)
            self._imbie_ant_gdf = self._imbie_ant_gdf[
                self._imbie_ant_gdf["Subregion"].notna()
                & (self._imbie_ant_gdf["Subregion"] != "")
            ]
            self._imbie_ant_regions = regionmask.from_geopandas(
                self._imbie_ant_gdf, names="Subregion", name="IMBIE_ANT"
            )
        return self._imbie_ant_regions

    @property
    def mouginot_grl_regions(self) -> regionmask.Regions:
        """Greenland drainage basins (Mouginot et al., 2019)."""
        if self._mouginot_grl_regions is None:

            ensure_data("MOUGINOT_GRL")

            path = DATADIR / "Greenland_Basins" / "Greenland_Basins_PS_v1.4.2.shp"
            self._mouginot_grl_gdf = gpd.read_file(path).to_crs(epsg=4326)
            self._mouginot_grl_gdf = self._mouginot_grl_gdf.dissolve(
                by="SUBREGION1"
            ).reset_index()
            self._mouginot_grl_regions = regionmask.from_geopandas(
                self._mouginot_grl_gdf, names="SUBREGION1", name="Mouginot_GRL"
            )
        return self._mouginot_grl_regions

    @property
    def hydrobasins_regions(self) -> regionmask.Regions:
        """Level 3 Hydrological Basins (HydroBASINS/HydroSHEDS)."""
        if self._hydrobasins_regions is None:

            ensure_data("HYDRO")

            path = DATADIR / "HydroBasins" / "HydroBASINS_L3_combined.shp"
            self._hydrobasins_gdf = gpd.read_file(path)
            self._hydrobasins_gdf["HYBAS_ID"] = self._hydrobasins_gdf[
                "HYBAS_ID"
            ].astype(str)
            self._hydrobasins_regions = regionmask.from_geopandas(
                self._hydrobasins_gdf, names="HYBAS_ID", name="HydroBASINS_L3"
            )
        return self._hydrobasins_regions

    @property
    def iho_seas_regions(self) -> regionmask.Regions:
        """IHO World Seas (Marine Regions, v3)."""
        if self._iho_seas_regions is None:

            ensure_data("IHO_SEAS")

            path = DATADIR / "World_Seas_IHO_v3" / "World_Seas_IHO_v3.shp"
            self._iho_seas_gdf = gpd.read_file(path).to_crs(epsg=4326)
            self._iho_seas_regions = regionmask.from_geopandas(
                self._iho_seas_gdf, names="NAME", name="IHO_Seas"
            )
        return self._iho_seas_regions

    @property
    def ne_ocean_regions(self) -> regionmask.Regions:
        """Built-in Natural Earth ocean basins (v5.1.2)."""
        if self._ne_ocean_regions is None:
            self._ne_ocean_regions = (
                regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50
            )
        return self._ne_ocean_regions

    # ---------------------------------------------------------#
    #             Internal Mask Applier                        #
    # ---------------------------------------------------------#

    def _apply_regionmask(
        self, dataset_key: str, region_name: str, value: float
    ) -> SHGrid:
        """
        Internal helper to apply a regionmask to the current grid.

        Handles the meshgrid creation and coordinate wrapping required to
        robustly mask regions across the antimeridian.
        """
        if dataset_key == "ANT":
            rm_obj = self.imbie_ant_regions
        elif dataset_key == "GRL":
            rm_obj = self.mouginot_grl_regions
        elif dataset_key == "HYDRO":
            rm_obj = self.hydrobasins_regions
        elif dataset_key == "OCEAN":
            rm_obj = self.iho_seas_regions
        elif dataset_key == "NE_OCEAN":
            rm_obj = self.ne_ocean_regions
        else:
            raise ValueError(
                "dataset_key must be 'ANT', 'GRL', 'HYDRO', 'OCEAN', or 'NE_OCEAN'"
            )

        try:
            region_id = rm_obj.map_keys(region_name)
        except KeyError:
            raise ValueError(f"Region '{region_name}' not found in {dataset_key}.")

        lons, lats = self.lons(), self.lats()

        # Robustness: Create a 2D meshgrid and manually wrap longitudes to [-180, 180].
        lon_mesh, lat_mesh = np.meshgrid(lons[:-1], lats)
        lon_mesh_180 = np.where(lon_mesh > 180, lon_mesh - 360, lon_mesh)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=".*overlapping regions.*"
            )
            mask_3d = rm_obj.mask_3D(lon_mesh_180, lat_mesh, wrap_lon=False)

        if region_id in mask_3d.region.values:
            specific_layer = mask_3d.sel(region=region_id).values
            if not np.any(specific_layer):
                warnings.warn(
                    f"Region '{region_name}' in {dataset_key} contains no grid points "
                    f"at lmax={self.lmax}. Consider increasing resolution.",
                    UserWarning,
                )
        else:
            specific_layer = np.zeros((len(lats), len(lons) - 1), dtype=bool)

        mask_data = np.where(specific_layer, 1.0, value)
        masked_data = np.hstack((mask_data, mask_data[:, 0:1]))

        return SHGrid.from_array(masked_data, grid=self.grid)

    # ---------------------------------------------------------#
    #             Public Projection Methods                    #
    # ---------------------------------------------------------#

    def regionmask_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
        """
        Returns a binary mask for an IPCC AR6 region.

        Args:
            region_name: The name of the AR6 region (e.g., 'Greenland/Iceland').
            value: The value for areas outside the region. Defaults to NaN.
        """
        try:
            region_id = self._ar6_regions.map_keys(region_name)
        except KeyError as exc:
            raise ValueError(
                f"Region '{region_name}' not found in the AR6 dataset."
            ) from exc
        lons, lats = self.lons(), self.lats()
        mask_unextended = self._ar6_regions.mask(lons[:-1], lats)
        masked_data_unextended = np.where(mask_unextended.data == region_id, 1, value)
        masked_data = np.hstack(
            (masked_data_unextended, masked_data_unextended[:, 0:1])
        )
        return SHGrid.from_array(masked_data, grid=self.grid)

    def greenland_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """Greenland projection using AR6 boundaries."""
        return self.regionmask_projection("Greenland/Iceland", value=value)

    def west_antarctic_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """West Antarctic projection using AR6 boundaries."""
        return self.regionmask_projection("W.Antarctica", value=value)

    def east_antarctic_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """East Antarctic projection using AR6 boundaries."""
        return self.regionmask_projection("E.Antarctica", value=value)

    def caspian_sea_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """Simple rectangular projection for the Caspian Sea."""
        lats, lons = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        caspian_mask = np.logical_and(
            np.logical_and(lats > 36, lats < 49.5),
            np.logical_and(lons > 45.5, lons < 55),
        )
        return SHGrid.from_array(np.where(caspian_mask, 1, value), grid=self.grid)

    def glacier_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """Glacier projection for North American mountain glaciers."""
        lats, lons = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        glacier_mask = np.logical_and(
            np.logical_and(lats > 30, lats < 70), np.logical_and(lons > 180, lons < 270)
        )
        return SHGrid.from_array(np.where(glacier_mask, 1, value), grid=self.grid)

    def imbie_ant_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
        """Project a specific IMBIE Antarctic basin."""
        return self._apply_regionmask("ANT", region_name, value)

    def mouginot_grl_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
        """Project a specific Mouginot Greenland basin."""
        return self._apply_regionmask("GRL", region_name, value)

    def hydrobasin_projection(
        self, region_id: str, /, *, value: float = np.nan
    ) -> SHGrid:
        """Project a specific Level 3 Hydrological Basin ID."""
        return self._apply_regionmask("HYDRO", region_id, value)

    def iho_sea_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
        """Project a specific IHO Sea area."""
        return self._apply_regionmask("OCEAN", region_name, value)

    def ne_ocean_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
        """Project a Natural Earth ocean basin."""
        return self._apply_regionmask("NE_OCEAN", region_name, value)

    # ---------------------------------------------------------#
    #             Listing and Plotting Methods                 #
    # ---------------------------------------------------------#

    def list_imbie_ant_regions(self) -> List[str]:
        """List available IMBIE Antarctic subregions."""
        return sorted(self.imbie_ant_regions.names)

    def plot_imbie_ant_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_names: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot IMBIE Antarctic basin boundaries onto a Cartopy axis."""
        _ = self.imbie_ant_regions
        gdf = self._imbie_ant_gdf
        if region_names is not None:
            gdf = gdf[gdf["Subregion"].isin(np.atleast_1d(region_names))]
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("linewidth", 1.0)
        return gdf.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)

    def list_mouginot_grl_regions(self) -> List[str]:
        """List available Greenland subregions."""
        return sorted(self.mouginot_grl_regions.names)

    def plot_mouginot_grl_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_names: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot Greenland basin boundaries onto a Cartopy axis."""
        _ = self.mouginot_grl_regions
        gdf = self._mouginot_grl_gdf
        if region_names is not None:
            gdf = gdf[gdf["SUBREGION1"].isin(np.atleast_1d(region_names))]
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("linewidth", 1.0)
        return gdf.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)

    def list_hydrobasins(self) -> List[str]:
        """List available HydroBASINS IDs (Level 3)."""
        return sorted(self.hydrobasins_regions.names)

    def plot_hydrobasin_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_ids: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot hydrological basin boundaries onto a Cartopy axis."""
        _ = self.hydrobasins_regions
        gdf = self._hydrobasins_gdf
        if region_ids is not None:
            gdf = gdf[gdf["HYBAS_ID"].isin(np.atleast_1d(region_ids))]
        kwargs.setdefault("edgecolor", "blue")
        kwargs.setdefault("linewidth", 0.5)
        return gdf.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)

    def list_iho_seas(self) -> List[str]:
        """List available IHO Sea names."""
        return sorted(self.iho_seas_regions.names)

    def plot_iho_sea_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_names: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot IHO sea boundaries onto a Cartopy axis."""
        _ = self.iho_seas_regions
        gdf = self._iho_seas_gdf
        if region_names is not None:
            gdf = gdf[gdf["NAME"].isin(np.atleast_1d(region_names))]
        kwargs.setdefault("edgecolor", "teal")
        kwargs.setdefault("linewidth", 0.8)
        return gdf.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)

    def list_ne_oceans(self) -> List[str]:
        """List available Natural Earth ocean basins."""
        return sorted(self.ne_ocean_regions.names)

    def plot_ne_ocean_boundaries(self, ax: Any, /, **kwargs) -> Any:
        """Plot Natural Earth ocean boundaries onto a Cartopy axis."""
        kwargs.setdefault("line_kws", dict(color="dodgerblue", linewidth=1.0))
        return self.ne_ocean_regions.plot(ax=ax, add_label=False, **kwargs)
