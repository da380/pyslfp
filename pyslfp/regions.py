"""
Module defining the Regions base class for spatial masking and dataset handling.
"""

import os
import warnings
import numpy as np
import geopandas as gpd
import regionmask
from cartopy import crs as ccrs
from pyshtools import SHGrid

from . import DATADIR


class Regions:
    """
    Base class providing spatial region masks and boundary plotting.
    Expects the child class to implement `lats()`, `lons()`, and `grid`.
    """

    def __init__(self):
        self._ar6_regions = regionmask.defined_regions.ar6.all

        # Placeholders for custom datasets (lazy loaded)
        self._imbie_ant_gdf = None
        self._imbie_ant_regions = None
        self._mouginot_grl_gdf = None
        self._mouginot_grl_regions = None
        self._hydrobasins_gdf = None
        self._hydrobasins_regions = None
        self._iho_seas_gdf = None
        self._iho_seas_regions = None
        self._ne_ocean_regions = None

    # -----------------------------------------------#
    #             Lazy Loading Properties            #
    # -----------------------------------------------#

    @property
    def imbie_ant_regions(self):
        if self._imbie_ant_regions is None:
            path = os.path.join(
                DATADIR, "ANT_Basins_IMBIE2", "ANT_Basins_IMBIE2_v1.6.shp"
            )
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
    def mouginot_grl_regions(self):
        if self._mouginot_grl_regions is None:
            path = os.path.join(
                DATADIR, "Greenland_Basins", "Greenland_Basins_PS_v1.4.2.shp"
            )
            self._mouginot_grl_gdf = gpd.read_file(path).to_crs(epsg=4326)
            self._mouginot_grl_gdf = self._mouginot_grl_gdf.dissolve(
                by="SUBREGION1"
            ).reset_index()
            self._mouginot_grl_regions = regionmask.from_geopandas(
                self._mouginot_grl_gdf, names="SUBREGION1", name="Mouginot_GRL"
            )
        return self._mouginot_grl_regions

    @property
    def hydrobasins_regions(self):
        if self._hydrobasins_regions is None:
            path = os.path.join(DATADIR, "HydroBasins", "HydroBASINS_L3_combined.shp")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Could not find combined shapefile at {path}")
            self._hydrobasins_gdf = gpd.read_file(path)
            self._hydrobasins_gdf["HYBAS_ID"] = self._hydrobasins_gdf[
                "HYBAS_ID"
            ].astype(str)
            self._hydrobasins_regions = regionmask.from_geopandas(
                self._hydrobasins_gdf, names="HYBAS_ID", name="HydroBASINS_L3"
            )
        return self._hydrobasins_regions

    @property
    def iho_seas_regions(self):
        if self._iho_seas_regions is None:
            path = os.path.join(DATADIR, "World_Seas_IHO_v3", "World_Seas_IHO_v3.shp")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Could not find IHO shapefile at {path}")
            self._iho_seas_gdf = gpd.read_file(path).to_crs(epsg=4326)
            self._iho_seas_regions = regionmask.from_geopandas(
                self._iho_seas_gdf, names="NAME", name="IHO_Seas"
            )
        return self._iho_seas_regions

    @property
    def ne_ocean_regions(self):
        if self._ne_ocean_regions is None:
            self._ne_ocean_regions = (
                regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50
            )
        return self._ne_ocean_regions

    # -----------------------------------------------#
    #             Internal Mask Applier              #
    # -----------------------------------------------#

    def _apply_regionmask(self, dataset_key, region_name, value):
        """Internal helper to apply a regionmask to the current grid."""
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
        # This bypasses regionmask's strict internal boundary checks that can crash
        # on shapefiles with floating-point anomalies (e.g., 180.000643).
        lon_mesh, lat_mesh = np.meshgrid(lons[:-1], lats)
        lon_mesh_180 = np.where(lon_mesh > 180, lon_mesh - 360, lon_mesh)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=".*overlapping regions.*"
            )

            # Use 2D wrapped mesh and EXPLICITLY set wrap_lon=False.
            mask_3d = rm_obj.mask_3D(lon_mesh_180, lat_mesh, wrap_lon=False)

        if region_id in mask_3d.region.values:
            specific_layer = mask_3d.sel(region=region_id).values

            # Robustness: Warn the user if no grid points fall inside the region.
            if not np.any(specific_layer):
                warnings.warn(
                    f"Region '{region_name}' in {dataset_key} contains no grid points "
                    f"at lmax={self.lmax}. Consider increasing resolution.",
                    UserWarning,
                )
        else:
            specific_layer = np.zeros((len(lats), len(lons) - 1), dtype=bool)

        mask_data = np.where(specific_layer, 1.0, value)

        # Re-extend the grid for SHGrid compatibility (add 360-degree column).
        masked_data = np.hstack((mask_data, mask_data[:, 0:1]))

        return SHGrid.from_array(masked_data, grid=self.grid)

    # -----------------------------------------------#
    #             Public Projection Methods          #
    # -----------------------------------------------#

    def regionmask_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
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
        return self.regionmask_projection("Greenland/Iceland", value=value)

    def west_antarctic_projection(self, /, *, value: float = np.nan) -> SHGrid:
        return self.regionmask_projection("W.Antarctica", value=value)

    def east_antarctic_projection(self, /, *, value: float = np.nan) -> SHGrid:
        return self.regionmask_projection("E.Antarctica", value=value)

    def caspian_sea_projection(self, /, *, value: float = np.nan) -> SHGrid:
        lats, lons = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        caspian_mask = np.logical_and(
            np.logical_and(lats > 36, lats < 49.5),
            np.logical_and(lons > 45.5, lons < 55),
        )
        return SHGrid.from_array(np.where(caspian_mask, 1, value), grid=self.grid)

    def glacier_projection(self, /, *, value: float = np.nan) -> SHGrid:
        lats, lons = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        glacier_mask = np.logical_and(
            np.logical_and(lats > 30, lats < 70), np.logical_and(lons > 180, lons < 270)
        )
        return SHGrid.from_array(np.where(glacier_mask, 1, value), grid=self.grid)

    def imbie_ant_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
        return self._apply_regionmask("ANT", region_name, value)

    def mouginot_grl_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
        return self._apply_regionmask("GRL", region_name, value)

    def hydrobasin_projection(
        self, region_id: str, /, *, value: float = np.nan
    ) -> SHGrid:
        return self._apply_regionmask("HYDRO", region_id, value)

    def iho_sea_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
        return self._apply_regionmask("OCEAN", region_name, value)

    def ne_ocean_projection(
        self, region_name: str, /, *, value: float = np.nan
    ) -> SHGrid:
        return self._apply_regionmask("NE_OCEAN", region_name, value)

    # -----------------------------------------------#
    #             Listing and Plotting               #
    # -----------------------------------------------#

    def list_imbie_ant_regions(self) -> list[str]:
        return sorted(self.imbie_ant_regions.names)

    def plot_imbie_ant_boundaries(self, ax, region_names=None, **kwargs):
        _ = self.imbie_ant_regions
        gdf = self._imbie_ant_gdf
        if region_names is not None:
            gdf = gdf[gdf["Subregion"].isin(np.atleast_1d(region_names))]
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("linewidth", 1.0)
        return gdf.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)

    def list_mouginot_grl_regions(self) -> list[str]:
        return sorted(self.mouginot_grl_regions.names)

    def plot_mouginot_grl_boundaries(self, ax, region_names=None, **kwargs):
        _ = self.mouginot_grl_regions
        gdf = self._mouginot_grl_gdf
        if region_names is not None:
            gdf = gdf[gdf["SUBREGION1"].isin(np.atleast_1d(region_names))]
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("linewidth", 1.0)
        return gdf.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)

    def list_hydrobasins(self) -> list[str]:
        return sorted(self.hydrobasins_regions.names)

    def plot_hydrobasin_boundaries(self, ax, region_ids=None, **kwargs):
        _ = self.hydrobasins_regions
        gdf = self._hydrobasins_gdf
        if region_ids is not None:
            gdf = gdf[gdf["HYBAS_ID"].isin(np.atleast_1d(region_ids))]
        kwargs.setdefault("edgecolor", "blue")
        kwargs.setdefault("linewidth", 0.5)
        return gdf.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)

    def list_iho_seas(self) -> list[str]:
        return sorted(self.iho_seas_regions.names)

    def plot_iho_sea_boundaries(self, ax, region_names=None, **kwargs):
        _ = self.iho_seas_regions
        gdf = self._iho_seas_gdf
        if region_names is not None:
            gdf = gdf[gdf["NAME"].isin(np.atleast_1d(region_names))]
        kwargs.setdefault("edgecolor", "teal")
        kwargs.setdefault("linewidth", 0.8)
        return gdf.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)

    def list_ne_oceans(self) -> list[str]:
        return sorted(self.ne_ocean_regions.names)

    def plot_ne_ocean_boundaries(self, ax, **kwargs):
        kwargs.setdefault("line_kws", dict(color="dodgerblue", linewidth=1.0))
        return self.ne_ocean_regions.plot(ax=ax, add_label=False, **kwargs)
