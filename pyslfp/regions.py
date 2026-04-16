"""
Module defining the Regions mixin class for spatial masking and dataset handling.
"""

from __future__ import annotations
import warnings
from typing import Optional, List, Union, Any, Tuple

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

    # ==================================================================== #
    #                  1. Initialization & Lazy Loaders                    #
    # ==================================================================== #

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

    # ==================================================================== #
    #                    2. Universal API Routers                          #
    # ==================================================================== #

    def list_all_regions(self) -> dict[str, List[str]]:
        """
        Returns a dictionary mapping every dataset key to a list of its
        available region names.

        Warning: Calling this will trigger the lazy-loading of ALL
        underlying shapefiles if they haven't been downloaded/parsed yet.
        """
        return {
            "AR6 (IPCC Climate Regions)": sorted(self._ar6_regions.names),
            "IMBIE (Antarctica)": self.list_imbie_ant_regions(),
            "Mouginot (Greenland)": self.list_mouginot_grl_regions(),
            "IHO (World Seas)": self.list_iho_seas(),
            "HydroBASINS (Level 3)": self.list_hydrobasins(),
            "Natural Earth (Oceans)": self.list_ne_oceans(),
            "Hardcoded Custom": ["Caspian Sea", "Glaciers"],
        }

    def get_projection(
        self, regions: Union[str, List[str]], /, *, value: float = np.nan
    ) -> SHGrid:
        """
        A universal projection router. Takes a single region name or a list of
        region names, automatically locates them across all available datasets
        (AR6, IHO, HydroBASINS, IMBIE, Mouginot), and returns a unified mask.

        Args:
            regions: A single region name or list of region names.
            value: The value to assign outside the unified mask. Defaults to NaN.
        """
        if isinstance(regions, str):
            regions = [regions]

        lats, lons = self.lats(), self.lons()
        accumulated_mask = np.zeros((len(lats), len(lons)), dtype=bool)

        lookup_methods = [
            self.regionmask_projection,
            self.iho_sea_projection,
            self.hydrobasin_projection,
            self.imbie_ant_projection,
            self.mouginot_grl_projection,
            self.ne_ocean_projection,
        ]

        for region in regions:
            matched = False

            if region == "Caspian Sea":
                mask_grid = self.caspian_sea_projection(value=0.0)
                accumulated_mask = accumulated_mask | (mask_grid.data == 1.0)
                continue
            if region == "Glaciers":
                mask_grid = self.glacier_projection(value=0.0)
                accumulated_mask = accumulated_mask | (mask_grid.data == 1.0)
                continue

            for lookup_method in lookup_methods:
                try:
                    mask_grid = lookup_method(region, value=0.0)
                    accumulated_mask = accumulated_mask | (mask_grid.data == 1.0)
                    matched = True
                    break
                except ValueError:
                    continue

            if not matched:
                raise ValueError(
                    f"Universal lookup failed: Region '{region}' could not be found "
                    f"in AR6, IHO, HydroBASINS, IMBIE, Mouginot, or Natural Earth."
                )

        final_data = np.where(accumulated_mask, 1.0, value)
        return SHGrid.from_array(final_data, grid=self.grid)

    def plot_boundaries(
        self,
        ax: Any,
        regions: Union[str, List[str]],
        /,
        **kwargs,
    ) -> None:
        """
        A universal boundary plotting router.
        """
        if isinstance(regions, str):
            regions = [regions]

        grouped = {
            "AR6": [],
            "IHO": [],
            "HYDRO": [],
            "ANT": [],
            "GRL": [],
            "NE_OCEAN": [],
        }

        for r in regions:
            if r == "Caspian Sea" or r == "Glaciers":
                warnings.warn(
                    f"'{r}' is a hardcoded array mask and has no vector boundaries to plot."
                )
                continue

            def check_and_add(dataset_obj, key, name):
                try:
                    dataset_obj.map_keys(name)
                    grouped[key].append(name)
                    return True
                except KeyError:
                    return False

            if check_and_add(self._ar6_regions, "AR6", r):
                continue
            if check_and_add(self.iho_seas_regions, "IHO", r):
                continue
            if check_and_add(self.hydrobasins_regions, "HYDRO", r):
                continue
            if check_and_add(self.imbie_ant_regions, "ANT", r):
                continue
            if check_and_add(self.mouginot_grl_regions, "GRL", r):
                continue
            if check_and_add(self.ne_ocean_regions, "NE_OCEAN", r):
                continue

            raise ValueError(
                f"Universal plot failed: Region '{r}' not found in any dataset."
            )

        # Repackage line arguments cleanly for regionmask plotting methods
        rm_kwargs = kwargs.copy()
        line_kws = rm_kwargs.pop("line_kws", {})

        # Pop all common Matplotlib styling arguments into line_kws
        if "edgecolor" in rm_kwargs:
            line_kws["color"] = rm_kwargs.pop("edgecolor")
        if "color" in rm_kwargs:
            line_kws["color"] = rm_kwargs.pop("color")
        if "linewidth" in rm_kwargs:
            line_kws["linewidth"] = rm_kwargs.pop("linewidth")
        if "lw" in rm_kwargs:
            line_kws["linewidth"] = rm_kwargs.pop("lw")
        if "zorder" in rm_kwargs:
            line_kws["zorder"] = rm_kwargs.pop("zorder")
        if "alpha" in rm_kwargs:
            line_kws["alpha"] = rm_kwargs.pop("alpha")
        if "linestyle" in rm_kwargs:
            line_kws["linestyle"] = rm_kwargs.pop("linestyle")
        if "ls" in rm_kwargs:
            line_kws["linestyle"] = rm_kwargs.pop("ls")

        if line_kws:
            rm_kwargs["line_kws"] = line_kws

        # Regionmask datasets get the safely repackaged kwargs
        if grouped["AR6"]:
            self.plot_ar6_boundaries(ax, region_names=grouped["AR6"], **rm_kwargs)

        if grouped["NE_OCEAN"]:
            self.plot_ne_ocean_boundaries(
                ax, region_names=grouped["NE_OCEAN"], **rm_kwargs
            )

        # Geopandas datasets can safely accept the raw, standard matplotlib kwargs
        if grouped["IHO"]:
            self.plot_iho_sea_boundaries(ax, region_names=grouped["IHO"], **kwargs)
        if grouped["HYDRO"]:
            self.plot_hydrobasin_boundaries(ax, region_ids=grouped["HYDRO"], **kwargs)
        if grouped["ANT"]:
            self.plot_imbie_ant_boundaries(ax, region_names=grouped["ANT"], **kwargs)
        if grouped["GRL"]:
            self.plot_mouginot_grl_boundaries(ax, region_names=grouped["GRL"], **kwargs)

    # ==================================================================== #
    #                         3. Listing Methods                           #
    # ==================================================================== #

    def list_imbie_ant_regions(self) -> List[str]:
        """List available IMBIE Antarctic subregions."""
        return sorted(self.imbie_ant_regions.names)

    def list_mouginot_grl_regions(self) -> List[str]:
        """List available Greenland subregions."""
        return sorted(self.mouginot_grl_regions.names)

    def list_hydrobasins(self) -> List[str]:
        """List available HydroBASINS IDs (Level 3)."""
        return sorted(self.hydrobasins_regions.names)

    def list_iho_seas(self) -> List[str]:
        """List available IHO Sea names."""
        return sorted(self.iho_seas_regions.names)

    def list_ne_oceans(self) -> List[str]:
        """List available Natural Earth ocean basins."""
        return sorted(self.ne_ocean_regions.names)

    # ==================================================================== #
    #                        4. Projection Methods                         #
    # ==================================================================== #

    def _apply_regionmask(
        self, dataset_key: str, region_names: Union[str, List[str]], value: float
    ) -> SHGrid:
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

        if isinstance(region_names, str):
            region_names = [region_names]

        region_ids = []
        for name in region_names:
            try:
                region_ids.append(rm_obj.map_keys(name))
            except KeyError as exc:
                raise ValueError(
                    f"Region '{name}' not found in {dataset_key}."
                ) from exc

        lons, lats = self.lons(), self.lats()
        lon_mesh, lat_mesh = np.meshgrid(lons[:-1], lats)
        lon_mesh_180 = np.where(lon_mesh > 180, lon_mesh - 360, lon_mesh)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=".*overlapping regions.*"
            )
            mask_3d = rm_obj.mask_3D(lon_mesh_180, lat_mesh, wrap_lon=False)

        valid_ids = [rid for rid in region_ids if rid in mask_3d.region.values]

        if valid_ids:
            specific_layers = mask_3d.sel(region=valid_ids).values
            combined_layer = np.any(specific_layers, axis=0)

            if not np.any(combined_layer):
                warnings.warn(
                    f"Regions '{region_names}' in {dataset_key} contain no grid points "
                    f"at lmax={self.lmax}. Consider increasing resolution.",
                    UserWarning,
                )
        else:
            combined_layer = np.zeros((len(lats), len(lons) - 1), dtype=bool)

        mask_data = np.where(combined_layer, 1.0, value)
        masked_data = np.hstack((mask_data, mask_data[:, 0:1]))

        return SHGrid.from_array(masked_data, grid=self.grid)

    def regionmask_projection(
        self, region_names: Union[str, List[str]], /, *, value: float = np.nan
    ) -> SHGrid:
        """Returns a binary mask for one or more IPCC AR6 regions."""
        if isinstance(region_names, str):
            region_names = [region_names]

        region_ids = []
        for name in region_names:
            try:
                region_ids.append(self._ar6_regions.map_keys(name))
            except KeyError as exc:
                raise ValueError(
                    f"Region '{name}' not found in the AR6 dataset."
                ) from exc

        lons, lats = self.lons(), self.lats()
        mask_unextended = self._ar6_regions.mask(lons[:-1], lats)
        combined_layer = np.isin(mask_unextended.data, region_ids)

        masked_data_unextended = np.where(combined_layer, 1.0, value)
        masked_data = np.hstack(
            (masked_data_unextended, masked_data_unextended[:, 0:1])
        )
        return SHGrid.from_array(masked_data, grid=self.grid)

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

    # --- Static & Convenience Projections ---

    def greenland_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """Greenland projection using AR6 boundaries."""
        return self.regionmask_projection("Greenland/Iceland", value=value)

    def west_antarctic_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """West Antarctic projection using AR6 boundaries."""
        return self.regionmask_projection("W.Antarctica", value=value)

    def east_antarctic_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """East Antarctic projection using AR6 boundaries."""
        return self.regionmask_projection("E.Antarctica", value=value)

    def antarctic_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """Antarctic projection using AR6 boundaries."""
        return self.west_antarctic_projection(
            value=value
        ) + self.east_antarctic_projection(value=value)

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

    # ==================================================================== #
    #                         5. Plotting Methods                          #
    # ==================================================================== #

    def _plot_gdf_boundaries(
        self,
        ax: Any,
        gdf: gpd.GeoDataFrame,
        name_column: str,
        region_names: Optional[Union[str, List[str]]],
        **kwargs,
    ) -> Any:
        """Internal helper to filter and plot GeoDataFrame boundaries."""
        if region_names is not None:
            gdf = gdf[gdf[name_column].isin(np.atleast_1d(region_names))]

        merged_geom = gdf.geometry.unary_union
        merged_series = gpd.GeoSeries([merged_geom], crs=gdf.crs)

        kwargs.pop("line_kws", None)
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("linewidth", 1.0)

        return merged_series.boundary.plot(
            ax=ax, transform=ccrs.PlateCarree(), aspect=None, **kwargs
        )

    def plot_ar6_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_names: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot AR6 region boundaries onto a Cartopy axis."""
        if region_names is not None:
            if isinstance(region_names, str):
                region_names = [region_names]
            idxs = [self._ar6_regions.map_keys(r) for r in region_names]
            return self._ar6_regions[idxs].plot(ax=ax, add_label=False, **kwargs)
        return self._ar6_regions.plot(ax=ax, add_label=False, **kwargs)

    def plot_ne_ocean_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_names: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot Natural Earth ocean boundaries onto a Cartopy axis."""
        if region_names is not None:
            if isinstance(region_names, str):
                region_names = [region_names]
            idxs = [self._ne_ocean_regions.map_keys(r) for r in region_names]
            return self._ne_ocean_regions[idxs].plot(ax=ax, add_label=False, **kwargs)
        return self._ne_ocean_regions.plot(ax=ax, add_label=False, **kwargs)

    def plot_imbie_ant_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_names: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot IMBIE Antarctic basin boundaries onto a Cartopy axis."""
        _ = self.imbie_ant_regions  # Trigger lazy loading
        return self._plot_gdf_boundaries(
            ax, self._imbie_ant_gdf, "Subregion", region_names, **kwargs
        )

    def plot_mouginot_grl_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_names: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot Greenland basin boundaries onto a Cartopy axis."""
        _ = self.mouginot_grl_regions  # Trigger lazy loading
        return self._plot_gdf_boundaries(
            ax, self._mouginot_grl_gdf, "SUBREGION1", region_names, **kwargs
        )

    def plot_hydrobasin_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_ids: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot hydrological basin boundaries onto a Cartopy axis."""
        _ = self.hydrobasins_regions  # Trigger lazy loading
        return self._plot_gdf_boundaries(
            ax, self._hydrobasins_gdf, "HYBAS_ID", region_ids, **kwargs
        )

    def plot_iho_sea_boundaries(
        self,
        ax: Any,
        /,
        *,
        region_names: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """Plot IHO sea boundaries onto a Cartopy axis."""
        _ = self.iho_seas_regions  # Trigger lazy loading
        return self._plot_gdf_boundaries(
            ax, self._iho_seas_gdf, "NAME", region_names, **kwargs
        )

    # ==================================================================== #
    #                    6. Composite Ice Groupings                        #
    # ==================================================================== #

    def ice_basin_groupings(self, /, *, scheme: str = "individual") -> List[List[str]]:
        """
        Provides predefined sensible groupings of ice sheet basins.

        Args:
            scheme (str): The grouping scheme to use ('individual', 'ice_sheets',
                          'macro_regions', 'grl_focused', 'ant_focused').
        """
        ant_names = [f"ANT_{name}" for name in self.list_imbie_ant_regions()]
        grl_names = [f"GRL_{name}" for name in self.list_mouginot_grl_regions()]

        if scheme == "individual":
            return [[name] for name in ant_names + grl_names]
        elif scheme == "ice_sheets":
            return [ant_names, grl_names]
        elif scheme == "macro_regions":
            wais = ["ANT_F-G", "ANT_G-H", "ANT_H-Hp"]
            ap = ["ANT_I-Ipp", "ANT_Ipp-J", "ANT_J-Jpp"]
            eais = [n for n in ant_names if n not in wais and n not in ap]
            return [wais, eais, ap, grl_names]
        elif scheme == "grl_focused":
            return [ant_names] + [[name] for name in grl_names]
        elif scheme == "ant_focused":
            return [[name] for name in ant_names] + [grl_names]
        else:
            raise ValueError(f"Unknown grouping scheme: '{scheme}'")

    def grouped_ice_projections(
        self, /, *, groupings: Optional[Union[str, List[List[str]]]] = None
    ) -> Tuple[List[SHGrid], List[str]]:
        """
        Returns combined SHGrid masks and labels for regional basin groupings.

        Args:
            groupings: A scheme name or a custom list of lists of prefixed region names.
        """
        if isinstance(groupings, str):
            groupings = self.ice_basin_groupings(scheme=groupings)

        if groupings is None:
            groupings = self.ice_basin_groupings(scheme="individual")

        combined_masks = []
        combined_labels = []

        for group in groupings:
            if not group:
                continue

            accumulated_data = None
            reference_grid = None

            for name in group:
                if name.startswith("ANT_"):
                    m = self.imbie_ant_projection(name[4:], value=0.0)
                elif name.startswith("GRL_"):
                    m = self.mouginot_grl_projection(name[4:], value=0.0)
                else:
                    raise ValueError(
                        f"Unknown region '{name}'. Must start with 'ANT_' or 'GRL_'."
                    )

                if accumulated_data is None:
                    accumulated_data = m.data.copy()
                    reference_grid = m
                else:
                    accumulated_data += m.data

            combined_grid = reference_grid.copy()
            combined_grid.data = accumulated_data

            combined_masks.append(combined_grid)
            combined_labels.append(" + ".join(group))

        return combined_masks, combined_labels
