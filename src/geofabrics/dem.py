# -*- coding: utf-8 -*-
"""
This module contains classes associated with loading, generating, and combining
DEMs.
"""
import rioxarray
import rioxarray.merge
import rasterio
import xarray
import numpy
import math
import typing
import pathlib
import geopandas
import pandas
import shapely
import dask
import dask.array
import pdal
import json
import abc
import gc
import logging
import scipy.interpolate
import scipy.spatial
from . import geometry


RBF_CACHE_SIZE = 1000


def chunk_mask(mask, chunk_size):
    arrs = []
    for i in range(0, mask.shape[0], chunk_size):
        sub_arrs = []
        for j in range(0, mask.shape[1], chunk_size):
            chunk = dask.array.from_array(
                mask[i : i + chunk_size, j : j + chunk_size].copy()
            )
            sub_arrs.append(chunk)
        arrs.append(sub_arrs)
    mask = dask.array.block(arrs)
    return mask


def clip_mask(arr, geometry, chunk_size, invert=False):
    mask = (
        xarray.ones_like(arr, dtype=numpy.float16)
        .compute()
        .rio.clip(geometry, drop=False, invert=invert)
        .notnull()
    )
    if chunk_size is not None:
        mask.data = chunk_mask(mask.data, chunk_size)
    return mask


class CoarseDem:
    """A class to manage coarse or background DEMs in the catchment context

    Specifically, clip within the catchment land and foreshore. There is the option to
    clip outside any LiDAR using the
    optional 'exclusion_extent' input.

    If set_foreshore is True all positive DEM values in the foreshore are set to zero.
    """

    def __init__(
        self,
        dem_file,
        extents: dict,
        set_foreshore: bool = True,
    ):
        """Load in the coarse DEM, clip and extract points"""

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.set_foreshore = set_foreshore
        # Drop the band coordinate added by rasterio.open()
        self._dem = rioxarray.rioxarray.open_rasterio(dem_file, masked=True).squeeze(
            "band", drop=True
        )

        self._extents = extents
        self._points = []

        self._set_up()

    def __del__(self):
        """Ensure the memory associated with netCDF files is properly freed."""

        # The overall DEM
        if self._dem is not None:
            self._dem.close()
            del self._dem

    @property
    def dem(self) -> xarray.Dataset:
        """Return the DEM over the catchment region"""
        return self._dem

    @property
    def resolution(self) -> float:
        """Return the largest dimension of the coarse DEM resolution"""

        resolution = self._dem.rio.resolution()
        resolution = max(abs(resolution[0]), abs(resolution[1]))
        return resolution

    @property
    def points(self) -> numpy.ndarray:
        """The coarse DEM points after any extent or foreshore value
        filtering."""

        return self._points

    @property
    def extents(self) -> geopandas.GeoDataFrame:
        """The extents for the coarse DEM"""

        return self._extents

    @property
    def empty(self) -> bool:
        """True if the DEM is empty"""

        return len(self._points) == 0 or self._dem is None

    def calculate_dem_bounds(self, dem):
        """Return the bounds for a DEM."""
        dem_bounds = geopandas.GeoDataFrame(
            geometry=[shapely.geometry.box(*dem.rio.bounds())],
            crs=dem.rio.crs,
        )
        return dem_bounds

    def _set_up(self):
        """Set DEM CRS and trim the DEM to size"""

        self._dem.rio.write_crs(self._extents["total"].crs, inplace=True)

        # Calculate DEM bounds and check for overlap before clip
        dem_bounds = self.calculate_dem_bounds(self._dem)
        self._extents["total"] = dem_bounds.overlay(
            self._extents["total"], how="intersection"
        )

        if self._extents["total"].area.sum() > self.resolution * self.resolution:
            # Try clip - catch if no DEM in clipping bounds
            try:
                self._dem = self._dem.rio.clip(
                    self._extents["total"]
                    .buffer(self.resolution * numpy.sqrt(2))
                    .geometry.values,
                    drop=True,
                    from_disk=True,
                )
                self._dem.load()

                # Update foreshore extents - buffer by resolution and clip to extents
                foreshore = self._extents["foreshore"].overlay(
                    self._extents["total"],
                    how="intersection",
                    keep_geom_type=True,
                )
                foreshore = geopandas.GeoDataFrame(
                    geometry=foreshore.buffer(self.resolution * numpy.sqrt(2))
                )
                if foreshore.is_empty.all():
                    self._extents["foreshore"] = foreshore
                else:
                    self._extents["foreshore"] = foreshore.overlay(
                        self._extents["land"],
                        how="difference",
                        keep_geom_type=True,
                    )
                # Update the land extents - buffer by resolution and clip to extents
                land = self._extents["land"].overlay(
                    self._extents["total"],
                    how="intersection",
                    keep_geom_type=True,
                )
                land = geopandas.GeoDataFrame(
                    geometry=land.buffer(self.resolution * numpy.sqrt(2))
                )
                if land.is_empty.all():
                    self._extents["land"] = land
                else:
                    self._extents["land"] = land.overlay(
                        self._extents["land"],
                        how="intersection",
                        keep_geom_type=True,
                    )
                # Calculate the points within the DEM
                self._points = self._extract_points(self._dem)

            except (
                rioxarray.exceptions.NoDataInBounds,
                ValueError,
            ) as caught_exception:
                self.logger.warning(
                    f"{caught_exception} in CoarseDEM. Will set to empty."
                )
                self._dem = None
                self._points = []
        else:
            self._dem = None
            self._points = []

    def _extract_points(self, dem):
        """Create a points list from the DEM. Treat the onland and foreshore
        poins separately"""

        # Take the values on land only - separately consider the buffered foreshore area
        if self._extents["land"].area.sum() > self.resolution * self.resolution:
            # Clip DEM to buffered land
            land_dem = dem.rio.clip(self._extents["land"].geometry.values, drop=True)
            # get coarse DEM points on land
            mask = land_dem.notnull().values
            grid_x, grid_y = numpy.meshgrid(land_dem.x, land_dem.y)

            land_x = grid_x[mask]
            land_y = grid_y[mask]
            land_z = land_dem.values[mask]
        else:  # If there is no DEM outside LiDAR/exclusion_extent and on land
            land_x = []
            land_y = []
            land_z = []
        # Take the values on foreshore only - separately consider the buffered land area
        if self._extents["foreshore"].area.sum() > self.resolution * self.resolution:
            # Clip DEM to buffered foreshore
            foreshore_dem = dem.rio.clip(
                self._extents["foreshore"].geometry.values, drop=True
            )

            # get coarse DEM points on the foreshore - with any positive set to zero
            if self.set_foreshore:
                foreshore_dem = foreshore_dem.where(foreshore_dem <= 0, 0)
            mask = foreshore_dem.notnull().values
            grid_x, grid_y = numpy.meshgrid(foreshore_dem.x, foreshore_dem.y)

            foreshore_x = grid_x[mask]
            foreshore_y = grid_y[mask]
            foreshore_z = foreshore_dem.values[mask]
        else:  # If there is no DEM outside LiDAR/exclusion_extent and on foreshore
            foreshore_x = []
            foreshore_y = []
            foreshore_z = []
        if len(land_x) + len(foreshore_x) == 0:
            # If no points - give a warning and then return an empty array
            self.logger.warning("The coarse DEM has no values on the land or foreshore")
            return []

        # combine in an single array
        points = numpy.empty(
            [len(land_x) + len(foreshore_x)],
            dtype=[
                ("X", geometry.RASTER_TYPE),
                ("Y", geometry.RASTER_TYPE),
                ("Z", geometry.RASTER_TYPE),
            ],
        )
        points["X"][: len(land_x)] = land_x
        points["Y"][: len(land_x)] = land_y
        points["Z"][: len(land_x)] = land_z

        points["X"][len(land_x) :] = foreshore_x
        points["Y"][len(land_x) :] = foreshore_y
        points["Z"][len(land_x) :] = foreshore_z

        return points


class DemBase(abc.ABC):
    """An abstract class to manage the different geofabric layers in a
    catchment context. The geofabruc has a z, and data_source layer and may
    sometimes also have a zo (roughness length) and lidar_source layer.

    It is represented by an XArray dataset and is expected to be saved as a
    netCDF file.

    Standard data catcegories are specified in the SOURCE_CLASSIFICATION
    variable.

    Parameters
    ----------

    catchment_geometry
        Defines the spatial extents of the catchment, land, foreshore, and offshore
        regions
    extents
        Defines the extents of any dense (LiDAR or refernence DEM) values already added.
    """

    SOURCE_CLASSIFICATION = {
        "LiDAR": 1,
        "ocean bathymetry": 2,
        "rivers and fans": 3,
        "waterways": 4,
        "coarse DEM": 5,
        "patch": 6,
        "stopbanks": 7,
        "masked feature": 8,
        "lakes": 9,
        "interpolated": 0,
        "no data": -1,
    }

    def __init__(self, catchment_geometry: geometry.CatchmentGeometry, chunk_size: int):
        """Setup base DEM to add future tiles too"""

        self.catchment_geometry = catchment_geometry
        self.chunk_size = chunk_size

    @property
    def dem(self) -> xarray.Dataset:
        """Return the DEM over the catchment region"""
        raise NotImplementedError("dem must be instantiated in the child class")

    def _load_dem(self, filename: pathlib.Path) -> xarray.Dataset:
        """Load in and replace the DEM with a previously cached version."""
        dem = rioxarray.rioxarray.open_rasterio(
            filename,
            masked=True,
            parse_coordinates=True,
            chunks={"x": self.chunk_size, "y": self.chunk_size},
        ).squeeze(
            "band", drop=True
        )  # remove band coordinate added by rasterio.open()
        self._write_netcdf_conventions_in_place(dem, self.catchment_geometry.crs)

        if "data_source" in dem.keys():
            dem["data_source"] = dem.data_source.astype(geometry.RASTER_TYPE)
        if "lidar_source" in dem.keys():
            dem["lidar_source"] = dem.lidar_source.astype(geometry.RASTER_TYPE)
        if "z" in dem.keys():
            dem["z"] = dem.z.astype(geometry.RASTER_TYPE)
        if "zo" in dem.keys():
            dem["zo"] = dem.zo.astype(geometry.RASTER_TYPE)

        return dem

    def save_dem(
        self, filename: pathlib.Path, dem: xarray.Dataset, compression: dict = None
    ):
        """Save the DEM to a netCDF file.

        :param filename: .nc or .tif file to save the DEM.
        :param dem: the DEM to save.
        :param compression: the compression instructions if compressing.
        """

        assert not any(
            array.rio.crs is None for array in dem.data_vars.values()
        ), "all DataArray variables of a xarray.Dataset must have a CRS"

        try:
            for key in dem.data_vars:
                dem[key] = dem[key].astype(geometry.RASTER_TYPE)
            self._write_netcdf_conventions_in_place(dem, self.catchment_geometry.crs)
            if filename.suffix.lower() == ".nc":
                if compression is not None:
                    encoding_keys = (
                        "_FillValue",
                        "dtype",
                        "scale_factor",
                        "add_offset",
                        "grid_mapping",
                    )
                    encoding = {}
                    for key in dem.data_vars:
                        encoding[key] = {
                            encoding_key: value
                            for encoding_key, value in dem[key].encoding.items()
                            if encoding_key in encoding_keys
                        }
                        if "dtype" not in encoding[key]:
                            encoding[key]["dtype"] = dem[key].dtype
                        encoding[key] = {**encoding[key], **compression}
                    dem.to_netcdf(
                        filename, format="NETCDF4", engine="netcdf4", encoding=encoding
                    )
                else:
                    dem.to_netcdf(filename, format="NETCDF4", engine="netcdf4")
            elif filename.suffix.lower() == ".tif":
                for key, array in dem.data_vars.items():
                    filename_layer = (
                        filename.parent / f"{filename.stem}_{key}{filename.suffix}"
                    )
                    array.encoding = {
                        "dtype": array.dtype,
                        "grid_mapping": array.encoding["grid_mapping"],
                        "rasterio_dtype": array.dtype,
                    }
                    if compression:
                        array.rio.to_raster(filename_layer, compress="deflate")
                    else:
                        array.rio.to_raster(filename_layer)
            dem.close()

        except (Exception, KeyboardInterrupt) as caught_exception:
            pathlib.Path(filename).unlink()
            self.logger.info(
                f"Caught error {caught_exception} and deleting"
                "partially created netCDF output "
                f"{filename} before re-raising error."
            )
            raise caught_exception

    def save_and_load_dem(
        self,
        filename: pathlib.Path,
    ):
        """Update the saved file cache for the DEM (self._dem) as a netCDF file."""

        self.logger.info(
            "In LidarBase.save_and_load_dem saving _dem as NetCDF file to "
            f"{filename}"
        )
        self.save_dem(filename=filename, dem=self._dem)
        del self._dem
        gc.collect()
        self._dem = self._load_dem(filename=filename)

    @staticmethod
    def _ensure_positive_indexing(
        dem: xarray.core.dataarray.DataArray,
    ) -> xarray.core.dataarray.DataArray:
        """A routine to check an xarray has positive dimension indexing and to reindex
        if needed."""

        x = dem.x
        y = dem.y
        if x[0] > x[-1]:
            x = x[::-1]
        if y[0] > y[-1]:
            y = y[::-1]
        dem = dem.reindex({"x": x, "y": y})
        dem.rio.write_transform(inplace=True)
        return dem

    @staticmethod
    def _write_netcdf_conventions_in_place(
        dem: xarray.core.dataarray.DataArray, crs_dict: dict
    ):
        """Write the CRS and transform associated with a netCDF file such that it is CF
        complient and meets the GDAL
        expectations for transform information.

        Parameters
        ----------

        dem
            The dataset to have its spatial data written in place.
        crs_dict
            A dict with horizontal and vertical CRS information.
        """

        dem.rio.write_crs(crs_dict["horizontal"], inplace=True)
        dem.rio.write_transform(inplace=True)
        for layer in ["z", "data_source", "lidar_source", "zo"]:
            if layer in dem:
                dem[layer] = dem[layer].rio.write_crs(crs_dict["horizontal"])
                dem[layer] = dem[layer].rio.write_nodata(numpy.nan, encoded=True)

    def _extents_from_mask(self, mask: numpy.ndarray, transform: dict):
        """Define the spatial extents of the pixels in the DEM as defined by the mask
        (i.e. what are the spatial extents of pixels in the DEM that are marked True in
         the mask).

        transform -> data_array.rio.transform()

         Remove holes as these can cause self intersection warnings."""

        dense_extents = [
            shapely.geometry.shape(polygon[0])
            for polygon in rasterio.features.shapes(numpy.uint8(mask))
            if polygon[1] == 1.0
        ]
        dense_extents = shapely.ops.unary_union(dense_extents)

        # Remove internal holes for select types as these may cause self-intersections
        if type(dense_extents) is shapely.geometry.Polygon:
            dense_extents = shapely.geometry.Polygon(dense_extents.exterior)
        elif type(dense_extents) is shapely.geometry.MultiPolygon:
            dense_extents = shapely.geometry.MultiPolygon(
                [
                    shapely.geometry.Polygon(polygon.exterior)
                    for polygon in dense_extents.geoms
                ]
            )
        # Convert into a Geopandas dataframe
        dense_extents = geopandas.GeoDataFrame(
            {"geometry": [dense_extents]},
            crs=self.catchment_geometry.crs["horizontal"],
        )

        # Move from image to the dem space & buffer(0) to reduce self-intersections
        dense_extents = dense_extents.affine_transform(
            [
                transform.a,
                transform.b,
                transform.d,
                transform.e,
                transform.xoff,
                transform.yoff,
            ]
        ).buffer(0)

        # And make our GeoSeries into a GeoDataFrame
        dense_extents = geopandas.GeoDataFrame(geometry=dense_extents)

        return dense_extents

    def _chunks_from_dem(self, chunk_size, dem: xarray.Dataset) -> (list, list):
        """Define the chunks to break the catchment into when reading in and
        downsampling LiDAR.

        Parameters
        ----------

        chunk_size
            The size in pixels of each chunk.
        """

        dim_x_all = dem.x.data
        dim_y_all = dem.y.data

        # Determine the number of chunks
        n_chunks_x = int(numpy.ceil(len(dim_x_all) / chunk_size))
        n_chunks_y = int(numpy.ceil(len(dim_y_all) / chunk_size))

        # Determine x coordinates rounded up to the nearest chunk
        dim_x = []
        for i in range(n_chunks_x):
            if i + 1 < n_chunks_x:
                # Add full sized chunk
                dim_x.append(dim_x_all[i * chunk_size : (i + 1) * chunk_size])
            else:
                # Add rest of array
                dim_x.append(dim_x_all[i * chunk_size :])

        # Determine y coordinates rounded up to the nearest chunk
        dim_y = []
        for i in range(n_chunks_y):
            if i + 1 < n_chunks_y:
                # Add full sized chunk
                dim_y.append(dim_y_all[i * chunk_size : (i + 1) * chunk_size])
            else:
                # Add rest of array
                dim_y.append(dim_y_all[i * chunk_size :])

        return dim_x, dim_y

    def _define_chunk_region(
        self,
        region_to_rasterise: geopandas.GeoDataFrame,
        dim_x: numpy.ndarray,
        dim_y: numpy.ndarray,
        radius: float,
    ):
        """Define the region to rasterise within a single chunk."""
        # Define the region to tile
        chunk_geometry = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.geometry.Polygon(
                        [
                            (dim_x.min(), dim_y.min()),
                            (dim_x.max(), dim_y.min()),
                            (dim_x.max(), dim_y.max()),
                            (dim_x.min(), dim_y.max()),
                        ]
                    )
                ]
            },
            crs=self.catchment_geometry.crs["horizontal"],
        )

        # Define region to rasterise inside the chunk area
        if region_to_rasterise.area.sum() > 0:
            chunk_region_to_tile = geopandas.GeoDataFrame(
                geometry=region_to_rasterise.buffer(radius).clip(
                    chunk_geometry.buffer(radius), keep_geom_type=True
                )
            )
        else:
            chunk_region_to_tile = region_to_rasterise
        # remove any subpixel polygons
        chunk_region_to_tile = chunk_region_to_tile[
            chunk_region_to_tile.area
            > self.catchment_geometry.resolution * self.catchment_geometry.resolution
        ]

        return chunk_region_to_tile


class HydrologicallyConditionedDem(DemBase):
    """A class to manage loading in an already created and saved dense DEM that has yet
    to have an offshore DEM associated with it.

    Parameters
    ----------

    Logic controlling behaviour
        interpolation_method
            If not None, interpolate using that method. Valid options are 'linear',
            'nearest', and 'cubic'
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        raw_dem_path: str | pathlib.Path,
        interpolation_method: str,
        chunk_size,
    ):
        """Load in the extents and dense DEM. Ensure the dense DEM is clipped within the
        extents"""
        super(HydrologicallyConditionedDem, self).__init__(
            catchment_geometry=catchment_geometry,
            chunk_size=chunk_size,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Read in the dense DEM raster - and free up file by performing a deep copy.
        raw_dem = rioxarray.rioxarray.open_rasterio(
            pathlib.Path(raw_dem_path), masked=True, parse_coordinates=True, chunks=True
        ).squeeze(
            "band", drop=True
        )  # remove band coordinate added by rasterio.open()
        self._write_netcdf_conventions_in_place(raw_dem, catchment_geometry.crs)

        # Clip to catchment and set the data_source layer to NaN where there is no data
        raw_dem = raw_dem.rio.clip_box(*tuple(catchment_geometry.catchment.total_bounds))
        raw_dem = raw_dem.where(
            clip_mask(raw_dem.z, catchment_geometry.catchment.geometry, self.chunk_size)
        )
        raw_dem["data_source"] = raw_dem.data_source.where(
            raw_dem.data_source != self.SOURCE_CLASSIFICATION["no data"],
            numpy.nan,
        )
        # Rerun as otherwise the no data as NaN seems to be lost for the data_source layer
        self._write_netcdf_conventions_in_place(raw_dem, catchment_geometry.crs)

        # Set attributes
        self._raw_dem = raw_dem
        self.interpolation_method = interpolation_method

        # Calculate extents of pre-hydrological conditioning DEM
        self._raw_extents = self._extents_from_mask(
            mask=self._raw_dem.z.notnull().values,
            transform=self._raw_dem.z.rio.transform(),
        )

        # The not yet created hydrologically conditioned DEM.
        self._dem = self._raw_dem

    def __del__(self):
        """Ensure the memory associated with netCDF files is properly freed."""

        # The dense DEM - may be opened from memory
        if self._raw_dem is not None:
            self._raw_dem.close()
            del self._raw_dem
        # The overall DEM
        if self._dem is not None:
            self._dem.close()
            del self._dem

    @property
    def raw_extents(self):
        """Return the combined DEM from tiles and any interpolated offshore values"""
        return self._raw_extents

    @property
    def dem(self):
        """Return the combined DEM from tiles and any interpolated offshore values"""
        self._write_netcdf_conventions_in_place(self._dem, self.catchment_geometry.crs)
        # Ensure valid name and increasing dimension indexing for the dem
        if (
            self.interpolation_method is not None
        ):  # methods are 'nearest', 'linear' and 'cubic'
            interpolation_mask = self._dem.z.isnull()
            self._dem["z"] = self._dem.z.rio.interpolate_na(
                method=self.interpolation_method
            )
            # If any NaN remain apply nearest neighbour interpolation
            if self._dem.z.isnull().any():
                self._dem["z"] = self._dem.z.rio.interpolate_na(method="nearest")
            # Only set areas with successful interpolation as interpolated
            interpolation_mask &= (
                self._dem.z.notnull()
            )  # Mask of values set in line above
            self._dem["data_source"] = self._dem.data_source.where(
                ~(interpolation_mask),
                self.SOURCE_CLASSIFICATION["interpolated"],
            )
            self._dem["lidar_source"] = self._dem.lidar_source.where(
                ~(interpolation_mask),
                self.SOURCE_CLASSIFICATION["no data"],
            )
        # Ensure all area's with NaN values are marked as no-data
        no_data_mask = self._dem.z.notnull()
        self._dem["data_source"] = self._dem.data_source.where(
            no_data_mask,
            self.SOURCE_CLASSIFICATION["no data"],
        )
        self._dem["lidar_source"] = self._dem.lidar_source.where(
            no_data_mask,
            self.SOURCE_CLASSIFICATION["no data"],
        )
        self._dem = self._dem.rio.clip(
            self.catchment_geometry.catchment.geometry, drop=True
        )
        # Some programs require positively increasing indices
        # Last as otherwise errors when merging (clipping resets defaults)
        self._dem = self._ensure_positive_indexing(self._dem)
        return self._dem

    def _sample_offshore_edge(self, resolution) -> numpy.ndarray:
        """Return the pixel values of the offshore edge to be used for offshore
        interpolation"""

        assert resolution >= self.catchment_geometry.resolution, (
            "_sample_offshore_edge only supports downsampling"
            f" and not  up-samping. The requested sampling resolution of {resolution} "
            "must be equal to or larger than the catchment resolution of "
            f" {self.catchment_geometry.resolution}"
        )

        offshore_dense_data_edge = self.catchment_geometry.offshore_dense_data_edge(
            self._raw_extents
        )
        if offshore_dense_data_edge.area.sum() == 0:
            # No offshore edge. Return an empty array.
            offshore_edge = numpy.empty(
                [0],
                dtype=[
                    ("X", geometry.RASTER_TYPE),
                    ("Y", geometry.RASTER_TYPE),
                    ("Z", geometry.RASTER_TYPE),
                ],
            )
            return offshore_edge
        # Otherwise proceed as normal
        offshore_edge_dem = self._raw_dem.rio.clip(offshore_dense_data_edge.geometry)

        # If the sampling resolution is coaser than the catchment_geometry resolution
        # resample the DEM - Align to the resolution (not the BBox).
        if resolution > self.catchment_geometry.resolution:
            x = numpy.arange(
                numpy.ceil(offshore_edge_dem.x.min() / resolution) * resolution,
                numpy.ceil(offshore_edge_dem.x.max() / resolution) * resolution,
                resolution,
            )
            y = numpy.arange(
                numpy.ceil(offshore_edge_dem.y.max() / resolution) * resolution,
                numpy.ceil(offshore_edge_dem.y.min() / resolution) * resolution,
                -resolution,
            )
            offshore_edge_dem = offshore_edge_dem.interp(x=x, y=y, method="nearest")
            offshore_edge_dem = offshore_edge_dem.rio.clip(
                offshore_dense_data_edge.geometry
            )  # Reclip to inbounds
        grid_x, grid_y = numpy.meshgrid(offshore_edge_dem.x, offshore_edge_dem.y)
        mask = offshore_edge_dem.z.notnull().values

        offshore_edge = numpy.empty(
            [mask.sum().sum()],
            dtype=[
                ("X", geometry.RASTER_TYPE),
                ("Y", geometry.RASTER_TYPE),
                ("Z", geometry.RASTER_TYPE),
            ],
        )

        offshore_edge["X"] = grid_x[mask]
        offshore_edge["Y"] = grid_y[mask]
        offshore_edge["Z"] = offshore_edge_dem.z.values[mask]

        return offshore_edge

    def interpolate_ocean_chunked(
        self,
        ocean_points,
        cache_path: pathlib.Path,
        k_nearest_neighbours: int,
        use_edge: bool,
        buffer: int,
        method: str,
    ) -> xarray.Dataset:
        """Create a 'raw'' DEM from a set of tiled LiDAR files. Read these in over
        non-overlapping chunks and then combine"""

        crs = self.catchment_geometry.crs
        raster_options = {
            "raster_type": geometry.RASTER_TYPE,
            "elevation_range": None,
            "k_nearest_neighbours": k_nearest_neighbours,
            "method": method,
            "crs": crs,
            "use_edge": use_edge,
            "strict": True,
        }
        if method == "rbf":
            raster_options["kernel"] = "thin_plate_spline"
        if use_edge:
            # Save point cloud as LAZ file
            offshore_edge_points = self._sample_offshore_edge(
                self.catchment_geometry.resolution
            )
            # Remove any ocean points on land and within the buffered distance
            # of the foreshore to avoid any sharp changes that may cause jumps
            offshore_region = ocean_points.boundary
            buffer_radius = self.catchment_geometry.resolution * buffer
            offshore_region = offshore_region.overlay(
                self.catchment_geometry.land_and_foreshore.buffer(
                    buffer_radius
                ).to_frame("geometry"),
                how="difference",
                keep_geom_type=True,
            )
            ocean_points._points = ocean_points._points.clip(
                offshore_region, keep_geom_type=True
            )
        offshore_points = ocean_points.sample()
        if len(offshore_points) < k_nearest_neighbours:
            self.logger.warning(
                f"Fewer ocean points ({len(offshore_points)}) than k_nearest_neighbours "
                f"{k_nearest_neighbours}. Skip offshore interpolation."
            )
            return
        if use_edge and len(offshore_edge_points) < k_nearest_neighbours:
            self.logger.warning(
                f"Fewer edge points ({len(offshore_edge_points)}) than "
                f"k_nearest_neighbours {k_nearest_neighbours}. Skip offshore interpolation."
            )
            return

        # Save offshore points in a temporary laz file
        offshore_file = cache_path / "offshore_points.laz"
        pdal_pipeline_instructions = [
            {
                "type": "writers.las",
                "a_srs": f"EPSG:" f"{crs['horizontal']}+" f"{crs['vertical']}",
                "filename": str(offshore_file),
                "compression": "laszip",
            }
        ]
        pdal_pipeline = pdal.Pipeline(
            json.dumps(pdal_pipeline_instructions), [offshore_points]
        )
        pdal_pipeline.execute()
        if use_edge:
            # Save edge points in a temporary laz file
            coast_edge_file = cache_path / "coast_edge_points.laz"
            pdal_pipeline_instructions = [
                {
                    "type": "writers.las",
                    "a_srs": f"EPSG:" f"{crs['horizontal']}+" f"{crs['vertical']}",
                    "filename": str(coast_edge_file),
                    "compression": "laszip",
                }
            ]
            pdal_pipeline = pdal.Pipeline(
                json.dumps(pdal_pipeline_instructions), [offshore_edge_points]
            )
            pdal_pipeline.execute()

        assert self.chunk_size is not None, "chunk_size must be defined"

        # get chunking information
        chunked_dim_x, chunked_dim_y = self._chunks_from_dem(self.chunk_size, self._dem)
        elevations = {}

        self.logger.info(f"Preparing {[len(chunked_dim_x), len(chunked_dim_y)]} chunks")

        # Define the region to rasterise
        region_to_rasterise = self.catchment_geometry.offshore_no_dense_data(
            self._raw_extents
        )

        # cycle through index chunks - and collect in a delayed array
        self.logger.info("Running over ocean chunked")
        delayed_chunked_matrix = []
        for i, dim_y in enumerate(chunked_dim_y):
            delayed_chunked_x = []
            for j, dim_x in enumerate(chunked_dim_x):
                self.logger.debug(f"\tLiDAR chunk {[i, j]}")
                # Check ROI to tile
                chunk_region_to_tile = self._define_chunk_region(
                    region_to_rasterise=region_to_rasterise,
                    dim_x=dim_x,
                    dim_y=dim_y,
                    radius=0,
                )
                if chunk_region_to_tile.area.sum() == 0:
                    self.logger.debug(f"\t\tReturning empty tile as out of RIO")
                    delayed_chunked_x.append(
                        dask.array.full(
                            shape=(len(dim_y), len(dim_x)),
                            fill_value=numpy.nan,
                            dtype=raster_options["raster_type"],
                        )
                    )
                    continue
                # Load in points
                chunk_offshore_points = delayed_load_tiles_in_chunk(
                    lidar_files=[offshore_file],
                    source_crs=raster_options["crs"],
                    chunk_region_to_tile=None,
                    crs=raster_options["crs"],
                )
                if use_edge:
                    chunk_coast_edge_points = delayed_load_tiles_in_chunk(
                        lidar_files=[coast_edge_file],
                        source_crs=raster_options["crs"],
                        chunk_region_to_tile=None,
                        crs=raster_options["crs"],
                    )
                else:
                    chunk_coast_edge_points = None
                # Rasterise tiles
                delayed_chunked_x.append(
                    dask.array.from_delayed(
                        delayed_elevation_over_chunk_from_nearest(
                            dim_x=dim_x,
                            dim_y=dim_y,
                            points=chunk_offshore_points,
                            edge_points=chunk_coast_edge_points,
                            options=raster_options,
                        ),
                        shape=(len(dim_y), len(dim_x)),
                        dtype=raster_options["raster_type"],
                    )
                )
            delayed_chunked_matrix.append(delayed_chunked_x)

        # Combine chunks into a dataset
        elevations = dask.array.block(delayed_chunked_matrix)

        # Update DEM layers - copy only where no offshore data
        no_values_mask = self._dem.z.isnull() & clip_mask(
            self._dem.z,
            region_to_rasterise.geometry,
            self.chunk_size,
        )
        no_values_mask.load()
        self._dem["z"] = self._dem.z.where(~no_values_mask, elevations)
        mask = ~(no_values_mask & self._dem.z.notnull())
        self._dem["data_source"] = self._dem.data_source.where(
            mask,
            self.SOURCE_CLASSIFICATION["ocean bathymetry"],
        )
        self._dem["lidar_source"] = self._dem.lidar_source.where(
            mask, self.SOURCE_CLASSIFICATION["no data"]
        )
        self._write_netcdf_conventions_in_place(self._dem, self.catchment_geometry.crs)

    def _interpolate_elevation_points(
        self,
        point_cloud: numpy.ndarray,
        flat_x_array: numpy.ndarray,
        flat_y_array: numpy.ndarray,
        method: str,
    ) -> numpy.ndarray:
        """Interpolate the elevation points at the specified locations using the
        specified method."""

        if method == "rbf":
            # Ensure the number of points is not too great for RBF interpolation
            if len(point_cloud) < RBF_CACHE_SIZE:
                self.logger.warning(
                    "The number of points to fit and RBF interpolant to is"
                    f" {len(point_cloud)}. We recommend using fewer "
                    f" than {RBF_CACHE_SIZE} for best performance and to. "
                    "avoid errors in the `scipy.interpolate.Rbf` function"
                )
            # Create RBF function
            self.logger.info("Creating RBF interpolant")
            rbf_function = scipy.interpolate.Rbf(
                point_cloud["X"],
                point_cloud["Y"],
                point_cloud["Z"],
                function="linear",
            )
            # Tile area - this limits the maximum memory required at any one time
            flat_z_array = numpy.ones_like(flat_x_array) * numpy.nan
            number_offshore_tiles = math.ceil(len(flat_x_array) / RBF_CACHE_SIZE)
            for i in range(number_offshore_tiles):
                self.logger.info(
                    f"Offshore intepolant tile {i+1} of {number_offshore_tiles}"
                )
                start_index = int(i * RBF_CACHE_SIZE)
                end_index = (
                    int((i + 1) * RBF_CACHE_SIZE)
                    if i + 1 != number_offshore_tiles
                    else len(flat_x_array)
                )

                flat_z_array[start_index:end_index] = rbf_function(
                    flat_x_array[start_index:end_index],
                    flat_y_array[start_index:end_index],
                )
        elif method == "linear" or method == "cubic" or method == "nearest":
            # Interpolate river area - use cubic or linear interpolation
            flat_z_array = scipy.interpolate.griddata(
                points=(point_cloud["X"], point_cloud["Y"]),
                values=point_cloud["Z"],
                xi=(flat_x_array, flat_y_array),
                method=method,  # linear or cubic
            )
        else:
            raise ValueError("method must be rbf, nearest, linear or cubic")
        return flat_z_array

    def interpolate_ocean_bathymetry(self, bathy_contours, method="linear"):
        """Performs interpolation offshore outside LiDAR extents using the SciPy RBF
        function."""

        # Reset the offshore DEM

        offshore_edge_points = self._sample_offshore_edge(
            self.catchment_geometry.resolution
        )
        bathy_points = bathy_contours.sample_contours(
            self.catchment_geometry.resolution
        )
        if len(bathy_points) == 0:
            offshore_points = offshore_edge_points
        else:
            offshore_points = numpy.concatenate([offshore_edge_points, bathy_points])

        # Resample at a lower resolution if too many offshore points
        if method == "rbf" and len(offshore_points) > RBF_CACHE_SIZE * 10:
            reduced_resolution = (
                self.catchment_geometry.resolution
                * len(offshore_points)
                / RBF_CACHE_SIZE
                * 10
            )
            self.logger.info(
                "Reducing the number of 'offshore_points' used to create the RBF "
                "function by increasing the resolution from "
                f" {self.catchment_geometry.resolution} to {reduced_resolution}"
            )
            offshore_edge_points = self._sample_offshore_edge(reduced_resolution)
            bathy_points = bathy_contours.sample_contours(reduced_resolution)
            if len(bathy_points) == 0:
                offshore_points = offshore_edge_points
            else:
                offshore_points = numpy.concatenate(
                    [offshore_edge_points, bathy_points]
                )
        # Setup the empty offshore area ready for interpolation
        offshore_no_dense_data = self.catchment_geometry.offshore_no_dense_data(
            self._raw_extents
        )
        offshore_dem = self._raw_dem.rio.clip(self.catchment_geometry.offshore.geometry)

        # set all zero (or to ocean bathy classification) then clip out dense region
        # where we don't need to interpolate
        offshore_dem.z.data[:] = 0
        offshore_dem.data_source.data[:] = self.SOURCE_CLASSIFICATION[
            "ocean bathymetry"
        ]
        offshore_dem.lidar_source.data[:] = self.SOURCE_CLASSIFICATION["no data"]
        offshore_dem = offshore_dem.rio.clip(offshore_no_dense_data.geometry)

        grid_x, grid_y = numpy.meshgrid(offshore_dem.x, offshore_dem.y)
        mask = offshore_dem.z.notnull().values

        # Set up the interpolation function
        self.logger.info("Offshore interpolation")
        flat_z_masked = self._interpolate_elevation_points(
            point_cloud=offshore_points,
            flat_x_array=grid_x[mask],
            flat_y_array=grid_y[mask],
            method=method,
        )
        flat_z = offshore_dem.z.values.flatten()
        flat_z[mask.flatten()] = flat_z_masked
        offshore_dem.z.data = flat_z.reshape(offshore_dem.z.shape)

        self._dem = rioxarray.merge.merge_datasets(
            [self._raw_dem, offshore_dem],
            method="first",
        )

    def clip_within_polygon(self, polygon_paths: list, label: str):
        """Clip existing DEM to remove areas within the polygons"""
        crs = self.catchment_geometry.crs
        dem_bounds = geopandas.GeoDataFrame(
            geometry=[shapely.geometry.box(*self._dem.rio.bounds())],
            crs=crs["horizontal"],
        )
        clip_polygon = []
        for path in polygon_paths:
            clip_polygon.append(geopandas.read_file(path).to_crs(crs["horizontal"]))
        clip_polygon = pandas.concat(clip_polygon).dissolve()
        clip_polygon = clip_polygon.clip(dem_bounds)
        if clip_polygon.area.sum() > self.catchment_geometry.resolution**2:
            self.logger.info(
                f"Clipping to remove all features in polygons {polygon_paths}"
            )
            mask = clip_mask(
                arr=self._dem.z,
                geometry=clip_polygon.geometry,
                chunk_size=self.chunk_size,
            )
            self._dem["z"] = self._dem.z.where(
                ~mask,
                numpy.nan,
            )
            self._dem["data_source"] = self._dem.data_source.where(
                ~mask,
                self.SOURCE_CLASSIFICATION[label],
            )
            self._dem["lidar_source"] = self._dem.lidar_source.where(
                ~mask, self.SOURCE_CLASSIFICATION["no data"]
            )
            self._write_netcdf_conventions_in_place(
                self._dem, self.catchment_geometry.crs
            )
        else:
            self.logger.warning(
                f"No clipping. Polygons {polygon_paths} do not overlap DEM."
            )

    def add_points_within_polygon_chunked(
        self,
        elevations: geometry.ElevationPoints,
        method: str,
        cache_path: pathlib.Path,
        label: str,
        include_edges: bool = True,
    ) -> xarray.Dataset:
        """Performs interpolation from estimated bathymetry points within a polygon
        using the specified interpolation approach after filtering the points based
        on the type label. The type_label also determines the source classification.
        """

        crs = self.catchment_geometry.crs
        raster_options = {
            "raster_type": geometry.RASTER_TYPE,
            "elevation_range": None,
            "method": method,
            "crs": crs,
            "radius": elevations.points["width"].max()
            + 2 * self.catchment_geometry.resolution,
            "strict": False,
        }
        if method == "rbf":
            raster_options["kernel"] = "linear"

        # Define the region to rasterise
        region_to_rasterise = elevations.polygons

        # Extract river elevations
        point_cloud = elevations.points_array

        if include_edges:
            # Get edge points - from DEM
            edge_roi = region_to_rasterise.dissolve().buffer(self.catchment_geometry.resolution)
            edge_dem = self._dem.rio.clip_box(*tuple(edge_roi.total_bounds))
            edge_dem = edge_dem.rio.clip(edge_roi)
            edge_dem = edge_dem.rio.clip(
                region_to_rasterise.dissolve().geometry,
                invert=True,
            )
            # Define the edge points
            grid_x, grid_y = numpy.meshgrid(edge_dem.x, edge_dem.y)
            mask = edge_dem.z.notnull().values
            # Define edge points and heights
            edge_points = numpy.empty(
                [mask.sum().sum()],
                dtype=[
                    ("X", geometry.RASTER_TYPE),
                    ("Y", geometry.RASTER_TYPE),
                    ("Z", geometry.RASTER_TYPE),
                ],
            )
            edge_points["X"] = grid_x[mask]
            edge_points["Y"] = grid_y[mask]
            edge_points["Z"] = edge_dem.z.values[mask]

            # Combine the estimated and edge points
            point_cloud = numpy.concatenate([edge_points, point_cloud])

        # Save river points in a temporary laz file
        lidar_file = cache_path / f"{label}_points.laz"
        pdal_pipeline_instructions = [
            {
                "type": "writers.las",
                "a_srs": f"EPSG:" f"{crs['horizontal']}+" f"{crs['vertical']}",
                "filename": str(lidar_file),
                "compression": "laszip",
            }
        ]
        pdal_pipeline = pdal.Pipeline(
            json.dumps(pdal_pipeline_instructions), [point_cloud]
        )
        pdal_pipeline.execute()

        if self.chunk_size is None:
            logging.warning("Chunksize of none. set to DEM shape.")
            self.chunk_size = max(len(self._dem.x), len(self._dem.y))

        # get chunking information
        chunked_dim_x, chunked_dim_y = self._chunks_from_dem(self.chunk_size, self._dem)
        elevations = {}

        self.logger.info(f"Preparing {[len(chunked_dim_x), len(chunked_dim_y)]} chunks")

        # cycle through index chunks - and collect in a delayed array
        self.logger.info(f"Running over {label} chunked")
        delayed_chunked_matrix = []
        for i, dim_y in enumerate(chunked_dim_y):
            delayed_chunked_x = []
            for j, dim_x in enumerate(chunked_dim_x):
                self.logger.debug(f"\tLiDAR chunk {[i, j]}")
                # Check ROI to tile
                chunk_region_to_tile = self._define_chunk_region(
                    region_to_rasterise=region_to_rasterise,
                    dim_x=dim_x,
                    dim_y=dim_y,
                    radius=0,
                )
                if chunk_region_to_tile.area.sum() == 0:
                    self.logger.debug(f"\t\tReturning empty tile as out of RIO")
                    delayed_chunked_x.append(
                        dask.array.full(
                            shape=(len(dim_y), len(dim_x)),
                            fill_value=numpy.nan,
                            dtype=raster_options["raster_type"],
                        )
                    )
                    continue

                # Load in points
                river_points = delayed_load_tiles_in_chunk(
                    lidar_files=[lidar_file],
                    source_crs=raster_options["crs"],
                    chunk_region_to_tile=None,
                    crs=raster_options["crs"],
                )

                # Rasterise tiles
                delayed_chunked_x.append(
                    dask.array.from_delayed(
                        delayed_elevation_over_chunk(
                            dim_x=dim_x,
                            dim_y=dim_y,
                            tile_points=river_points,
                            options=raster_options,
                        ),
                        shape=(len(dim_y), len(dim_x)),
                        dtype=raster_options["raster_type"],
                    )
                )
            delayed_chunked_matrix.append(delayed_chunked_x)

        # Combine chunks into a dataset
        elevations = dask.array.block(delayed_chunked_matrix)

        # Update DEM layers - copy everyhere within the region to rasterise
        region_mask = clip_mask(
            self._dem.z,
            region_to_rasterise.geometry,
            self.chunk_size,
        )
        region_mask.load()
        self._dem["z"] = self._dem.z.where(~region_mask, elevations)
        mask = ~(region_mask & self._dem.z.notnull())
        self._dem["data_source"] = self._dem.data_source.where(
            mask,
            self.SOURCE_CLASSIFICATION[label],
        )
        self._dem["lidar_source"] = self._dem.lidar_source.where(
            mask, self.SOURCE_CLASSIFICATION["no data"]
        )
        self._write_netcdf_conventions_in_place(self._dem, self.catchment_geometry.crs)

    def add_points_within_polygon_nearest_chunked(
        self,
        elevations: geometry.ElevationPoints,
        method: str,
        cache_path: pathlib.Path,
        label: str,
        k_nearest_neighbours: int,
        include_edges: bool = True,
    ) -> xarray.Dataset:
        """Performs interpolation from estimated bathymetry points within a polygon
        using the specified interpolation approach after filtering the points based
        on the type label. The type_label also determines the source classification.
        """

        crs = self.catchment_geometry.crs
        raster_options = {
            "raster_type": geometry.RASTER_TYPE,
            "elevation_range": None,
            "method": method,
            "crs": crs,
            "k_nearest_neighbours": k_nearest_neighbours,
            "use_edge": include_edges,
            "strict": False,
        }
        if method == "rbf":
            raster_options["kernel"] = "linear"
        # Define the region to rasterise
        region_to_rasterise = elevations.polygons

        # Tempoarily save the points to add
        points = elevations.points_array
        points_file = cache_path / f"{label}_points.laz"
        pdal_pipeline_instructions = [
            {
                "type": "writers.las",
                "a_srs": f"EPSG:" f"{crs['horizontal']}+" f"{crs['vertical']}",
                "filename": str(points_file),
                "compression": "laszip",
            }
        ]
        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [points])
        pdal_pipeline.execute()

        # Tempoarily save the adjacent points from the DEM - ensure no NaN through NN interpolation
        if include_edges:
            edge_roi = region_to_rasterise.dissolve().buffer(
                    self.catchment_geometry.resolution
                )
            edge_dem = self._dem.rio.clip_box(*tuple(edge_roi.total_bounds))
            #edge_dem = edge_dem.rio.clip(edge_roi)
            self._write_netcdf_conventions_in_place(
                edge_dem, self.catchment_geometry.crs
            )
            edge_dem["z"] = edge_dem.z.rio.interpolate_na(method="nearest")
            edge_dem = edge_dem.rio.clip(edge_roi, drop=True, )
            edge_dem = edge_dem.rio.clip(
                region_to_rasterise.dissolve().geometry,
                invert=True,
                drop=True,
            )

            # Save provided points
            grid_x, grid_y = numpy.meshgrid(edge_dem.x, edge_dem.y)
            flat_x = grid_x.flatten()
            flat_y = grid_y.flatten()
            flat_z = edge_dem.z.values.flatten()
            mask_z = ~numpy.isnan(flat_z)

            # Interpolate the estimated bank heights around the polygon if they exist
            if elevations.bank_heights_exist():
                # Get the estimated  bank heights and define a mask where nan
                bank_points = elevations.bank_height_points()
                bank_nan_mask = numpy.logical_not(numpy.isnan(bank_points["Z"]))
                # Interpolate from the estimated bank heights
                xy_out = numpy.concatenate(
                    [[flat_x[mask_z]], [flat_y[mask_z]]], axis=0
                ).transpose()
                options = {
                    "radius": elevations.points["width"].max(),
                    "raster_type": geometry.RASTER_TYPE,
                    "method": "linear",
                    "strict": False,
                }
                estimated_edge_z = elevation_from_points(
                    point_cloud=bank_points[bank_nan_mask],
                    xy_out=xy_out,
                    options=options,
                )

                # Use the estimated bank heights where lower than the DEM edge values
                mask_z_edge = mask_z.copy()
                mask_z_edge[:] = False
                mask_z_edge[mask_z] = flat_z[mask_z] > estimated_edge_z
                flat_z[mask_z_edge] = estimated_edge_z[
                    flat_z[mask_z] > estimated_edge_z
                ]

            # Use the flat_x/y/z to define edge points and heights
            edge_points = numpy.empty(
                [mask_z.sum().sum()],
                dtype=[
                    ("X", geometry.RASTER_TYPE),
                    ("Y", geometry.RASTER_TYPE),
                    ("Z", geometry.RASTER_TYPE),
                ],
            )
            edge_points["X"] = flat_x[mask_z]
            edge_points["Y"] = flat_y[mask_z]
            edge_points["Z"] = flat_z[mask_z]

            edge_file = cache_path / f"{label}_edge_points.laz"
            pdal_pipeline_instructions = [
                {
                    "type": "writers.las",
                    "a_srs": f"EPSG:" f"{crs['horizontal']}+" f"{crs['vertical']}",
                    "filename": str(edge_file),
                    "compression": "laszip",
                }
            ]
            pdal_pipeline = pdal.Pipeline(
                json.dumps(pdal_pipeline_instructions), [edge_points]
            )
            pdal_pipeline.execute()

        if len(points) < raster_options["k_nearest_neighbours"] or (
            include_edges and len(edge_points) < raster_options["k_nearest_neighbours"]
        ):
            k_nearest_neighbours = (
                min(len(points), len(edge_points)) if include_edges else len(points)
            )
            logging.info(
                f"Fewer points or edge points than the default expected {raster_options['k_nearest_neighbours']}. "
                f"Updating k_nearest_neighbours to {k_nearest_neighbours}."
            )
            raster_options["k_nearest_neighbours"] = k_nearest_neighbours
        if raster_options["k_nearest_neighbours"] < 3:
            logging.warning(
                f"Not enough points or edge points to meaningfully include {raster_options['k_nearest_neighbours']}. "
                f"Exiting without including the points and edge points."
            )
            return

        if self.chunk_size is None:
            logging.warning("Chunksize of none. set to DEM shape.")
            self.chunk_size = max(len(self._dem.x), len(self._dem.y))

        # get chunking information
        chunked_dim_x, chunked_dim_y = self._chunks_from_dem(self.chunk_size, self._dem)
        elevations = {}

        self.logger.info(f"Preparing {[len(chunked_dim_x), len(chunked_dim_y)]} chunks")

        # cycle through index chunks - and collect in a delayed array
        self.logger.info(
            "Running over points chunked - nearest of points & edge points"
        )
        delayed_chunked_matrix = []
        for i, dim_y in enumerate(chunked_dim_y):
            delayed_chunked_x = []
            for j, dim_x in enumerate(chunked_dim_x):
                self.logger.debug(f"\tLiDAR chunk {[i, j]}")
                # Check ROI to tile
                chunk_region_to_tile = self._define_chunk_region(
                    region_to_rasterise=region_to_rasterise,
                    dim_x=dim_x,
                    dim_y=dim_y,
                    radius=0,
                )
                if chunk_region_to_tile.area.sum() == 0:
                    self.logger.debug(f"\t\tReturning empty tile as out of RIO")
                    delayed_chunked_x.append(
                        dask.array.full(
                            shape=(len(dim_y), len(dim_x)),
                            fill_value=numpy.nan,
                            dtype=raster_options["raster_type"],
                        )
                    )
                    continue

                # Load in points
                points = delayed_load_tiles_in_chunk(
                    lidar_files=[points_file],
                    source_crs=raster_options["crs"],
                    chunk_region_to_tile=None,
                    crs=raster_options["crs"],
                )
                if include_edges:
                    edge_points = delayed_load_tiles_in_chunk(
                        lidar_files=[edge_file],
                        source_crs=raster_options["crs"],
                        chunk_region_to_tile=None,
                        crs=raster_options["crs"],
                    )
                else:
                    edge_points = None

                # Rasterise tiles
                delayed_chunked_x.append(
                    dask.array.from_delayed(
                        delayed_elevation_over_chunk_from_nearest(
                            dim_x=dim_x,
                            dim_y=dim_y,
                            points=points,
                            edge_points=edge_points,
                            options=raster_options,
                        ),
                        shape=(len(dim_y), len(dim_x)),
                        dtype=raster_options["raster_type"],
                    )
                )
            delayed_chunked_matrix.append(delayed_chunked_x)

        # Combine chunks into a dataset
        elevations = dask.array.block(delayed_chunked_matrix)

        # Update DEM layers - copy everyhere within the region to rasterise
        polygon_mask = clip_mask(
            self._dem.z,
            region_to_rasterise.geometry,
            self.chunk_size,
        )
        polygon_mask.load()
        self._dem["z"] = self._dem.z.where(~polygon_mask, elevations)
        mask = ~(polygon_mask & self._dem.z.notnull())
        self._dem["data_source"] = self._dem.data_source.where(
            mask,
            self.SOURCE_CLASSIFICATION[label],
        )
        self._dem["lidar_source"] = self._dem.lidar_source.where(
            mask, self.SOURCE_CLASSIFICATION["no data"]
        )
        self._write_netcdf_conventions_in_place(self._dem, self.catchment_geometry.crs)


class LidarBase(DemBase):
    """A class with some base methods for reading in LiDAR data.

    Parameters
    ----------

    catchment_geometry
        Defines the geometry of the catchment
    elevation_range
        The range of valid LiDAR elevations. i.e. define elevation filtering to apply.
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        chunk_size: int,
        elevation_range: list = None,
    ):
        """Setup base DEM to add future tiles too"""

        super(LidarBase, self).__init__(
            catchment_geometry=catchment_geometry,
            chunk_size=chunk_size,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.elevation_range = elevation_range
        assert elevation_range is None or (
            type(elevation_range) is list and len(elevation_range) == 2
        ), "Error the 'elevation_range' must either be none, or a two entry list"

        self._dem = None

    def __del__(self):
        """Ensure the memory associated with netCDF files is properly freed."""

        # The dense DEM - may be opened from memory
        if self._dem is not None:
            self._dem.close()
            del self._dem

    @property
    def dem(self):
        """Return the positivly indexed DEM from tiles"""

        # Ensure positively increasing indices as required by some programs
        self._dem = self._ensure_positive_indexing(self._dem)
        return self._dem

    def _tile_index_column_name(
        self,
        tile_index_file: str | pathlib.Path,
        region_to_rasterise: geopandas.GeoDataFrame,
    ):
        """Read in LiDAR tile index file and determine the column name of the
        tile geometries"""
        # Check to see if a extents file was added
        tile_index_extents = geopandas.read_file(tile_index_file)

        # Remove tiles outside the catchment & get the 'file name' column
        tile_index_extents = tile_index_extents.to_crs(
            self.catchment_geometry.crs["horizontal"]
        )
        # ensure there is no overlap in the name columns
        region_to_rasterise = region_to_rasterise.copy(deep=True)
        if "filename" in region_to_rasterise.columns:
            region_to_rasterise = region_to_rasterise.drop(columns=["filename"])
        elif "file_name" in region_to_rasterise.columns:
            region_to_rasterise = region_to_rasterise.drop(columns=["file_name"])
        elif "name" in region_to_rasterise.columns:
            region_to_rasterise = region_to_rasterise.drop(columns=["name"])
        tile_index_extents = geopandas.sjoin(tile_index_extents, region_to_rasterise)
        tile_index_extents = tile_index_extents.reset_index(drop=True)

        column_names = tile_index_extents.columns
        tile_index_name_column = column_names[
            [
                "filename" == name.lower()
                or "file_name" == name.lower()
                or "name" == name.lower()
                for name in column_names
            ]
        ][0]
        return tile_index_extents, tile_index_name_column

    def _check_valid_inputs(self, lidar_datasets_info):
        """Check the combination of inputs for adding LiDAR is valid.

        Parameters
        ----------

        lidar_datasets_info
            A dictionary of dictionaties of LiDAR dataset information. The CRS, list of
            LAS files, and tile index file are included for each dataset.
        """
        empty_datasets = []
        for dataset_name in lidar_datasets_info:
            # Check the source_crs is valid
            source_crs = lidar_datasets_info[dataset_name]["crs"]
            if source_crs is not None:
                assert "horizontal" in source_crs, (
                    "The horizontal component of the source CRS is not specified. Both "
                    "horizontal and vertical CRS need to be defined. The source_crs "
                    f"specified is: {source_crs} for {dataset_name}"
                )
                assert "vertical" in source_crs, (
                    "The vertical component of the source CRS is not specified. Both "
                    "horizontal and vertical CRS need to be defined. The source_crs "
                    f"specified is: {self.source_crs} for {dataset_name}"
                )
            # Check some LiDAR files are specified
            lidar_files = lidar_datasets_info[dataset_name]["file_paths"]
            if len(lidar_files) == 0:
                self.logger.warning(
                    f"Ignoring LiDAR dataset {dataset_name} as there are no LiDAR files within the ROI."
                )
                empty_datasets.append(dataset_name)
                continue
            # Check for valid combination of chunk_size, lidar_files and tile_index_file
            if self.chunk_size is None:
                assert len(lidar_files) == 1, (
                    "If there is no chunking there must be only one LiDAR file. This "
                    f"isn't the case in dataset {dataset_name}"
                )
            else:
                assert (
                    self.chunk_size > 0 and type(self.chunk_size) is int
                ), "chunk_size must be a positive integer"
                tile_index_file = lidar_datasets_info[dataset_name]["tile_index_file"]
                assert tile_index_file is not None, (
                    "A tile index file must be provided if chunking is "
                    f"defined for {dataset_name}"
                )
        # Re move a dataset if no LIDAR
        if len(empty_datasets) > 0:
            for empty_dataset in empty_datasets:
                lidar_datasets_info.pop(empty_dataset)
        # There should only be one dataset if there is no chunking information
        if self.chunk_size is None:
            assert len(lidar_datasets_info) == 1, (
                "If there is no chunking there must only be one LiDAR file."
                f" Instead there is {len(lidar_datasets_info)} "
                f"with keys f{lidar_datasets_info.keys()}"
            )

    def add_lidar(
        self,
        lidar_datasets_info: dict,
        chunk_size: int,
        lidar_classifications_to_keep: list,
        metadata: dict,
    ):
        """Read in all LiDAR files and use to create a 'raw' DEM.

        Parameters
        ----------

        lidar_datasets_info
            A dictionary of information for each specified LIDAR dataset - For
            each this includes: a list of LAS files, CRS, and tile index file.
        chunk_size
            The chunk size in pixels for parallel/staged processing
        lidar_classifications_to_keep
            A list of LiDAR classifications to keep - '2' for ground, '9' for water.
            See https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf for
            standard list
        meta_data
            Information to include in the created DEM - must include
            `dataset_mapping` key if datasets (not a single LAZ file) included.
        """
        raise NotImplementedError("add_lidar must be instantiated in the child class")

    def _add_tiled_lidar_chunked(
        self,
        lidar_files: typing.List[typing.Union[str, pathlib.Path]],
        tile_index_file: str | pathlib.Path,
        source_crs: dict,
        region_to_rasterise: geopandas.GeoDataFrame,
        metadata: dict,
        raster_options: dict,
    ) -> xarray.Dataset:
        """Create/Update dataset from a set of tiled LiDAR files. Read these in over
        non-overlapping chunks and then combine"""

        raise NotImplementedError(
            "_add_tiled_lidar_chunked must be instantiated in the" " child class"
        )

    def _add_lidar_no_chunking(
        self,
        lidar_datasets_info: dict,
        region_to_rasterise: geopandas.GeoDataFrame,
        options: dict,
        metadata: dict,
    ) -> xarray.Dataset:
        """Create/Update dataset from a single LiDAR file."""

        raise NotImplementedError(
            "_add_lidar_no_chunking must be instantiated in the " "child class"
        )


class RawDem(LidarBase):
    """A class to manage the creation of a 'raw' DEM from LiDAR tiles, and/or a
    coarse DEM.

    Parameters
    ----------

    drop_offshore_lidar
        If True only keep LiDAR values within the foreshore and land regions defined by
        the catchment_geometry. If False keep all LiDAR values.
    elevation_range
        Optitionally specify a range of valid elevations. Any LiDAR points with
        elevations outside this range will be filtered out.
    lidar_interpolation_method
        The interpolation method to apply to LiDAR during downsampling/averaging.
        Options are: mean, median, IDW, max, min, STD.
    buffer_cells - the number of empty cells to keep around LiDAR cells for
        interpolation after the coarse DEM added to ensure a smooth boundary.
    chunk_size
        The chunk size in pixels for parallel/staged processing
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        lidar_interpolation_method: str,
        drop_offshore_lidar: dict,
        zero_positive_foreshore: bool,
        buffer_cells: int,
        elevation_range: list | None = None,
        chunk_size: int | None = None,
    ):
        """Setup base DEM to add future tiles too"""

        super(RawDem, self).__init__(
            catchment_geometry=catchment_geometry,
            chunk_size=chunk_size,
            elevation_range=elevation_range,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.drop_offshore_lidar = drop_offshore_lidar
        self.zero_positive_foreshore = zero_positive_foreshore
        self.lidar_interpolation_method = lidar_interpolation_method
        self.buffer_cells = buffer_cells
        self._dem = None

    def _set_up_chunks(self) -> (list, list):
        """Define the chunks to break the catchment into when reading in and
        downsampling LiDAR.
        """

        bounds = self.catchment_geometry.catchment.geometry.bounds
        resolution = self.catchment_geometry.resolution

        # Determine the number of chunks
        minx = bounds.minx.min()
        maxx = bounds.maxx.max()
        miny = bounds.miny.min()
        maxy = bounds.maxy.max()
        n_chunks_x = int(
            numpy.ceil((bounds.maxx.max() - minx) / (self.chunk_size * resolution))
        )
        n_chunks_y = int(
            numpy.ceil((maxy - bounds.miny.min()) / (self.chunk_size * resolution))
        )

        # x coordinates rounded up to the nearest chunk - resolution aligned
        dim_x = []
        aligned_min_x = numpy.ceil(minx / resolution) * resolution
        for i in range(n_chunks_x):
            chunk_min_x = aligned_min_x + i * self.chunk_size * resolution
            if i + 1 < n_chunks_x:
                chunk_max_x = aligned_min_x + (i + 1) * self.chunk_size * resolution
            else:
                chunk_max_x = numpy.ceil(maxx / resolution) * resolution + resolution
            dim_x.append(
                numpy.arange(
                    chunk_min_x,
                    chunk_max_x,
                    resolution,
                    dtype=geometry.RASTER_TYPE,
                )
            )
        # y coordinates rounded up to the nearest chunk - resolution aligned
        dim_y = []
        aligned_max_y = numpy.ceil(maxy / resolution) * resolution
        for i in range(n_chunks_y):
            chunk_max_y = aligned_max_y - i * self.chunk_size * resolution
            if i + 1 < n_chunks_y:
                chunk_min_y = aligned_max_y - (i + 1) * self.chunk_size * resolution
            else:
                chunk_min_y = numpy.ceil(miny / resolution) * resolution - resolution
            dim_y.append(
                numpy.arange(
                    chunk_max_y,
                    chunk_min_y,
                    -resolution,
                    dtype=geometry.RASTER_TYPE,
                )
            )
        return dim_x, dim_y

    def _calculate_raw_extents(self):
        """Define the extents of the DEM with values (i.e. what are the spatial extents
        of pixels in the DEM that are defined from LiDAR or a coarse DEM)."""

        # Defines extents where raw DEM values exist
        mask = numpy.logical_not(numpy.isnan(self._dem.z.data))
        extents = self._extents_from_mask(
            mask=mask, transform=self._dem.rio.transform()
        )
        return extents

    def add_lidar(
        self,
        lidar_datasets_info: dict,
        lidar_classifications_to_keep: list,
        metadata: dict,
    ):
        """Read in all LiDAR files and use to create a 'raw' DEM.

        Parameters
        ----------

        lidar_datasets_info
            A dictionary of information for each specified LIDAR dataset - For
            each this includes: a list of LAS files, CRS, and tile index file.
        lidar_classifications_to_keep
            A list of LiDAR classifications to keep - '2' for ground, '9' for water.
            See https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf for
            standard list
        meta_data
            Information to include in the created DEM - must include
            `dataset_mapping` key if datasets (not a single LAZ file) included.
        """

        # Check valid inputs
        self._check_valid_inputs(lidar_datasets_info=lidar_datasets_info)

        # create dictionary defining raster options
        raster_options = {
            "lidar_classifications_to_keep": lidar_classifications_to_keep,
            "raster_type": geometry.RASTER_TYPE,
            "elevation_range": self.elevation_range,
            "radius": self.catchment_geometry.resolution / numpy.sqrt(2),
            "method": self.lidar_interpolation_method,
            "crs": self.catchment_geometry.crs,
            "strict": True,
        }
        if self.lidar_interpolation_method == "rbf":
            raster_options["kernel"] = "linear"

        # Don't use dask delayed if there is no chunking
        if len(lidar_datasets_info) == 0:
            # Create an empty dataset as no LiDAR
            self.logger.warning("No LiDAR dataset. Creating an empty raw DEM dataset.")
            bounds = self.catchment_geometry.catchment.geometry.bounds
            resolution = self.catchment_geometry.resolution
            x = numpy.arange(
                numpy.ceil(bounds.minx.min() / resolution) * resolution,
                numpy.ceil(bounds.maxx.max() / resolution) * resolution,
                resolution,
                dtype=raster_options["raster_type"],
            )
            y = numpy.arange(
                numpy.ceil(bounds.maxy.max() / resolution) * resolution,
                numpy.ceil(bounds.miny.min() / resolution) * resolution,
                -resolution,
                dtype=raster_options["raster_type"],
            )
            dem = self._create_empty_data_set(
                x=x,
                y=y,
                raster_type=raster_options["raster_type"],
                metadata=metadata,
            )
        elif self.chunk_size is None:
            dem = self._add_lidar_no_chunking(
                lidar_datasets_info=lidar_datasets_info,
                options=raster_options,
                metadata=metadata,
            )
        else:
            dem = self._add_tiled_lidar_chunked(
                lidar_datasets_info=lidar_datasets_info,
                raster_options=raster_options,
                metadata=metadata,
            )

        self._dem = dem

    def clip_lidar(
        self,
    ):
        """Clip the  a 'raw' DEM. Should be called immediately after the add_lidar function."""

        # Clip DEM to Catchment and ensure NaN outside region to rasterise
        catchment = self.catchment_geometry.catchment
        self._dem = self._dem.rio.clip_box(*tuple(catchment.total_bounds))
        self._dem = self._dem.where(
            clip_mask(self._dem.z, catchment.geometry, self.chunk_size)
        )

        # Check if the ocean is clipped or not (must be in all datasets)
        drop_offshore_lidar = all(self.drop_offshore_lidar.values())
        land_and_foreshore = self.catchment_geometry.land_and_foreshore
        if drop_offshore_lidar and land_and_foreshore.area.sum() > 0:
            # If area of 0 size, all will be NaN anyway
            mask = clip_mask(self._dem.z, land_and_foreshore.geometry, self.chunk_size)
            self._dem = self._dem.where(mask)

        # If drop offshore LiDAR ensure the foreshore values are 0 or negative
        foreshore = self.catchment_geometry.foreshore
        if (
            self.drop_offshore_lidar
            and foreshore.area.sum() > 0
            and self.zero_positive_foreshore
        ):
            buffer_radius = self.catchment_geometry.resolution * numpy.sqrt(2)
            buffered_foreshore = (
                foreshore.buffer(buffer_radius)
                .to_frame("geometry")
                .overlay(
                    self.catchment_geometry.full_land,
                    how="difference",
                    keep_geom_type=True,
                )
            )

            # Mask to delineate DEM outside of buffered foreshore or below 0
            mask = ~(
                (self._dem.z > 0)
                & clip_mask(self._dem.z, buffered_foreshore.geometry, self.chunk_size)
            )

            # Set any positive LiDAR foreshore points to zero
            self._dem["z"] = self._dem.z.where(mask, 0)

            self._dem["data_source"] = self._dem.data_source.where(
                mask, self.SOURCE_CLASSIFICATION["ocean bathymetry"]
            )
            self._dem["lidar_source"] = self._dem.lidar_source.where(
                mask, self.SOURCE_CLASSIFICATION["no data"]
            )

    def _add_tiled_lidar_chunked(
        self,
        lidar_datasets_info: dict,
        metadata: dict,
        raster_options: dict,
    ) -> xarray.Dataset:
        """Create a 'raw'' DEM from a set of tiled LiDAR files. Read these in over
        non-overlapping chunks and then combine"""

        assert self.chunk_size is not None, "chunk_size must be defined"

        # get chunking information
        chunked_dim_x, chunked_dim_y = self._set_up_chunks()
        elevations = {}

        self.logger.info(f"Preparing {[len(chunked_dim_x), len(chunked_dim_y)]} chunks")
        for dataset_name, dataset_info in lidar_datasets_info.items():
            # Pull out the dataset information
            lidar_files = dataset_info["file_paths"]
            tile_index_file = dataset_info["tile_index_file"]
            source_crs = dataset_info["crs"]

            # Define the region to rasterise
            region_to_rasterise = (
                self.catchment_geometry.land_and_foreshore
                if self.drop_offshore_lidar[dataset_name]
                else self.catchment_geometry.catchment
            )

            # create a map from tile name to tile file name
            lidar_files_map = {
                lidar_file.name: lidar_file for lidar_file in lidar_files
            }

            # remove all tiles entirely outside the region to raserise
            (
                tile_index_extents,
                tile_index_name_column,
            ) = self._tile_index_column_name(
                tile_index_file=tile_index_file,
                region_to_rasterise=self.catchment_geometry.catchment,
            )

            # cycle through index chunks - and collect in a delayed array
            self.logger.info(f"Running over dataset {dataset_name}")
            delayed_chunked_matrix = []
            for i, dim_y in enumerate(chunked_dim_y):
                delayed_chunked_x = []
                for j, dim_x in enumerate(chunked_dim_x):
                    self.logger.debug(f"\tLiDAR chunk {[i, j]}")

                    # Define the region to tile
                    chunk_region_to_tile = self._define_chunk_region(
                        region_to_rasterise=region_to_rasterise,
                        dim_x=dim_x,
                        dim_y=dim_y,
                        radius=raster_options["radius"],
                    )

                    # Load in files into tiles
                    chunk_lidar_files = select_lidar_files(
                        tile_index_extents=tile_index_extents,
                        tile_index_name_column=tile_index_name_column,
                        chunk_region_to_tile=chunk_region_to_tile,
                        lidar_files_map=lidar_files_map,
                    )

                    # Return empty if no files
                    if len(chunk_lidar_files) == 0:
                        self.logger.debug(
                            f"\t\tReturning empty tile as no LiDAR or out of ROI"
                        )
                        delayed_chunked_x.append(
                            dask.array.full(
                                shape=(len(dim_y), len(dim_x)),
                                fill_value=numpy.nan,
                                dtype=raster_options["raster_type"],
                            )
                        )
                        continue

                    # Get point cloud
                    chunk_points = delayed_load_tiles_in_chunk(
                        lidar_files=chunk_lidar_files,
                        source_crs=source_crs,
                        chunk_region_to_tile=chunk_region_to_tile,
                        crs=raster_options["crs"],
                    )
                    # Rasterise tiles
                    delayed_chunked_x.append(
                        dask.array.from_delayed(
                            delayed_elevation_over_chunk(
                                dim_x=dim_x,
                                dim_y=dim_y,
                                tile_points=chunk_points,
                                options=raster_options,
                            ),
                            shape=(len(dim_y), len(dim_x)),
                            dtype=raster_options["raster_type"],
                        )
                    )
                delayed_chunked_matrix.append(delayed_chunked_x)

            # Combine chunks into a dataset
            elevations[dataset_name] = dask.array.block(delayed_chunked_matrix)
        chunked_dem = self._create_data_set(
            x=numpy.concatenate(chunked_dim_x),
            y=numpy.concatenate(chunked_dim_y),
            elevations=elevations,
            metadata=metadata,
        )

        return chunked_dem

    def _add_lidar_no_chunking(
        self,
        lidar_datasets_info: dict,
        options: dict,
        metadata: dict,
    ) -> xarray.Dataset:
        """Create a 'raw' DEM from a single LiDAR file with no chunking."""

        assert self.chunk_size is None, "chunk_size should not be defined"

        # Note only support for a single LiDAR file without tile information
        lidar_name = list(lidar_datasets_info.keys())[0]
        lidar_file = lidar_datasets_info[lidar_name]["file_paths"][0]
        source_crs = lidar_datasets_info[lidar_name]["crs"]
        self.logger.info(f"On LiDAR tile 1 of 1: {lidar_file}")

        # Define the region to rasterise
        region_to_rasterise = (
            self.catchment_geometry.land_and_foreshore
            if self.drop_offshore_lidar[lidar_name]
            else self.catchment_geometry.catchment
        )

        # Use PDAL to load in file
        pdal_pipeline = read_file_with_pdal(
            lidar_file,
            source_crs=source_crs,
            region_to_tile=region_to_rasterise,
            crs=options["crs"],
        )

        # Load LiDAR points from pipeline
        tile_points = pdal_pipeline.arrays[0]

        # Define the raster/DEM dimensions - Align resolution (not BBox)
        bounds = self.catchment_geometry.catchment.geometry.bounds
        resolution = self.catchment_geometry.resolution
        dim_x = numpy.arange(
            numpy.ceil(bounds.minx.min() / resolution) * resolution,
            numpy.ceil(bounds.maxx.max() / resolution) * resolution,
            resolution,
            dtype=options["raster_type"],
        )
        dim_y = numpy.arange(
            numpy.ceil(bounds.maxy.max() / resolution) * resolution,
            numpy.ceil(bounds.miny.min() / resolution) * resolution,
            -resolution,
            dtype=options["raster_type"],
        )

        # Create elevation raster
        raster_values = self._elevation_over_tile(
            dim_x=dim_x, dim_y=dim_y, tile_points=tile_points, options=options
        )
        elevation = raster_values.reshape((len(dim_y), len(dim_x)))

        # Create xarray
        dem = self._create_data_set(
            x=dim_x,
            y=dim_y,
            elevations={lidar_name: elevation},
            metadata=metadata,
        )

        return dem

    def _elevation_over_tile(
        self,
        dim_x: numpy.ndarray,
        dim_y: numpy.ndarray,
        tile_points: numpy.ndarray,
        options: dict,
    ):
        """Rasterise all points within a tile to give elevation. Does not require a tile
        index file."""

        # keep only the specified classifications (should be ground / water)
        classification_mask = numpy.zeros_like(
            tile_points["Classification"], dtype=bool
        )
        for classification in options["lidar_classifications_to_keep"]:
            classification_mask[tile_points["Classification"] == classification] = True
        tile_points = tile_points[classification_mask]

        if len(tile_points) == 0:
            self.logger.warning(
                "In RawDem._elevation_over_tile the tile has no data and is being "
                "ignored."
            )
            return
        # Get the grided locations overwhich to perform IDW
        grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
        xy_out = numpy.concatenate(
            [[grid_x.flatten()], [grid_y.flatten()]], axis=0
        ).transpose()

        # Perform the specified rasterisation over the grid locations
        z_flat = elevation_from_points(
            point_cloud=tile_points,
            xy_out=xy_out,
            options=options,
        )
        grid_z = z_flat.reshape(grid_x.shape)

        return grid_z

    def _create_data_set(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        elevations: dict,
        metadata: dict,
    ) -> xarray.Dataset:
        """A function to create a new dataset from x, y and z arrays.

        Parameters
        ----------

            x
                X coordinates of the dataset.
            y
                Y coordinates of the dataset.
            elevations
                A dictionary of elevations over the x, and y coordiantes.Keyed
                by the dataset name.
            metadata
                Used to pull out the dataset mapping for creating the
                lidar_source layer
        """

        # Cycle over each elevation dataset populating a DEM - merge after
        dems = []
        dataset_mapping = metadata["instructions"]["dataset_mapping"]["lidar"]
        for dataset_name, z in elevations.items():
            # Set NaN where not z values so merging occurs correctly
            z_values_mask = dask.array.notnull(z)

            # Create source variable - assume all values are defined from LiDAR
            data_source = dask.array.where(
                z_values_mask,
                self.SOURCE_CLASSIFICATION["LiDAR"],
                numpy.nan,
            )

            # Create LiDAR id variable - name and value info in the metadata
            lidar_source = dask.array.where(
                z_values_mask,
                dataset_mapping[dataset_name],
                numpy.nan,
            )

            dem = xarray.Dataset(
                data_vars=dict(
                    z=(
                        ["y", "x"],
                        z,
                        {
                            "units": "m",
                            "long_name": "ground elevation",
                            "vertical_datum": "EPSG:"
                            f"{self.catchment_geometry.crs['vertical']}",
                        },
                    ),
                    data_source=(
                        ["y", "x"],
                        data_source,
                        {
                            "units": "",
                            "long_name": "source data classification",
                            "mapping": f"{self.SOURCE_CLASSIFICATION}",
                        },
                    ),
                    lidar_source=(
                        ["y", "x"],
                        lidar_source,
                        {
                            "units": "",
                            "long_name": "source lidar ID",
                            "mapping": f"{dataset_mapping}",
                        },
                    ),
                ),
                coords=dict(x=(["x"], x), y=(["y"], y)),
                attrs={
                    "title": "Geofabric representing elevation and roughness",
                    "source": f"{metadata['library_name']} version "
                    f"{metadata['library_version']}",
                    "description": f"{metadata['library_name']}:"
                    f"{metadata['class_name']} resolution "
                    f"{self.catchment_geometry.resolution}",
                    "history": f"{metadata['utc_time']}:"
                    f"{metadata['library_name']}:{metadata['class_name']} "
                    f"version {metadata['library_version']} "
                    f"resolution {self.catchment_geometry.resolution};",
                    "geofabrics_instructions": f"{metadata['instructions']}",
                },
            )
            # ensure the expected CF conventions are followed
            self._write_netcdf_conventions_in_place(dem, self.catchment_geometry.crs)
            dems.append(dem)

        if len(dems) == 1:
            dem = dems[0]
        else:
            dem = rioxarray.merge.merge_datasets(dems, method="first")

        # After merging LiDAR datasets set remaining NaN to no data/LiDAR
        # data_source: set areas with no values to No Data
        dem["data_source"] = dem.data_source.where(
            dem.data_source.notnull(),
            self.SOURCE_CLASSIFICATION["no data"],
        )

        # lidar_source: Set areas with no LiDAR to "No LiDAR"
        dem["lidar_source"] = dem.lidar_source.where(
            dem.lidar_source.notnull(),
            dataset_mapping["no LiDAR"],
        )

        return dem

    def _create_empty_data_set(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        raster_type: str,
        metadata: dict,
    ) -> xarray.Dataset:
        """A function to create a new but empty dataset from x, y arrays.

        Parameters
        ----------

            x
                X coordinates of the dataset.
            y
                Y coordinates of the dataset.
            elevations
                A dictionary of elevations over the x, and y coordiantes.Keyed
                by the dataset name.
            metadata
                Used to pull out the dataset mapping for creating the
                lidar_source layer
        """

        # Create the empty layers - z, data_source, lidar_source
        # Set NaN where not z values so merging occurs correctly
        z = dask.array.full(
            fill_value=numpy.nan,
            shape=(len(y), len(x)),
            dtype=raster_type,
            chunks={"x": self.chunk_size, "y": self.chunk_size},
        )

        # Create source variable - assume all values are defined from LiDAR
        data_source = dask.array.full(
            fill_value=self.SOURCE_CLASSIFICATION["no data"],
            shape=(len(y), len(x)),
            dtype=raster_type,
            chunks={"x": self.chunk_size, "y": self.chunk_size},
        )

        # Create LiDAR id variable - name and value info in the metadata
        lidar_source = dask.array.full(
            fill_value=self.SOURCE_CLASSIFICATION["no data"],
            shape=(len(y), len(x)),
            dtype=raster_type,
            chunks={"x": self.chunk_size, "y": self.chunk_size},
        )

        dem = xarray.Dataset(
            data_vars=dict(
                z=(
                    ["y", "x"],
                    z,
                    {
                        "units": "m",
                        "long_name": "ground elevation",
                        "vertical_datum": "EPSG:"
                        f"{self.catchment_geometry.crs['vertical']}",
                    },
                ),
                data_source=(
                    ["y", "x"],
                    data_source,
                    {
                        "units": "",
                        "long_name": "source data classification",
                        "mapping": f"{self.SOURCE_CLASSIFICATION}",
                    },
                ),
                lidar_source=(
                    ["y", "x"],
                    lidar_source,
                    {
                        "units": "",
                        "long_name": "source lidar ID",
                        "mapping": f"{{'no LiDAR': self.SOURCE_CLASSIFICATION['no data']}}",
                    },
                ),
            ),
            coords=dict(x=(["x"], x), y=(["y"], y)),
            attrs={
                "title": "Geofabric representing elevation and roughness",
                "source": f"{metadata['library_name']} version "
                f"{metadata['library_version']}",
                "description": f"{metadata['library_name']}:"
                f"{metadata['class_name']} resolution "
                f"{self.catchment_geometry.resolution}",
                "history": f"{metadata['utc_time']}:"
                f"{metadata['library_name']}:{metadata['class_name']} "
                f"version {metadata['library_version']} "
                f"resolution {self.catchment_geometry.resolution};",
                "geofabrics_instructions": f"{metadata['instructions']}",
            },
        )
        # ensure the expected CF conventions are followed
        self._write_netcdf_conventions_in_place(dem, self.catchment_geometry.crs)
        dem = dem.chunk(chunks={"x": self.chunk_size, "y": self.chunk_size})

        return dem


class PatchDem(LidarBase):
    """A class to manage the addition of a DEM to the foreground or background
    of a preexisting DEM.

    Parameters
    ----------

    patch_on_top
        If True only patch the DEM values on top of the initial DEM. If False
        patch only where values are NaN.
    drop_patch_offshore
        If True only keep patch values on land and the foreshore.
    elevation_range
        Optitionally specify a range of valid elevations. Any LiDAR points with
        elevations outside this range will be filtered out.
    initial_dem_path
        The DEM to patch the other DEM on top / only where values are NaN.
    buffer_cells - the number of empty cells to keep around LiDAR cells for
        interpolation after the coarse DEM added to ensure a smooth boundary.
    chunk_size
        The chunk size in pixels for parallel/staged processing
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        patch_on_top: bool,
        drop_patch_offshore: bool,
        zero_positive_foreshore: bool,
        buffer_cells: int,
        initial_dem_path: str | pathlib.Path,
        elevation_range: list | None = None,
        chunk_size: int | None = None,
    ):
        """Setup base DEM to add future tiles too"""

        super(PatchDem, self).__init__(
            catchment_geometry=catchment_geometry,
            chunk_size=chunk_size,
            elevation_range=elevation_range,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.drop_patch_offshore = drop_patch_offshore
        self.zero_positive_foreshore = zero_positive_foreshore
        self.patch_on_top = patch_on_top
        self.buffer_cells = buffer_cells
        # Read in the DEM raster
        initial_dem = rioxarray.rioxarray.open_rasterio(
            pathlib.Path(initial_dem_path),
            masked=True,
            parse_coordinates=True,
            chunks=True,
        ).squeeze(
            "band", drop=True
        )  # remove band coordinate added by rasterio.open()
        self._write_netcdf_conventions_in_place(initial_dem, catchment_geometry.crs)
        self._dem = initial_dem

    def add_patch(self, patch_path: pathlib.Path, label: str, layer: str):
        """Check if gaps in DEM on land, if so iterate through coarse DEMs
        adding missing detail.

        Note the Coarse DEM values are only applied on land and not
        in the ocean.

        Parameters
        ----------

            patch_path - patch file path to try add
            label - either "coarse DEM" or "patch". Defines the data_source
            layer - either 'z' or 'zo' and the layer to set the patch on
        """

        if layer not in self._dem.keys():
            self.logger.error(
                f"Invalid 'layer' option {layer}. Valid layers include "
                f"{self._dem.keys()} excluding the 'data_source' and "
                "lidar_source' layers."
            )
            raise ValueError

        # Check for overlap with the Coarse DEM
        patch = rioxarray.rioxarray.open_rasterio(
            patch_path,
            masked=True,
            chunks=True,
        ).squeeze("band", drop=True)
        patch.rio.write_crs(self.catchment_geometry.crs["horizontal"], inplace=True)
        patch_resolution = patch.rio.resolution()
        patch_resolution = max(abs(patch_resolution[0]), abs(patch_resolution[1]))

        # Define region to patch within
        if self.drop_patch_offshore:
            roi = self.catchment_geometry.land_and_foreshore
        else:
            roi = self.catchment_geometry.catchment
        try:  # Use try catch as otherwise crash if patch does not overlap roi
            patch = patch.rio.clip_box(*tuple(roi.buffer(patch_resolution).total_bounds))
            patch = patch.where(
                clip_mask(patch, roi.buffer(patch_resolution).geometry, self.chunk_size)
            )
        except (  # If exception skip and proceed to the next patch
            rioxarray.exceptions.NoDataInBounds,
            ValueError,
        ) as caught_exception:
            self.logger.warning(
                f"NoDataInDounds in PatchDem.add_patchs. Will skip {patch_path}."
                f"Exception: {caught_exception}."
            )
            return False
        patch_bounds = geopandas.GeoDataFrame(
            geometry=[shapely.geometry.box(*patch.rio.bounds())],
            crs=self.catchment_geometry.crs["horizontal"],
        )
        if not self.patch_on_top:
            # patch DEM data where there's no LiDAR updating the extents
            no_values_mask = self.no_values_mask & clip_mask(
                self._dem.z, patch_bounds.geometry, self.chunk_size
            )
            no_values_mask.load()
            # Early return if there is nowhere to add patch DEM data
            if not no_values_mask.any():
                return False

        self.logger.info(f"\t\tAdd data from coarse DEM: {patch_path.name}")

        # Check if same resolution
        if all(patch.x.isin(self._dem.x)) and all(patch.y.isin(self._dem.y)):
            # Do a stright replacement if grid aligned
            patch = patch.reindex_like(self._dem, fill_value=numpy.nan)
        elif (
            self.chunk_size is not None
            and max(len(self._dem.x), len(self._dem.y)) > self.chunk_size
        ):  # Ensure parallelisation if chunks specified
            # Expect xarray dims (y, x), not (x, y) as default for rioxarray
            interpolator = scipy.interpolate.RegularGridInterpolator(
                (patch.y.values, patch.x.values),
                patch.values,
                bounds_error=False,
                fill_value=numpy.nan,
                method="linear",
            )

            def dask_interpolation(y, x):
                yx_array = numpy.stack(numpy.meshgrid(y, x, indexing="ij"), axis=-1)
                return interpolator(yx_array)

            # Explicitly redefine x & y
            x = dask.array.from_array(self._dem.x.values, chunks=self.chunk_size)
            y = dask.array.from_array(self._dem.y.values, chunks=self.chunk_size)
            patch_interp = dask.array.blockwise(
                dask_interpolation, "ij", y, "i", x, "j"
            )
            patch = xarray.DataArray(
                patch_interp,
                dims=("y", "x"),
                coords={"x": self._dem.x, "y": self._dem.y},
            )
            patch.rio.write_transform(inplace=True)
        else:  # No chunking use built in method
            patch = patch.interp(x=self._dem.x, y=self._dem.y, method="linear")
        patch.rio.write_crs(self.catchment_geometry.crs["horizontal"], inplace=True)
        patch.rio.write_nodata(numpy.nan, encoded=True, inplace=True)

        # Ensure clipped in region of interest (catchment, or land & foreshore)
        mask = clip_mask(patch, roi.geometry, self.chunk_size)
        patch = patch.where(mask)
        if self.patch_on_top:
            self._dem[layer] = self._dem.z.where(patch.isnull(), patch)
            mask = patch.isnull()
            self._dem["lidar_source"] = self._dem.lidar_source.where(
                mask,
                self.SOURCE_CLASSIFICATION["no data"],
            )
        else:  # patch on bottom (where NaN)
            self._dem[layer] = self._dem.z.where(~no_values_mask, patch)
            mask = ~(no_values_mask & self._dem.z.notnull())

        # Update the data source layer
        self._dem["data_source"] = self._dem.data_source.where(
            mask,
            self.SOURCE_CLASSIFICATION[label],
        )

        if label == "coarse DEM" and self.zero_positive_foreshore:
            # Ensure Coarse DEM values along the foreshore are less than zero
            foreshore = self.catchment_geometry.foreshore
            if foreshore.area.sum() > 0:
                buffer_radius = self.catchment_geometry.resolution * numpy.sqrt(2)
                buffered_foreshore = (
                    foreshore.buffer(buffer_radius)
                    .to_frame("geometry")
                    .overlay(
                        self.catchment_geometry.full_land,
                        how="difference",
                        keep_geom_type=True,
                    )
                )

                # Clip coarse DEM patch to buffered foreshore
                patch_mask = (
                    self._dem.data_source == self.SOURCE_CLASSIFICATION["coarse DEM"]
                )
                foreshore_mask = clip_mask(
                    self._dem.z, buffered_foreshore.geometry, self.chunk_size
                )
                mask = ~((self._dem.z > 0) & foreshore_mask & patch_mask)

                # Set any positive Coarse DEM foreshore points to zero
                self._dem["data_source"] = self._dem.data_source.where(
                    mask, self.SOURCE_CLASSIFICATION["ocean bathymetry"]
                )
                self._dem["lidar_source"] = self._dem.lidar_source.where(
                    mask,
                    self.SOURCE_CLASSIFICATION["no data"],
                )
                self._dem["z"] = self._dem.z.where(mask, 0)
        return True

    @property
    def no_values_mask(self):
        """No values mask from DEM within land and foreshore region"""

        if self.catchment_geometry.land_and_foreshore.area.sum() > 0:
            no_values_mask = (
                self._dem.z.rolling(
                    dim={
                        "x": self.buffer_cells * 2 + 1,
                        "y": self.buffer_cells * 2 + 1,
                    },
                    min_periods=1,
                    center=True,
                )
                .count()
                .isnull()
            )
            no_values_mask &= clip_mask(
                self._dem.z,
                self.catchment_geometry.land_and_foreshore.geometry,
                self.chunk_size,
            )
        else:
            no_values_mask = xarray.zeros_like(self._dem.z, dtype=bool)

        return no_values_mask


class RoughnessDem(LidarBase):
    """A class to add a roughness (zo) layer to a hydrologically conditioned DEM.

    They STD and mean height of ground cover classified points are calculated from the
    LiDAR data and z (elevation) layer of the hydrologically conditioned DEM, and used
    to estimate roughness emperically.

    RoughnessDem logic can be controlled by the constructor inputs.

    Parameters
    ----------

    catchment_geometry
        Defines the geometry of the catchment
    hydrological_dem_path
        The path to the hydrologically conditioned DEM.
    interpolation_method
        If not None, interpolate using that method. Valid options are 'linear',
        'nearest', and 'cubic'.
    lidar_interpolation_method
        The interpolation method to apply to LiDAR. Options are: mean, median, IDW.
    """

    def __init__(
        self,
        catchment_geometry: geometry.CatchmentGeometry,
        hydrological_dem_path: str | pathlib.Path,
        temp_folder: pathlib.Path,
        interpolation_method: str,
        default_values: dict,
        drop_offshore_lidar: dict,
        chunk_size: int | None = None,
        elevation_range: list = None,
    ):
        """Setup base DEM to add future tiles too"""

        super(RoughnessDem, self).__init__(
            catchment_geometry=catchment_geometry,
            elevation_range=elevation_range,
            chunk_size=chunk_size,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Load hyrdological DEM.
        hydrological_dem = rioxarray.rioxarray.open_rasterio(
            pathlib.Path(hydrological_dem_path),
            masked=True,
            parse_coordinates=True,
            chunks=True,
        ).squeeze(
            "band", drop=True
        )  # remove band coordinate added by rasterio.open()
        self._write_netcdf_conventions_in_place(
            hydrological_dem, catchment_geometry.crs
        )

        # Ensure the resolution of the hydrological DEM matches the input DEM
        assert (
            abs(float(hydrological_dem.x[1] - hydrological_dem.x[0]))
            == self.catchment_geometry.resolution
        ), (
            "The input DEM resolution doesn't match the input resolution. These must "
            "match"
        )

        # Clip to the catchment extents to ensure performance
        catchment = self.catchment_geometry.catchment
        hydrological_dem = hydrological_dem.rio.clip_box(*tuple(catchment.total_bounds))
        mask = clip_mask(hydrological_dem.z, catchment.geometry, self.chunk_size)
        hydrological_dem = hydrological_dem.where(mask)
        # Rerun as otherwise the no data as NaN seems to be lost for the data_source layer
        self._write_netcdf_conventions_in_place(
            hydrological_dem, catchment_geometry.crs
        )

        self.temp_folder = temp_folder
        self.interpolation_method = interpolation_method
        self.default_values = default_values
        self.drop_offshore_lidar = drop_offshore_lidar
        self._dem = hydrological_dem

    def _calculate_lidar_extents(self):
        """Calculate the extents of the LiDAR data."""

        # Defines extents where raw DEM values exist
        mask = self._dem.data_source.data == self.SOURCE_CLASSIFICATION["LiDAR"]
        extents = self._extents_from_mask(
            mask=mask, transform=self._dem.rio.transform()
        )
        return extents

    def add_lidar(
        self,
        lidar_datasets_info: dict,
        lidar_classifications_to_keep: list,
        metadata: dict,
        parameters: dict,
    ):
        """Read in all LiDAR files and use the point cloud distribution,
        data_source layer, and hydrologiaclly conditioned elevations to
        estimate the roughness across the DEM.

        Parameters
        ----------

        lidar_datasets_info
            A dictionary of information for each specified LIDAR dataset - For
            each this includes: a list of LAS files, CRS, and tile index file.
        lidar_classifications_to_keep
            A list of LiDAR classifications to keep - '2' for ground, '9' for water.
            See https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf for
            standard list
        meta_data
            Information to include in the created DEM - must include
            `dataset_mapping` key if datasets (not a single LAZ file) included.
        parameters
            The roughness equation parameters.
        """

        # Check valid inputs
        self._check_valid_inputs(lidar_datasets_info=lidar_datasets_info)

        # create dictionary defining raster options
        raster_options = {
            "lidar_classifications_to_keep": lidar_classifications_to_keep,
            "raster_type": geometry.RASTER_TYPE,
            "elevation_range": self.elevation_range,
            "radius": self.catchment_geometry.resolution / numpy.sqrt(2),
            "crs": self.catchment_geometry.crs,
            "parameters": parameters,
        }

        # Calculate roughness from LiDAR
        if len(lidar_datasets_info) == 0:
            # Create an empty dataset as no LiDAR
            self.logger.warning("No LiDAR dataset. Creating an empty roughness layer.")
            zo = xarray.ones_like(self._dem.z)
            zo = zo.where(False, numpy.nan)
            zo = zo.assign_attrs(long_name="Roughness length")
            zo.rio.write_transform(inplace=True)
            zo.rio.write_nodata(numpy.nan, encoded=True, inplace=True)
            self._dem["zo"] = zo
            # ensure the expected CF conventions are followed
            self._write_netcdf_conventions_in_place(
                self._dem, self.catchment_geometry.crs
            )

        elif self.chunk_size is None:  # If one file it's ok if there is no tile_index
            self._dem = self._add_lidar_no_chunking(
                lidar_datasets_info=lidar_datasets_info,
                options=raster_options,
                metadata=metadata,
            )
        else:
            self._dem = self._add_tiled_lidar_chunked(
                lidar_datasets_info=lidar_datasets_info,
                raster_options=raster_options,
                metadata=metadata,
            )

        self.save_and_load_dem(
            filename=self.temp_folder / "raw_lidar_zo.nc",
        )
        # Set roughness where water
        self._dem["zo"] = self._dem.zo.where(
            self._dem.data_source != self.SOURCE_CLASSIFICATION["ocean bathymetry"],
            self.default_values["ocean"],
        )
        self._dem["zo"] = self._dem.zo.where(
            self._dem.data_source != self.SOURCE_CLASSIFICATION["rivers and fans"],
            self.default_values["rivers"],
        )

        if self.default_values["waterways"] is not None:
            self._dem["zo"] = self._dem.zo.where(
                self._dem.data_source != self.SOURCE_CLASSIFICATION["waterways"],
                self.default_values["waterways"],
            )
        # Set roughness where land and no LiDAR
        self._dem["zo"] = self._dem.zo.where(
            self._dem.data_source != self.SOURCE_CLASSIFICATION["coarse DEM"],
            self.default_values["land"],
        )  # or LiDAR with no roughness estimate
        # Ensure the defaults are re-added
        self._write_netcdf_conventions_in_place(self._dem, self.catchment_geometry.crs)
        # Interpolate any missing roughness values
        if self.interpolation_method is not None:
            self._dem["zo"] = self._dem.zo.rio.interpolate_na(
                method=self.interpolation_method
            )
            # If any NaN remain apply nearest neighbour interpolation
            if numpy.isnan(self._dem.zo.data).any():
                self._dem["zo"] = self._dem.zo.rio.interpolate_na(method="nearest")

        mask = clip_mask(
            self._dem.z, self.catchment_geometry.catchment.geometry, self.chunk_size
        )
        self._dem = self._dem.where(mask)
        self._write_netcdf_conventions_in_place(self._dem, self.catchment_geometry.crs)

    def add_roads(self, roads_polygon: dict):
        """Set roads to paved and unpaved roughness values.

        Parameters
        ----------

        roads_polygon
            Dataframe with polygon and associated roughness values
        """

        # Set unpaved roads
        mask = clip_mask(
            self._dem.z,
            roads_polygon[roads_polygon["surface"] == "unpaved"].geometry,
            self.chunk_size,
        )
        self._dem["zo"] = self._dem.zo.where(~mask, self.default_values["unpaved"])
        # Then set paved roads
        mask = clip_mask(
            self._dem.z,
            roads_polygon[roads_polygon["surface"] == "paved"].geometry,
            self.chunk_size,
        )
        self._dem["zo"] = self._dem.zo.where(~mask, self.default_values["paved"])
        self._write_netcdf_conventions_in_place(self._dem, self.catchment_geometry.crs)

    def _add_tiled_lidar_chunked(
        self,
        lidar_datasets_info: dict,
        metadata: dict,
        raster_options: dict,
    ) -> xarray.Dataset:
        """Create a roughness layer with estimates where there is LiDAR from a set of
        tiled LiDAR files. Read these in over non-overlapping chunks and then combine.
        """

        # get chunks to tile over
        chunked_dim_x, chunked_dim_y = self._chunks_from_dem(self.chunk_size, self._dem)

        roughnesses = []

        self.logger.info(f"Preparing {[len(chunked_dim_x), len(chunked_dim_y)]} chunks")
        for dataset_name in lidar_datasets_info.keys():
            # Pull out the dataset information
            lidar_files = lidar_datasets_info[dataset_name]["file_paths"]
            tile_index_file = lidar_datasets_info[dataset_name]["tile_index_file"]
            source_crs = lidar_datasets_info[dataset_name]["crs"]

            # create a map from tile name to tile file name
            lidar_files_map = {
                lidar_file.name: lidar_file for lidar_file in lidar_files
            }

            # Define the region to rasterise
            region_to_rasterise = (
                self.catchment_geometry.land_and_foreshore
                if self.drop_offshore_lidar[dataset_name]
                else self.catchment_geometry.catchment
            )
            # Remove all tiles entirely outside the region to raserise
            (
                tile_index_extents,
                tile_index_name_column,
            ) = self._tile_index_column_name(
                tile_index_file=tile_index_file,
                region_to_rasterise=region_to_rasterise,
            )

            # cycle through chunks - and collect in a delayed array
            self.logger.info(f"Running over dataset {dataset_name}")
            delayed_chunked_matrix = []
            for i, dim_y in enumerate(chunked_dim_y):
                delayed_chunked_x = []
                for j, dim_x in enumerate(chunked_dim_x):
                    self.logger.debug(f"\tChunk {[i, j]}")

                    # Define the region to tile
                    chunk_region_to_tile = self._define_chunk_region(
                        region_to_rasterise=region_to_rasterise,
                        dim_x=dim_x,
                        dim_y=dim_y,
                        radius=raster_options["radius"],
                    )

                    # Load in files into tiles
                    chunk_lidar_files = select_lidar_files(
                        tile_index_extents=tile_index_extents,
                        tile_index_name_column=tile_index_name_column,
                        chunk_region_to_tile=chunk_region_to_tile,
                        lidar_files_map=lidar_files_map,
                    )

                    # Return empty if no files
                    if len(chunk_lidar_files) == 0:
                        self.logger.debug(
                            f"\t\tReturning empty tile as no LiDAR or out of ROI"
                        )
                        delayed_chunked_x.append(
                            dask.array.empty(
                                shape=(len(dim_y), len(dim_x)),
                                dtype=raster_options["raster_type"],
                            )
                        )
                        continue

                    chunk_points = delayed_load_tiles_in_chunk(
                        lidar_files=chunk_lidar_files,
                        source_crs=source_crs,
                        chunk_region_to_tile=chunk_region_to_tile,
                        crs=raster_options["crs"],
                    )
                    # Rasterise tiles
                    xy_ground = self._dem.z.sel(
                        x=dim_x, y=dim_y, method="nearest"
                    ).data.flatten()
                    delayed_chunked_x.append(
                        dask.array.from_delayed(
                            delayed_roughness_over_chunk(
                                dim_x=dim_x,
                                dim_y=dim_y,
                                tile_points=chunk_points,
                                xy_ground=xy_ground,
                                options=raster_options,
                            ),
                            shape=(len(dim_y), len(dim_x)),
                            dtype=geometry.RASTER_TYPE,
                        )
                    )
                delayed_chunked_matrix.append(delayed_chunked_x)
            # Combine chunks and add to dataset
            roughnesses.append(dask.array.block(delayed_chunked_matrix))
        chunked_dem = self._add_roughness_to_data_set(
            x=numpy.concatenate(chunked_dim_x),
            y=numpy.concatenate(chunked_dim_y),
            roughnesses=roughnesses,
            metadata=metadata,
        )

        return chunked_dem

    def _add_lidar_no_chunking(
        self,
        lidar_datasets_info: dict,
        options: dict,
        metadata: dict,
    ) -> xarray.Dataset:
        """Create a roughness layer with estimates where there is LiDAR from a single
        LiDAR file with no chunking."""

        # Note only support for a single LiDAR file without tile information
        lidar_name = list(lidar_datasets_info.keys())[0]
        lidar_file = lidar_datasets_info[lidar_name]["file_paths"][0]
        source_crs = lidar_datasets_info[lidar_name]["crs"]
        self.logger.info(f"On LiDAR tile 1 of 1: {lidar_file}")

        # Define the region to rasterise
        region_to_rasterise = (
            self.catchment_geometry.land_and_foreshore
            if self.drop_offshore_lidar[lidar_name]
            else self.catchment_geometry.catchment
        )

        # Use PDAL to load in file
        pdal_pipeline = read_file_with_pdal(
            lidar_file,
            source_crs=source_crs,
            region_to_tile=region_to_rasterise,
            crs=options["crs"],
        )

        # Load LiDAR points from pipeline
        tile_array = pdal_pipeline.arrays[0]

        # Get the locations to rasterise
        dim_x = self._dem.x.data
        dim_y = self._dem.y.data

        # Estimate roughness over the region
        raster_values = self._roughness_over_tile(
            dim_x=dim_x,
            dim_y=dim_y,
            tile_points=tile_array,
            xy_ground=self._dem.z.data.flatten(),
            options=options,
        )
        roughness = raster_values.reshape((len(dim_y), len(dim_x)))

        # Create xarray
        dem = self._add_roughness_to_data_set(
            x=dim_x,
            y=dim_y,
            roughnesses=[roughness],
            metadata=metadata,
        )

        return dem

    def _roughness_over_tile(
        self,
        dim_x: numpy.ndarray,
        dim_y: numpy.ndarray,
        xy_ground: numpy.ndarray,
        tile_points: numpy.ndarray,
        options: dict,
    ):
        """Rasterise all points within a tile to give roughness. Does not require a tile
        index file."""

        # keep only the specified classifications (should be ground cover)
        classification_mask = numpy.zeros_like(
            tile_points["Classification"], dtype=bool
        )
        for classification in options["lidar_classifications_to_keep"]:
            classification_mask[tile_points["Classification"] == classification] = True
        tile_points = tile_points[classification_mask]

        if len(tile_points) == 0:
            self.logger.warning(
                "In RoughnessDem._roughness_over_tile the tile has no data and is being"
                " ignored."
            )
            return
        # Get the locations overwhich to estimate roughness
        grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
        xy_out = numpy.concatenate(
            [[grid_x.flatten()], [grid_y.flatten()]], axis=0
        ).transpose()

        # Perform roughness estimation within the extents of the tile
        z_flat = roughness_from_points(
            point_cloud=tile_points,
            xy_out=xy_out,
            xy_ground=xy_ground,
            options=options,
        )
        grid_z = z_flat.reshape(grid_x.shape)

        return grid_z

    def _add_roughness_to_data_set(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        roughnesses: list,
        metadata: dict,
    ) -> xarray.Dataset:
        """A function to add zo to the existing DEM as a new variable.

        Parameters
        ----------

            x
                X coordinates of the dataset.
            y
                Y coordinates of the dataset.
            roughnesses
                A list of roughnesses over the x, and y coordiantes for each dataset.
        """

        self.logger.info(
            "In RoughnessDem._add_roughness_to_data_set creating and "
            "adding the Zo layer to the dataset."
        )

        # Create a DataArray of zo
        zos = []
        for zo_array in roughnesses:
            zo = xarray.DataArray(
                data=zo_array,
                dims=["y", "x"],
                coords=dict(x=(["x"], x), y=(["y"], y)),
                attrs=dict(
                    long_name="ground roughness",
                    units="",
                ),
            )
            zo.rio.write_crs(self.catchment_geometry.crs["horizontal"], inplace=True)
            zo.rio.write_transform(inplace=True)
            zo.rio.write_nodata(numpy.nan, encoded=True, inplace=True)
            zos.append(zo)
        if len(zos) == 1:
            zo = zos[0]
        else:
            zo = rioxarray.merge.merge_arrays(zos, method="first")
        # Resize zo to share the same dimensions at the DEM
        self._dem["zo"] = zo.sel(x=self._dem.x, y=self._dem.y, method="nearest")
        # Ensure roughness values are bounded by the defaults
        if self.default_values["minimum"] is not None:
            self._dem["zo"] = self._dem.zo.where(
                self._dem.zo > self.default_values["minimum"],
                self.default_values["minimum"],
            )
        if self.default_values["maximum"] is not None:
            self._dem["zo"] = self._dem.zo.where(
                self._dem.zo < self.default_values["maximum"],
                self.default_values["maximum"],
            )

        # update metadata
        history = self._dem.attrs["history"]
        self._dem.attrs["history"] = (
            f"{metadata['utc_time']}:{metadata['library_name']}"
            f":{metadata['class_name']} version {metadata['library_version']} "
            f" resolution {self.catchment_geometry.resolution}; {history}"
        )
        self._dem.attrs[
            "source"
        ] = f"{metadata['library_name']} version {metadata['library_version']}"
        self._dem.attrs["description"] = (
            f"{metadata['library_name']}:{metadata['class_name']} resolution "
            f"{self.catchment_geometry.resolution}"
        )
        self._dem.attrs["geofabrics_instructions"] = f"{metadata['instructions']}"

        # ensure the expected CF conventions are followed
        self._write_netcdf_conventions_in_place(self._dem, self.catchment_geometry.crs)

        return self._dem


def read_file_with_pdal(
    lidar_file: str | pathlib.Path,
    region_to_tile: geopandas.GeoDataFrame,
    crs: dict,
    source_crs: dict = None,
):
    """Read a tile file in using PDAL with input and output CRS specified."""

    # Define instructions for loading in LiDAR
    pdal_pipeline_instructions = [{"type": "readers.las", "filename": str(lidar_file)}]

    # Specify reprojection - if a source_crs is specified use this to define the
    # 'in_srs'
    if source_crs is None:
        pdal_pipeline_instructions.append(
            {
                "type": "filters.reprojection",
                "out_srs": f"EPSG:{crs['horizontal']}+" f"{crs['vertical']}",
            }
        )
    else:
        pdal_pipeline_instructions.append(
            {
                "type": "filters.reprojection",
                "in_srs": f"EPSG:{source_crs['horizontal']}+"
                f"{source_crs['vertical']}",
                "out_srs": f"EPSG:{crs['horizontal']}+" f"{crs['vertical']}",
            }
        )
    # Add instructions for clip within either the catchment, or the land and foreshore
    if region_to_tile is not None:
        pdal_pipeline_instructions.append(
            {
                "type": "filters.crop",
                "polygon": str(region_to_tile.loc[0].geometry),
            }
        )

    # Load in LiDAR and perform operations
    pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions))
    pdal_pipeline.execute()
    return pdal_pipeline


def roughness_from_points(
    point_cloud: numpy.ndarray,
    xy_out: numpy.ndarray,
    xy_ground: numpy.ndarray,
    options: dict,
    eps: float = 0,
    leaf_size: int = 10,
) -> numpy.ndarray:
    """Calculate DEM elevation values at the specified locations using the selected
    approach. Options include: mean, median, and inverse distance weighing (IDW). This
    implementation is based on the scipy.spatial.KDTree"""

    assert len(xy_out) == len(xy_ground), (
        f"xy_out and xy_ground arrays differ in length: {len(xy_out)} vs "
        "{len(xy_ground)}"
    )

    xy_in = numpy.empty((len(point_cloud), 2))
    xy_in[:, 0] = point_cloud["X"]
    xy_in[:, 1] = point_cloud["Y"]

    tree = scipy.spatial.KDTree(xy_in, leafsize=leaf_size)  # build the tree
    tree_index_list = tree.query_ball_point(xy_out, r=options["radius"], eps=eps)
    z_out = numpy.zeros(len(xy_out), dtype=options["raster_type"])

    for i, (near_indicies, ground) in enumerate(zip(tree_index_list, xy_ground)):
        if len(near_indicies) == 0:  # Set NaN if no values in search region
            z_out[i] = numpy.nan
        else:
            height = numpy.mean(point_cloud["Z"][near_indicies]) - ground
            std = numpy.std(point_cloud["Z"][near_indicies])
            parameters = options["parameters"]

            # if building/plantation - set value based on classification
            # Emperical relationship between mean and std above the ground
            z_out[i] = max(std * parameters["std"], height * parameters["mean"])
    return z_out


def elevation_from_points(
    point_cloud: numpy.ndarray,
    xy_out,
    options: dict,
    eps: float = 0,
    leaf_size: int = 10,
) -> numpy.ndarray:
    """Calculate DEM elevation values at the specified locations using the selected
    approach. Options include: mean, median, and inverse distance weighing (IDW). This
    implementation is based on the scipy.spatial.KDTree"""

    xy_in = numpy.empty((len(point_cloud), 2))
    xy_in[:, 0] = point_cloud["X"]
    xy_in[:, 1] = point_cloud["Y"]

    tree = scipy.spatial.KDTree(xy_in, leafsize=leaf_size)  # build the tree
    tree_index_list = tree.query_ball_point(
        xy_out, r=options["radius"], eps=eps
    )  # , eps=0.2)
    z_out = numpy.zeros(len(xy_out), dtype=options["raster_type"])

    for i, (near_indices, point) in enumerate(zip(tree_index_list, xy_out)):
        near_points = tree.data[near_indices]
        near_z = point_cloud["Z"][near_indices]

        z_out[i] = point_elevation(
            near_z=near_z,
            near_points=near_points,
            point=point,
            options=options,
        )
    return z_out


def elevation_from_nearest_points(
    point_cloud: numpy.ndarray,
    edge_point_cloud: numpy.ndarray,
    xy_out,
    options: dict,
    eps: float = 0,
    leaf_size: int = 10,
) -> numpy.ndarray:
    """Calculate DEM elevation values at the specified locations using the selected
    approach. Options include: mean, median, and inverse distance weighing (IDW). This
    implementation is based on the scipy.spatial.KDTree"""

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if len(point_cloud) == 0 and (options["use_edge"] and len(edge_point_cloud) == 0):
        logger.warning("No points provided. Returning NaN array.")
        z_out = numpy.ones(len(xy_out), dtype=options["raster_type"]) * numpy.nan
        return z_out
    k = options["k_nearest_neighbours"]
    if k > len(point_cloud) or (options["use_edge"] and k > len(edge_point_cloud)):
        logger.warning(
            f"Fewer points than the nearest k to search for provided: k = {k} "
            f"> points {len(point_cloud)} or edge points "
            f"{len(edge_point_cloud)}. Returning NaN array."
        )
        z_out = numpy.ones(len(xy_out), dtype=options["raster_type"]) * numpy.nan
        return z_out
    xy_in = numpy.empty((len(point_cloud), 2))
    xy_in[:, 0] = point_cloud["X"]
    xy_in[:, 1] = point_cloud["Y"]
    tree = scipy.spatial.KDTree(xy_in, leafsize=leaf_size)  # build the tree
    (tree_distance_list, tree_index_list) = tree.query(xy_out, k=k, eps=eps)

    if options["use_edge"]:
        xy_in = numpy.empty((len(edge_point_cloud), 2))
        xy_in[:, 0] = edge_point_cloud["X"]
        xy_in[:, 1] = edge_point_cloud["Y"]
        edge_tree = scipy.spatial.KDTree(xy_in, leafsize=leaf_size)  # build the tree
        (edge_tree_distance_list, edge_tree_index_list) = edge_tree.query(
            xy_out, k=k, eps=eps
        )

    z_out = numpy.zeros(len(xy_out), dtype=options["raster_type"])

    for i, point in enumerate(xy_out):
        near_indices = tree_index_list[i]
        near_z = point_cloud["Z"][near_indices]
        near_points = tree.data[near_indices]
        if options["use_edge"]:
            # Add in the edge values as they are nearby
            edge_near_indices = edge_tree_index_list[i]
            near_z = numpy.concatenate(
                (near_z, edge_point_cloud["Z"][edge_near_indices])
            )
            near_points = numpy.concatenate(
                (near_points, edge_tree.data[edge_near_indices])
            )

        z_out[i] = point_elevation(
            near_z=near_z,
            near_points=near_points,
            point=point,
            options=options,
        )

    return z_out


def point_elevation(
    near_z: numpy.ndarray,
    near_points: numpy.ndarray,
    point: numpy.ndarray,
    options: dict,
) -> float:
    """Calculate DEM elevation values at the specified locations using the selected
    approach. Options include: mean, median, and inverse distance weighing (IDW). This
    implementation is based on the scipy.spatial.KDTree"""

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if len(near_z) == 0:  # Set NaN if no values in search region
        z_out = numpy.nan
    else:
        if options["method"] == "mean":
            z_out = numpy.mean(near_z)
        elif options["method"] == "median":
            z_out = numpy.median(near_z)
        elif options["method"] == "idw":
            z_out = calculate_idw(
                near_points=near_points,
                near_z=near_z,
                point=point,
            )
        elif options["method"] in ["cubic", "nearest", "linear"]:
            z_out = calculate_interpolate_griddata(
                near_points=near_points,
                near_z=near_z,
                point=point,
                strict=options["strict"],
                method=options["method"],
            )
        elif options["method"] == "rbf":
            z_out = calculate_rbf(
                near_points=near_points,
                near_z=near_z,
                point=point,
                kernel=options["kernel"],
            )
        elif options["method"] == "min":
            z_out = numpy.min(near_z)
        elif options["method"] == "max":
            z_out = numpy.max(near_z)
        elif options["method"] == "std":
            z_out = numpy.std(near_z)
        elif options["method"] == "count":
            z_out = numpy.len(near_z)
        else:
            assert (
                False
            ), f"An invalid lidar_interpolation_method of '{options['method']}' was"
            " provided"
    return z_out


def calculate_idw(
    near_points: numpy.ndarray,
    near_z: numpy.ndarray,
    point: numpy.ndarray,
    smoothing: float = 0,
    power: int = 2,
):
    """Calculate the IDW mean of the 'near_indices' points. This implementation is based
    on the scipy.spatial.KDTree"""

    # Near points will be ordered by proximinty. Close to far.
    distance_vectors = point - near_points
    smoothed_distances = numpy.sqrt(
        ((distance_vectors**2).sum(axis=1) + smoothing**2)
    )
    if smoothed_distances.min() == 0:  # in the case of an exact match
        idw = near_z[smoothed_distances.argmin()]
    else:
        idw = (near_z / (smoothed_distances**power)).sum(axis=0) / (
            1 / (smoothed_distances**power)
        ).sum(axis=0)
    return idw


def calculate_interpolate_griddata(
    near_points: numpy.ndarray,
    near_z: numpy.ndarray,
    point: numpy.ndarray,
    strict: bool,
    method: str,
):
    """Calculate linear interpolation of the 'near_indices' points. Take the straight
    mean if the points are co-linear or too few for linear interpolation."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if len(near_z) >= 3:  # There are enough points for a linear interpolation
        try:
            value = scipy.interpolate.griddata(
                points=near_points,
                values=near_z,
                xi=point,
                method=method,
            )[0]
        except (scipy.spatial.QhullError, Exception) as caught_exception:
            logger.warning(
                f"Exception {caught_exception} during "
                "linear interpolation. Set to NaN."
            )
            value = numpy.nan

    elif len(near_z) == 1:
        value = near_z[0]
    elif len(near_z) == 2:
        # take the distance weighted average
        distance_vectors = point - near_points
        distances = numpy.sqrt((distance_vectors**2).sum(axis=1))
        value = (near_z / distances).sum(axis=0) / (1 / distances).sum(axis=0)
    else:
        value = numpy.nan
    if numpy.isnan(value) and len(near_z) > 0 and strict:
        logger.warning(
            "NaN - this will occur if colinear points or outside convex hull"
        )
    elif numpy.isnan(value) and len(near_z) > 0 and not strict:
        logger.warning("Was NaN - will estimate as distance weighted mean")
        distance_vectors = point - near_points
        distances = numpy.sqrt((distance_vectors**2).sum(axis=1))
        value = (near_z / distances).sum(axis=0) / (1 / distances).sum(axis=0)
    return value


def calculate_rbf(
    near_points: numpy.ndarray, near_z: numpy.ndarray, point: numpy.ndarray, kernel: str
):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if len(near_z) >= 3:
        if len(near_z) < RBF_CACHE_SIZE:
            logging.warning(
                "Using RBFInterpolator 'neighbors' option "
                f"as {len(near_z)} points nearby"
            )
        try:
            rbf_function = scipy.interpolate.RBFInterpolator(
                y=near_points,
                d=near_z,
                kernel=kernel,
                smoothing=0,
                neighbors=RBF_CACHE_SIZE,
            )
            value = rbf_function([point])
        except (ValueError, Exception) as caught_exception:
            logger.warning(
                f"Exception {caught_exception} during "
                "RBF interpolation. Apply cubic."
            )
            value = calculate_interpolate_griddata(
                near_points=near_points,
                near_z=near_z,
                point=point,
                strict=True,
                method="cubic",
            )
    else:
        logger.warning(
            "Too many or few points for RBF "
            f"interpolation: {len(near_z)}. "
            "Instead applying cubic interpolation."
        )
        value = calculate_interpolate_griddata(
            near_points=near_points,
            near_z=near_z,
            point=point,
            strict=True,
            method="cubic",
        )
    return value


def select_lidar_files(
    tile_index_extents: geopandas.GeoDataFrame,
    tile_index_name_column: str,
    chunk_region_to_tile: geopandas.GeoDataFrame,
    lidar_files_map: typing.Dict[str, pathlib.Path],
) -> typing.List[pathlib.Path]:
    if chunk_region_to_tile.empty:
        return []
    # clip the tile indices to only include those within the chunk region
    chunk_tile_index_extents = geopandas.sjoin(
        chunk_region_to_tile, tile_index_extents.drop(columns=["index_right"])
    )

    # get the LiDAR file with the tile_index_name
    filtered_lidar_files = [
        lidar_files_map[tile_index_name]
        for tile_index_name in chunk_tile_index_extents[tile_index_name_column]
    ]

    return filtered_lidar_files


def load_tiles_in_chunk(
    lidar_files: typing.List[pathlib.Path],
    source_crs: dict,
    chunk_region_to_tile: geopandas.GeoDataFrame,
    crs: dict,
):
    """Read in all LiDAR files within the chunked region - clipped to within
    the region within which to rasterise."""

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.debug(f"Reading all {len(lidar_files)} files in chunk.")

    # Initialise LiDAR points
    lidar_points = []

    # Cycle through each file loading it in an adding it to a numpy array
    if (
        chunk_region_to_tile is None
        or len(chunk_region_to_tile) > 0
        and chunk_region_to_tile.area.sum() > 0
    ):
        for lidar_file in lidar_files:
            logger.debug(f"Loading in file {lidar_file}")

            # read in the LiDAR file
            pdal_pipeline = read_file_with_pdal(
                lidar_file=lidar_file,
                region_to_tile=chunk_region_to_tile,
                source_crs=source_crs,
                crs=crs,
            )
            lidar_points.append(pdal_pipeline.arrays[0])
    if len(lidar_points) > 0:
        lidar_points = numpy.concatenate(lidar_points)
    return lidar_points


def roughness_over_chunk(
    dim_x: numpy.ndarray,
    dim_y: numpy.ndarray,
    tile_points: numpy.ndarray,
    xy_ground: numpy.ndarray,
    options: dict,
) -> numpy.ndarray:
    """Rasterise all points within a chunk."""

    # Get the indicies overwhich to perform IDW
    grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
    xy_out = numpy.concatenate(
        [[grid_x.flatten()], [grid_y.flatten()]], axis=0
    ).transpose()
    grid_z = numpy.ones(grid_x.shape) * numpy.nan

    # If no points return an array of NaN
    if len(tile_points) == 0:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.debug("The latest chunk has no data and is being ignored.")
        return grid_z
    # keep only the specified classifications (should be ground cover)
    classification_mask = numpy.zeros_like(tile_points["Classification"], dtype=bool)
    for classification in options["lidar_classifications_to_keep"]:
        classification_mask[tile_points["Classification"] == classification] = True
    tile_points = tile_points[classification_mask]

    # optionally filter to within the specified elevation range
    elevation_range = options["elevation_range"]
    if elevation_range is not None:
        tile_points = tile_points[tile_points["Z"] >= elevation_range[0]]
        tile_points = tile_points[tile_points["Z"] <= elevation_range[1]]
    # Check again - if no points return an array of NaN
    if len(tile_points) == 0:
        return grid_z
    # Perform the point cloud roughness estimation method over chunk
    z_flat = roughness_from_points(
        point_cloud=tile_points,
        xy_out=xy_out,
        xy_ground=xy_ground,
        options=options,
    )
    grid_z = z_flat.reshape(grid_x.shape)

    return grid_z


def elevation_over_chunk(
    dim_x: numpy.ndarray,
    dim_y: numpy.ndarray,
    tile_points: numpy.ndarray,
    options: dict,
) -> numpy.ndarray:
    """Rasterise all points within a chunk."""

    # Get the indicies overwhich to perform IDW
    grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
    xy_out = numpy.concatenate(
        [[grid_x.flatten()], [grid_y.flatten()]], axis=0
    ).transpose()
    grid_z = numpy.ones(grid_x.shape) * numpy.nan

    # If no points return an array of NaN
    if len(tile_points) == 0:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.debug(" The latest chunk has no data and is being ignored.")
        return grid_z
    # keep only the specified classifications (should be ground / water)
    # Not used for coarse DEM
    if "lidar_classifications_to_keep" in options:
        classification_mask = numpy.zeros_like(
            tile_points["Classification"], dtype=bool
        )
        for classification in options["lidar_classifications_to_keep"]:
            classification_mask[tile_points["Classification"] == classification] = True
        tile_points = tile_points[classification_mask]

    # optionally filter to within the specified elevation range
    elevation_range = options["elevation_range"]
    if elevation_range is not None:
        tile_points = tile_points[tile_points["Z"] >= elevation_range[0]]
        tile_points = tile_points[tile_points["Z"] <= elevation_range[1]]
    # Check again - if no points return an array of NaN
    if len(tile_points) == 0:
        return grid_z
    # Perform the specified averaging method over the dense DEM within the extents of
    # this point cloud tile
    z_flat = elevation_from_points(
        point_cloud=tile_points, xy_out=xy_out, options=options
    )
    grid_z = z_flat.reshape(grid_x.shape)

    return grid_z


def elevation_over_chunk_from_nearest(
    dim_x: numpy.ndarray,
    dim_y: numpy.ndarray,
    points: numpy.ndarray,
    edge_points: numpy.ndarray,
    options: dict,
) -> numpy.ndarray:
    """Rasterise all points within a chunk."""

    # Get the indicies overwhich to perform IDW
    grid_x, grid_y = numpy.meshgrid(dim_x, dim_y)
    xy_out = numpy.concatenate(
        [[grid_x.flatten()], [grid_y.flatten()]], axis=0
    ).transpose()
    grid_z = numpy.ones(grid_x.shape) * numpy.nan

    # If no points return an array of NaN
    if len(points) == 0 and len(edge_points) == 0:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.debug(" The latest chunk has no data and is being ignored.")
        return grid_z

    # Perform the specified averaging method over the dense DEM within the extents of
    # this point cloud tile
    z_flat = elevation_from_nearest_points(
        point_cloud=points, edge_point_cloud=edge_points, xy_out=xy_out, options=options
    )
    grid_z = z_flat.reshape(grid_x.shape)

    return grid_z


""" Wrap the `roughness_over_chunk` routine in dask.delayed """
delayed_roughness_over_chunk = dask.delayed(roughness_over_chunk)

""" Wrap the `elevation_over_chunk` routine in dask.delayed """
delayed_elevation_over_chunk = dask.delayed(elevation_over_chunk)

""" Wrap the `elevation_over_chunk` routine in dask.delayed """
delayed_elevation_over_chunk_from_nearest = dask.delayed(
    elevation_over_chunk_from_nearest
)

""" Wrap the `load_tiles_in_chunk` routine in dask.delayed """
delayed_load_tiles_in_chunk = dask.delayed(load_tiles_in_chunk)
