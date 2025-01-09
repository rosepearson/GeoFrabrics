# -*- coding: utf-8 -*-
"""
This module contains classes associated with manipulating vector data.
"""
import geopandas
import pandas
import shapely
import numpy
import pathlib
import typing
import logging

RASTER_TYPE = numpy.float32


class CatchmentGeometry:
    """A class defining revelant catchment regions by polygons.

    The CRS is a dictionary containing the EPSG code for a 'horizontal' and
    'vertical' datum.

    Specifically, this defines polygon regions like 'land', 'foreshore',
    'offshore', and ensures all regions are defined in the same CRS.

    The 'land', 'foreshore' and 'offshore' regions are clipped within the
    'catchment', but do not overlap. The 'land' is polygon is defined by a
    specified polygon which is clipped within the 'catchment'. The 'foreshore'
    is defined as a 'foreshore_buffer' x 'resolution' outward buffer from the
    'land' and within the 'catchment' extents. The 'offshore' region is any
    remaining region within the 'catchment' and outside the 'land' and
    'foreshore' polygons.

    It also supports functions for determining how much of a region is outside an
    exclusion zone. I.E. is outside
    `lidar_extents` see class method land_and_foreshore_without_lidar for an example.

    It is initalised with the catchment. The land must be added before the other regions
    (i.e. land, offshore,
    foreshore, etc) can be accessed."""

    def __init__(
        self,
        catchment_file: typing.Union[str, pathlib.Path],
        crs: dict,
        resolution: float,
        foreshore_buffer: int = 2,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._catchment = geopandas.read_file(catchment_file)
        self.crs = crs
        self.resolution = resolution
        self.foreshore_buffer = foreshore_buffer

        # set catchment CRS
        self._catchment = self._catchment.to_crs(self.crs["horizontal"])

        # values set in setup once land has been specified
        self._land = None
        self._full_land = None
        self._foreshore = None
        self._land_and_foreshore = None
        self._foreshore_and_offshore = None
        self._offshore = None

    def _set_up(self):
        """Define the main catchment regions and ensure the CRS is set for
        each region"""

        self._full_land = self._full_land.to_crs(self.crs["horizontal"])

        # Clip and remove any sub pixel regions
        self._land = self._catchment.clip(self._full_land, keep_geom_type=True)
        self._land = self._land[self._land.area > self.resolution * self.resolution]
        if self._land.area.sum() > 0:
            self._foreshore_and_offshore = self.catchment.overlay(
                self.land, how="difference"
            )
            # Buffer and clip and remove any sub pixel regions
            self._land_and_foreshore = geopandas.GeoDataFrame(
                {"geometry": self.land.buffer(self.resolution * self.foreshore_buffer)},
                crs=self.crs["horizontal"],
            )
            self._land_and_foreshore = self._catchment.clip(
                self._land_and_foreshore, keep_geom_type=True
            )
            self._land_and_foreshore = self._land_and_foreshore[
                self._land_and_foreshore.area > self.resolution * self.resolution
            ]

            self._foreshore = self._land_and_foreshore.overlay(
                self.land, how="difference"
            )

            self._offshore = self.catchment.overlay(
                self.land_and_foreshore, how="difference"
            )
        else:
            # Assume no foreshore
            self._foreshore_and_offshore = self.catchment
            self._foreshore = self._land
            self._land_and_foreshore = self._land
            self._offshore = self.catchment

        assert len(self._catchment) == 1, (
            "The catchment is made of multiple separate polygons it must be a single "
            + "polygon object - a MultiPolygon is fine"
        )

    def _assert_land_set(self):
        """Check to make sure the 'land' has been set."""
        assert self._land is not None, (
            "Land has not been set yet. This must be set before anything other than the"
            " `catchment` can be returned from a `CatchmentGeometry` object"
        )

    @property
    def catchment(self):
        """Return the catchment region"""

        return self._catchment

    @property
    def land(self):
        """Return the catchment land region"""

        self._assert_land_set()
        return self._land

    @property
    def full_land(self):
        """Return the full land region"""

        self._assert_land_set()
        return self._full_land

    @land.setter
    def land(self, land_file: typing.Union[str, pathlib.Path]):
        """Set the land region and finish setup."""

        self._full_land = geopandas.read_file(land_file)

        self._set_up()

    @property
    def foreshore(self):
        """Return the catchment foreshore region"""

        self._assert_land_set()
        return self._foreshore

    @property
    def land_and_foreshore(self):
        """Return the catchment land and foreshore region"""

        self._assert_land_set()
        return self._land_and_foreshore

    @property
    def foreshore_and_offshore(self):
        """Return the catchment foreshore and offshore region"""

        self._assert_land_set()
        return self._foreshore_and_offshore

    @property
    def offshore(self):
        """Return the catchment offshore region"""

        self._assert_land_set()
        return self._offshore

    def land_and_foreshore_without_lidar(self, dense_extents: geopandas.GeoDataFrame):
        """Return the land and foreshore region without LiDAR"""

        self._assert_land_set()

        if dense_extents is None or dense_extents.is_empty.all():
            self.logger.warning(
                "In CatchmentGeometry dense extents are `None` so "
                " `land_and_foreshore_without_lidar` is returning `land_and_foreshore`"
            )
            return self.land_and_foreshore
        elif self.land_and_foreshore.area.sum() == 0:
            # There is no land and foreshore region - return an empty dataframe
            return self.land_and_foreshore
        # Clip to remove any offshore regions before doing a difference overlay. Drop
        # any sub-pixel polygons.
        land_and_foreshore_with_lidar = dense_extents.clip(
            self.land_and_foreshore, keep_geom_type=True
        )
        land_and_foreshore_with_lidar = land_and_foreshore_with_lidar[
            land_and_foreshore_with_lidar.area > self.resolution * self.resolution
        ]
        land_and_foreshore_without_lidar = self.land_and_foreshore.overlay(
            land_and_foreshore_with_lidar, how="difference"
        )

        return land_and_foreshore_without_lidar

    def offshore_without_lidar(self, dense_extents: geopandas.GeoDataFrame):
        """Return the offshore region without LiDAR"""

        self._assert_land_set()

        if dense_extents is None or dense_extents.is_empty.all():
            self.logger.warning(
                "In CatchmentGeometry dense extents are `None` so "
                "`offshore_without_lidar` is returning `offshore`"
            )
            return self.offshore
        elif (
            self.offshore.area.sum() == 0
        ):  # There is no offshore region - return an empty dataframe
            return self.offshore
        # Clip to remove any offshore regions before doing a difference overlay. Drop
        # any sub-pixel polygons.
        offshore_with_lidar = dense_extents.clip(self.offshore, keep_geom_type=True)
        offshore_with_lidar = offshore_with_lidar[
            offshore_with_lidar.area > self.resolution**2
        ]
        offshore_without_lidar = geopandas.overlay(
            self.offshore, offshore_with_lidar, how="difference"
        )

        return offshore_without_lidar

    def offshore_dense_data_edge(self, dense_extents: geopandas.GeoDataFrame):
        """Return the offshore edge of where there is 'dense data' i.e. LiDAR or
        reference DEM"""

        self._assert_land_set()

        if dense_extents is None or dense_extents.is_empty.all():
            self.logger.warning(
                "In CatchmentGeometry dense extents are `None` so "
                "`offshore_dense_data_edge` is returning `None`"
            )
            return dense_extents
        elif self.foreshore_and_offshore.area.sum() == 0:
            # If no offshore and foreshore just return the emptry geometry
            return self.foreshore

        # the foreshore and whatever lidar extents are offshore
        offshore_foreshore_dense_data_extents = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.ops.unary_union(
                        [
                            self.foreshore.loc[0].geometry,
                            dense_extents.loc[0].geometry,
                        ]
                    )
                ]
            },
            crs=self.crs["horizontal"],
        )
        offshore_foreshore_dense_data_extents = (
            offshore_foreshore_dense_data_extents.clip(
                self.foreshore_and_offshore, keep_geom_type=True
            )
        )

        # deflate this - this will be taken away from the
        # offshore_foreshore_dense_data_extents to give the edge
        deflated_dense_data_extents = geopandas.GeoDataFrame(
            {
                "geometry": offshore_foreshore_dense_data_extents.buffer(
                    self.resolution * -self.foreshore_buffer
                )
            },
            crs=self.crs["horizontal"],
        )

        # get the difference between them
        if deflated_dense_data_extents.area.sum() > 0:
            offshore_dense_data_edge = offshore_foreshore_dense_data_extents.overlay(
                deflated_dense_data_extents, how="difference"
            )
        else:
            offshore_dense_data_edge = offshore_foreshore_dense_data_extents
        offshore_dense_data_edge = offshore_dense_data_edge.clip(
            self.foreshore_and_offshore, keep_geom_type=True
        )
        return offshore_dense_data_edge

    def offshore_no_dense_data(self, lidar_extents):
        """Return the offshore area where there is no 'dense data' i.e. LiDAR"""

        assert len(lidar_extents) == 1, "LiDAR extents has a length greater than 1"

        assert self._land is not None, (
            "Land has not been set yet. This must be set before anything other than the"
            + " `catchment` can be returned from a `CatchmentGeometry` object"
        )

        # lidar extents are offshore - drop any sub pixel areas
        offshore_dense_data = lidar_extents.clip(self.offshore, keep_geom_type=True)
        offshore_dense_data = offshore_dense_data[
            offshore_dense_data.area > self.resolution * self.resolution
        ]

        # get the difference between them
        offshore_no_dense_data = self.offshore.overlay(
            offshore_dense_data, how="difference"
        )

        return offshore_no_dense_data


class BathymetryContours:
    """A class for sampling from bathymetry contours.

    Assumes contours to be sampled to the catchment_geometry resolution"""

    def __init__(
        self,
        contour_file: str,
        catchment_geometry: CatchmentGeometry,
        z_label=None,
        exclusion_extent=None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._contour = geopandas.read_file(contour_file)
        self.catchment_geometry = catchment_geometry
        self.z_label = z_label

        self._extent = None

        self._set_up(exclusion_extent)

    def _set_up(self, exclusion_extent):
        """Set CRS and clip to catchment"""

        self._contour = self._contour.to_crs(self.catchment_geometry.crs["horizontal"])

        # Define a large offshore area to keep ocean contours in - to ensure correct
        # interpolation behaviour near the edges
        offshore = self.catchment_geometry.offshore
        offshore = geopandas.GeoDataFrame(
            geometry=offshore.buffer(numpy.sqrt(offshore.area.sum()))
        )
        offshore = offshore.overlay(self.catchment_geometry.land, how="difference")

        if exclusion_extent is not None:
            # Remove areas already covered by LiDAR - drop any polygons less than a
            # pixel in area
            exclusion_extent = exclusion_extent.clip(offshore, keep_geom_type=True)
            exclusion_extent = exclusion_extent[
                exclusion_extent.area > self.catchment_geometry.resolution**2
            ]
            self._extent = offshore.overlay(exclusion_extent, how="difference")
        else:
            self._extent = offshore
        # Keep only contours in the 'extents' i.e. inside the catchment and outside any
        # exclusion_extent
        self._contour = self._contour.clip(self._extent, keep_geom_type=True)
        self._contour = self._contour.reset_index(drop=True)

        # Convert any 'GeometryCollection' objects to 'MultiLineString' objects
        # - dropping any points
        if (self._contour.geometry.type == "GeometryCollection").any():
            geometry_list = []
            for geometry_row in self._contour.geometry:
                if geometry_row.geometryType() in {
                    "LineString",
                    "MultiLineString",
                }:
                    geometry_list.append(geometry_row)
                elif geometry_row.geometryType() == "GeometryCollection":
                    geometry_list.append(
                        shapely.geometry.MultiLineString(
                            [
                                geometry_element
                                for geometry_element in geometry_row
                                if geometry_element.geometryType() == "LineString"
                            ]
                        )
                    )
            self._contour.set_geometry(geometry_list, inplace=True)

    def sample_contours(self, resolution: float) -> numpy.ndarray:
        """Sample the contours at the specified resolution."""

        assert (
            resolution > 0
        ), f"The sampling resolution must be greater than 0, but is {resolution}."

        points_df = self._contour.geometry.apply(
            lambda row: shapely.geometry.MultiPoint(
                [
                    row.interpolate(i * resolution)
                    for i in range(int(numpy.ceil(row.length / resolution)))
                ]
            )
        )
        if len(points_df) == 0:
            # No data, return an empty array
            return []

        points = numpy.empty(
            [points_df.apply(lambda row: len(row.geoms)).sum()],
            dtype=[("X", RASTER_TYPE), ("Y", RASTER_TYPE), ("Z", RASTER_TYPE)],
        )

        # Extract the x, y and z values from the Shapely MultiPoints and possibly a
        # depth column
        points["X"] = numpy.concatenate(
            points_df.apply(lambda row: [row_i.x for row_i in row.geoms]).to_list()
        )
        points["Y"] = numpy.concatenate(
            points_df.apply(lambda row: [row_i.y for row_i in row.geoms]).to_list()
        )
        if self.z_label is None:
            points["Z"] = (
                numpy.concatenate(
                    points_df.apply(
                        lambda row: [row_i.z for row_i in row.geoms]
                    ).to_list()
                )
                * -1
            )
        else:
            points["Z"] = (
                numpy.concatenate(
                    [
                        numpy.ones(len(points_df.loc[i].geoms))
                        * self._contour[self.z_label].loc[i]
                        for i in range(len(points_df))
                    ]
                )
                * -1
            )
        return points


class OceanPoints:
    """A class for accesing marine bathymetry points. These can be used as
    depths to interpolate elevations offshore."""

    def __init__(
        self,
        points_file: str,
        catchment_geometry: CatchmentGeometry,
        is_depth: bool = False,
        z_label: str = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._points = geopandas.read_file(points_file)
        self.catchment_geometry = catchment_geometry
        self.is_depth = is_depth
        self.z_label = z_label

        self._set_up()

    def _set_up(self):
        """Set CRS and clip to > 2x catchment area."""

        self._points = self._points.to_crs(self.catchment_geometry.crs["horizontal"])
        self._points = self._points.explode(ignore_index=True)
        self._points = self._points.clip(
            self.catchment_geometry.catchment.buffer(
                numpy.sqrt(self.catchment_geometry.catchment.area.max())
            )
        )

    @property
    def boundary(self):
        """Return the bounding box containing data"""

        bounds = self._points.total_bounds
        boundary = geopandas.GeoDataFrame(
            geometry=[
                shapely.geometry.Polygon(
                    [
                        [bounds[0], bounds[1]],
                        [bounds[2], bounds[1]],
                        [bounds[2], bounds[3]],
                        [bounds[0], bounds[3]],
                    ]
                )
            ],
            crs=self._points.crs,
        )
        return boundary

    @property
    def points(self):
        """Return the points"""

        return self._points

    @property
    def x(self):
        """The x values"""

        if self._x is None:
            self._x = self._points.apply(
                lambda row: row["geometry"].x, axis=1
            ).to_numpy()
        return self._x

    @property
    def y(self):
        """The y values"""

        if self._y is None:
            self._y = self._points.apply(
                lambda row: row["geometry"].y, axis=1
            ).to_numpy()
        return self._y

    @property
    def z(self):
        """The z values"""

        if self._z is None:
            # map depth to elevation
            multiplier = 1 if not self.is_depth else -1
            if self.z_label is None:
                self._z = (
                    self._points.apply(lambda row: row["geometry"].z, axis=1).to_numpy()
                    * multiplier
                )
            else:
                self._z = self._points[self.z_label].to_numpy()
        return self._z

    def sample(self) -> numpy.ndarray:
        """Return points - rename in future."""
        points = numpy.empty(
            [len(self._points)],
            dtype=[("X", RASTER_TYPE), ("Y", RASTER_TYPE), ("Z", RASTER_TYPE)],
        )

        # Extract the x, y and z values from the Shapely MultiPoints and possibly a
        # depth column
        points["X"] = self._points.apply(
            lambda row: row["geometry"].x, axis=1
        ).to_numpy()
        points["Y"] = self._points.apply(
            lambda row: row["geometry"].y, axis=1
        ).to_numpy()
        if self.is_depth:
            points["Z"] = (
                self._points.apply(lambda row: row["geometry"].z, axis=1).to_numpy()
                * -1
            )
        else:
            points["Z"] = self._points.apply(
                lambda row: row["geometry"].z, axis=1
            ).to_numpy()

        return points


class ElevationPoints:
    """A class for accessing estimated or measured river, mouth and waterway
    elevations as points. Paired elevation and polygon files are expected.
    The elevations are used to interpolate elevations within the polygons.
    """

    Z_LABEL = "z"
    BANK_HEIGHT_LABEL = "bank_height"
    WIDTH_LABEL = "width"
    OSM_ID = "OSM_id"

    def __init__(
        self,
        points_files: list,
        polygon_files: list,
        catchment_geometry: CatchmentGeometry,
        filter_osm_ids: list = [],
        z_labels: list = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.catchment_geometry = catchment_geometry

        self.z_label = z_labels is not None
        self._points = None
        self._polygon = None

        self._set_up(points_files, polygon_files, z_labels, filter_osm_ids)

    def _set_up(
        self,
        points_files: list,
        polygon_files: list,
        z_labels: list,
        filter_osm_ids: list,
    ):
        """Load point and polygon files and concatentate and clip to the catchment."""

        if len(points_files) != len(polygon_files):
            raise ValueError(
                "The polygon and point lists should all be the same length. Instead there "
                f"are {len(points_files)} points files and {len(polygon_files)} polygon "
                "files"
            )

        if z_labels is not None and type(z_labels) is str:
            z_labels = [z_labels for i in range(len(points_files))]

        if z_labels is not None and len(z_labels) != len(points_files):
            raise ValueError(
                "There is a mismatch in length between the provided z_labels "
                f"and the point files: {z_labels} {points_files}"
            )

        # Points - rename depth labels to standard if a non standard one is specified
        points_list = []
        polygon_list = []
        for i in range(0, len(points_files)):
            # Points - rename depth labels to standard if  specified
            points_i = geopandas.read_file(points_files[i])
            if z_labels is not None and z_labels[i] != self.Z_LABEL:
                points_i = points_i.rename(columns={z_labels[i]: self.Z_LABEL})
            columns_i = [self.Z_LABEL, "geometry"]
            if self.BANK_HEIGHT_LABEL in points_i.columns:
                columns_i.append(self.BANK_HEIGHT_LABEL)
            if self.WIDTH_LABEL in points_i.columns:
                columns_i.append(self.WIDTH_LABEL)
            if self.OSM_ID in points_i.columns:
                columns_i.append(self.OSM_ID)
            # Only add the points and polygons if their are points
            if len(points_i) > 0:
                points_i = points_i[columns_i]
                points_list.append(points_i)
                # Polygon where the points are relevent
                polygon_i = geopandas.read_file(polygon_files[i])
                polygon_list.append(polygon_i)
        # Set CRS, clip to size and reset index
        if len(points_list) == 0:
            self.logger.warning("No elevations. Ignoring.")
            self._points = []
            self._polygon = []
            return
        points = geopandas.GeoDataFrame(
            pandas.concat(points_list, ignore_index=True),
            crs=points_list[0].crs,
        )
        points = points.to_crs(self.catchment_geometry.crs["horizontal"])
        polygon = geopandas.GeoDataFrame(
            pandas.concat(polygon_list, ignore_index=True),
            crs=polygon_list[0].crs,
        )
        polygon = polygon.to_crs(self.catchment_geometry.crs["horizontal"])
        points = points.clip(polygon.buffer(0), keep_geom_type=True)
        points = points.clip(self.catchment_geometry.catchment, keep_geom_type=True)
        points = points.reset_index(drop=True)

        # Filter if there are OSM IDs specified to filter
        if len(filter_osm_ids) > 0:
            for osm_id in filter_osm_ids:
                points = points[points[self.OSM_ID] != osm_id]
                polygon = polygon[polygon[self.OSM_ID] != osm_id]

        # Set to class members
        self._points = points
        self._polygon = polygon

    @property
    def polygons(self) -> geopandas.GeoDataFrame:
        """Return the polygons."""
        return self._polygon

    @property
    def points_array(self) -> numpy.ndarray:
        """Return the points as a single array."""

        points = self._points

        points_array = numpy.empty(
            [len(points)],
            dtype=[("X", RASTER_TYPE), ("Y", RASTER_TYPE), ("Z", RASTER_TYPE)],
        )

        if len(points_array) == 0:
            return points_array
        # Extract the x, y and z values from the Shapely MultiPoints and possibly a
        # depth column
        points_array["X"] = points.apply(lambda row: row.geometry.x, axis=1).to_list()
        points_array["Y"] = points.apply(lambda row: row.geometry.y, axis=1).to_list()
        if self.z_label:
            points_array["Z"] = points.apply(
                lambda row: row[self.Z_LABEL], axis=1
            ).to_list()
        else:
            points_array["Z"] = points.apply(
                lambda row: row.geometry.z, axis=1
            ).to_list()
        return points_array

    @property
    def points(self):
        """Return the points"""

        return self._points

    def bank_heights_exist(self) -> bool:
        """Return true if the bank heights are defined in array"""
        return (
            self.BANK_HEIGHT_LABEL in self._points.columns
            and self._points[self.BANK_HEIGHT_LABEL].any()
        )

    def bank_height_points(self) -> numpy.ndarray:
        """Return the points with 'Z' being the estimated bank height as a single array."""

        # Filter the points by the type label
        points = self._points

        points_array = numpy.empty(
            [len(points)],
            dtype=[("X", RASTER_TYPE), ("Y", RASTER_TYPE), ("Z", RASTER_TYPE)],
        )

        # Extract the x, y and z values from the Shapely MultiPoints and possibly a
        # depth column
        points_array["X"] = points.apply(lambda row: row.geometry.x, axis=1).to_list()
        points_array["Y"] = points.apply(lambda row: row.geometry.y, axis=1).to_list()

        # Pull out the bank heights
        points_array["Z"] = points.apply(
            lambda row: row[self.BANK_HEIGHT_LABEL], axis=1
        ).to_list()
        return points_array

    @property
    def x(self):
        """The x values"""

        if self._x is None:
            self._x = self._points.points.apply(
                lambda row: row["geometry"][0].x, axis=1
            ).to_numpy()
        return self._x

    @property
    def y(self):
        """The y values"""

        if self._y is None:
            self._y = self._points.points.apply(
                lambda row: row["geometry"][0].y, axis=1
            ).to_numpy()
        return self._y

    @property
    def z(self):
        """The z values"""

        if self.z_label:
            self._z = self._points.apply(
                lambda row: row[self.Z_LABEL], axis=1
            ).to_list()
        else:
            self._z = self._points.apply(lambda row: row.geometry.z, axis=1).to_list()
        return self._z


class ElevationContours(ElevationPoints):
    """Resample at spatial resolution at points"""

    def __init__(
        self,
        points_files: list,
        polygon_files: list,
        catchment_geometry: CatchmentGeometry,
        z_labels: list = None,
    ):
        super(ElevationContours, self).__init__(
            points_files=points_files,
            polygon_files=polygon_files,
            catchment_geometry=catchment_geometry,
            filter_osm_ids=[],
            z_labels=z_labels,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # convert contoursto samples points at resolution
        self.sample_contours(self.catchment_geometry.resolution)

    def sample_contours(self, resolution: float) -> numpy.ndarray:
        """Sample the contours at the specified resolution."""

        # convert contours to multipoints
        self._points.loc[:, "geometry"] = self._points.geometry.apply(
            lambda row: shapely.geometry.MultiPoint(
                [
                    row.interpolate(i * resolution)
                    for i in range(int(numpy.ceil(row.length / resolution)))
                ]
            )
        )
        self._points = self._points.explode(index_parts=True, ignore_index=True)


class TileInfo:
    """A class for working with tiling information."""

    def __init__(self, tile_file: str, catchment_geometry: CatchmentGeometry):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._tile_info = geopandas.read_file(tile_file)
        self.catchment_geometry = catchment_geometry

        self.file_name = None

        self._set_up()

    def _set_up(self):
        """Set CRS and select all tiles partially within the catchment, and look up the
        file column name"""

        self._tile_info = self._tile_info.to_crs(
            self.catchment_geometry.crs["horizontal"]
        )
        self._tile_info = geopandas.sjoin(
            self._tile_info, self.catchment_geometry.catchment
        )
        self._tile_info = self._tile_info.reset_index(drop=True)

        column_names = self._tile_info.columns

        column_name_matches = [
            name for name in column_names if "filename" == name.lower()
        ]
        column_name_matches.extend(
            [name for name in column_names if "file_name" == name.lower()]
        )

        assert len(column_name_matches) == 1, (
            "No single `file name` column detected in the tile file with"
            + f" columns: {column_names}"
        )
        self.file_name = column_name_matches[0]

    @property
    def tile_names(self):
        """Return the names of all tiles within the catchment"""

        return self._tile_info[self.file_name]


class RiverMouthFan:
    """A class for creating an appropiate river mouth fan to transition from
    river depth estimates to the ocean bathymetry values. This fan region defines a
    transition from river to coast within a fan shaped polygon (15 degrees on each
                                                                side).
    The fan begins with the most downstream river width estimate, and ends with the
    first contour of either more than 2x the depth of the mouth. This aims to ensure
    their is a defined fan that is slightly deeper than the surrounding during
    the transition. A fan is caculated for both river bed elevation approach.
    TODO In future,it may move to defining the width as 10x the mouth width.
    If no width at mouth - move back to where there is a valid width

    Attributes:
        crs  The horizontal CRS to be used. i.e. EPSG:2193
        cross_section_spacing  The spacing in (m) of the sampled cross sections.
        aligned_channel_file  Thefile name for the aligned river channel file.
        river_bathymetry_file  The file name for the river bathymetry values.
        ocean_contour_file  The file name for the ocean contours.
        ocean_contour_depth_label  The column label for the depth values.
    """

    FAN_ANGLE = 30
    FAN_MAX_LENGTH = 10_000

    def __init__(
        self,
        aligned_channel_file: str,
        river_bathymetry_file: str,
        river_polygon_file: str,
        ocean_contour_file: str,
        ocean_points_file: str,
        crs: int,
        cross_section_spacing: float,
        elevation_labels: list,
        ocean_contour_depth_label: str = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.crs = crs
        self.cross_section_spacing = cross_section_spacing
        self.aligned_channel_file = aligned_channel_file
        self.river_polygon_file = river_polygon_file
        self.river_bathymetry_file = river_bathymetry_file
        self.ocean_contour_file = ocean_contour_file
        self.ocean_contour_depth_label = ocean_contour_depth_label
        self.elevation_labels = elevation_labels
        self.ocean_points_file = ocean_points_file

    def _get_mouth_alignment(self):
        """Get the location and alignment of the river mouth."""

        # Get the river alignment and clip to the river polygon
        aligned_channel = geopandas.read_file(self.aligned_channel_file)
        river_polygon = geopandas.read_file(self.river_polygon_file).make_valid()
        aligned_channel = aligned_channel.clip(river_polygon)
        # Explode incase the aligned channel is clipped into a MultiPolyLine
        (x, y) = aligned_channel.explode().iloc[0].geometry.xy

        # Calculate the normal and tangent to the channel segment at the mouth
        segment_dx = x[0] - x[1]
        segment_dy = y[0] - y[1]
        segment_length = numpy.sqrt(segment_dx**2 + segment_dy**2)
        mouth_tangent = shapely.geometry.Point(
            [segment_dx / segment_length, segment_dy / segment_length]
        )
        mouth_normal = shapely.geometry.Point([-mouth_tangent.y, mouth_tangent.x])

        # Get the midpoint of the river mouth from the river bathymetry
        river_bathymetry = geopandas.read_file(self.river_bathymetry_file)
        river_bathymetry = river_bathymetry.clip(
            river_polygon.buffer(self.cross_section_spacing / 2)
        ).sort_index(ascending=True)
        mouth_point = river_bathymetry.iloc[0].geometry

        return mouth_point, mouth_tangent, mouth_normal

    def _get_mouth_bathymetry(self):
        """Get the width and depth at the river mouth."""

        # Get river bathymetry values and clip to the river polygon
        river_bathymetry = geopandas.read_file(self.river_bathymetry_file)
        river_polygon = geopandas.read_file(self.river_polygon_file)
        river_bathymetry = river_bathymetry.clip(
            river_polygon.buffer(self.cross_section_spacing / 2)
        ).sort_index(ascending=True)

        river_mouth_elevations = [
            river_bathymetry[elevation_label].iloc[0]
            for elevation_label in self.elevation_labels
        ]
        river_mouth_width = river_bathymetry["width"].iloc[0]
        return river_mouth_width, river_mouth_elevations

    def _get_ocean_contours(
        self,
        river_mouth_depth,
        depth_sign: int = -1,
        depth_multiplier: int = 2,
    ) -> geopandas.GeoDataFrame:
        """Load in the ocean contours and remove those that are too close to the
        shoreline to consider as the slope change would be too great.

        Parameters:
            river_mouth_depth  The depth in m of the river mouth
            depth_sign  The sign of the depths (-1 means depths are positive)
            depth_multiplier  Number of times deeped the end contour should be
                than the river mouth
        """

        assert (
            self.ocean_contour_depth_label is not None
        ), "Ocean z-values must currently be stored a separate column currently."

        # Load in the ocean contours and find the contours to terminate against
        ocean_contours = geopandas.read_file(self.ocean_contour_file).to_crs(self.crs)
        depth_label = self.ocean_contour_depth_label

        # Determine the end depth and filter the contours to include only these contours
        ocean_contours = ocean_contours[
            ocean_contours[depth_label]
            > depth_multiplier * river_mouth_depth * depth_sign
        ].reset_index(drop=True)

        assert (
            len(ocean_contours) > 0
        ), f"No contours exist with a depth {depth_multiplier}x the river mouth depth. "

        return ocean_contours
    
    def _get_distance_to_nearest_ocean_point_in_fan(
        self,
        river_mouth_z,
        fan,
        mouth_point,
        depth_multiplier,
    ) -> geopandas.GeoDataFrame:
        """Load in the ocean points and get the nearest one to the coast that is deeper than
        the river.

        Parameters:
            river_mouth_depth  The depth in m of the river mouth
            depth_multiplier  Number of times deeped the end contour should be
                than the river mouth
        """

        # Load in the ocean contours and find the contours to terminate against
        ocean_points = geopandas.read_file(self.ocean_points_file).to_crs(self.crs)

        # Determine the end depth and filter the contours to include only these contours
        ocean_points = ocean_points.clip(fan).reset_index(drop=True)
        if river_mouth_z > 0:
            river_mouth_z = 0
        ocean_points = ocean_points[
            ocean_points["Z"] < depth_multiplier * river_mouth_z
        ].reset_index(drop=True)

        if len(ocean_points) == 0:
            raise ValueError(f"No points exist within fan at a depth {depth_multiplier}x the river mouth depth. ")
        
        distance = ocean_points.distance(mouth_point).min()
        elevation = ocean_points.iloc[ocean_points.distance(mouth_point).idxmin()]["Z"]

        return distance, elevation

    def _bathymetry(
        self,
        intersection_line: shapely.geometry.LineString,
        river_mouth_elevations: list,
        end_depth: float,
        mouth_point: shapely.geometry.Point,
        mouth_tangent: shapely.geometry.Point,
    ):
        """Calculate and return the fan bathymetry values for both river bed
        estimation approaches.

        Parameters:
            intersection_line  The contour line defining the end of the fan
            river_mouth_depth  The depth in m of the river mouth
            end_depth  The depth of the end contour of the fan in m
            mouth_point  The location of the centre of the river mouth
            mouth_tangent The tangent to the river mouth (along channel axis)
        """

        # Get the length of the fan centreline
        fan_centre = shapely.geometry.LineString(
            [
                mouth_point,
                [
                    mouth_point.x + self.FAN_MAX_LENGTH * mouth_tangent.x,
                    mouth_point.y + self.FAN_MAX_LENGTH * mouth_tangent.y,
                ],
            ]
        )
        distance = fan_centre.intersection(intersection_line).distance(mouth_point)

        # Setup the fan data values
        fan_depths = {
            **{"geometry": []},
            **dict((elevation_label, []) for elevation_label in self.elevation_labels),
        }
        number_of_samples = int(distance / self.cross_section_spacing)
        depth_increments = [
            (-1 * end_depth - river_mouth_elevation) / number_of_samples
            for river_mouth_elevation in river_mouth_elevations
        ]

        # Iterate through creating fan bathymetry
        for i in range(1, number_of_samples):
            fan_depths["geometry"].append(
                shapely.geometry.Point(
                    [
                        mouth_point.x
                        + mouth_tangent.x * i * self.cross_section_spacing,
                        mouth_point.y
                        + mouth_tangent.y * i * self.cross_section_spacing,
                    ]
                )
            )
            for elevation_label, river_mouth_elevation, depth_increment in zip(
                self.elevation_labels, river_mouth_elevations, depth_increments
            ):
                fan_depths[elevation_label].append(
                    river_mouth_elevation + i * depth_increment
                )
        fan_depths = geopandas.GeoDataFrame(fan_depths, crs=self.crs)
        return fan_depths

    def _fixed_length_fan_polygon(
        self,
        river_mouth_width: float,
        mouth_point: float,
        mouth_tangent: float,
        mouth_normal: float,
        length: float
    ):
        """Return the fan polygon of maximum length. This will be used to
        produce data to the first contour at least 2x the depth of the river
        mouth.

        Parameters:
            river_mouth_depth  The depth in m of the river mouth
            mouth_point  The location of the centre of the river mouth
            mouth_tangent The tangent to the river mouth (along channel axis)
            mouth_normal  The normal to the river mouth (cross channel axis)
        """

        end_width = river_mouth_width + 2 * length * numpy.tan(
            numpy.pi / 180 * self.FAN_ANGLE / 2
        )
        fan_end_point = shapely.geometry.Point(
            [
                mouth_point.x + length * mouth_tangent.x,
                mouth_point.y + length * mouth_tangent.y,
            ]
        )
        fan_polygon = shapely.geometry.Polygon(
            [
                [
                    mouth_point.x - mouth_normal.x * river_mouth_width / 2,
                    mouth_point.y - mouth_normal.y * river_mouth_width / 2,
                ],
                [
                    mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                    mouth_point.y + mouth_normal.y * river_mouth_width / 2,
                ],
                [
                    fan_end_point.x + mouth_normal.x * end_width / 2,
                    fan_end_point.y + mouth_normal.y * end_width / 2,
                ],
                [
                    fan_end_point.x - mouth_normal.x * end_width / 2,
                    fan_end_point.y - mouth_normal.y * end_width / 2,
                ],
            ]
        )
        return fan_polygon

    def polygon_and_bathymetry(self):
        """Calculate and return the fan polygon values. Perform calculations for both
        rive bed estimation approaches, so both have a smooth transition to offshore
        bathyemtry."""

        # Load in river mouth alignment and bathymetry
        mouth_point, mouth_tangent, mouth_normal = self._get_mouth_alignment()
        (
            river_mouth_width,
            river_mouth_elevations,
        ) = self._get_mouth_bathymetry()

        # Create maximum fan polygon
        fan_polygon = self._fixed_length_fan_polygon(
            river_mouth_width=river_mouth_width,
            mouth_point=mouth_point,
            mouth_tangent=mouth_tangent,
            mouth_normal=mouth_normal,
            length=self.FAN_MAX_LENGTH,
        )

        # Define fan centre line
        fan_centre = shapely.geometry.LineString(
            [
                mouth_point,
                [
                    mouth_point.x + self.FAN_MAX_LENGTH * mouth_tangent.x,
                    mouth_point.y + self.FAN_MAX_LENGTH * mouth_tangent.y,
                ],
            ]
        )

        if self.ocean_points_file is not None:
            length, end_depth = self._get_distance_to_nearest_ocean_point_in_fan(
                river_mouth_z=max(river_mouth_elevations),
                fan=fan_polygon,
                mouth_point=mouth_point,
                depth_multiplier=2
            )
            fan_polygon = self._fixed_length_fan_polygon(
                river_mouth_width=river_mouth_width,
                mouth_point=mouth_point,
                mouth_tangent=mouth_tangent,
                mouth_normal=mouth_normal,
                length=length,
            )
            end_width = river_mouth_width + 2 * length * numpy.tan(
                numpy.pi / 180 * self.FAN_ANGLE / 2
            )
            fan_end_point = shapely.geometry.Point(
                [
                    mouth_point.x + length * mouth_tangent.x,
                    mouth_point.y + length * mouth_tangent.y,
                ]
            )
            intersection_line = shapely.geometry.LineString(
                [
                    [
                        fan_end_point.x + mouth_normal.x * end_width / 2,
                        fan_end_point.y + mouth_normal.y * end_width / 2,
                    ],
                    [
                        fan_end_point.x - mouth_normal.x * end_width / 2,
                        fan_end_point.y - mouth_normal.y * end_width / 2,
                    ],
                ]
            )
            
        else:
            # Load in ocean depth contours and keep only those intersecting the fan centre
            ocean_contours = self._get_ocean_contours(max(river_mouth_elevations))
            end_depth = ocean_contours[self.ocean_contour_depth_label].min()
            ocean_contours = ocean_contours.explode(ignore_index=True)
            ocean_contours = ocean_contours[
                ~ocean_contours.intersection(fan_centre).is_empty
            ]

            # Keep the closest contour intersecting
            if len(ocean_contours) > 0:
                ocean_contours = ocean_contours.clip(fan_polygon).explode(ignore_index=True)
                min_index = ocean_contours.distance(mouth_point).idxmin()
                intersection_line = ocean_contours.loc[min_index].geometry
                end_depth = ocean_contours.loc[min_index][self.ocean_contour_depth_label]
            else:
                self.logger.warning(
                    "No ocean contour intersected. Instaed assumed fan geoemtry"
                )
                intersection_line = shapely.geometry.LineString(
                    [
                        [fan_polygon.exterior.xy[0][2], fan_polygon.exterior.xy[1][2]],
                        [fan_polygon.exterior.xy[0][3], fan_polygon.exterior.xy[1][3]],
                    ]
                )
                end_depth = max(river_mouth_elevations) * 10

            # Construct a fan ending at the contour
            (x, y) = intersection_line.xy
            polygon_points = [[xi, yi] for (xi, yi) in zip(x, y)]

            # Check if the intersected contour and mouth normal are roughtly parallel or
            # anti-parallel
            unit_vector_contour = numpy.array([x[-1] - x[0], y[-1] - y[0]])
            unit_vector_contour = unit_vector_contour / numpy.linalg.norm(
                unit_vector_contour
            )
            unit_vector_mouth = numpy.array([mouth_normal.x, mouth_normal.y])

            # they have the oposite direction
            if (
                numpy.arccos(numpy.dot(unit_vector_contour, unit_vector_mouth))
                > numpy.pi / 2
            ):
                # keep line order
                polygon_points.extend(
                    [
                        [
                            mouth_point.x - mouth_normal.x * river_mouth_width / 2,
                            mouth_point.y - mouth_normal.y * river_mouth_width / 2,
                        ],
                        [
                            mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                            mouth_point.y + mouth_normal.y * river_mouth_width / 2,
                        ],
                    ]
                )
            else:  # The have the same direction, so reverse
                # reverse fan order
                polygon_points.extend(
                    [
                        [
                            mouth_point.x + mouth_normal.x * river_mouth_width / 2,
                            mouth_point.y + mouth_normal.y * river_mouth_width / 2,
                        ],
                        [
                            mouth_point.x - mouth_normal.x * river_mouth_width / 2,
                            mouth_point.y - mouth_normal.y * river_mouth_width / 2,
                        ],
                    ]
                )
            fan_polygon = shapely.geometry.Polygon(polygon_points)
        fan_polygon = geopandas.GeoDataFrame(geometry=[fan_polygon], crs=self.crs)

        # Get bathymetry values
        bathymetry = self._bathymetry(
            intersection_line=intersection_line,
            end_depth=end_depth,
            river_mouth_elevations=river_mouth_elevations,
            mouth_point=mouth_point,
            mouth_tangent=mouth_tangent,
        )
        return fan_polygon, bathymetry
