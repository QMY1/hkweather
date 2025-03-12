from typing import Any

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.plot
import rasterio.mask
import datetime as dt
import json
import os
import re

from rasterio.enums import Resampling
from rasterio.windows import Window


class SceneCollection:
    def __init__(self, data_folder=None):
        """
        Class to handle Landsat scenes
        """

        self._data_folder = data_folder
        self._scenes = {}

    @property
    def data_folder(self):
        return self._data_folder

    @property
    def scenes(self):
        return self._scenes

    @scenes.setter
    def scenes(self, value: tuple):
        key, val = value
        self._scenes[key] = val

    def get_scenes(self):
        for scene in os.listdir(self.data_folder):
            if os.path.isdir(os.path.join(self.data_folder, scene)):
                self.scenes = (
                    scene,
                    Scene(scene, os.path.join(self.data_folder, scene)),
                )

    def process_scenes(self):
        for scene in self.scenes.values():
            scene.process()


class Scene:
    def __init__(self, scene: str, folder: str = None, vcid: int = 2):
        """
        Class to handle Landsat scenes
        """

        # Regular expression to validate the scene reference
        f_r = re.compile(
            r"^L"  # First character is 'L' = Landsat
            r"[COTEM]"  # Second character is either 'C' [=OLI+TIRS], 'O' [=OLI], 'T' [L05: =TM; L08-09: =TIRS], 'E' [=ETM+], or 'M' [=MSS] (sensor)
            r"(05|07|08|09)"  # Third and fourth characters are either '05', '07', '08' or '09' = satellite
            r"_"  # Fifth character is '_'
            r"(L1|L2)"  # Sixth and seventh characters are 'L1' or L2' = processing level
            r"(TP|GT|GS|SP|SR)"  # Eighth and ninth characters are either 'TP', 'GT', or 'GS' (L05-07); or 'SR' [=SR only] or 'ST' [=SR+ST] (L08-09)
            r"_"  # Tenth character is '_'
            r"\d{3}"  # Characters eleven to thirteen are a group of 3 digits = WRS path
            r"\d{3}"  # Characters fourteen to sixteen are a group of 3 digits = WRS row
            r"_"  # Seventeenth character is '_'
            r"\d{4}"  # Characters eighteen to twenty-one are a group of four digits representing a valid year = acquisition year
            r"(0[1-9]|1[0-2])"  # Characters twenty-two and twenty-three are a group of two digits representing a zero-padded month = acquisition month
            r"(0[1-9]|[12]\d|3[01])"  # Characters twenty-four and twenty-five are a group of two digits representing a zero-padded day = acquisition day
            r"_"  # The twenty-fifth character is '_'
            r"\d{4}"  # Characters twenty-six to twenty-nine are a group of four digits representing a valid year = processing year
            r"(0[1-9]|1[0-2])"  # Characters thirty and thirty-one are a group of two digits representing a zero-padded month = processing month
            r"(0[1-9]|[12]\d|3[01])"  # Characters thirty-two and thirty-three are a group of two digits representing a zero-padded day = processing day
            r"_"  # The thirty-fourth character is '_'
            r"\d{2}"  # Characters thirty-five and thirty-six are a group of two digits = collection number
            r"_"  # The thirty-seventh character is '_'
            r"(RT|T1|T2)$"  # Characters thirty-eight and thirty-nine are either 'RT', 'T1', or 'T2' = collection category
        )

        if f_r.match(scene) is not None:
            try:
                dt.datetime.strptime(scene.split("_")[3].strip(), "%Y%m%d").date()
            except ValueError as e:
                raise ValueError("Invalid scene reference") from e

        self._scene = scene

        if folder is not None:
            if os.path.isdir(folder):
                self._folder = folder
            elif os.path.isdir(os.path.join("data", scene)):
                self._folder = os.path.join("data", scene)
            elif os.path.isdir(os.path.join("../data", scene)):
                self._folder = os.path.join("../data", scene)
            else:
                raise ValueError(
                    "Cannot find scene folder: please declare a valid folder"
                )

        self.metadata = os.path.join(self.folder, f"{self.scene}_MTL.JSON")

        with open(self.metadata, "r") as j:
            mtl = json.loads(j.read())

            self._spacecraft_id = mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"][
                "SPACECRAFT_ID"
            ]

            self._sensor_id = mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"][
                "SENSOR_ID"
            ]
            self._date_acquired = dt.date.fromisoformat(
                mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["DATE_ACQUIRED"]
            )
            self._scene_center_time = dt.time.fromisoformat(
                mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"]
            )
            self._scene_center_datetime = dt.datetime.combine(
                self._date_acquired, self._scene_center_time
            )
            self._sun_azimuth = float(
                mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SUN_AZIMUTH"]
            )
            self._sun_elevation = float(
                mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"]
            )
            self._earth_sun_distance = float(
                mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["EARTH_SUN_DISTANCE"]
            )

            self._processing_level = mtl["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]["PROCESSING_LEVEL"]

            '''
            Options:
            1. match processing level here, and have self.get_L1R_info(), self.get_L1T_info(), self.get_L2R_info(), self.get_L2T_info()
                Would mean new/modified band subclasses ReflectanceBandL1, ReflectanceBandL2, ThermalBandL1, ThermalBandL2, ProcessedBandL1, ProcessedBandL2 
            2. match only spacecraft here, and have self.get_reflectance_band_info() and self.get_thermal_band_info() return L2 metadata if L2
                Would mean changing band subclass args to include L2 params, and have methods check for L1/L2 and use L2 params if L2
            '''

            match self._spacecraft_id:
                case "LANDSAT_5":
                    self._lambda_thermal = 0.0001145
                    match self._processing_level[:2]:
                            case "L1":
                                self.B1 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "1"))
                                self.B2 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "2"))
                                self.B3 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "3"))
                                self.B4 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "4"))
                                self.B5 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "5"))
                                self.B7 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "7"))
                                self.B6: ThermalBand = ThermalBand(*self.get_thermal_band_info(mtl, "6"))
                            case "L2":
                                pass
                case "LANDSAT_7":
                    self._lambda_thermal = 0.0001145
                    match self._processing_level[:2]:
                            case "L1":
                                self.B1 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "1"))
                                self.B2 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "2"))
                                self.B3 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "3"))
                                self.B4 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "4"))
                                self.B5 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "5"))
                                self.B7 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "7"))
                                self.B8 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "8"))
                                self.B6_VCID_1: ThermalBand = ThermalBand(*self.get_thermal_band_info(mtl, "6_VCID_1"))
                                self.B6_VCID_2: ThermalBand = ThermalBand(*self.get_thermal_band_info(mtl, "6_VCID_2"))
                                if vcid == 1:
                                    self.B6 = self.B6_VCID_1
                                elif vcid == 2:
                                    self.B6 = self.B6_VCID_2
                            case "L2":
                                pass
                case "LANDSAT_8":
                    self._lambda_thermal = 0.0001090
                    match self._processing_level[:2]:
                            case "L1":
                                # needs to be added
                                pass
                            case "L2":
                                # needs to be added
                                pass
                case "LANDSAT_9":
                    self._lambda_thermal = 0.0001090
                    match self._processing_level[:2]:
                            case "L1":
                                # needs to be added, Thermal is Band 10
                                pass
                            case "L2":
                                # needs to be added, Thermal is Band 10
                                pass
            

            
        self.processed_bands = {}

        self.ndvi_min = 0.2
        self.ndvi_max = 0.5

    @property
    def scene(self):
        return self._scene

    @property
    def folder(self):
        return self._folder

    @property
    def spacecraft_id(self):
        return self._spacecraft_id

    @property
    def sensor_id(self):
        return self._sensor_id

    @property
    def date_acquired(self):
        return self._date_acquired

    @property
    def scene_center_time(self):
        return self._scene_center_time

    @property
    def scene_center_datetime(self):
        return self._scene_center_datetime

    @property
    def sun_azimuth(self):
        return self._sun_azimuth

    @property
    def sun_elevation(self):
        return self._sun_elevation

    @property
    def earth_sun_distance(self):
        return self._earth_sun_distance

    @property
    def lambda_thermal(self):
        return self._lambda_thermal

    def get_shared_band_info(self, mtl, band):
        filename = mtl["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"][
            f"FILE_NAME_BAND_{band}"
        ]
        path = os.path.join(self.folder, filename)
        dtype = mtl["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"][
            f"DATA_TYPE_BAND_{band}"
        ]
        maxradiance = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_RADIANCE"][
                f"RADIANCE_MAXIMUM_BAND_{band}"
            ]
        )
        minradiance = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_RADIANCE"][
                f"RADIANCE_MINIMUM_BAND_{band}"
            ]
        )
        maxqcal = int(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_PIXEL_VALUE"][
                f"QUANTIZE_CAL_MAX_BAND_{band}"
            ]
        )
        minqcal = int(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_PIXEL_VALUE"][
                f"QUANTIZE_CAL_MIN_BAND_{band}"
            ]
        )
        gain_rad = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_RADIOMETRIC_RESCALING"][
                f"RADIANCE_MULT_BAND_{band}"
            ]
        )
        offset_rad = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_RADIOMETRIC_RESCALING"][
                f"RADIANCE_ADD_BAND_{band}"
            ]
        )

        return (
            filename,
            path,
            dtype,
            maxradiance,
            minradiance,
            maxqcal,
            minqcal,
            gain_rad,
            offset_rad,
        )

    def get_reflectance_info(self, mtl, band):
        maxreflectance = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_REFLECTANCE"][
                f"REFLECTANCE_MAXIMUM_BAND_{band}"
            ]
        )
        minreflectance = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_REFLECTANCE"][
                f"REFLECTANCE_MINIMUM_BAND_{band}"
            ]
        )
        gain_ref = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_RADIOMETRIC_RESCALING"][
                f"REFLECTANCE_MULT_BAND_{band}"
            ]
        )
        offset_ref = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_RADIOMETRIC_RESCALING"][
                f"REFLECTANCE_ADD_BAND_{band}"
            ]
        )

        return maxreflectance, minreflectance, gain_ref, offset_ref

    def get_reflectance_band_info(self, mtl, bandkey):
        (
            filename,
            path,
            dtype,
            maxradiance,
            minradiance,
            maxqcal,
            minqcal,
            gain_rad,
            offset_rad,
        ) = self.get_shared_band_info(mtl, bandkey)
        (
            maxreflectance,
            minreflectance,
            gain_ref,
            offset_ref,
        ) = self.get_reflectance_info(mtl, bandkey)
        return (
            bandkey,
            self.scene,
            self.spacecraft_id,
            self.sensor_id,
            self.date_acquired,
            self.scene_center_time,
            self.scene_center_datetime,
            self.sun_azimuth,
            self.sun_elevation,
            self.earth_sun_distance,
            filename,
            self.folder,
            path,
            dtype,
            maxradiance,
            minradiance,
            maxreflectance,
            minreflectance,
            maxqcal,
            minqcal,
            gain_rad,
            offset_rad,
            gain_ref,
            offset_ref,
        )

    def get_thermal_info(self, mtl, band):
        k1 = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_THERMAL_CONSTANTS"][
                f"K1_CONSTANT_BAND_{band}"
            ]
        )
        k2 = float(
            mtl["LANDSAT_METADATA_FILE"]["LEVEL1_THERMAL_CONSTANTS"][
                f"K2_CONSTANT_BAND_{band}"
            ]
        )
        return k1, k2

    def get_thermal_band_info(self, mtl, bandkey):
        (
            filename,
            path,
            dtype,
            maxradiance,
            minradiance,
            maxqcal,
            minqcal,
            gain_rad,
            offset_rad,
        ) = self.get_shared_band_info(mtl, bandkey)
        k1, k2 = self.get_thermal_info(mtl, bandkey)
        return (
            bandkey,
            self.scene,
            self.spacecraft_id,
            self.sensor_id,
            self.date_acquired,
            self.scene_center_time,
            self.scene_center_datetime,
            self.sun_azimuth,
            self.sun_elevation,
            self.earth_sun_distance,
            filename,
            self.folder,
            path,
            dtype,
            maxradiance,
            minradiance,
            maxqcal,
            minqcal,
            gain_rad,
            offset_rad,
            k1,
            k2,
            self.lambda_thermal,
        )

    def create_processed_band(self, name: str, meta: dict, info: str):
        '''
        Creates an object of Class ProcessedBand from args and self_attrs:
        name -> band: str,
        self.scene -> scene: str,
        self.spacecraft_id -> spacecraft_id: str,
        self.sensor_id -> sensor_id: str,
        self.date_acquired -> date_acquired: dt.date,
        self.scene_center_time -> scene_center_time: dt.time,
        self.scene_center_datetime -> scene_center_datetime: dt.datetime,
        self.sun_azimuth -> sun_azimuth: float,
        self.sun_elevation -> sun_elevation: float,
        self.earth_sun_distance -> earth_sun_distance: float,
        f"{self.scene}_{name}.TIF" -> filename: str,
        self.folder ->  folder: str,
        os.path.join(self.folder, f"{self.scene}_{name}.TIF") -> path: object,
        meta["dtype"] -> dtype: object,
        meta -> meta: dict = None,
        info -> info: str = None,
        '''
        self.processed_bands[name] = ProcessedBand(
            name,
            self.scene,
            self.spacecraft_id,
            self.sensor_id,
            self.date_acquired,
            self.scene_center_time,
            self.scene_center_datetime,
            self.sun_azimuth,
            self.sun_elevation,
            self.earth_sun_distance,
            f"{self.scene}_{name}.TIF",
            self.folder,
            os.path.join(self.folder, f"{self.scene}_{name}.TIF"),
            meta["dtype"],
            meta,
            info,
        )

    def save(self, name: str, out_meta: dict, data: Any) -> None:
        file_path = os.path.join(self.folder, f"{self.scene}_{name}.TIF")
        with rasterio.open(file_path, "w", **out_meta) as dst:
            dst.write(data, 1)

    def calculate_ndvi(self, save: bool = True, ret: bool = False):
        b3_float = self.B3.to_float()
        b4_float = self.B4.to_float()
        ndvi = (b4_float - b3_float) / (b4_float + b3_float)
        ndvi_meta = self.B3.meta.copy()
        ndvi_meta.update(dtype=np.float64, nodata=np.nan)
        self.ndvi_min = ndvi.min()
        self.ndvi_max = ndvi.max()
        if save:
            self.save("NDVI", ndvi_meta, ndvi)
            self.create_processed_band("NDVI", ndvi_meta, info="NDVI")
        if ret:
            return ndvi, ndvi_meta

    def calculate_pv(
        self,
        ndvi=None,
        ndvi_min=None,
        ndvi_max=None,
        save: bool = True,
        ret: bool = False,
    ):
        if ndvi is None:
            ndvi, pv_meta = self.calculate_ndvi(ret=True)
        if ndvi_min is None:
            ndvi_min = self.ndvi_min
        if ndvi_max is None:
            ndvi_max = self.ndvi_max
        ndvi_clip = np.clip(ndvi, ndvi_min, ndvi_max)
        pv = (ndvi_clip - ndvi_min) / (ndvi_max - ndvi_min)
        pv_clip = np.clip(pv, 0, 1)
        if save:
            self.save("PV", pv_meta, pv_clip)
            self.create_processed_band("PV", pv_meta, info="PV")
        if ret:
            return pv_clip, pv_meta

    def calculate_lse(self, pv=None, save: bool = True, ret: bool = False):
        if pv is None:
            pv, lse_meta = self.calculate_pv(ret=True)
        lse = 0.004 * pv + 0.986
        lse_clip = np.clip(lse, 0.98, 1.0)
        if save:
            self.save("LSE", lse_meta, lse_clip)
            self.create_processed_band("LSE", lse_meta, info="LSE")
        if ret:
            return lse_clip, lse_meta

    def calculate_lst(
        self, bt=None, lse=None, c2=1.438, save: bool = True, ret: bool = False
    ):
        if bt is None:
            bt, bt_meta = self.B6.calculate_bt(ret=True)
        if lse is None:
            lse, lse_meta = self.calculate_lse(ret=True)
        lambda_bt_ratio = (self.lambda_thermal * bt) / c2
        log_lse = np.log(lse)
        log_lse = np.clip(log_lse, -20, 20)
        lst_kelvin = bt / (1 + lambda_bt_ratio * log_lse)
        lst = lst_kelvin - 273.15
        if save:
            self.save("LST", lse_meta, lst)
            self.create_processed_band("LST", lse_meta, info="LST")
        if ret:
            return lst, lse_meta

    def calculate_ndsi(self, save: bool = True, ret: bool = False):
        b2_float = self.B2.to_float()
        b5_float = self.B5.to_float()
        ndsi = (b2_float - b5_float) / (b2_float + b5_float)
        ndsi_meta = self.B2.meta.copy()
        ndsi_meta.update(dtype=np.float64, nodata=np.nan)
        if save:
            self.save("NDSI", ndsi_meta, ndsi)
            self.create_processed_band("NDSI", ndsi_meta, info="NDSI")
        if ret:
            return ndsi, ndsi_meta

    def get_points(self, shapefile: str = None):
        if shapefile is None:
            shapefile = os.path.join("data", "locations.feather")
        match shapefile.split(".")[1].strip():
            case "feather":
                gdf = gpd.read_feather(shapefile)
            case "parquet":
                gdf = gpd.read_parquet(shapefile)
            case _:
                gdf = gpd.read_file(shapefile)
        return gdf

    def get_average_around_points(
        self,
        band,
        cells: int,
        gdf: gpd.GeoDataFrame = None,
        save: bool = True,
        fmt: str = "parquet",
        ret: bool = False,
    ):
        if gdf is None:
            gdf = self.get_points()

        if gdf.crs != band.meta["crs"]:
            gdf = gdf.to_crs(band.meta["crs"])
        with rasterio.open(band.path) as src:
            for index, row in gdf.iterrows():
                # Get the point coordinates
                point = row.geometry
                # Convert point coordinates to raster space
                src_row, src_col = src.index(point.x, point.y)
                # Calculate the half window size
                half_window = cells // 2
                # define the window
                window = Window(
                    src_col - half_window, src_row - half_window, cells, cells
                )
                # read the data from the window
                data = src.read(1, window=window)

                # Calculate the average value, ignoring nodata values
                average_value = np.nanmean(data)

                # Add the average value to the geodataframe
                gdf.loc[index, f"{band.band}_{cells}px"] = average_value

                # add the date and time to the geodataframe
                gdf.loc[index, "date"] = band.scene_center_datetime

        # re-index the geodataframe by the date
        gdf = gdf.set_index("date")

        if save:
            match fmt:
                case "feather":
                    gdf.to_feather(
                        os.path.join(
                            self.folder, f"{self.scene}_{band.band}_{cells}px.feather"
                        )
                    )
                case "parquet":
                    gdf.to_parquet(
                        os.path.join(
                            self.folder, f"{self.scene}_{band.band}_{cells}px.parquet"
                        )
                    )
                case "_":
                    try:
                        gdf.to_file(
                            os.path.join(
                                self.folder, f"{self.scene}_{band.band}_{cells}px.{fmt}"
                            ),
                            driver=fmt,
                        )
                    except Exception:
                        gdf.to_file(
                            os.path.join(
                                self.folder, f"{self.scene}_{band.band}_{cells}px.gpkg"
                            ),
                            driver="GPKG",
                        )
            gdf.to_csv(
                os.path.join(self.folder, f"{self.scene}_{band.band}_{cells}px.csv")
            )

        if ret:
            return gdf

    def process(self):
        self.calculate_lst()
        self.calculate_ndsi()
        for band in self.processed_bands.values():
            self.get_average_around_points(band, 5)


class Band:
    def __init__(
        self,
        band: str,
        scene: str,
        spacecraft_id: str,
        sensor_id: str,
        date_acquired: dt.date,
        scene_center_time: dt.time,
        scene_center_datetime: dt.datetime,
        sun_azimuth: float,
        sun_elevation: float,
        earth_sun_distance: float,
        filename: str,
        folder: str,
        path: object,
        dtype: object,
        meta: dict = None,
        crs: object = None,
    ) -> None:
        """
        Class to handle Landsat bands, with child classes for reflected, thermal, and processed bands
        """
        self._band = band
        self._scene = scene
        self._spacecraft_id = spacecraft_id
        self._sensor_id = sensor_id
        self._date_acquired = date_acquired
        self._scene_center_time = scene_center_time
        self._scene_center_datetime = scene_center_datetime
        self._sun_azimuth = sun_azimuth
        self._sun_elevation = sun_elevation
        self._earth_sun_distance = earth_sun_distance
        self._filename = filename
        self._folder = folder
        self._path = path
        self._dtype = dtype
        if meta is not None:
            self._meta = meta
        else:
            with rasterio.open(self.path) as src:
                self._meta = src.meta
        self._crs = crs if crs is not None else self._meta["crs"]
        self._versions = {self._band: "original"}

    @property
    def band(self):
        return self._band

    @property
    def scene(self):
        return self._scene

    @property
    def spacecraft_id(self):
        return self._spacecraft_id

    @property
    def sensor_id(self):
        return self._sensor_id

    @property
    def date_acquired(self):
        return self._date_acquired

    @property
    def scene_center_time(self):
        return self._scene_center_time

    @property
    def scene_center_datetime(self):
        return self._scene_center_datetime

    @property
    def sun_azimuth(self):
        return self._sun_azimuth

    @property
    def sun_elevation(self):
        return self._sun_elevation

    @property
    def earth_sun_distance(self):
        return self._earth_sun_distance

    @property
    def folder(self):
        return self._folder

    @property
    def filename(self):
        return self._filename

    @property
    def path(self):
        return self._path

    @property
    def dtype(self):
        return self._dtype

    @property
    def read(self):
        with rasterio.open(self.path) as src:
            band_data = src.read(1)
        return band_data

    @property
    def meta(self):
        return self._meta

    @property
    def versions(self):
        return self._versions

    @versions.setter
    def versions(self, value: tuple):
        key, val = value
        self._versions[key] = val

    def return_resampled(
        self,
        height: int,
        width: int,
        mode=Resampling.nearest,
    ):
        with rasterio.open(self.path) as src:
            scale_h = src.height / height
            scale_w = src.width / width
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "transform": src.transform * src.transform.scale(scale_w, scale_h),
                    "height": height,
                    "width": width,
                }
            )
            out_data = src.read(
                out_shape=(src.count, height, width),
                resampling=mode,
            )
            return out_data, out_meta

    def to_float(self) -> Any:
        band64 = self.read.astype(np.float64)
        band64[band64 == 0] = np.nan
        return band64

    def save(self, name: str, out_meta: dict, data: Any) -> None:
        file_path = os.path.join(self.folder, f"{self.scene}_{name}.TIF")
        with rasterio.open(file_path, "w", **out_meta) as dst:
            dst.write(data, 1)


class ReflectanceBand(Band):
    def __init__(
        self,
        band: str,
        scene: str,
        spacecraft_id: str,
        sensor_id: str,
        date_acquired: dt.date,
        scene_center_time: dt.time,
        scene_center_datetime: dt.datetime,
        sun_azimuth: float,
        sun_elevation: float,
        earth_sun_distance: float,
        filename: str,
        folder: str,
        path: object,
        dtype: object,
        maxradiance: float,
        minradiance: float,
        maxreflectance: float,
        minreflectance: float,
        maxqcal: int,
        minqcal: int,
        gain_rad: float,
        offset_rad: float,
        gain_ref: float,
        offset_ref: float,
        meta: dict = None,
        reflectance: bool = False,
    ) -> None:
        """
        Class to handle Landsat reflectance bands
        """

        '''
        super().__init__() sends to __init__() of parent Class Band:
            band -> band: str,
            scene -> scene: str,
            spacecraft_id -> spacecraft_id: str,
            sensor_id -> sensor_id: str,
            date_acquired -> date_acquired: dt.date,
            scene_center_time -> scene_center_time: dt.time,
            scene_center_datetime -> scene_center_datetime: dt.datetime,
            sun_azimuth -> sun_azimuth: float,
            sun_elevation -> sun_elevation: float,
            earth_sun_distance -> earth_sun_distance: float,
            filename -> filename: str,
            folder -> folder: str,
            path -> path: object,
            dtype -> dtype: object,
            meta -> meta: dict = None,
             -> crs: object = None,
        '''
        super().__init__(
            band,
            scene,
            spacecraft_id,
            sensor_id,
            date_acquired,
            scene_center_time,
            scene_center_datetime,
            sun_azimuth,
            sun_elevation,
            earth_sun_distance,
            filename,
            folder,
            path,
            dtype,
            meta,
        )
        self._maxradiance = maxradiance
        self._minradiance = minradiance
        self._maxreflectance = maxreflectance
        self._minreflectance = minreflectance
        self._maxqcal = maxqcal
        self._minqcal = minqcal
        self._gain_rad = gain_rad
        self._offset_rad = offset_rad
        self._gain_ref = gain_ref
        self._offset_ref = offset_ref

        self._reflectance = reflectance

    @property
    def maxradiance(self):
        return self._maxradiance

    @property
    def minradiance(self):
        return self._minradiance

    @property
    def maxreflectance(self):
        return self._maxreflectance

    @property
    def minreflectance(self):
        return self._minreflectance

    @property
    def maxqcal(self):
        return self._maxqcal

    @property
    def minqcal(self):
        return self._minqcal

    @property
    def gain_rad(self):
        return self._gain_rad

    @property
    def offset_rad(self):
        return self._offset_rad

    @property
    def gain_ref(self):
        return self._gain_ref

    @property
    def offset_ref(self):
        return self._offset_ref

    @property
    def reflectance(self):
        return self._reflectance

    @reflectance.setter
    def reflectance(self, value: bool = True):
        self._reflectance = value

    def create_reflectance_band(self, name: str, meta: dict):
        '''
        Creates a new object of Class ReflectanceBand by sending:
        name -> band: str,
        self.scene -> scene: str,
        self.spacecraft_id -> spacecraft_id: str,
        self.sensor_id -> sensor_id: str,
        self.date_acquired -> date_acquired: dt.date,
        self.scene_center_time -> scene_center_time: dt.time,
        self.scene_center_datetime -> scene_center_datetime: dt.datetime,
        self.sun_azimuth -> sun_azimuth: float,
        self.sun_elevation -> sun_elevation: float,
        self.earth_sun_distance -> earth_sun_distance: float,
        f"{self.scene}_{name}.TIF" -> filename: str,
        self.folder -> folder: str,
        os.path.join(self.folder, f"{self.scene}_{name}.TIF") -> path: object,
        meta["dtype"] -> dtype: object,
        self.maxradiance -> maxradiance: float,
        self.minradiance -> minradiance: float,
        self.maxreflectance -> maxreflectance: float,
        self.minreflectance -> minreflectance: float,
        self.maxqcal -> maxqcal: int,
        self.minqcal -> minqcal: int,
        self.gain_rad -> gain_rad: float,
        self.offset_rad -> offset_rad: float,
        self.gain_ref -> gain_ref: float,
        self.offset_ref -> offset_ref: float,
        meta -> meta: dict = None,
         -> reflectance: bool = False,
        '''
        globals()[name] = ReflectanceBand(
            name,
            self.scene,
            self.spacecraft_id,
            self.sensor_id,
            self.date_acquired,
            self.scene_center_time,
            self.scene_center_datetime,
            self.sun_azimuth,
            self.sun_elevation,
            self.earth_sun_distance,
            f"{self.scene}_{name}.TIF",
            self.folder,
            os.path.join(self.folder, f"{self.scene}_{name}.TIF"),
            meta["dtype"],
            self.maxradiance,
            self.minradiance,
            self.maxreflectance,
            self.minreflectance,
            self.maxqcal,
            self.minqcal,
            self.gain_rad,
            self.offset_rad,
            self.gain_ref,
            self.offset_ref,
            meta,
        )

    def resample(
        self,
        height: int,
        width: int,
        mode=Resampling.nearest,
        name: str = None,
        save: bool = True,
        ret: bool = False,
    ):
        out_data, out_meta = self.return_resampled(height, width, mode)

        if save:
            if not name:
                name = f"{self.band}_resampled_{height}x{width}"
            self.save(name, out_meta, out_data)
            self.create_reflectance_band(name, out_meta)
            self.versions = (name, "resampled")

        if ret:
            return out_data, out_meta

    def to_reflectance(self, name: str = None, save: bool = True, ret: bool = False):
        band_float = self.to_float()
        band_reflectance = (band_float * self.gain_ref) + self.offset_ref
        meta_reflectance = self.meta.copy()
        meta_reflectance.update(dtype=np.float64, nodata=np.nan)

        if save:
            if not name:
                name = f"{self.band}_TOA"
            self.save(name, meta_reflectance, band_reflectance)
            self.create_reflectance_band(name, meta_reflectance)
            self.versions = (name, "reflectance")
            self.reflectance = True

        if ret:
            return band_reflectance, meta_reflectance


class ThermalBand(Band):
    def __init__(
        self,
        band: str,
        scene: str,
        spacecraft_id: str,
        sensor_id: str,
        date_acquired: dt.date,
        scene_center_time: dt.time,
        scene_center_datetime: dt.datetime,
        sun_azimuth: float,
        sun_elevation: float,
        earth_sun_distance: float,
        filename: str,
        folder: str,
        path: object,
        dtype: object,
        maxradiance: float,
        minradiance: float,
        maxqcal: int,
        minqcal: int,
        gain_rad: float,
        offset_rad: float,
        k1: float,
        k2: float,
        lambda_thermal: float,
        meta: dict = None,
    ):
        """
        Class to handle Landsat Thermal Bands
        """

        '''
        super().__init__() sends to __init__() of parent Class Band:
            band -> band: str,
            scene -> scene: str,
            spacecraft_id -> spacecraft_id: str,
            sensor_id -> sensor_id: str,
            date_acquired -> date_acquired: dt.date,
            scene_center_time -> scene_center_time: dt.time,
            scene_center_datetime -> scene_center_datetime: dt.datetime,
            sun_azimuth -> sun_azimuth: float,
            sun_elevation -> sun_elevation: float,
            earth_sun_distance -> earth_sun_distance: float,
            filename -> filename: str,
            folder -> folder: str,
            path -> path: object,
            dtype -> dtype: object,
            meta -> meta: dict = None,
             -> crs: object = None,
        '''
        super().__init__(
            band,
            scene,
            spacecraft_id,
            sensor_id,
            date_acquired,
            scene_center_time,
            scene_center_datetime,
            sun_azimuth,
            sun_elevation,
            earth_sun_distance,
            filename,
            folder,
            path,
            dtype,
            meta,
        )
        self._maxradiance = maxradiance
        self._minradiance = minradiance
        self._maxqcal = maxqcal
        self._minqcal = minqcal
        self._gain_rad = gain_rad
        self._offset_rad = offset_rad
        self._k1 = k1
        self._k2 = k2
        self._lambda_thermal = lambda_thermal
        self._resampled = [band]
        self._resample_index = 0
        self._toa = None

        if meta is not None:
            self._meta = meta
        else:
            with rasterio.open(self.path) as src:
                self._meta = src.meta

    @property
    def maxradiance(self):
        return self._maxradiance

    @property
    def minradiance(self):
        return self._minradiance

    @property
    def maxqcal(self):
        return self._maxqcal

    @property
    def minqcal(self):
        return self._minqcal

    @property
    def gain_rad(self):
        return self._gain_rad

    @property
    def offset_rad(self):
        return self._offset_rad

    @property
    def k1(self):
        return self._k1

    @property
    def k2(self):
        return self._k2

    @property
    def lambda_thermal(self):
        return self._lambda_thermal

    @property
    def resampled(self):
        return self._resampled

    @resampled.setter
    def resampled(self, value: str):
        self._resampled.append(value)

    @property
    def resample_index(self):
        return self._resample_index

    @resample_index.setter
    def resample_index(self, value: bool):
        if value:
            self._resample_index += 1

    @property
    def toa(self):
        return self._toa

    @toa.setter
    def toa(self, value: str):
        self._toa = value

    def create_thermal_band(self, name: str, meta: dict):
        '''
        Creates a new object of Class ThermalBand by sending:
            name -> band: str,
            self.scene -> scene: str,
            self.spacecraft_id -> spacecraft_id: str,
            self.sensor_id -> sensor_id: str,
            self.date_acquired -> date_acquired: dt.date,
            self.scene_center_time -> scene_center_time: dt.time,
            self.scene_center_datetime -> scene_center_datetime: dt.datetime,
            self.sun_azimuth -> sun_azimuth: float,
            self.sun_elevation -> sun_elevation: float,
            self.earth_sun_distance -> earth_sun_distance: float,
            f"{self.scene}_{name}.TIF" -> filename: str,
            self.folder -> folder: str,
            os.path.join(self.folder, f"{self.scene}_{name}.TIF") -> path: object,
            meta["dtype"] -> dtype: object,
            self.maxradiance -> maxradiance: float,
            self.minradiance -> minradiance: float,
            self.maxqcal -> maxqcal: int,
            self.minqcal -> minqcal: int,
            self.gain_rad -> gain_rad: float,
            self.offset_rad -> offset_rad: float,
            self.k1 -> k1: float,
            self.k2 -> k2: float,
            self.lambda_thermal -> lambda_thermal: float,
            meta -> meta: dict = None,
        '''
        globals()[name] = ThermalBand(
            name,
            self.scene,
            self.spacecraft_id,
            self.sensor_id,
            self.date_acquired,
            self.scene_center_time,
            self.scene_center_datetime,
            self.sun_azimuth,
            self.sun_elevation,
            self.earth_sun_distance,
            f"{self.scene}_{name}.TIF",
            self.folder,
            os.path.join(self.folder, f"{self.scene}_{name}.TIF"),
            meta["dtype"],
            self.maxradiance,
            self.minradiance,
            self.maxqcal,
            self.minqcal,
            self.gain_rad,
            self.offset_rad,
            self.k1,
            self.k2,
            self.lambda_thermal,
            meta,
        )

    def resample(
        self,
        height: int,
        width: int,
        mode=Resampling.nearest,
        name: str = None,
        save: bool = True,
        ret: bool = False,
    ):
        out_data, out_meta = self.return_resampled(height, width, mode)

        if save:
            if not name:
                name = f"{self.band}_resampled_{self.resample_index}"
            self.save(name, out_meta, out_data)
            self.create_thermal_band(name, out_meta)
            self.resampled = name
            self.resample_index = True

        if ret:
            return out_data, out_meta

    def to_radiance(self, name: str = None, save: bool = True, ret: bool = False):
        band_float = self.to_float()
        band_rescaled = (band_float * self.gain_rad) + self.offset_rad
        rescaled_meta = self.meta.copy()
        rescaled_meta.update(dtype=np.float64, nodata=np.nan)
        if save:
            if not name:
                name = f"{self.band}_TOA"
            self.save(name, rescaled_meta, band_rescaled)
            self.create_thermal_band(name, rescaled_meta)
            self.toa = name

        if ret:
            return band_rescaled, rescaled_meta

    def calculate_bt(self, name: str = None, save: bool = True, ret: bool = False):
        """Converts TOA radiance to brightness temperature using the Planck equation."""
        # Add a small constant to avoid taking log of zero or negative values
        radiance, radiance_meta = self.to_radiance(ret=True)
        radiance_clip = np.clip(radiance, 1e-10, None)

        bt = self.k2 / np.log((self.k1 / radiance_clip) + 1)
        bt_meta = radiance_meta.copy()
        bt_meta.update(dtype=np.float64, nodata=np.nan)
        if save:
            if not name:
                name = f"{self.band}_BT"
            self.save(name, bt_meta, bt)
            self.create_thermal_band(name, bt_meta)
            self.toa = name
        if ret:
            return bt, bt_meta


class ProcessedBand(Band):
    def __init__(
        self,
        band: str,
        scene: str,
        spacecraft_id: str,
        sensor_id: str,
        date_acquired: dt.date,
        scene_center_time: dt.time,
        scene_center_datetime: dt.datetime,
        sun_azimuth: float,
        sun_elevation: float,
        earth_sun_distance: float,
        filename: str,
        folder: str,
        path: object,
        dtype: object,
        meta: dict = None,
        info: str = None,
    ) -> None:
        """
        Class to handle Landsat reflectance bands
        """
        
        '''
        super().__init__() sends to __init__() of parent Class Band:
            band -> band: str,
            scene -> scene: str,
            spacecraft_id -> spacecraft_id: str,
            sensor_id -> sensor_id: str,
            date_acquired -> date_acquired: dt.date,
            scene_center_time -> scene_center_time: dt.time,
            scene_center_datetime -> scene_center_datetime: dt.datetime,
            sun_azimuth -> sun_azimuth: float,
            sun_elevation -> sun_elevation: float,
            earth_sun_distance -> earth_sun_distance: float,
            filename -> filename: str,
            folder -> folder: str,
            path -> path: object,
            dtype -> dtype: object,
            meta -> meta: dict = None,
             -> crs: object = None,
        '''
        super().__init__(
            band,
            scene,
            spacecraft_id,
            sensor_id,
            date_acquired,
            scene_center_time,
            scene_center_datetime,
            sun_azimuth,
            sun_elevation,
            earth_sun_distance,
            filename,
            folder,
            path,
            dtype,
            meta,
        )
        self._info = info

    @property
    def info(self):
        return self._info

    def create_processed_band(self, name: str, meta: dict, info: str):
        globals()[name] = ProcessedBand(
            name,
            self.scene,
            self.spacecraft_id,
            self.sensor_id,
            self.date_acquired,
            self.scene_center_time,
            self.scene_center_datetime,
            self.sun_azimuth,
            self.sun_elevation,
            self.earth_sun_distance,
            f"{self.scene}_{name}.TIF",
            self.folder,
            os.path.join(self.folder, f"{self.scene}_{name}.TIF"),
            self.dtype,
            meta,
            info,
        )

    def resample(
        self,
        height: int,
        width: int,
        mode=Resampling.nearest,
        name: str = None,
        save: bool = True,
        ret: bool = False,
    ):
        out_data, out_meta = self.return_resampled(height, width, mode)

        if save:
            if not name:
                name = f"{self.band}_resampled_{height}x{width}"
            self.save(name, out_meta, out_data)
            self.create_processed_band(name, out_meta, f"{self.band}_resampled")
            self.versions = (name, "resampled")

        if ret:
            return out_data, out_meta
