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
from collections import namedtuple


class Scene:
    def __init__(self, scene: str, folder: str = None, vcid: int = 2):
        """
        Class to handle Landsat scenes
        """

        f_r = re.compile("LE07_L1TP_149035_\d{8}_20200917_02_T1")
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
            self._sun_azimuth = mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"][
                "SUN_AZIMUTH"
            ]
            self._sun_elevation = mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"][
                "SUN_ELEVATION"
            ]
            self._earth_sun_distance = mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"][
                "EARTH_SUN_DISTANCE"
            ]

            self.B1 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "1"))
            self.B2 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "2"))
            self.B3 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "3"))
            self.B4 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "4"))
            self.B5 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "5"))
            self.B7 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "7"))
            self.B8 = ReflectanceBand(*self.get_reflectance_band_info(mtl, "8"))

            self.B6_VCID_1: ThermalBand = ThermalBand(
                *self.get_thermal_band_info(mtl, "6_VCID_1")
            )
            self.B6_VCID_2: ThermalBand = ThermalBand(
                *self.get_thermal_band_info(mtl, "6_VCID_2")
            )
            if vcid == 1:
                self.B6 = self.B6_VCID_1
            elif vcid == 2:
                self.B6 = self.B6_VCID_2

        self.ndvi_min = 0.2
        self.ndvi_max = 0.5

    @property
    def scene(self):
        return self._scene

    @property
    def folder(self):
        return self._folder

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
            filename,
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
            filename,
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
        )

    def calculate_ndvi(self, save: bool = True):
        b4 = self.B4.read
        b3 = self.B3.read
        b3_float = self.B3.to_float(b3)
        b4_float = self.B4.to_float(b4)
        ndvi = (b4_float - b3_float) / (b4_float + b3_float)
        ndvi_meta = self.B3.meta.copy()
        ndvi_meta.update(dtype=np.float64, nodata=np.nan)
        if save:
            self.B3.save("NDVI", ndvi_meta, ndvi)
            self.B3.toa = "NDVI"
        return ndvi

    def calculate_pv(self, ndvi=None, ndvi_min=None, ndvi_max=None):
        if ndvi is None:
            ndvi = self.calculate_ndvi()
        if ndvi_min is None:
            ndvi_min = self.ndvi_min
        if ndvi_max is None:
            ndvi_max = self.ndvi_max
        ndvi_clip = np.clip(ndvi, ndvi_min, ndvi_max)
        pv = (ndvi_clip - ndvi_min) / (ndvi_max - ndvi_min)
        return np.clip(pv, 0, 1)

    def calculate_lse(self, pv):
        emissivity = 0.004 * pv + 0.986
        return np.clip(emissivity, 0.98, 1.0)

    def calculate_lst(self, bt, emissivity, lambda_thermal, c2=1.438):
        lambda_bt_ratio = (lambda_thermal * bt) / c2
        log_emissivity = np.log(emissivity)
        log_emissivity = np.clip(log_emissivity, -20, 20)
        return bt / (1 + lambda_bt_ratio * log_emissivity)

    def calculate_ndsi(self, save: bool = True):
        b2 = self.B2.read
        b5 = self.B5.read
        b2_float = self.B2.to_float(b2)
        b5_float = self.B5.to_float(b5)
        ndsi = (b2_float - b5_float) / (b2_float + b5_float)
        ndsi_meta = self.B2.meta.copy()
        ndsi_meta.update(dtype=np.float64, nodata=np.nan)
        if save:
            self.B2.save("NDSI", ndsi_meta, ndsi)
            self.B2.toa = "NDSI"
        return ndsi

    def get_points(self, shapefile: str):
        return gpd.read_file(shapefile)

    def get_average_around_points(self, gdf, band, cells):
        band_data = band.read
        band_data_float = band.to_float(band_data)
        points = []
        for index, row in gdf.iterrows():
            point = row.geometry
            row_data = band_data_float[point.y, point.x]
            points.append(row_data)
        return points


class ReflectanceBand:
    def __init__(
        self,
        band: str,
        filename: str,
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
    ) -> None:
        """
        Class to handle Landsat reflectance bands
        """

        self._band = band
        self._filename = filename
        self._path = path
        self._dtype = dtype
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
        self._resampled = [band]
        self._resample_index = 0
        self._toa = None

        if meta is not None:
            self._meta = meta
        else:
            with rasterio.open(self.path) as src:
                self._meta = src.meta

    @property
    def band(self):
        return self._band

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

    @property
    def read(self):
        with rasterio.open(self.path) as src:
            band_data = src.read
        return band_data

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value: dict):
        self._meta.update(value)

    def resample(
        self,
        height: int,
        width: int,
        mode=Resampling.nearest,
        name: str = None,
        save: bool = False,
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

        if not name:
            name = f"{self.band}_resampled_{self.resample_index}"

        if save:
            self.save(name, out_meta, out_data)
            globals()[name] = ReflectanceBand(
                name,
                f"{name}.TIF",
                os.path.join(self.folder, f"{name}.TIF"),
                self.dtype,
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
                out_meta,
            )
            self.resampled = name
            self.resample_index = True

    def reflectance_rescale(self, name: str = None, save: bool = False):
        gain = self.gain_ref
        offset = self.offset_ref
        banddata = self.read
        band_float = self.to_float(banddata)
        band_rescaled = (band_float * gain) + offset
        rescaled_meta = self.meta.copy()
        rescaled_meta.update(dtype=np.float64, nodata=np.nan)
        if save:
            if not name:
                name = f"{self.band}_TOA"
            self.save(name, rescaled_meta, band_rescaled)
            globals()[name] = ReflectanceBand(
                name,
                f"{name}.TIF",
                os.path.join(self.folder, f"{name}.TIF"),
                self.dtype,
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
                rescaled_meta,
            )
            self.toa = name
        return band_rescaled

    def to_float(self, raster: np.uint8) -> np.float64:
        raster64 = raster.astype(np.float64)
        raster64[raster64 == 0] = np.nan
        return raster64

    def save(self, name: str, out_meta: dict, data: Any):
        file_path = os.path.join(self.folder, f"{name}.TIF")
        with rasterio.open(file_path, "w", **out_meta) as dst:
            dst.write(data, 1)


class ThermalBand:
    def __init__(
        self,
        band: str,
        filename: str,
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
        meta: dict = None,
    ):
        """
        Class to handle Landsat bands
        """

        self._path = path
        self._band = band
        self._filename = filename
        self._path = path
        self._dtype = dtype
        self._maxradiance = maxradiance
        self._minradiance = minradiance
        self._maxqcal = maxqcal
        self._minqcal = minqcal
        self._gain_rad = gain_rad
        self._offset_rad = offset_rad
        self._k1 = k1
        self._k2 = k2
        self._resampled = [band]
        self._resample_index = 0
        self._toa = None

        if meta is not None:
            self._meta = meta
        else:
            with rasterio.open(self.path) as src:
                self._meta = src.meta

    @property
    def band(self):
        return self._band

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

    @property
    def read(self):
        with rasterio.open(self.path) as src:
            band_data = src.read
        return band_data

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value: dict):
        self._meta.update(value)

    def resample(
        self,
        height: int,
        width: int,
        mode=Resampling.nearest,
        name: str = None,
        save: bool = False,
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

        if not name:
            name = f"{self.band}_resampled_{self.resample_index}"

        if save:
            self.save(name, out_meta, out_data)
            globals()[name] = ThermalBand(
                name,
                f"{name}.TIF",
                os.path.join(self.folder, f"{name}.TIF"),
                self.dtype,
                self.maxradiance,
                self.minradiance,
                self.maxqcal,
                self.minqcal,
                self.gain_rad,
                self.offset_rad,
                self.k1,
                self.k2,
                out_meta,
            )
            self.resampled = name
            self.resample_index = True

    def radiance_rescale(self, name: str = None, save: bool = False):
        gain = self.gain_rad
        offset = self.offset_rad
        banddata = self.read
        band_float = self.to_float(banddata)
        band_rescaled = (band_float * gain) + offset
        rescaled_meta = self.meta.copy()
        rescaled_meta.update(dtype=np.float64, nodata=np.nan)
        if save:
            if not name:
                name = f"{self.band}_TOA"
            self.save(name, rescaled_meta, band_rescaled)
            globals()[name] = ThermalBand(
                name,
                f"{name}.TIF",
                os.path.join(self.folder, f"{name}.TIF"),
                self.dtype,
                self.maxradiance,
                self.minradiance,
                self.maxqcal,
                self.minqcal,
                self.gain_rad,
                self.offset_rad,
                self.k1,
                self.k2,
                rescaled_meta,
            )
            self.toa = name
        return band_rescaled

    def to_float(self, raster: np.uint8) -> np.float64:
        raster64 = raster.astype(np.float64)
        raster64[raster64 == 0] = np.nan
        return raster64

    def save(self, name: str, out_meta: dict, data: Any):
        file_path = os.path.join(self.folder, f"{name}.TIF")
        with rasterio.open(file_path, "w", **out_meta) as dst:
            dst.write(data, 1)

    def radiance_to_bt(self, radiance, k1, k2):
        return k2 / np.log((k1 / radiance) + 1)

    def calculate_bt(self, name: str = None, save: bool = False):
        radiance = self.radiance_rescale()
        bt = self.radiance_to_bt(radiance, self.k1, self.k2)
        bt_meta = self.meta.copy()
        bt_meta.update(dtype=np.float64, nodata=np.nan)
        if save:
            if not name:
                name = f"{self.band}_BT"
            self.save(name, bt_meta, bt)
            globals()[name] = ThermalBand(
                name,
                f"{name}.TIF",
                os.path.join(self.folder, f"{name}.TIF"),
                self.dtype,
                self.maxradiance,
                self.minradiance,
                self.maxqcal,
                self.minqcal,
                self.gain_rad,
                self.offset_rad,
                self.k1,
                self.k2,
                bt_meta,
            )
            self.toa = name
        return bt


BAND = namedtuple(
    "band",
    "band filename path dtype maxradiance minradiance maxreflectance minreflectance maxqcal minqcal radiancemult radianceadd reflectancemult reflectanceadd k1 k2",
    defaults=(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
)


def read_raster(file_path):
    """Reads a raster file and returns the data of the first band."""
    with rasterio.open(file_path) as src:
        band_data = src.read  # Read first band
    return band_data


def read_raster_and_meta(file_path):
    """Reads a raster file and returns the data of the first band."""
    with rasterio.open(file_path) as src:
        band_data = src.read  # Read first band
        meta = src.meta
    return band_data, meta


class Landsat:
    def __init__(
        self,
        folder=None,
        B6="B6_VCID_2",
    ):
        """
        docstring to update
        """

    def read_band(self, band_key: str) -> tuple[np.float64 | float | np.uint8, dict]:
        """Reads a raster file and returns the data of the first band."""
        with rasterio.open(self.band[band_key].path) as src:
            band_data = src.read  # Read first band
            band_meta: dict = src.meta
        return band_data, band_meta

    def resample_band(
        self, band_key: str, height: int, width: int, mode: object = Resampling.nearest
    ) -> tuple[np.float64 | float | np.uint8, dict]:
        # sourcery skip: dict-assign-update-to-union
        """Reads a raster file and resamples with the specified dimensions and mode."""
        with rasterio.open(self.band[band_key].path) as src:
            band_meta: dict = src.meta
            scale_h = height / band_meta["height"]
            scale_w = width / band_meta["width"]
            out_meta: dict = src.meta.copy()
            out_meta.update(
                {
                    "transform": src.transform * src.transform.scale(scale_w, scale_h),
                    "height": height,
                    "width": width,
                }
            )
            out_data = src.read

        return out_data, out_meta

    def save_raster(
        self,
        name: str,
        meta: dict,
        band_data: np.float64 | float | np.uint8,
        add: bool = False,
    ) -> None:
        """Reads a raster file and returns the data of the first band."""
        file_name = f"{self.root}_{name}.TIF"
        file_path = os.path.join(self.folder, file_name)
        with rasterio.open(file_path, "w", **meta) as dst:
            dst.write(band_data, 1)  # Write first band
        if add:
            self.band[name] = BAND(
                name,
                file_name,
                file_path,
                meta["dtype"],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    def save_multiband_raster(self, name, meta, *band_data):
        """Reads a raster file and returns the data of the first band."""
        file_path = os.path.join(self.folder, f"{self.root}_{name}.TIF")
        bands = len(band_data)
        meta.update(count=bands, dtype=band_data[0].dtype)
        with rasterio.open(file_path, "w", **meta) as dst:
            for bandn, band in enumerate(band_data, start=1):
                dst.write(band, bandn)  # Write band data

    # method to radiometrically rescale reflectance band data
    def reflectance_rescale(self, band_key: str) -> tuple[np.float64, dict]:
        """Radiometrically rescales reflectance band data using the provided constants."""
        gain = self.band[band_key].reflectancemult
        offset = self.band[band_key].reflectanceadd
        banddata, bandmeta = self.read_band(band_key)
        band_float = self.raster_to_float(banddata)
        band_rescaled = (band_float * gain) + offset
        return band_rescaled, bandmeta

    # method to radiometrically rescale radiance band data
    def radiance_rescale(self, band_key: str) -> tuple[np.float64 | float, dict]:
        """Radiometrically rescales radiance band data using the provided constants."""
        gain = self.band[band_key].radiancemult
        offset = self.band[band_key].radianceadd
        banddata, bandmeta = self.read_band(band_key)
        band_float = self.raster_to_float(banddata)
        band_rescaled = (band_float * gain) + offset
        rescaled_meta: dict[str, Any] = bandmeta.copy()
        rescaled_meta.update(dtype=np.float64, nodata=np.nan)
        return band_rescaled, rescaled_meta

    def calculate_bt(self, band_key: str) -> tuple[np.float64, dict]:
        """Calculates Brightness Temperature (BT) from a thermal band."""
        # Radiometrically rescale the thermal band data
        radiance, radiance_meta = self.radiance_rescale(band_key)
        # Calculate Brightness Temperature (BT) from the radiance values
        k1 = self.band[band_key].k1
        k2 = self.band[band_key].k2
        bt = self.radiance_to_brightness_temp(radiance, k1, k2)
        return bt, radiance_meta

    # Function to convert DN values to TOA Radiance
    def dn_to_radiance(self, dn, gain, offset):
        """Converts the DN values to TOA radiance using the gain and offset."""
        return (gain * dn) + offset

    # Function to convert TOA Radiance to Brightness Temperature with error handling for low radiance
    def radiance_to_brightness_temp(self, radiance, K1, K2):
        """Converts TOA radiance to brightness temperature using the Planck equation."""
        # Add a small constant to avoid taking log of zero or negative values
        radiance_clip = np.clip(
            radiance, 1e-10, None
        )  # Clip radiance to avoid negative or zero values
        return K2 / np.log((K1 / radiance_clip) + 1)

    # Function to calculate Proportion of Vegetation (Pv)
    def calculate_pv(self, ndvi=None, ndvi_min=None, ndvi_max=None):
        """Calculates the proportion of vegetation (Pv) based on NDVI."""
        if ndvi is None:
            try:
                ndvi_meta: dict
                ndvi_data, ndvi_meta = self.read_band("NDVI")
            except ValueError:
                ndvi_data, ndvi_meta = self.calculate_ndvi()
        else:
            ndvi_data, ndvi_meta = self.read_band(ndvi)

        if ndvi_min is None:
            ndvi_min = self.ndvi_min
        if ndvi_max is None:
            ndvi_max = self.ndvi_max

        # Clip the NDVI to the valid range
        ndvi_clip = np.clip(ndvi, ndvi_min, ndvi_max)
        # Ensure Pv stays between 0 and 1
        pv = (ndvi_clip - ndvi_min) / (ndvi_max - ndvi_min)
        return np.clip(pv, 0, 1), ndvi_meta

    # Function to calculate Land Surface Emissivity (LSE)
    def calculate_lse(self, pv):
        """Calculates land surface emissivity (LSE) based on Pv."""
        # Revisit emissivity formula to ensure proper emissivity behavior
        emissivity = 0.004 * pv + 0.986
        return np.clip(
            emissivity, 0.98, 1.0
        )  # Limiting emissivity to be between 0.98 and 1.0

    # Function to calculate Land Surface Temperature (LST) with a small radiance adjustment for safety
    def calculate_lst(self, bt, emissivity, lambda_thermal, c2=1.438):
        """Calculates Land Surface Temperature using the Single-Channel Algorithm."""

        # Ensure BT values are within expected ranges
        lambda_bt_ratio = (lambda_thermal * bt) / c2
        log_emissivity = np.log(emissivity)

        # Preventing logarithmic errors by adding a small value to emissivity if necessary
        log_emissivity = np.clip(log_emissivity, -20, 20)  # Prevent extreme values

        return bt / (1 + lambda_bt_ratio * log_emissivity)

    # method to convert raster to float
    def raster_to_float(self, raster: np.uint8) -> np.float64:
        """Converts a raster to float64 data type."""
        raster64 = raster.astype(np.float64)  # Convert to float64
        raster64[raster64 == 0] = np.nan  # Set zero values to NaN
        return raster64

    # method to calculate NDVI
    def calculate_ndvi(self, save: bool = True) -> tuple[np.float64 | float, dict]:
        """Calculates NDVI"""

        b3, b3meta = self.read_band(self.band["B3"])
        b4, b4meta = self.read_band(self.band["B4"])
        b3 = self.raster_to_float(b3)
        b4 = self.raster_to_float(b4)
        ndvi_meta = b3meta
        ndvi_meta.update(dtype=np.float64, nodata=np.nan)

        # needs reflectance rescale

        ndvi = (b4 - b3) / (b4 + b3)

        if save:
            self.save_raster("NDVI", ndvi_meta, ndvi, add=True)
            self.ndvi_meta = ndvi_meta

        return ndvi, ndvi_meta

    # Function to resample raster data to match the thermal band dimensions
    def resample_raster_to_match(
        self, to_match, to_resample, mode=Resampling.nearest, name=None, save=True
    ):
        """Resamples a raster to match a second raster's dimensions."""

        # Read band to be matched
        to_match_data, to_match_meta = self.read_band(self.band[to_match].path)
        to_match_width = to_match_meta["width"]
        to_match_height = to_match_meta["height"]

        # Resample the NDVI raster to match the thermal band's size
        resampled_data, resampled_meta = self.resample_band(
            to_resample, to_match_height, to_match_width, mode=mode
        )

        if save is True:
            if name is None:
                name = f"{self.band[to_resample].band}_resampled"
            self.save_raster(name, resampled_meta, resampled_data, add=True)

        return resampled_data, resampled_meta

    # Main function to process the data and calculate LST
    def process_landsat_data(self):
        # check NDVI data exists
        try:
            os.path.isfile(self.read_band(self.band["NDVI"].path))
        except ValueError:
            self.calculate_ndvi(save=True)

        # Get thermal band data
        thermal_band, thermal_meta = self.read_band("B6")

        # Inspect thermal band DN values
        thermal_band_min = np.nanmin(thermal_band)
        thermal_band_max = np.nanmax(thermal_band)
        print(
            f"Thermal Band DN Values: Min = {thermal_band_min}, Max = {thermal_band_max}"
        )

        # Resample NDVI to match the thermal band's dimensions
        ndvi_band = self.resample_raster_to_match(
            "B6", "NDVI", save=True, name="NDVI_resampled"
        )

        # Step 2: Read constants
        gain = self.band["B6"].radiancemult
        offset = self.band["B6"].radianceadd
        K1 = self.band["B6"].k1
        K2 = self.band["B6"].k2

        print(f"Constants: Gain={gain}, Offset={offset}, K1={K1}, K2={K2}")

        # Step 3: Convert DN to TOA Radiance
        radiance, radiance_meta = self.radiance_rescale("B6")
        print(f"Radiance: Min = {np.nanmin(radiance)}, Max = {np.nanmax(radiance)}")

        # Step 4: Convert TOA Radiance to Brightness Temperature (BT)
        bt, bt_meta = self.calculate_bt("B6")
        print(f"Brightness Temperature: Min = {np.nanmin(bt)}, Max = {np.nanmax(bt)}")

        # Step 5: Calculate Proportion of Vegetation (Pv) and LSE using pre-calculated NDVI
        pv, pv_meta = self.calculate_pv("NDVI_resampled")
        print(f"Pv Values: Min = {np.nanmin(pv)}, Max = {np.nanmax(pv)}")
        emissivity = self.calculate_lse(pv)
        print(
            f"Emissivity Values: Min = {np.nanmin(emissivity)}, Max = {np.nanmax(emissivity)}"
        )

        # Step 6: Calculate LST (in Kelvin)
        lambda_thermal = 0.0001145  # µm for Landsat 7 Band 6
        lst_kelvin = self.calculate_lst(bt, emissivity, lambda_thermal)
        print(
            f"LST (Kelvin): Min = {np.nanmin(lst_kelvin)}, Max = {np.nanmax(lst_kelvin)}"
        )

        # Step 7: Convert LST from Kelvin to Celsius
        lst_celsius = lst_kelvin - 273.15
        print(
            f"LST (Celsius): Min = {np.nanmin(lst_celsius)}, Max = {np.nanmax(lst_celsius)}"
        )

        # Save the LST output as a new GeoTIFF file (optional)
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        output_file_path = os.path.join(output_dir, "LST_output.tif")

        with rasterio.open(
            output_file_path,
            "w",
            driver="GTiff",
            count=1,
            dtype="float32",
            crs="+proj=latlong",
            transform=rasterio.open(thermal_band_path).transform,
            width=lst_celsius.shape[1],
            height=lst_celsius.shape[0],
        ) as dst:
            dst.write(lst_celsius, 1)

        print(f"LST calculation complete and saved as {output_file_path}.")
        return lst_celsius
