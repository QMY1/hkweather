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
        band_data = src.read(1)  # Read first band
    return band_data


def read_raster_and_meta(file_path):
    """Reads a raster file and returns the data of the first band."""
    with rasterio.open(file_path) as src:
        band_data = src.read(1)  # Read first band
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

        self.f_r = re.compile("LE07_L1TP_149035_\d{8}_20200917_02_T1")

        if folder is None:
            raise ValueError("Please declare a valid folder")
        elif self.f_r.match(folder) is not None:
            try:
                self.date = dt.datetime.strptime(
                    folder.split("_")[3].strip(), "%Y%m%d"
                ).date()
            except ValueError:
                raise ValueError("Please declare a valid folder")

            if os.path.isdir(folder):
                self.folder = folder
                self.root = folder
            elif os.path.isdir(os.path.join("data", folder)):
                self.folder = os.path.join("data", folder)
                self.root = folder
            elif os.path.isdir(os.path.join("../data", folder)):
                self.folder = os.path.join("../data", folder)
                self.root = folder
            else:
                raise ValueError("Please declare a valid folder")

        self.metadata = os.path.join(self.folder, f"{self.root}_MTL.JSON")
        self.band = {}
        bandlist = ["1", "2", "3", "4", "5", "6_VCID_1", "6_VCID_2", "7", "8"]

        for b in bandlist:
            with open(self.metadata, "r") as j:
                mtl = json.loads(j.read())
                bandkey = f"B{b}"
                filename = mtl["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"][
                    f"FILE_NAME_BAND_{b}"
                ]
                path = os.path.join(self.folder, filename)
                dtype = mtl["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"][
                    f"DATA_TYPE_BAND_{b}"
                ]
                maxradiance = float(
                    mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_RADIANCE"][
                        f"RADIANCE_MAXIMUM_BAND_{b}"
                    ]
                )
                minradiance = float(
                    mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_RADIANCE"][
                        f"RADIANCE_MINIMUM_BAND_{b}"
                    ]
                )
                try:
                    maxreflectance = float(
                        mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_REFLECTANCE"][
                            f"REFLECTANCE_MAXIMUM_BAND_{b}"
                        ]
                    )
                except KeyError:
                    maxreflectance = None
                try:
                    minreflectance = float(
                        mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_REFLECTANCE"][
                            f"REFLECTANCE_MINIMUM_BAND_{b}"
                        ]
                    )
                except KeyError:
                    minreflectance = None
                maxqcal = float(
                    mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_PIXEL_VALUE"][
                        f"QUANTIZE_CAL_MAX_BAND_{b}"
                    ]
                )
                minqcal = float(
                    mtl["LANDSAT_METADATA_FILE"]["LEVEL1_MIN_MAX_PIXEL_VALUE"][
                        f"QUANTIZE_CAL_MIN_BAND_{b}"
                    ]
                )
                radiancemult = float(
                    mtl["LANDSAT_METADATA_FILE"]["LEVEL1_RADIOMETRIC_RESCALING"][
                        f"RADIANCE_MULT_BAND_{b}"
                    ]
                )
                radianceadd = float(
                    mtl["LANDSAT_METADATA_FILE"]["LEVEL1_RADIOMETRIC_RESCALING"][
                        f"RADIANCE_ADD_BAND_{b}"
                    ]
                )
                try:
                    reflectancemult = float(
                        mtl["LANDSAT_METADATA_FILE"]["LEVEL1_RADIOMETRIC_RESCALING"][
                            f"REFLECTANCE_MULT_BAND_{b}"
                        ]
                    )
                except KeyError:
                    reflectancemult = None
                try:
                    reflectanceadd = float(
                        mtl["LANDSAT_METADATA_FILE"]["LEVEL1_RADIOMETRIC_RESCALING"][
                            f"REFLECTANCE_ADD_BAND_{b}"
                        ]
                    )
                except KeyError:
                    reflectanceadd = None
                try:
                    k1 = float(
                        mtl["LANDSAT_METADATA_FILE"]["LEVEL1_THERMAL_CONSTANTS"][
                            f"K1_CONSTANT_BAND_{b}"
                        ]
                    )
                except KeyError:
                    k1 = None
                try:
                    k2 = float(
                        mtl["LANDSAT_METADATA_FILE"]["LEVEL1_THERMAL_CONSTANTS"][
                            f"K2_CONSTANT_BAND_{b}"
                        ]
                    )
                except KeyError:
                    k2 = None

                if bandkey == B6:
                    bandkey = "B6"

                bandinfo = BAND(
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
                    radiancemult,
                    radianceadd,
                    reflectancemult,
                    reflectanceadd,
                    k1,
                    k2,
                )
            self.band[bandkey] = bandinfo

        self.ndvi_min = 0.2
        self.ndvi_max = 0.5

    def read_band(self, band_key: str) -> tuple[np.float64 | float | np.uint8, dict]:
        """Reads a raster file and returns the data of the first band."""
        with rasterio.open(self.band[band_key].path) as src:
            band_data = src.read(1)  # Read first band
            band_meta: dict = src.meta
        return band_data, band_meta

    def resample_band(
        self, band_key: str, height: int, width: int, mode: object = Resampling.nearest
    ) -> tuple[np.float64 | float | np.uint8, dict]:
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
            out_data = src.read(
                1,
                out_shape=(height, width),
                resampling=mode,
            )

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
        if add is True:
            bandinfo = BAND(
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
            self.band[name] = bandinfo

    def save_multiband_raster(self, name, meta, *band_data):
        """Reads a raster file and returns the data of the first band."""
        file_path = os.path.join(self.folder, f"{self.root}_{name}.TIF")
        bands = len(band_data)
        meta.update(count=bands, dtype=band_data[0].dtype)
        with rasterio.open(file_path, "w", **meta) as dst:
            bandn = 0
            for band in band_data:
                bandn += 1
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
        bandmeta = bandmeta.update(dtype=np.float64, nodata=np.nan)
        return band_rescaled, bandmeta

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

        lst = bt / (1 + lambda_bt_ratio * log_emissivity)

        return lst

    # method to convert raster to float
    def raster_to_float(self, raster: np.uint8) -> np.float64:
        """Converts a raster to float64 data type."""
        raster64 = raster.astype(np.float64)  # Convert to float64
        raster64[raster64 == 0] = np.nan  # Set zero values to NaN
        return raster64

    # method to calculate NDVI
    def calculate_ndvi(self, save: bool = True) -> tuple[np.float64 | float, dict]:
        """Calculates NDVI"""

        b3, b3meta = self.read_band(self.band["B3"].path)
        b4, b4meta = self.read_band(self.band["B4"].path)
        b3 = self.raster_to_float(b3)
        b4 = self.raster_to_float(b4)
        ndvi_meta = b3meta
        ndvi_meta.update(dtype=np.float64, nodata=np.nan)

        # needs reflectance rescale

        ndvi = (b4 - b3) / (b4 + b3)

        if save is True:
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
