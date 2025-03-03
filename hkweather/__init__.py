"""
`hkweather`: Analyse Landsat NDVI, NDSI, LST and weather station data for the Hindu Kush region, Pakistan
"""

from .landsat import SceneCollection, Scene, ThermalBand, ReflectanceBand, ProcessedBand

name = "hkweather"
__version__ = "0.0.1"
