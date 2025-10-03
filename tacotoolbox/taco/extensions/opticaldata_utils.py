from collections.abc import Callable

SensorFunc = Callable[[str | list[str]], dict]


def landsat1mss_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 1 - MSS sensor.

    Args:
        bands (str | list, optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the Landsat 1 - MSS sensor.
    """

    landsat1_mss_bands = {
        "B04": {
            "index": 0,
            "common_name": "Green",
            "description": "Band 4 - Green - 60m",
            "center_wavelength": 553.000,
            "full_width_half_max": 98.000,
        },
        "B05": {
            "index": 1,
            "common_name": "Red",
            "description": "Band 5 - Red - 60m",
            "center_wavelength": 652.500,
            "full_width_half_max": 95.000,
        },
        "B06": {
            "index": 2,
            "common_name": "NIR 1",
            "description": "Band 6 - Near infrared 1 - 60m",
            "center_wavelength": 747.500,
            "full_width_half_max": 105.000,
        },
        "B07": {
            "index": 3,
            "common_name": "NIR 2",
            "description": "Band 7 - Near infrared 2 - 60m",
            "center_wavelength": 900.000,
            "full_width_half_max": 178.000,
        },
    }

    if bands != "all":
        landsat1_mss_bands = {k: landsat1_mss_bands[k] for k in bands}

    return landsat1_mss_bands


def landsat2mss_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 2 - MSS sensor.

    Args:
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".
    Returns:
        dict: The bands of the Landsat 2 - MSS sensor.
    """

    landsat2_mss_bands = {
        "B04": {
            "index": 0,
            "common_name": "Green",
            "description": "Band 4 - Green - 60m",
            "center_wavelength": 549.500,
            "full_width_half_max": 101.000,
        },
        "B05": {
            "index": 1,
            "common_name": "Red",
            "description": "Band 5 - Red - 60m",
            "center_wavelength": 659.000,
            "full_width_half_max": 104.000,
        },
        "B06": {
            "index": 2,
            "common_name": "NIR 1",
            "description": "Band 6 - Near infrared 1 - 60m",
            "center_wavelength": 750.500,
            "full_width_half_max": 105.000,
        },
        "B07": {
            "index": 3,
            "common_name": "NIR 2",
            "description": "Band 7 - Near infrared 2 - 60m",
            "center_wavelength": 897.000,
            "full_width_half_max": 184.000,
        },
    }

    if bands != "all":
        landsat2_mss_bands = {k: landsat2_mss_bands[k] for k in bands}

    return landsat2_mss_bands


def landsat3mss_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 3 - MSS sensor.

    Args:
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".
    Returns:
        dict: The bands of the Landsat 3 - MSS sensor.
    """

    landsat3_mss_bands = {
        "B04": {
            "index": 0,
            "common_name": "Green",
            "description": "Band 4 - Green - 60m",
            "center_wavelength": 544.000,
            "full_width_half_max": 96.000,
        },
        "B05": {
            "index": 1,
            "common_name": "Red",
            "description": "Band 5 - Red - 60m",
            "center_wavelength": 653.500,
            "full_width_half_max": 101.000,
        },
        "B06": {
            "index": 2,
            "common_name": "NIR1",
            "description": "Band 6 - Near infrared 1 - 60m",
            "center_wavelength": 743.000,
            "full_width_half_max": 100.000,
        },
        "B07": {
            "index": 3,
            "common_name": "NIR 2",
            "description": "Band 7 - Near infrared 2 - 60m",
            "center_wavelength": 896.500,
            "full_width_half_max": 167.000,
        },
    }

    if bands != "all":
        landsat3_mss_bands = {k: landsat3_mss_bands[k] for k in bands}

    return landsat3_mss_bands


def landsat4mss_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 4 - MSS sensor.

    Args:

        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".
    Returns:
        dict: The bands of the Landsat 4 - MSS sensor.
    """

    landsat4_mss_bands = {
        "B01": {
            "index": 0,
            "common_name": "Green",
            "description": "Band 1 - Green - 60m",
            "center_wavelength": 549.500,
            "full_width_half_max": 109.000,
        },
        "B02": {
            "index": 1,
            "common_name": "Red",
            "description": "Band 2 - Red - 60m",
            "center_wavelength": 650.500,
            "full_width_half_max": 93.000,
        },
        "B03": {
            "index": 2,
            "common_name": "NIR 1",
            "description": "Band 3 - Near infrared 1 - 60m",
            "center_wavelength": 756.500,
            "full_width_half_max": 111.000,
        },
        "B04": {
            "index": 3,
            "common_name": "NIR 2",
            "description": "Band 4 - Near infrared 2 - 30m",
            "center_wavelength": 914.000,
            "full_width_half_max": 216.000,
        },
    }

    if bands != "all":
        landsat4_mss_bands = {k: landsat4_mss_bands[k] for k in bands}

    return landsat4_mss_bands


def landsat4tm_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 4 - TM sensor.

    Args:
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the Landsat 4 - TM sensor.
    """

    landsat4_tm_bands = {
        "B01": {
            "index": 0,
            "common_name": "Blue",
            "description": "Band 1 - Blue - 30m",
            "center_wavelength": 484.500,
            "full_width_half_max": 65.000,
        },
        "B02": {
            "index": 1,
            "common_name": "Green",
            "description": "Band 2 - Green - 30m",
            "center_wavelength": 569.000,
            "full_width_half_max": 80.000,
        },
        "B03": {
            "index": 2,
            "common_name": "Red",
            "description": "Band 3 - Red - 30m",
            "center_wavelength": 659.000,
            "full_width_half_max": 68.000,
        },
        "B04": {
            "index": 3,
            "common_name": "NIR",
            "description": "Band 4 - Near infrared - 30m",
            "center_wavelength": 841.000,
            "full_width_half_max": 128.000,
        },
        "B05": {
            "index": 4,
            "common_name": "SWIR 1",
            "description": "Band 5 - Shortwave infrared 1 - 30m",
            "center_wavelength": 1678.000,
            "full_width_half_max": 214.000,
        },
        "B06": {
            "index": 5,
            "common_name": "TIR",
            "description": "Band 6 - Thermal infrared - 30m",
            "center_wavelength": 11080.000,
            "full_width_half_max": 1300.000,
        },
        "B07": {
            "index": 6,
            "common_name": "SWIR 2",
            "description": "Band 7 - Shortwave infrared 2 - 30m",
            "center_wavelength": 2222.500,
            "full_width_half_max": 247.000,
        },
    }

    if bands != "all":
        landsat4_tm_bands = {k: landsat4_tm_bands[k] for k in bands}

    return landsat4_tm_bands


def landsat5mss_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 5 - MSS sensor.

    Args:

        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the Landsat 5 - MSS sensor.
    """

    landsat5_mss_bands = {
        "B01": {
            "index": 0,
            "common_name": "Green",
            "description": "Band 1 - Green - 60m",
            "center_wavelength": 552.500,
            "full_width_half_max": 109.000,
        },
        "B02": {
            "index": 1,
            "common_name": "Red",
            "description": "Band 2 - Red - 60m",
            "center_wavelength": 648.500,
            "full_width_half_max": 93.000,
        },
        "B03": {
            "index": 2,
            "common_name": "NIR 1",
            "description": "Band 3 - Near infrared 1 - 60m",
            "center_wavelength": 760.000,
            "full_width_half_max": 110.000,
        },
        "B04": {
            "index": 3,
            "common_name": "NIR 2",
            "description": "Band 4 - Near infrared 2 - 30m",
            "center_wavelength": 925.000,
            "full_width_half_max": 230.000,
        },
    }

    if bands != "all":
        landsat5_mss_bands = {k: landsat5_mss_bands[k] for k in bands}

    return landsat5_mss_bands


def landsat5tm_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 5 - TM sensor.

    Args:

        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the Landsat 5 - TM sensor.
    """

    landsat5_tm_bands = {
        "B01": {
            "index": 0,
            "common_name": "Blue",
            "description": "Band 1 - Blue - 30m",
            "center_wavelength": 485.000,
            "full_width_half_max": 64.000,
        },
        "B02": {
            "index": 1,
            "common_name": "Green",
            "description": "Band 2 - Green - 30m",
            "center_wavelength": 569.000,
            "full_width_half_max": 80.000,
        },
        "B03": {
            "index": 2,
            "common_name": "Red",
            "description": "Band 3 - Red - 30m",
            "center_wavelength": 660.000,
            "full_width_half_max": 66.000,
        },
        "B04": {
            "index": 3,
            "common_name": "NIR",
            "description": "Band 4 - Near infrared - 30m",
            "center_wavelength": 840.000,
            "full_width_half_max": 127.000,
        },
        "B05": {
            "index": 4,
            "common_name": "SWIR 1",
            "description": "Band 5 - Shortwave infrared 1 - 30m",
            "center_wavelength": 1676.000,
            "full_width_half_max": 214.000,
        },
        "B06": {
            "index": 5,
            "common_name": "TIR",
            "description": "Band 6 - Thermal infrared - 30m",
            "center_wavelength": 11435.000,
            "full_width_half_max": 1970.000,
        },
        "B07": {
            "index": 6,
            "common_name": "SWIR 2",
            "description": "Band 7 - Shortwave infrared 2 - 30m",
            "center_wavelength": 2223.500,
            "full_width_half_max": 249.000,
        },
    }

    if bands != "all":
        landsat5_tm_bands = {k: landsat5_tm_bands[k] for k in bands}

    return landsat5_tm_bands


def landsat7etm_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 7 - ETM+ sensor.

    Args:

        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the Landsat 7 - ETM+ sensor.
    """

    landsat7_bands = {
        "B01": {
            "index": 0,
            "common_name": "Blue",
            "description": "Band 1 - Blue - 30m",
            "center_wavelength": 477.500,
            "full_width_half_max": 71.000,
        },
        "B02": {
            "index": 1,
            "common_name": "Green",
            "description": "Band 2 - Green - 30m",
            "center_wavelength": 560.000,
            "full_width_half_max": 80.000,
        },
        "B03": {
            "index": 2,
            "common_name": "Red",
            "description": "Band 3 - Red - 30m",
            "center_wavelength": 661.500,
            "full_width_half_max": 61.000,
        },
        "B04": {
            "index": 3,
            "common_name": "NIR",
            "description": "Band 4 - Near infrared - 30m",
            "center_wavelength": 835.000,
            "full_width_half_max": 126.000,
        },
        "B05": {
            "index": 4,
            "common_name": "SWIR 1",
            "description": "Band 5 - Shortwave infrared 1 - 30m",
            "center_wavelength": 1648.000,
            "full_width_half_max": 198.000,
        },
        "B06": {
            "index": 5,
            "common_name": "TIR",
            "description": "Band 6 - Longwave infrared - 30m",
            "center_wavelength": 11345.000,
            "full_width_half_max": 2030.000,
        },
        "B07": {
            "index": 6,
            "common_name": "SWIR 2",
            "description": "Band 7 - Shortwave infrared 2 - 30m",
            "center_wavelength": 2205.000,
            "full_width_half_max": 278.000,
        },
        "B08": {
            "index": 7,
            "common_name": "Pan",
            "description": "Band 8 - Panchromatic - 15m",
            "center_wavelength": 705.000,
            "full_width_half_max": 378.000,
        },
    }

    if bands != "all":
        landsat7_bands = {k: landsat7_bands[k] for k in bands}

    return landsat7_bands


def landsat8oli_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 8 - OLI sensor.

    Args:
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the Landsat 8 - OLI sensor.
    """

    landsat8_bands = {
        "B01": {
            "index": 0,
            "common_name": "coastal aerosol",
            "description": "Band 1 - Coastal aerosol - 30m",
            "center_wavelength": 442.500,
            "full_width_half_max": 15.000,
        },
        "B02": {
            "index": 1,
            "common_name": "blue",
            "description": "Band 2 - Blue - 30m",
            "center_wavelength": 482.500,
            "full_width_half_max": 59.000,
        },
        "B03": {
            "index": 2,
            "common_name": "green",
            "description": "Band 3 - Green - 30m",
            "center_wavelength": 561.500,
            "full_width_half_max": 57.000,
        },
        "B04": {
            "index": 3,
            "common_name": "red",
            "description": "Band 4 - Red - 30m",
            "center_wavelength": 654.500,
            "full_width_half_max": 37.000,
        },
        "B05": {
            "index": 4,
            "common_name": "NIR",
            "description": "Band 5 - Near infrared - 30m",
            "center_wavelength": 864.500,
            "full_width_half_max": 27.000,
        },
        "B06": {
            "index": 5,
            "common_name": "SWIR 1",
            "description": "Band 6 - Shortwave infrared 1 - 30m",
            "center_wavelength": 1609.000,
            "full_width_half_max": 84.000,
        },
        "B07": {
            "index": 6,
            "common_name": "SWIR 2",
            "description": "Band 7 - Shortwave infrared 2 - 30m",
            "center_wavelength": 2201.000,
            "full_width_half_max": 186.000,
        },
        "B08": {
            "index": 7,
            "common_name": "Pan",
            "description": "Band 8 - Panchromatic - 15m",
            "center_wavelength": 589.500,
            "full_width_half_max": 171.000,
        },
        "B09": {
            "index": 8,
            "common_name": "Cirrus",
            "description": "Band 9 - Cirrus - 30m",
            "center_wavelength": 1373.500,
            "full_width_half_max": 19.000,
        },
        "B10": {
            "index": 9,
            "common_name": "TIR 1",
            "description": "Band 10 - Thermal infrared 1 - 30 m",
            "center_wavelength": 10875.000,
            "full_width_half_max": 550.000,
        },
        "B11": {
            "index": 10,
            "common_name": "TIR 2",
            "description": "Band 11 - Thermal infrared 2 - 30 m",
            "center_wavelength": 12025.000,
            "full_width_half_max": 950.000,
        },
    }

    if bands != "all":
        landsat8_bands = {k: landsat8_bands[k] for k in bands}

    return landsat8_bands


def landsat9oli_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Landsat 9 - OLI sensor.

    Args:
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the Landsat 9 - OLI sensor.
    """

    landsat9_bands = {
        "B01": {
            "index": 0,
            "common_name": "coastal aerosol",
            "description": "Band 1 - Coastal aerosol - 30m",
            "center_wavelength": 443.000,
            "full_width_half_max": 14.000,
        },
        "B02": {
            "index": 1,
            "common_name": "blue",
            "description": "Band 2 - Blue - 30m",
            "center_wavelength": 481.500,
            "full_width_half_max": 59.000,
        },
        "B03": {
            "index": 2,
            "common_name": "green",
            "description": "Band 3 - Green - 30m",
            "center_wavelength": 561.000,
            "full_width_half_max": 56.000,
        },
        "B04": {
            "index": 3,
            "common_name": "red",
            "description": "Band 4 - Red - 30m",
            "center_wavelength": 654.000,
            "full_width_half_max": 36.000,
        },
        "B05": {
            "index": 4,
            "common_name": "NIR",
            "description": "Band 5 - Near infrared - 30m",
            "center_wavelength": 864.500,
            "full_width_half_max": 27.000,
        },
        "B06": {
            "index": 5,
            "common_name": "SWIR 1",
            "description": "Band 6 - Shortwave infrared 1 - 30m",
            "center_wavelength": 1608.500,
            "full_width_half_max": 85.000,
        },
        "B07": {
            "index": 6,
            "common_name": "SWIR 2",
            "description": "Band 7 - Shortwave infrared 2 - 30m",
            "center_wavelength": 2200.000,
            "full_width_half_max": 188.000,
        },
        "B08": {
            "index": 7,
            "common_name": "Pan",
            "description": "Band 8 - Panchromatic - 15m",
            "center_wavelength": 589.000,
            "full_width_half_max": 172.000,
        },
        "B09": {
            "index": 8,
            "common_name": "Cirrus",
            "description": "Band 9 - Cirrus - 30m",
            "center_wavelength": 1374.000,
            "full_width_half_max": 20.000,
        },
        "B10": {
            "index": 9,
            "common_name": "TIR 1",
            "description": "Band 10 - Thermal infrared 1 - 30 m",
            "center_wavelength": 10825.000,
            "full_width_half_max": 650.000,
        },
        "B11": {
            "index": 10,
            "common_name": "TIR 2",
            "description": "Band 11 - Thermal infrared 2 - 30 m",
            "center_wavelength": 12025.000,
            "full_width_half_max": 850.000,
        },
    }

    if bands != "all":
        landsat9_bands = {k: landsat9_bands[k] for k in bands}

    return landsat9_bands


def sentinel2msi_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the Sentinel 2 sensor.

    Args:
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the Sentinel 2 sensor.
    """

    sentinel2_bands = {
        "B01": {
            "index": 0,
            "common_name": "coastal aerosol",
            "description": "Band 1 - Coastal aerosol - 60m",
            "center_wavelength": 443.500,
            "full_width_half_max": 17.000,
        },
        "B02": {
            "index": 1,
            "common_name": "blue",
            "description": "Band 2 - Blue - 10m",
            "center_wavelength": 496.500,
            "full_width_half_max": 53.000,
        },
        "B03": {
            "index": 2,
            "common_name": "green",
            "description": "Band 3 - Green - 10m",
            "center_wavelength": 560.000,
            "full_width_half_max": 34.000,
        },
        "B04": {
            "index": 3,
            "common_name": "red",
            "description": "Band 4 - Red - 10m",
            "center_wavelength": 664.500,
            "full_width_half_max": 29.000,
        },
        "B05": {
            "index": 4,
            "common_name": "red edge 1",
            "description": "Band 5 - Vegetation red edge 1 - 20m",
            "center_wavelength": 704.500,
            "full_width_half_max": 13.000,
        },
        "B06": {
            "index": 5,
            "common_name": "red edge 2",
            "description": "Band 6 - Vegetation red edge 2 - 20m",
            "center_wavelength": 740.500,
            "full_width_half_max": 13.000,
        },
        "B07": {
            "index": 6,
            "common_name": "red edge 3",
            "description": "Band 7 - Vegetation red edge 3 - 20m",
            "center_wavelength": 783.000,
            "full_width_half_max": 18.000,
        },
        "B08": {
            "index": 7,
            "common_name": "NIR",
            "description": "Band 8 - Near infrared - 10m",
            "center_wavelength": 840.000,
            "full_width_half_max": 114.000,
        },
        "B8A": {
            "index": 8,
            "common_name": "red edge 4",
            "description": "Band 8A - Vegetation red edge 4 - 20m",
            "center_wavelength": 864.500,
            "full_width_half_max": 19.000,
        },
        "B09": {
            "index": 9,
            "common_name": "water vapor",
            "description": "Band 9 - Water vapor - 60m",
            "center_wavelength": 945.000,
            "full_width_half_max": 18.000,
        },
        "B10": {
            "index": 10,
            "common_name": "cirrus",
            "description": "Band 10 - Cirrus - 60m",
            "center_wavelength": 1375.500,
            "full_width_half_max": 31.000,
        },
        "B11": {
            "index": 11,
            "common_name": "SWIR 1",
            "description": "Band 11 - Shortwave infrared 1 - 20m",
            "center_wavelength": 1613.500,
            "full_width_half_max": 89.000,
        },
        "B12": {
            "index": 12,
            "common_name": "SWIR 2",
            "description": "Band 12 - Shortwave infrared 2 - 20m",
            "center_wavelength": 2199.500,
            "full_width_half_max": 173.000,
        },
    }

    if bands != "all":
        sentinel2_bands = {k: sentinel2_bands[k] for k in bands}
    return sentinel2_bands


def eo1ali_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the EO1 - ALI sensor.

    Args:
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the EO1 - ALI sensor.
    """

    eo1_ali_bands = {
        "B01": {
            "index": 0,
            "common_name": "Pan",
            "description": "Band 1 - Panchromatic - 30m",
            "center_wavelength": 588.500,
            "full_width_half_max": 175.000,
        },
        "B02": {
            "index": 1,
            "common_name": "MS-1p",
            "description": "Band 2 - Blue - 30m",
            "center_wavelength": 441.500,
            "full_width_half_max": 19.000,
        },
        "B03": {
            "index": 2,
            "common_name": "MS-1",
            "description": "Band 3 - Blue - 30m",
            "center_wavelength": 566.000,
            "full_width_half_max": 74.000,
        },
        "B04": {
            "index": 3,
            "common_name": "MS-2",
            "description": "Band 4 - Green - 30m",
            "center_wavelength": 482.500,
            "full_width_half_max": 61.000,
        },
        "B05": {
            "index": 4,
            "common_name": "MS-3",
            "description": "Band 5 - Red - 30m",
            "center_wavelength": 660.000,
            "full_width_half_max": 58.000,
        },
        "B06": {
            "index": 5,
            "common_name": "MS-4",
            "description": "Band 6 - Near infrared - 30m",
            "center_wavelength": 790.000,
            "full_width_half_max": 30.000,
        },
        "B07": {
            "index": 6,
            "common_name": "MS-4p",
            "description": "Band 7 - Shortwave infrared - 30m",
            "center_wavelength": 866.000,
            "full_width_half_max": 44.000,
        },
        "B08": {
            "index": 7,
            "common_name": "MS-5",
            "description": "Band 8 - Shortwave infrared - 30m",
            "center_wavelength": 1638.000,
            "full_width_half_max": 184.000,
        },
        "B09": {
            "index": 8,
            "common_name": "MS-5p",
            "description": "Band 9 - Shortwave infrared - 30m",
            "center_wavelength": 1244.500,
            "full_width_half_max": 89.000,
        },
        "B10": {
            "index": 9,
            "common_name": "MS-7",
            "description": "Band 10 - Shortwave infrared - 30m",
            "center_wavelength": 2224.000,
            "full_width_half_max": 276.000,
        },
    }
    if bands != "all":
        eo1_ali_bands = {k: eo1_ali_bands[k] for k in bands}
    return eo1_ali_bands


def aster_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the ASTER sensor.

    Args:
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the ASTER sensor.
    """

    # Define ASTER bands with their properties
    aster_bands = {
        "B01": {
            "index": 0,
            "common_name": "VNIR",
            "description": "B01 - Visible and near infrared - 15m",
            "center_wavelength": 558.000,
            "full_width_half_max": 84.000,
        },
        "B02": {
            "index": 1,
            "common_name": "VNIR",
            "description": "B02 - Visible and near infrared - 15m",
            "center_wavelength": 659.000,
            "full_width_half_max": 60.000,
        },
        "B3N": {
            "index": 2,
            "common_name": "VNIR",
            "description": "B3N - Visible and near infrared - 15m",
            "center_wavelength": 806.000,
            "full_width_half_max": 100.000,
        },
        "B3B": {
            "index": 3,
            "common_name": "VNIR",
            "description": "B3B - Visible and near infrared - 15m",
            "center_wavelength": 806.000,
            "full_width_half_max": 108.000,
        },
        "B04": {
            "index": 4,
            "common_name": "SWIR",
            "description": "B04 - Shortwave infrared - 30m",
            "center_wavelength": 1655.000,
            "full_width_half_max": 90.000,
        },
        "B05": {
            "index": 5,
            "common_name": "SWIR",
            "description": "B05 - Shortwave infrared - 30m",
            "center_wavelength": 2165.000,
            "full_width_half_max": 35.000,
        },
        "B06": {
            "index": 6,
            "common_name": "SWIR",
            "description": "B06 - Shortwave infrared - 30m",
            "center_wavelength": 2208.500,
            "full_width_half_max": 37.500,
        },
        "B07": {
            "index": 7,
            "common_name": "SWIR",
            "description": "B07 - Shortwave infrared - 30m",
            "center_wavelength": 2262.500,
            "full_width_half_max": 45.000,
        },
        "B08": {
            "index": 8,
            "common_name": "SWIR",
            "description": "B08 - Shortwave infrared - 30m",
            "center_wavelength": 2333.500,
            "full_width_half_max": 67.500,
        },
        "B09": {
            "index": 9,
            "common_name": "SWIR",
            "description": "B09 - Shortwave infrared - 30m",
            "center_wavelength": 2397.500,
            "full_width_half_max": 65.000,
        },
        "B10": {
            "index": 10,
            "common_name": "TIR",
            "description": "B10 - Thermal infrared - 90m",
            "center_wavelength": 8280.000,
            "full_width_half_max": 340.000,
        },
        "B11": {
            "index": 11,
            "common_name": "TIR",
            "description": "B11 - Thermal infrared - 90m",
            "center_wavelength": 8630.000,
            "full_width_half_max": 340.000,
        },
        "B12": {
            "index": 12,
            "common_name": "TIR",
            "description": "B12 - Thermal infrared - 90m",
            "center_wavelength": 9075.000,
            "full_width_half_max": 350.000,
        },
        "B13": {
            "index": 13,
            "common_name": "TIR",
            "description": "B13 - Thermal infrared - 90m",
            "center_wavelength": 10655.000,
            "full_width_half_max": 650.000,
        },
        "B14": {
            "index": 14,
            "common_name": "TIR",
            "description": "B14 - Thermal infrared - 90m",
            "center_wavelength": 11300.000,
            "full_width_half_max": 580.000,
        },
    }

    if bands != "all":
        aster_bands = {k: aster_bands[k] for k in bands}
    return aster_bands


def modis_bands(bands: str | list[str] = "all") -> dict:
    """Gets the bands of the MODIS sensor.

    Args:
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the MODIS sensor.
    """

    # Define MODIS bands with their properties
    modis_bands = {
        "B01": {
            "index": 0,
            "common_name": "B01",
            "description": "B01 - Land/Cloud/Aerosols Boundaries - 250m",
            "center_wavelength": 645.690,
            "full_width_half_max": 45.010,
        },
        "B02": {
            "index": 1,
            "common_name": "B02",
            "description": "B02 - Land/Cloud/Aerosols Boundaries - 250m",
            "center_wavelength": 856.880,
            "full_width_half_max": 37.000,
        },
        "B03": {
            "index": 2,
            "common_name": "B03",
            "description": "B03 - Land/Cloud/Aerosols Properties - 500m",
            "center_wavelength": 465.700,
            "full_width_half_max": 17.979,
        },
        "B04": {
            "index": 3,
            "common_name": "B04",
            "description": "B04 - Land/Cloud/Aerosols Properties - 500m",
            "center_wavelength": 553.190,
            "full_width_half_max": 18.989,
        },
        "B05": {
            "index": 4,
            "common_name": "B05",
            "description": "B05 - Land/Cloud/Aerosols Properties - 500m",
            "center_wavelength": 1241.500,
            "full_width_half_max": 22.000,
        },
        "B06": {
            "index": 5,
            "common_name": "B06",
            "description": "B06 - Land/Cloud/Aerosols Properties - 500m",
            "center_wavelength": 1629.000,
            "full_width_half_max": 27.900,
        },
        "B07": {
            "index": 6,
            "common_name": "B07",
            "description": "B07 - Land/Cloud/Aerosols Properties - 500m",
            "center_wavelength": 2113.200,
            "full_width_half_max": 45.000,
        },
        "B08": {
            "index": 7,
            "common_name": "B08",
            "description": "B08 - Ocean Color/ Phytoplankton/Biogeochemistry - 1000m",
            "center_wavelength": 411.670,
            "full_width_half_max": 11.000,
        },
        "B09": {
            "index": 8,
            "common_name": "B09",
            "description": "B09 - Ocean Color/ Phytoplankton/Biogeochemistry - 1000m",
            "center_wavelength": 442.160,
            "full_width_half_max": 8.010,
        },
        "B10": {
            "index": 9,
            "common_name": "B10",
            "description": "B10 - Ocean Color/ Phytoplankton/Biogeochemistry - 1000m",
            "center_wavelength": 486.690,
            "full_width_half_max": 9.980,
        },
        "B11": {
            "index": 10,
            "common_name": "B11",
            "description": "B11 - Ocean Color/ Phytoplankton/Biogeochemistry - 1000m",
            "center_wavelength": 529.669,
            "full_width_half_max": 11.010,
        },
        "B12": {
            "index": 11,
            "common_name": "B12",
            "description": "B12 - Ocean Color/ Phytoplankton/Biogeochemistry - 1000m",
            "center_wavelength": 546.710,
            "full_width_half_max": 9.019,
        },
        "B13": {
            "index": 12,
            "common_name": "B13",
            "description": "B13 - Ocean Color/ Phytoplankton/Biogeochemistry - 1000m",
            "center_wavelength": 665.190,
            "full_width_half_max": 8.020,
        },
        "B14": {
            "index": 13,
            "common_name": "B14",
            "description": "B14 - Ocean Color/ Phytoplankton/Biogeochemistry - 1000m",
            "center_wavelength": 677.219,
            "full_width_half_max": 10.020,
        },
        "B15": {
            "index": 14,
            "common_name": "B15",
            "description": "B15 - Ocean Color/ Phytoplankton/Biogeochemistry - 1000m",
            "center_wavelength": 746.789,
            "full_width_half_max": 9.000,
        },
        "B16": {
            "index": 15,
            "common_name": "B16",
            "description": "B16 - Ocean Color/ Phytoplankton/Biogeochemistry - 1000m",
            "center_wavelength": 866.809,
            "full_width_half_max": 15.011,
        },
        "B17": {
            "index": 16,
            "common_name": "B17",
            "description": "B17 - Atmospheric Water Vapor - 1000m",
            "center_wavelength": 903.750,
            "full_width_half_max": 33.000,
        },
        "B18": {
            "index": 17,
            "common_name": "B18",
            "description": "B18 - Atmospheric Water Vapor - 1000m",
            "center_wavelength": 935.750,
            "full_width_half_max": 13.010,
        },
        "B19": {
            "index": 18,
            "common_name": "B19",
            "description": "B19 - Atmospheric Water Vapor - 1000m",
            "center_wavelength": 935.340,
            "full_width_half_max": 44.990,
        },
        "B20": {
            "index": 19,
            "common_name": "B20",
            "description": "B20 - Surface/Cloud Temperature - 1000m",
            "center_wavelength": 3786.000,
            "full_width_half_max": 190.000,
        },
        "B21": {
            "index": 20,
            "common_name": "B21",
            "description": "B21 - Surface/Cloud Temperature - 1000m",
            "center_wavelength": 3991.000,
            "full_width_half_max": 80.000,
        },
        "B22": {
            "index": 21,
            "common_name": "B22",
            "description": "B22 - Surface/Cloud Temperature - 1000m",
            "center_wavelength": 3971.000,
            "full_width_half_max": 80.000,
        },
        "B23": {
            "index": 22,
            "common_name": "B23",
            "description": "B23 - Atmospheric Temperature - 1000m",
            "center_wavelength": 4057.500,
            "full_width_half_max": 85.000,
        },
        "B24": {
            "index": 23,
            "common_name": "B24",
            "description": "B24 - Atmospheric Temperature - 1000m",
            "center_wavelength": 4471.500,
            "full_width_half_max": 85.000,
        },
        "B25": {
            "index": 24,
            "common_name": "B25",
            "description": "B25 - Atmospheric Temperature - 1000m",
            "center_wavelength": 4546.000,
            "full_width_half_max": 90.000,
        },
        "B26": {
            "index": 25,
            "common_name": "B26",
            "description": "B26 - Cirrus Clouds Water Vapor - 1000m",
            "center_wavelength": 1383.600,
            "full_width_half_max": 34.000,
        },
        "B27": {
            "index": 26,
            "common_name": "B27",
            "description": "B27 - Cirrus Clouds Water Vapor - 1000m",
            "center_wavelength": 6758.500,
            "full_width_half_max": 225.000,
        },
        "B28": {
            "index": 27,
            "common_name": "B28",
            "description": "B28 - Cirrus Clouds Water Vapor - 1000m",
            "center_wavelength": 7334.500,
            "full_width_half_max": 316.000,
        },
        "B29": {
            "index": 28,
            "common_name": "B29",
            "description": "B29 - Cloud Properties - 1000m",
            "center_wavelength": 8526.000,
            "full_width_half_max": 300.000,
        },
        "B30": {
            "index": 29,
            "common_name": "B30",
            "description": "B30 - Cloud Properties - 1000m",
            "center_wavelength": 9752.000,
            "full_width_half_max": 270.000,
        },
        "B31": {
            "index": 30,
            "common_name": "B31",
            "description": "B31 - Surface/Cloud Temperature - 1000m",
            "center_wavelength": 11013.000,
            "full_width_half_max": 500.000,
        },
        "B32": {
            "index": 31,
            "common_name": "B32",
            "description": "B32 - Surface/Cloud Temperature - 1000m",
            "center_wavelength": 12039.500,
            "full_width_half_max": 475.000,
        },
        "B33": {
            "index": 32,
            "common_name": "B33",
            "description": "B33 - Cloud Top Altitude - 1000m",
            "center_wavelength": 13362.000,
            "full_width_half_max": 299.000,
        },
        "B34": {
            "index": 33,
            "common_name": "B34",
            "description": "B34 - Cloud Top Altitude - 1000m",
            "center_wavelength": 13682.500,
            "full_width_half_max": 301.000,
        },
        "B35": {
            "index": 34,
            "common_name": "B35",
            "description": "B35 - Cloud Top Altitude - 1000m",
            "center_wavelength": 13893.000,
            "full_width_half_max": 281.000,
        },
        "B36": {
            "index": 35,
            "common_name": "B36",
            "description": "B36 - Cloud Top Altitude - 1000m",
            "center_wavelength": 14143.500,
            "full_width_half_max": 139.000,
        },
    }

    if bands != "all":
        modis_bands = {k: modis_bands[k] for k in bands}
    return modis_bands


def get_sensor_bands(sensor: str, bands: str | list[str] = "all") -> dict:
    """Gets the bands of a specific sensor.

    Args:
        sensor (str): The sensor to get the bands from.
        bands (str | list[str], optional): The bands to be added.
            If "all", all bands are added. Defaults to "all".

    Returns:
        dict: The bands of the sensor.
    """

    # Mapping of sensors to their corresponding band functions
    sensor_functions: dict[str, SensorFunc] = {
        "landsat1mss": landsat1mss_bands,
        "landsat2mss": landsat2mss_bands,
        "landsat3mss": landsat3mss_bands,
        "landsat4mss": landsat4mss_bands,
        "landsat5mss": landsat5mss_bands,
        "landsat4tm": landsat4tm_bands,
        "landsat5tm": landsat5tm_bands,
        "landsat7etm": landsat7etm_bands,
        "landsat8oli": landsat8oli_bands,
        "landsat9oli": landsat9oli_bands,
        "sentinel2msi": sentinel2msi_bands,
        "eo1ali": eo1ali_bands,
        "aster": aster_bands,
        "modis": modis_bands,
    }

    sensor_lower = sensor.lower()

    if sensor_lower not in sensor_functions:
        raise ValueError(f"Unsupported sensor: {sensor}")

    return sensor_functions[sensor_lower](bands)
