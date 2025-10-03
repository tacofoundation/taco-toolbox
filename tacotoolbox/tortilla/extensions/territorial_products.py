import ee


def create_single_product(
    name, path, reducer, band=None, collection_type="Image", unmask_value=0
):
    """Create a single territorial product."""
    if collection_type == "ImageCollection":
        image = ee.ImageCollection(path).mosaic().unmask(unmask_value)
    else:
        image = ee.Image(path).unmask(unmask_value)

    if band:
        image = image.select(band)

    image = image.rename(name)

    return {"name": name, "image": image, "reducer": reducer}


def create_soil_products():
    """Create all soil-related products."""
    soil_datasets = {
        "soil_clay": "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02",
        "soil_sand": "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02",
        "soil_carbon": "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
        "soil_bulk_density": "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02",
        "soil_ph": "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02",
    }

    products = []
    for name, path in soil_datasets.items():
        products.append(
            create_single_product(
                name=name, path=path, reducer=ee.Reducer.mean(), band="b0"
            )
        )
    return products


def create_admin_products():
    """Create administrative boundary products."""
    admin_datasets = {
        "admin_countries": "projects/ee-csaybar-real/assets/admin0",
        "admin_states": "projects/ee-csaybar-real/assets/admin1",
        "admin_districts": "projects/ee-csaybar-real/assets/admin2",
    }

    products = []
    for name, path in admin_datasets.items():
        products.append(
            create_single_product(
                name=name, path=path, reducer=ee.Reducer.mode(), unmask_value=65535
            )
        )
    return products


def get_territorial_products():
    """Returns list of all territorial products."""
    products = []

    # 1. Physical/topographic features
    products.append(
        create_single_product(
            "elevation",
            "projects/sat-io/open-datasets/GLO-30",
            ee.Reducer.mean(),
            collection_type="ImageCollection",
        )
    )
    products.append(
        create_single_product(
            "cisi", "projects/sat-io/open-datasets/CISI/global_CISI", ee.Reducer.mean()
        )
    )

    # 2. Climate variables
    products.append(
        create_single_product(
            "precipitation",
            "projects/ee-csaybar-real/assets/precipitation",
            ee.Reducer.mean(),
        )
    )
    products.append(
        create_single_product(
            "temperature",
            "projects/ee-csaybar-real/assets/temperature",
            ee.Reducer.mean(),
        )
    )

    # 3. Soil properties
    products.extend(create_soil_products())

    # 4. Socioeconomic and human impact
    products.append(
        create_single_product(
            "gdp",
            "projects/sat-io/open-datasets/GRIDDED_HDI_GDP/total_gdp_perCapita_1990_2022_5arcmin",
            ee.Reducer.mean(),
            band="PPP_2022",
        )
    )
    products.append(
        create_single_product(
            "human_modification",
            "projects/sat-io/open-datasets/GHM/HM_1990_2020_OVERALL_300M",
            ee.Reducer.mean(),
            band="constant",
            collection_type="ImageCollection",
        )
    )
    products.append(
        create_single_product(
            "population",
            "projects/sat-io/open-datasets/hrsl/hrslpop",
            ee.Reducer.mean(),
            collection_type="ImageCollection",
        )
    )

    # 5. Administrative boundaries (at the end)
    products.extend(create_admin_products())

    return products
