# Training Data Format Exploration

## Background

This document captures explorations of the AI4Arctic training dataset performed by Gordon Logie in June 2026.

Both the Raw and Ready-to-Train (RTT) datasets were examined. The goals of these explorations were as follows:
- To understand what variables were included with both datasets.
- To understand the spatial reference information included with both files.
- To understand the processing applied to the RTT dataset, in order to later facilitate development of a similar preprocessing chain if we opted to use the raw dataset.

### Resources Used

To do this work, the following resources were used:
- A notebook was developed to explore the dataset, located in this repo at `notebooks/training_data_exploration/training_data_format_exploration.ipynb`
- Select datasets were downloaded from the AI4Arctic Sea Ice Challenge Dataset, both raw and RTT. Specifically within the "Test" collection for each.
    - [Raw Test Collection](https://data.dtu.dk/articles/dataset/Raw_AI4Arctic_Sea_Ice_Challenge_Test_Dataset/21762848?backTo=/collections/AI4Arctic_Sea_Ice_Challenge_Dataset/6244065)
    - [RTT Test Collection](https://data.dtu.dk/articles/dataset/Ready-To-Train_AI4Arctic_Sea_Ice_Challenge_Test_Dataset/21762830?backTo=/collections/AI4Arctic_Sea_Ice_Challenge_Dataset/6244065)
- The [AI4Arctic Challenge Dataset Manual](../../ai4arctic_docs/AI4Arctic_challenge-dataset-manual_21.12.2022.pdf) - a copy is included with each the dataset downloads. It has been copied into this repo for reference.
- [AI4ArcticSeaIceChallenge repo](https://github.com/astokholm/AI4ArcticSeaIceChallenge/tree/main) - a repository containing getting started and utility scripts for working with the challenge dataset.

## Dataset Overview
- The Sea Ice Challenge dataset consists of 533 files (513 training, 20 test) - in the dataset download pages, these are referred to as the "Challenge" and "Test" datasets, respectively.
- Each dataset version is provided in NetCDF format.
- Each dataset have differing file naming conventions:
    - **Raw**: `S1X_EW_GRDM_1SDH_YYYYMMDDThhmmssYYYYMMDDThhmmss_xxxxxx_xxxxxx_xxXX_icechart_icechartname_depending_on_provider.nc`. Where:
        - `S1X_EW_GRDM_1SDH_YYYYMMDDThhmmssYYYYMMDDThhmmss_xxxxxx_xxxxxx_xxXX` is the name of the Sentinel-1 file containing the original SAR imagery
        - `S1X` is the Sentinel-1 Mission identifier: "S1A" or "S1B"
        - `YYYYMMDDThhmmssYYYYMMDDThhmmss` is the Sentinel-1 Mission image acquisition start and stop timestamps, respectively
        - `icechartname_depending_on_provider` is the ice chart name used by the providing ice service (CIS or DMI)
        - For example: `S1A_EW_GRDM_1SDH_20180124T194759_20180124T194859_020301_022AA4_1F75_icechart_dmi_201801241950_SouthEast_RIC.nc`
    - **RTT**: `YYYYMMDDThhmmss_icechartname_depending_on_provider_prep.nc`. Where:
        - `YYYYMMDDThhmmss` is the Sentinel-1 image acquisition start timestamp.
        - `icechartname_depending_on_provider` is the ice chart name used by the providing ice service (CIS or DMI)
    - Important note: in the RTT `Test` dataset, a secondary NetCDF file is provided for each image. The naming convention is the same as for the regular RTT files, but instead of `prep.nc`, these files end with `_reference.nc`. 
        - These contain the sea ice chart label data, which are separated out from the regular sensor data. 
        - In the `Training` dataset, the sea ice chart labels are included in the same NetCDF file as the sensor data. It is assumed the labels were kept separate to be held back from challenge participants until the challenge had concluded.
- The main variables in the dataset can be grouped broadly as follows:
    - Sentinel-1 SAR data and auxiliary information
    - AMSR2 sensor data and auxiliary information
    - ERA5 climate data 
    - Sea ice chart data

### SAR Data
Sentinel-1 SAR data is collected in Extra Wide Swath Mode (EW) at Level-1 Ground Range Detected in Medium Resolution (GRDM) in dual polarizations (HH and HV). Each scene covers 400 x 400 km, at a nominal pixel space of 40 x 40 m.

#### SAR Variables

This section lists the SAR variables included in each dataset. For each variable, the Raw name is listed first, followed by the RTT name. If an equivalent variable doesn't exist in one dataset or another, `None` is substituted for the name.

- `nersc_sar_primary` / `nersc_sar_primary`
    - HH polarization sar backscatter channel
    - raw shape ~(10,000x10,000)
    - rtt shape ~(5000x5000)
- `nersc_sar_secondary` / `nersc_sar_secondary`
    - HV polarization sar backscatter channel
    - raw shape ~(10,000x10,000)
    - rtt shape ~(5000x5000)
- `sar_grid_incidenceangle` / `sar_incidenceangle`
    - incidence angle for sar backscatter
    - in the raw dataset, this data is a 1D array of ~441 values which correspond with the georeferencing points provided in the georeferencing variables (see `Anciliary Data` section).
    - In the RTT data, the incidence angle data has been converted to a 2D grid and interpolated to match the pixel grid of the sar backscatter (~5000 x 5000)

#### SAR Preprocessing (both Raw and RTT)
The following is applied to both the raw and RTT datasets (detailed in the manual on page 8):
- A noise algorithm correction is applied (see ####NERSC Noise Correction)
- Negative values are replaced using a two step process (see manual page 8).

#### NERSC Noise Correction
The noise correction algorithm used on the SAR data is provided by Nansen Environmental and Remote Sensing Center (NERSC) and is considered superior to the standard noise correction provided by ESA.

Sentinel-1 Extra Wide (EW) mode imagery is subject to significant additive thermal noise in the cross-polarization (HV) channel, which produces visible inter-subswath intensity discontinuities. The NERSC denoising algorithm applies a correction that addresses this inter-subswath radiometric bias, producing a more radiometrically consistent HV backscatter product than the standard ESA processing. The methodology is described in: *Thermal Denoising of Cross-Polarized Sentinel-1 Data in Interferometric and Extra Wide Swath Modes*, DOI: 10.1109/TGRS.2021.3131036.

#### SAR RTT Preprocessing
The RTT SAR data is additionally processed (Manual page 26). This includes:
- Downsampling the data from 40 m to 80 m pixel spacing. This is performed using a 2x2 averaging kernel.
- Normalization of SAR backscatter and incidence angle to the range [-1, 1] using per-variable minimum and maximum values. Per-variable statistics (means, standard deviations, and the min/max values used) are available in the `misc/` folder of the [AI4ArcticSeaIceChallenge repository](https://github.com/astokholm/AI4ArcticSeaIceChallenge/tree/main).
- Masking is applied to invalidate over-land pixels

### AMSR2 Data

#### AMSR2 Variables

This section lists the AMSR2 variables included in each dataset, using the same Raw / RTT naming convention as above.

- `btemp_6_9h` / `btemp_6_9h`
    - AMSR2 brightness temperature at 6.9 GHz, horizontal polarization. 
    - Shape ~(200, 209)
- `btemp_6_9v` / `btemp_6_9v`
    - AMSR2 brightness temperature at 6.9 GHz, vertical polarization. 
    - Shape ~(200, 209)
- `btemp_7_3h` / `btemp_7_3h` 
    - 7.3 GHz, horizontal polarization.
    - Shape ~(200, 209)
- `btemp_7_3v` / `btemp_7_3v`
    - 7.3 GHz, vertical polarization.
- `btemp_10_7h` / `btemp_10_7h`
    - 10.7 GHz, horizontal polarization.
- `btemp_10_7v` / `btemp_10_7v`
    - 10.7 GHz, vertical polarization.
- `btemp_18_7h` / `btemp_18_7h`
    - 18.7 GHz, horizontal polarization.
- `btemp_18_7v` / `btemp_18_7v`
    - 18.7 GHz, vertical polarization.
- `btemp_23_8h` / `btemp_23_8h`
    - 23.8 GHz, horizontal polarization.
- `btemp_23_8v` / `btemp_23_8v`
    - 23.8 GHz, vertical polarization.
- `btemp_36_5h` / `btemp_36_5h`
    - 36.5 GHz, horizontal polarization.
- `btemp_36_5v` / `btemp_36_5v`
    - 36.5 GHz, vertical polarization.
- `btemp_89_0h` / `btemp_89_0h`
    - 89.0 GHz, horizontal polarization.
- `btemp_89_0v` / `btemp_89_0v`
    - 89.0 GHz, vertical polarization.
- `amsr2_swath_map` / `None`
    - map indicating which AMSR2 swath pass contributed to each 2 km grid cell. Raw only.
- `swath_segmentation` / `None`
    - segmentation of the scene into individual AMSR2 swath passes. Raw only.

#### AMSR2 Preprocessing (both Raw and RTT)

The AMSR2 brightness temperature data is provided at a native resolution of approximately 2 km and is gridded to a 2 km grid aligned to the SAR scene extent. This grid is shared with the ERA5 data. Both the raw and RTT datasets contain the same AMSR2 variables at the same ~(200, 209) shape. In the RTT dataset, values are stored as `float32` rather than the `float64` used in the raw files.

The two raw-only variables (`amsr2_swath_map`, `swath_segmentation`) are not carried through to the RTT dataset. These appear to be auxiliary swath metadata used during the gridding process and are not required as model inputs.

#### AMSR2 RTT Preprocessing

AMSR2 brightness temperature variables are normalized to the range [-1, 1] using per-variable minimum and maximum values in the RTT dataset.

### ERA5 Data

#### ERA5 Variables

This section lists the ERA5 variables included in each dataset, using the same Raw / RTT naming convention as above. All ERA5 variables share the 2 km grid dimensions ~(200, 209).

- `u10m_rotated` / `u10m_rotated`
    - eastward component of 10 m wind speed, rotated to align with the SAR scene flight direction.
- `v10m_rotated` / `v10m_rotated`
    - northward component of 10 m wind speed, rotated to align with the SAR scene flight direction.
- `t2m` / `t2m`
    - 2 m air temperature (K).
- `skt` / `skt`
    - skin temperature (K).
- `tcwv` / `tcwv`
    - total column water vapour (kg/m²).
- `tclw` / `tclw`
    - total column cloud liquid water (kg/m²).

#### ERA5 Preprocessing (both Raw and RTT)

ERA5 reanalysis data is sourced from the Copernicus Climate Data Store and resampled to the shared 2 km grid used by the AMSR2 variables. The wind components (`u10m_rotated`, `v10m_rotated`) are rotated from their native geographic orientation to align with the Sentinel-1 satellite flight direction, as noted in the raw file's `geometric_info` attribute. As with the AMSR2 data, values are stored as `float64` in the raw dataset and `float32` in the RTT dataset.

#### ERA5 RTT Preprocessing

ERA5 variables are normalized to the range [-1, 1] using per-variable minimum and maximum values in the RTT dataset.

### Sea Ice Data

#### Sea Ice Variables

This section lists the sea ice variables included in each dataset, using the same Raw / RTT naming convention as above.

- `polygon_icechart` / `None` — per-pixel polygon ID, gridded to the native SAR resolution ~(10,000, 10,000). Each value is an integer ID that indexes into the `polygon_codes` coordinate to retrieve the full sea ice chart parameters for that pixel. See `#### Raw Sea Ice Lookup Table` for details on the encoding. Raw only.
- `None` / `SIC` — Sea Ice Concentration, derived from the polygon lookup table. Stored as `uint8`, shape ~(5,000, 5,000). Fill value: 255.
- `None` / `SOD` — Stage of Development, derived from the polygon lookup table. Stored as `uint8`, shape ~(5,000, 5,000). Fill value: 255.
- `None` / `FLOE` — Floe Size, derived from the polygon lookup table. Stored as `uint8`, shape ~(5,000, 5,000). Fill value: 255.

#### Raw Sea Ice Lookup Table

The raw dataset encodes sea ice chart information via a lookup table stored in the `polygon_codes` coordinate. Each entry is a semicolon-delimited string with the following fields (in order):

`poly_id;CT;CA;SA;FA;CB;SB;FB;CC;SC;FC;CN;CD;CF;POLY_TYPE`

Where the fields follow the SIGRID-3 standard used by operational ice services:
- `poly_id` — polygon identifier, corresponding to values in `polygon_icechart`
- `CT` — total sea ice concentration
- `CA`, `CB`, `CC` — concentration of the 1st, 2nd, and 3rd thickest ice types
- `SA`, `SB`, `SC` — stage of development (ice type) of each ice layer
- `FA`, `FB`, `FC` — form of ice (floe size) of each ice layer
- `CN` — concentration of new/nilas/frazil ice
- `CD`, `CF` — stage and form of the thinnest ice layer
- `POLY_TYPE` — polygon type (`I` for ice chart polygon, `O` for open water)

A value of `-9` indicates that a field is not applicable or not reported for that polygon. To recover the sea ice properties for any pixel, look up its `polygon_icechart` value in the `poly_id` field of this table.

#### Sea Ice Data RTT Preprocessing

The RTT label variables (`SIC`, `SOD`, `FLOE`) are derived by remapping the SIGRID-3 encoded values from the `polygon_codes` lookup table to a reduced class scheme. The full conversion tables are defined in Table 8 of the dataset manual (page 27), and are also implemented as lookup dictionaries in `utils.py` in the [AI4ArcticSeaIceChallenge repository](https://github.com/astokholm/AI4ArcticSeaIceChallenge/tree/main) — that file is the authoritative reference for the exact code-to-class mappings.

In summary, each variable is remapped as follows:
- **SIC** — derived from the `CT` field. The SIGRID-3 concentration codes are remapped to 11 integer classes (0–10), representing ice-free through full coverage. Codes for bergy water, unknown, and unfilled values are mapped to fill value 255.
- **SOD** — derived from the `SA` field. Stage-of-development codes are grouped into 6 classes: 0 (ice free), 1 (new ice), 2 (young ice), 3 (thin first-year ice), 4 (thick first-year ice), 5 (old/multi-year ice). Glacier ice, unknown, and unfilled codes are mapped to fill value 255.
- **FLOE** — derived from the `FA` field. Floe size codes are grouped into 7 classes: 0 (ice free), 1 (small ice cake), 2 (ice cake / small floe), 3 (medium floe), 4 (big floe), 5 (vast / giant floe), 6 (bergs). Pancake ice, fast ice, level ice, unknown, and unfilled codes are mapped to fill value 255.

The fill value 255 must be masked out during training to prevent it from contributing to the loss. Not all SIGRID-3 codes appear in the dataset — some are excluded from the class scheme because they occur in too few polygons to be adequately represented.

### Anciliary Data
There are several anciliary variables included in the raw NetCDF files. These are primarily spatial information for georeferencing.

#### Anciliary Variables
This section lists the anciliary variables included in each dataset. For each variable, the Raw name is listed first, followed by the RTT name. If an equivalent variable doesn't exist in one dataset or another, `None` is substituted for the name.

- `sar_grid_latitude` / `sar_grid2d_latitude`
    - Latitude spatial information for sar data
    - ~ 441 grid point locations evenly distributed across the scene
    - 1D vector in raw, converted to 2D in RTT (~21 x 21)
- `sar_grid_longitude` / `sar_grid2d_longitude`
    - Longitude spatial information for sar data
    - ~ 441 grid point locations evenly distributed across the scene
    - 1D vector in raw, converted to 2D in RTT (~21 x 21)
- `sar_grid_line` / `None`
    - Line (row) number corresponding to the 441 geographic grid points in the SAR image.
    - 1D vector in raw, no equivalent in RTT
- `sar_grid_sample` / `None`
    - Sample (column) number corresponding to the 441 geographic grid points in the SAR
    - 1D vector in raw, no equivalent in RTT
- `sar_grid_height` / `None`
    - Height above sea level for the 441 geographic grid points
    - 1D vector in raw, no equivalent in RTT
- `distance_map` / `distance_map`
    - Distance-to-land classified map. Gives distance to land per sar pixel
    - raw shape ~(10,000x10,000)
    - rtt shape ~(5000x5000)

#### Distance Map Processing
The `distance_map` variable is generated by computing the per-pixel distance to the nearest land mass, using OpenStreetMap coastlines as the land reference. Pixels are classified into discrete distance bands rather than storing a continuous distance value. The resulting classes encode proximity to land (e.g. near-coast, open water), which is useful as a model input since SAR backscatter near coastlines can be contaminated by land returns. In the RTT dataset, pixels classified as over-land are masked out (set to a fill value) as part of the SAR preprocessing step described in `#### RTT Preprocessing`. The full classification scheme and band boundaries are defined in the dataset manual. In the RTT dataset, the `distance_map` is also normalized to the range [-1, 1] using the same per-variable min/max approach applied to all other sensor variables.

#### Georeferencing Image Workflow
Neither the raw nor RTT datasets include an embedded CRS or affine geotransform. Instead, geographic location is encoded through sparse Ground Control Points (GCPs). The notebook at `notebooks/training_data_exploration/training_data_format_exploration.ipynb` implements a workflow to georeference and export any variable as a GeoTIFF using these GCPs.

The process differs slightly between the two datasets:
- **Raw**: GCPs are constructed from the 1D `sar_grid_latitude`, `sar_grid_longitude`, `sar_grid_line`, and `sar_grid_sample` arrays (~441 points). For variables on the 2 km grid, row/column positions are scaled by the ratio of 2 km grid dimensions to SAR grid dimensions before constructing GCPs.
- **RTT**: GCPs are constructed from the 2D `sar_grid2d_latitude` and `sar_grid2d_longitude` arrays (~21×21 points). Row and column positions are interpolated linearly across the image extent.

In both cases, GCPs are embedded into the output GeoTIFF in WGS84 (EPSG:4326). Warping or reprojection to a regular grid can then be performed downstream using tools such as `gdalwarp`.

## Get Started Tooling

The [AI4ArcticSeaIceChallenge repository](https://github.com/astokholm/AI4ArcticSeaIceChallenge/tree/main) provides reference tooling for working with the challenge dataset. Key utilities include:
- Data loader implementations that handle the multi-resolution structure of the dataset (SAR at ~5,000×5,000 and AMSR2/ERA5 at ~200×209).
- Class lookup dictionaries that map the `uint8` encoded values in `SIC`, `SOD`, and `FLOE` to their WMO/SIGRID-3 class labels.
- A reference U-Net model architecture used as the challenge baseline.

These utilities are useful as a reference when building our own preprocessing chain, particularly for the sea ice class encoding and the handling of the fill value (255) during loss computation.