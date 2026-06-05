# Prescient Ice: Automated Arctic Sea Ice Concentration Mapping

## Background and Motivation

The Arctic is changing rapidly. Sea ice is melting earlier and freezing later each year, with significant implications for ecosystems, northern communities, and global shipping as previously impassable routes become seasonally navigable. Accurate, timely sea ice information is critical for safe navigation and environmental monitoring in this evolving landscape.

Today, operational sea ice monitoring is carried out by national ice services — primarily the Canadian Ice Service (CIS) and the U.S. National Ice Center (USNIC). These organizations produce sea ice concentration maps through manual interpretation of synthetic aperture radar (SAR) imagery. Analysts examine SAR scenes, delineate regions of similar ice conditions, and assign concentration values to each polygon. While this approach benefits from expert knowledge, it introduces limitations: analyst subjectivity, coarse spatial resolution, publication latency, and distribution in formats poorly suited to real-time use.

Prescient Ice seeks to automate and improve upon this process. Using SAR imagery as the primary input, a geospatial foundation model (GeoFM) classifies ice concentration on a 320m × 320m grid into one of eleven discrete classes (0–10 tenths) matching the SIGRID-3 chart scheme — a substantial resolution improvement over existing ice chart polygons. The 320m resolution is defined by the Clay foundation model's patch footprint at Sentinel-1 EW's native ~40m ground sampling distance, and represents the honest effective resolution of the model rather than a chosen round-number target. Automation removes analyst subjectivity and enables faster turnaround from satellite acquisition to actionable information.

The project also serves as an internal showcase for Prescient, our cloud-native geospatial data management platform. It demonstrates the full data lifecycle — ingesting diverse sources, managing them as STAC collections, and serving both source data and derived products through a common interface. It showcases our ability to work with geospatial foundation models and multi-sensor data fusion, combining SAR, passive microwave, lidar altimetry, climate reanalysis, and optical remote sensing into a coherent analytical pipeline.

## Objectives

1. **Produce high-resolution SIC estimates from SAR imagery.** Using Clay v1.5 as a feature extractor, classify sea ice concentration into eleven discrete classes (0–10 tenths) from Sentinel-1 EW SAR on a 320m grid, with ERA5 and AMSR2 as ancillary inputs. The primary approach (Phase 1) uses Clay as a frozen extractor with downstream classifiers (Random Forest and XGBoost evaluated in parallel); end-to-end Clay fine-tuning is a conditional second phase if the accuracy gap warrants it.

2. **Demonstrate end-to-end data management through Prescient.** Ingest 2025–26 Hudson Bay data sources as STAC-compliant collections (COGs for rasters, GeoParquet + PMTiles dual assets for vectors), and demonstrate the full round-trip: source data served out to analytics, derived products (Clay patch token embeddings, SIC class outputs) ingested back in. Showcase Prescient's format flexibility, multi-source catalog management, and analytics integration across a realistic, complex project.

3. **Build a multi-sensor training and validation framework.** Train and evaluate on the AI4Arctic Sea Ice Challenge Dataset, which bundles co-registered Sentinel-1 EW SAR, AMSR2, ERA5, and CIS/DMI chart labels across 513 scenes. Conduct prospective evaluation on a 2025–26 Hudson Bay dataset acquired and processed through the Prescient pipeline, with USNIC charts as the evaluation label source. ICESat-2 altimetry serves as an independent physical validation overlay and as a candidate supplementary label source if 2025–26 data is incorporated into a retraining run. HLS optical imagery provides seasonal visual validation and context.

4. **Deliver results through an interactive visualization interface.** Serve SIC class predictions alongside source and validation layers through a MapLibre-powered web viewer, demonstrating Prescient's visualization capabilities with a multi-layer, multi-source display.

5. **Establish a cloud-native analytics pipeline on AWS.** Implement ingestion, embedding extraction, training, inference, and delivery using AWS infrastructure, with Prescient as the data layer throughout for the 2025–26 pipeline.

## Project Overview

Prescient Ice is structured as a six-stage pipeline. Prescient sits at the centre as the shared data layer for the 2025–26 Hudson Bay prospective evaluation and inference pipeline, providing a single source of truth for source imagery, labels, ancillary data, Clay patch token embeddings, and SIC class outputs through a common STAC API. The primary training pipeline consumes the AI4Arctic dataset directly, since AI4Arctic ships as a pre-curated, scene-co-registered NetCDF dataset whose primary value would be lost if re-ingested as separate per-source STAC collections; AI4Arctic remains outside Prescient by design.

**Data acquisition** gathers source datasets from their providers. The primary training and evaluation source is AI4Arctic, downloaded from DTU Data or accessed via TorchGeo. For the 2025–26 Hudson Bay pipeline, Sentinel-1 EW SAR is acquired from CDSE, AMSR2 brightness temperature from JAXA G-Portal, ERA5 from the Copernicus Climate Data Store, USNIC weekly Arctic charts from the USNIC archive, ICESat-2 ATL07/ATL10 from NSIDC, and HLS L30/S30 optical imagery from NASA Earthdata.

**Ingestion** converts the 2025–26 source files into Prescient-compatible formats (COGs for rasters, GeoParquet + PMTiles dual assets for vectors) and registers them as STAC collections. The NERSC noise correction is applied to Sentinel-1 EW HH/HV scenes during ingestion to maintain input distribution consistency with the NERSC-corrected AI4Arctic training data.

**Embedding generation** passes Sentinel-1 EW scenes through Clay v1.5's frozen encoder, extracting 32×32 patch token grids per chip (each 1024-dim token covering ~320m × 320m at EW ~40m GSD) along with a class token serving as the chip-level embedding. For the 2025–26 pipeline, these patch token grids are ingested into Prescient as a derived STAC collection, decoupling the compute-intensive encoding step from downstream classifier iteration.

**Training** operates primarily on AI4Arctic: chips are extracted, Clay encoded, paired with rasterised CIS/DMI chart class labels at the 320m grid, and used to train downstream classifiers (Random Forest and XGBoost) across three feature configurations — raw HH/HV backscatter baseline, patch tokens alone, and patch tokens combined with the class token chip embedding. If Phase 1 accuracy falls below threshold or fails to outperform the non-embedding baseline, Phase 2 initiates end-to-end Clay fine-tuning on GPU.

**Inference** applies the trained classifier to new 2025–26 Sentinel-1 scenes and ingests the resulting 320m SIC class grids into Prescient as a derived collection, completing the data round-trip from source imagery through derived analytical products under a single catalogue interface.

**Visualization** serves predictions and supporting data through a MapLibre-powered web viewer, where analysts can overlay SIC class output on source SAR, compare against USNIC charts for prospective evaluation, and inspect ICESat-2 tracks as an independent physical validation reference.
