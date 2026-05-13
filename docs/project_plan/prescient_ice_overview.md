# Prescient Ice: Automated Arctic Sea Ice Concentration Mapping

## Background and Motivation

The Arctic is changing rapidly. Sea ice is melting earlier and freezing later each year, with significant implications for ecosystems, northern communities, and global shipping as previously impassable routes become seasonally navigable. Accurate, timely sea ice information is critical for safe navigation and environmental monitoring in this evolving landscape.

Today, operational sea ice monitoring is carried out by national ice services — primarily the Canadian Ice Service (CIS) and the U.S. National Ice Center (USNIC). These organizations produce sea ice concentration maps through manual interpretation of synthetic aperture radar (SAR) imagery. Analysts examine SAR scenes, delineate regions of similar ice conditions, and assign concentration values to each polygon. While this approach benefits from expert knowledge, it introduces limitations: analyst subjectivity, coarse spatial resolution, publication latency, and distribution in formats poorly suited to real-time use.

Prescient Ice seeks to automate and improve upon this process. Using SAR imagery as the primary input, a geospatial foundation model (GeoFM) will be used to estimate ice concentration on a 500m × 500m grid — a substantial resolution improvement over existing ice chart polygons. Automation removes analyst subjectivity and enables faster turnaround from satellite acquisition to actionable information.

The project also serves as an internal showcase for Prescient, our cloud-native geospatial data management platform. It demonstrates the full data lifecycle — ingesting diverse sources, managing them as STAC collections, and serving both source data and derived products through a common interface. It showcases our ability to work with geospatial foundation models and multi-sensor data fusion, combining SAR, passive microwave, lidar altimetry, climate reanalysis, and optical remote sensing into a coherent analytical pipeline.

## Objectives

1. **Produce high-resolution SIC estimates from SAR imagery.** Using the Clay foundation model as a feature extractor, generate continuous sea ice concentration estimates (0–100%) from Sentinel-1 SAR on a 500m grid, with ERA5 and AMSR2 as ancillary inputs. The primary approach uses Clay as a frozen extractor with a simpler downstream model; end-to-end fine-tuning is a conditional second phase if the accuracy gap warrants it.

2. **Demonstrate end-to-end data management through Prescient.** Ingest all data sources as STAC-compliant collections (COGs, PMTiles) and demonstrate the full round-trip: source data served out to analytics, derived products ingested back in. Showcase Prescient's format flexibility, multi-source catalog management, and analytics integration across a realistic, complex project.

3. **Build a multi-sensor training and validation framework.** Combine USNIC ice charts as primary training labels, ICESat-2 as supplementary high-confidence labels, AMSR2 as a physics-based ancillary input feature, and HLS optical imagery as independent seasonal validation.

4. **Deliver results through an interactive visualization interface.** Serve SIC predictions alongside source and validation layers through a MapLibre-powered web viewer, demonstrating Prescient's visualization capabilities with a multi-layer, multi-source display.

5. **Establish a cloud-native analytics pipeline on AWS.** Implement ingestion, embedding extraction, training, inference, and delivery using AWS infrastructure, with Prescient as the data layer throughout.

## Project Overview

Prescient Ice is structured as a five-stage pipeline with Prescient at the center.

**Data acquisition** gathers source datasets from their providers. The primary model input is Sentinel-1 SAR, supplemented by ERA5 climate reanalysis and AMSR2 passive microwave SIC as ancillary features. Training labels come from USNIC weekly Arctic ice charts — which incorporate CIS analysis for Canadian waters, providing complete Canadian Arctic coverage from a single source — supplemented by ICESat-2 altimetry for high-confidence lead and freeboard observations. HLS optical imagery is included for seasonal validation and as a visualization context layer.

**Ingestion** converts these heterogeneous sources into Prescient-compatible formats (COGs for rasters, PMTiles for vectors) and registers them as STAC collections with spatial and temporal metadata.

**Data management** through Prescient provides a single source of truth. All source imagery, labels, ancillary data, model embeddings, and derived products are discoverable and accessible via the STAC API.

**Analytics** consists of two pipelines. The embedding pipeline passes Sentinel-1 scenes through Clay's frozen encoder, generating patch embeddings that are ingested back into Prescient as a derived STAC collection. The training pipeline then assembles temporally aligned embedding/label pairs, rasterizes USNIC ice chart polygons to a 500m grid, and trains a downstream regression model on the embedding features. The inference pipeline applies the trained model to new SAR acquisitions and ingests the resulting SIC grids into Prescient as a derived collection.

**Delivery** serves predictions and supporting data through a web viewer, where analysts can overlay SIC output on source SAR, compare against USNIC charts, and inspect validation layers.
