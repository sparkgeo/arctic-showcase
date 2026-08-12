# Training Pipeline (Track B)

This is the code that turns raw AI4Arctic satellite scenes into the tables our sea-ice
classifier actually trains on. In short: it reads a satellite image, cuts it into small
tiles, runs each tile through a pretrained vision model (Clay) to get a feature
representation, computes some extra statistics per tile, attaches the correct
sea-ice-concentration answer for each tile, and writes everything out to two tables in
S3, ready for a model to consume.

The two output tables are:

- **Patch table** (`training_data/ai4arctic/features/patch_table/`) - one row per small
  tile ("patch"). This is the main training table.
- **Chip table** (`training_data/ai4arctic/features/chip_table/`) - one row per larger
  tile ("chip"), holding a single whole-chip embedding.

The rest of this doc walks through the pipeline in the order data actually flows through
it, then explains how to run it yourself.

## Some vocabulary used throughout

- **Scene**: one full Sentinel-1 satellite image (a NetCDF file), covering a large area.
- **Chip**: a 256x256-pixel square cut out of a scene. Chips are the unit Clay's model
  operates on.
- **Patch**: an 8x8-pixel square within a chip. Each chip is a 32x32 grid of patches
  (1024 patches per chip). This is the unit our final training rows are at.
- **SIC (sea ice concentration)**: how much of a given area is covered by ice, on a
  0-10 scale (tenths). This is the thing the model is ultimately trying to predict.
- **Clay**: a pretrained "foundation model" for satellite imagery that we use as a
  frozen feature extractor - we don't train it, we just run our chips through it and
  keep its output as input features for our own, much smaller classifier.
- **GCP (ground control point)**: a known lat/lon reference point tied to a specific
  pixel in the scene, used to work out the real-world location of every other pixel.

## Step 1: Reading scenes and cutting them into chips (`data_loader/`)

| What it does | Where |
|--------|--------|
| Opens the satellite NetCDF file once and pulls everything else from the in-memory arrays it returns, so we don't keep re-reading the file from disk | `scene_reader.py` |
| Works out where each 256x256 chip should be placed across the scene (a regular grid, with the last row/column shifted inward so it still fits) | `tiling.py` |
| Resizes the lower-resolution supporting data (AMSR2 brightness temperatures, ERA5 weather variables, distance-to-land, incidence angle) up to match the SAR image's resolution | `ancillary.py` |
| Works out, pixel by pixel, which pixels are real data we can trust versus land or missing/no-data pixels - done *before* any fill-in values are substituted, so the fill-ins never get miscounted as real data | `valid_mask.py` |
| Converts the scene's sparse ground control points into a smooth function so we can look up the real-world lat/lon and acquisition time for any pixel in the chip | `geolocation.py` |
| Walks through a loaded scene and hands back one chip at a time, skipping any chip that turns out to be entirely invalid (logging how many got skipped) | `loader.py` (`yield_chips`) |
| The `Chip` data object that everything downstream reads from - the resampled ancillary data, valid mask, and ice-chart values for one chip, all bundled together | `chip.py` |

## Step 2: Running each chip through Clay (`encoding/`)

Clay expects the SAR image, a bit of metadata about the sensor, and location/time
information, and gives back an embedding - a big vector of numbers that summarizes the
visual content of the chip in a way a downstream classifier can learn from.

| File | What's in it |
|------|------|
| `metadata.py` | Builds and stores the "SarMetadata" entry Clay needs to know how to interpret our Sentinel-1 input (band names, wavelengths, normalisation stats, etc.) |
| `model.py` | Loads the actual Clay checkpoint. Deliberately pins it to the `"large"` model size, no masking, and no shuffling - the smaller default model produces a differently-sized output that would silently break everything downstream expecting 1024 numbers per chip |
| `encoder.py` | `encode_chip(...)` - hands one chip to Clay and gets back two things: a grid of 1024 small "patch" embeddings (one per 8x8 patch, arranged 32x32) and one "class token" embedding summarizing the whole chip |

## Step 3: Per-patch SAR statistics (`patch_features.py`)

For every one of the 1024 patches in a chip, this computes a handful of plain
statistics directly from the radar data - the average and spread of the
HH and HV radar backscatter values, how much of the patch is trustworthy real data
versus filled-in (`valid_fraction`), the average incidence angle, distance to land, and
the averaged AMSR2/ERA5 values over that patch's footprint. These sit alongside the
Clay embedding as extra, simpler features for the classifier to use.

## Step 4: Per-patch labels (`label_prep.py`)

This does the same per-patch split, but for the labels: for each
patch, it looks at the ice-chart data and works out what fraction of the patch has a
known ice-concentration reading (`valid_class_fraction`), what the overall label for
that patch should be (an area-weighted average of the SIC classes present, rounded to
the nearest whole class), whether the patch is "pure" (entirely one class), and the full
breakdown of how much of the patch falls into each of the 11 SIC classes
(`frac_sic`).

## Step 5: Working out where things actually are on the map (`geometry.py`)

Converts each chip's and each patch's pixel location into an actual geographic footprint
(a polygon) in our project's standard map projection (EPSG:3978 / Canada Atlas
Lambert), using the ground control points from Step 1. This is what lets the output
tables be treated as proper geospatial data (e.g. viewed on a map, or queried by
location) rather than just a flat table of numbers.

## Step 6: Putting one chip's row(s) together (`feature_assembly.py`)

Takes everything computed above for a single chip - the Clay embeddings, the SAR
statistics, the labels, and the geometry - and combines them into the final table rows,
in the exact column order our project's Feature Contract requires (see
`prescient_ice_model_architecture.md` § Feature Contract). Getting this column order
wrong wouldn't throw an error, but it would silently make the model see a different
feature at inference time than it saw during training - so this is the one place all
those column names and their order are pinned down.

## Step 7: Writing the results to S3 (`parquet_writer.py`)

Saves a batch of rows for a scene out to S3 as a GeoParquet file. Each scene gets its
own folder (named by `scene_id`); a scene's rows can be written out gradually in
several batches rather than all at once, which keeps memory use bounded no matter how
big the scene is.

## Step 8: The script that runs all of the above (`main.py`)

This is the actual entry point - it ties every step above into one loop. For each
scene, it downloads the file from S3, walks through its chips, runs each chip through
Clay + the statistics/label/geometry steps, assembles the rows, and periodically writes
them out to S3. If a scene has already been fully processed in a previous run
(detected by its presence in the chip table), it's skipped - so if a run gets
interrupted partway through, you can just run it again and it'll pick up where it left
off instead of redoing finished scenes. Scenes that fail for any reason are logged and
skipped rather than stopping the whole run.

One thing worth calling out: the project plan implementation doc describes needing a
separate step to join labels onto the test scenes. In practice, this isn't necessary -
the test data already comes with labels included.

## How to run it

```bash
# One-time setup: installs all workspace packages plus dev tools (pytest, ruff, mypy, etc.)
uv sync --all-packages --group dev

# Runs the full pipeline end to end.
uv run python -m training.main --profile <your-aws-profile>
```

You'll need AWS credentials with access to the `prescient-ice-data` S3 bucket. The Clay
model checkpoint and its metadata config are expected at the repo root (`clay-v1.5.ckpt`)
and in `configs/metadata.yaml`; if the checkpoint isn't there already, `main.py` will
download it from S3 automatically on first run.

Command-line options (see `parse_args()` in `main.py`):

- `--bucket` - which S3 bucket to read/write (defaults to `prescient-ice-data`)
- `--profile` - which AWS CLI profile to use for credentials; leave it out to fall back
  to whatever the default credential chain finds (e.g. an IAM role, if running in the
  cloud rather than locally)
- `--max-scenes` - how many scenes to process before stopping. Defaults to just `2`,
  which is meant for a quick local smoke test rather than a real run. Pass `0` or a
  negative number to process every scene in the corpus

While it runs, progress is logged scene by scene and every 50 chips within a scene, so
you can follow along and see roughly how far through it is.

### Checking the output

After a run (even a small `--max-scenes 2` test run), open
`notebooks/training_data_exploration/parquet_output_review.ipynb` to sanity-check what
got written: it checks the two output tables against the Feature Contract, i.e. that
all the expected columns are present with the right types, that value ranges look
sane (e.g. labels are 0-10, fractions are between 0 and 1), and that null/invalid rates
look reasonable rather than suspiciously high.
