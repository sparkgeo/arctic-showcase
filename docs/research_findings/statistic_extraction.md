# Statistic Extraction

## Purpose

Computes dataset-wide per-variable normalisation constants (mean and std) across all 512 AI4Arctic training scenes. These constants are consumed by both the training encoding path (B3) and the inference encoding path (A8) — see the feature contract in `prescient_ice_model_architecture.md`.

The script also writes a per-scene CSV (`scene_stats.csv`) with mean, std, min, and max for each variable, which is useful for EDA and identifying outlier scenes.

## Outputs

| File | Description |
|---|---|
| `dataset_stats.json` | Dataset-wide mean, std, and pixel count per variable |
| `scene_stats.csv` | Per-scene, per-variable mean, std, min, max, n_pixels |

## Usage

```bash
# Local (uses spk_data AWS profile) | SageMaker (IAM role — remove profile_name from boto3.Session)
uv run python scripts/statistic_extraction/statistic_extraction.py

# Limit to N scenes for testing
# Slice keys list at line 111: keys = list_scene_keys(...)[:10]

# Override defaults
python statistic_extraction.py \
  --bucket prescient-ice-data \
  --prefix training_data/ai4arctic/raw_train/ \
  --workers 8 \
  --output dataset_stats.json \
  --csv scene_stats.csv
```

## Algorithm

Per-scene statistics are computed in parallel using `ThreadPoolExecutor`. Each worker downloads one `.nc` file, strips NaN and sentinel values (255, 2), and returns per-variable `(n, mean, M2)` accumulators.

The accumulators are merged in the main thread using **Chan's parallel algorithm**, which combines two `(n, mean, M2)` accumulators without catastrophic cancellation:

```
delta = mean_b - mean_a
mean  = mean_a + delta * n_b / (n_a + n_b)
M2    = M2_a + M2_b + delta² * n_a * n_b / (n_a + n_b)
```

Final std is derived from the merged M2: `std = sqrt(M2 / n)`.

## Sentinel / nodata exclusions

Three value classes are excluded before computing statistics:

| Value | Meaning |
|---|---|
| `NaN` | No-data fill in the NetCDF |
| `255` | Land / out-of-swath mask |
| `2` | Error / default returns |

## Performance notes

- Default concurrency is 8 workers (`--workers`). Reduce to 4 on SageMaker or when processing many large scenes to avoid S3 throttling.
- Scene processing time varies with file size (~200–520 MB) and spatial extent. Occasional outlier scenes (900s+) are caused by S3 bandwidth contention under concurrent load, not anomalous data.
- The CSV is flushed after each scene so it is recoverable if the run is interrupted.

## Variables processed

| Variable | Description |
|---|---|
| `nersc_sar_primary` | HH-polarisation SAR backscatter (σ⁰, NERSC-corrected) |
| `nersc_sar_secondary` | HV-polarisation SAR backscatter (σ⁰, NERSC-corrected) |
| `btemp_6_9h/v` … `btemp_89_0h/v` | AMSR2 brightness temperatures (8 channels, H+V) |
| `u10m_rotated`, `v10m_rotated` | ERA5 10m wind components (rotated to scene grid) |
| `t2m` | ERA5 2m air temperature |
| `skt` | ERA5 skin temperature |
| `tcwv` | ERA5 total column water vapour |
| `tclw` | ERA5 total column liquid water |

## Findings

- Dataset-wide mean/std per variable

| Variable | Mean | Std | n_pixels |
|---|---:|---:|---:|
| `nersc_sar_primary` | -14.5039 | 5.6629 | 51,081,951,135 |
| `nersc_sar_secondary` | -24.6993 | 4.7507 | 51,081,488,014 |
| `btemp_6_9h` | 148.7839 | 61.6890 | 20,842,647 |
| `btemp_6_9v` | 203.4578 | 38.6797 | 20,839,627 |
| `btemp_18_7h` | 165.6028 | 52.6352 | 20,842,523 |
| `btemp_18_7v` | 218.5310 | 28.0599 | 20,839,756 |
| `btemp_36_5h` | 185.1192 | 38.6376 | 20,842,390 |
| `btemp_36_5v` | 228.1566 | 17.6345 | 20,840,626 |
| `btemp_89_0h` | 212.4604 | 22.5406 | 20,841,971 |
| `btemp_89_0v` | 241.2092 | 16.2968 | 20,839,289 |
| `u10m_rotated` | 0.6823 | 4.7516 | 20,842,669 |
| `v10m_rotated` | 0.6134 | 5.1997 | 20,842,669 |
| `t2m` | 268.6147 | 9.4758 | 20,842,665 |
| `skt` | 268.9006 | 10.3142 | 20,842,665 |
| `tcwv` | 7.7288 | 5.2560 | 20,842,669 |
| `tclw` | 0.0410 | 0.0709 | 20,842,669 |

**SAR vs AMSR2 pixel counts.** SAR variables accumulate ~51 billion valid pixels vs ~20.8 million for AMSR2 — a ~2,450× ratio consistent with the resolution difference between Sentinel-1 EW (~80 m native) and AMSR2 (~12.5 km).

**SAR HH/HV completeness.** `nersc_sar_primary` and `nersc_sar_secondary` pixel counts differ by only ~463K out of 51B (< 0.001%), indicating negligible missing data across the dataset.

**V-pol brightness temperatures are consistently higher than H-pol.** At each frequency, V-pol mean exceeds H-pol by 40–55 K (e.g. 6.9 GHz: H=148.8 K, V=203.5 K). H-pol channels also carry higher std, indicating greater sensitivity to surface type — most pronounced at 6.9 GHz H (std = 61.7 K).

**ERA5 temperatures confirm Hudson Bay winter conditions.** Mean 2m air temperature of 268.6 K (−4.5 °C) and skin temperature of 268.9 K (−4.0 °C) are consistent with freeze-up periods. `tclw` mean of 0.041 kg/m² with a right-skewed std (0.071) reflects predominantly clear-sky Arctic conditions.

**All 16 variables present.** No variables are missing from the dataset-wide accumulation; all scenes contributed to all 16 variable accumulators.

Total wall time is 18815.3s (~5.2 hours) run locally and 10392.8s (~2.9 hours) on Sagemaker.
