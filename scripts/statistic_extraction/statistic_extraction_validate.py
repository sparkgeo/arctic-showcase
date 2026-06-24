import xarray as xr
import numpy as np
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

VARIABLES = [
    'nersc_sar_primary', 'nersc_sar_secondary',
    'btemp_6_9h', 'btemp_6_9v', 'btemp_18_7h', 'btemp_18_7v',
    'btemp_36_5h', 'btemp_36_5v', 'btemp_89_0h', 'btemp_89_0v',
    'u10m_rotated', 'v10m_rotated', 't2m', 'skt', 'tcwv', 'tclw',
]

LOCAL_FILES = [
    "S1A_EW_GRDM_1SDH_20180124T194759_20180124T194859_020301_022AA4_1F75_icechart_dmi_201801241950_SouthEast_RIC.nc",
    "S1A_EW_GRDM_1SDH_20210430T205436_20210430T205537_037685_047252_CBB0_icechart_dmi_202104302055_SouthWest_RIC.nc",
]


def compute_scene_stats(filepath):
    t0 = time.time()
    try:
        stats = {}
        with xr.open_dataset(filepath) as ds:
            for var in VARIABLES:
                if var not in ds:
                    continue
                arr   = ds[var].values.astype(np.float32).flatten()
                valid = arr[~np.isnan(arr)]
                valid = valid[valid != 255]
                valid = valid[valid != 2]
                if len(valid) == 0:
                    continue
                stats[var] = {
                    'n':      len(valid),
                    'sum':    float(np.sum(valid)),
                    'sum_sq': float(np.sum(valid ** 2)),
                    'min':    float(np.min(valid)),
                    'max':    float(np.max(valid)),
                }
        elapsed = time.time() - t0
        print(f"  OK: {filepath.split('/')[-1][:50]}  ({elapsed:.1f}s)")
        return stats, elapsed
    except Exception as e:
        print(f"  ERROR {filepath}: {e}")
        return None, 0.0


def merge_stats(accumulated, scene_stats):
    for var, s in scene_stats.items():
        if var not in accumulated:
            accumulated[var] = {'n': 0, 'sum': 0.0, 'sum_sq': 0.0,
                                'min': np.inf, 'max': -np.inf}
        accumulated[var]['n']      += s['n']
        accumulated[var]['sum']    += s['sum']
        accumulated[var]['sum_sq'] += s['sum_sq']
        accumulated[var]['min']     = min(accumulated[var]['min'], s['min'])
        accumulated[var]['max']     = max(accumulated[var]['max'], s['max'])
    return accumulated


def finalise_stats(accumulated):
    results = {}
    for var, s in accumulated.items():
        mean = s['sum'] / s['n']
        std  = np.sqrt(s['sum_sq'] / s['n'] - mean ** 2)
        results[var] = {
            'mean':     round(float(mean), 6),
            'std':      round(float(std),  6),
            'min':      round(float(s['min']), 6),
            'max':      round(float(s['max']), 6),
            'n_pixels': s['n'],
        }
    return results


if __name__ == '__main__':
    total_start = time.time()
    accumulated = {}
    scene_times = {}

    print(f"Processing {len(LOCAL_FILES)} scenes...\n")

    scene_stats_by_file = {}

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(compute_scene_stats, fp): fp
                   for fp in LOCAL_FILES}
        for future in as_completed(futures):
            result, elapsed = future.result()
            fp = futures[future]
            if result is not None:
                accumulated   = merge_stats(accumulated, result)
                scene_times[fp.split('/')[-1]] = elapsed
                scene_stats_by_file[fp.split('/')[-1]] = result

    finalise_start = time.time()
    final = finalise_stats(accumulated)
    finalise_time = time.time() - finalise_start

    with open('dataset_stats.json', 'w') as f:
        json.dump(final, f, indent=2)

    total_elapsed = time.time() - total_start

    # ── Timing summary ────────────────────────────────────────────────────────
    print("\n── Timing ───────────────────────────────────────")
    for scene, t in scene_times.items():
        print(f"  {scene[:50]:50s}  {t:.1f}s")
    print(f"  Finalise stats:                                    {finalise_time:.2f}s")
    print(f"  Total wall time:                                   {total_elapsed:.1f}s")

    # ── Per-scene means ───────────────────────────────────────────────────────
    print("\n── Per-scene means ──────────────────────────────")
    for fname, scene_s in scene_stats_by_file.items():
        print(f"\n  {fname[:60]}")
        for var in VARIABLES:
            if var not in scene_s:
                continue
            s = scene_s[var]
            mean = s['sum'] / s['n']
            std  = np.sqrt(s['sum_sq'] / s['n'] - mean ** 2)
            print(f"    {var:25s}  mean={mean:8.3f}  std={std:7.3f}")

    # ── Overall means ─────────────────────────────────────────────────────────
    print("\n── Overall means (all scenes) ───────────────────")
    for var, s in final.items():
        print(f"  {var:25s}  mean={s['mean']:8.3f}  std={s['std']:7.3f}")

    # ── Equivalence validation ────────────────────────────────────────────────
    # Path A: per-scene (n, sum, sum_sq) accumulators → merged combined stat.
    #         This is `final`, already computed above.
    # Path B: all raw valid pixels pooled into one array → np.mean / np.std.
    # Both paths apply identical masking; any mismatch is a bug in the accumulator.
    print("\n── Equivalence validation ───────────────────────")
    print("  path A = per-scene accumulators → merged")
    print("  path B = all pixels pooled → numpy")
    print()

    all_pixels: dict[str, list[np.ndarray]] = {var: [] for var in VARIABLES}
    for fp in LOCAL_FILES:
        with xr.open_dataset(fp) as ds:
            for var in VARIABLES:
                if var not in ds:
                    continue
                arr   = ds[var].values.astype(np.float32).flatten()
                valid = arr[~np.isnan(arr)]
                valid = valid[valid != 255]
                valid = valid[valid != 2]
                if len(valid):
                    all_pixels[var].append(valid)

    header = f"  {'variable':25s}  {'A mean':>10}  {'B mean':>10}  {'Δmean':>10}  {'A std':>10}  {'B std':>10}  {'Δstd':>10}  result"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_pass = True
    for var in VARIABLES:
        if var not in final or not all_pixels[var]:
            continue
        pooled = np.concatenate(all_pixels[var])
        b_mean = float(np.mean(pooled))
        b_std  = float(np.std(pooled))
        a_mean = final[var]['mean']
        a_std  = final[var]['std']
        mean_ok = np.isclose(a_mean, b_mean, rtol=1e-5)
        std_ok  = np.isclose(a_std,  b_std,  rtol=1e-5)
        ok      = mean_ok and std_ok
        all_pass = all_pass and ok
        print(
            f"  {var:25s}  {a_mean:10.4f}  {b_mean:10.4f}  {a_mean-b_mean:10.2e}"
            f"  {a_std:10.4f}  {b_std:10.4f}  {a_std-b_std:10.2e}  {'PASS' if ok else 'FAIL'}"
        )

    print()
    print(f"  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")