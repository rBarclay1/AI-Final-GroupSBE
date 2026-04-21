"""
Preprocessing Step 1 — Load, Parse, Resample, Align
=====================================================
Loads NWM forecast CSVs and USGS observation CSVs for both stations,
parses timestamps, resamples USGS from 15-min to hourly, and saves
intermediate files ready for Step 2 (feature engineering + error computation).

Run from the RunoffForcastingProject/ directory:
    python load_align.py

Outputs (saved to intermediate/):
    nwm_s1.csv          NWM forecasts for Station 1 (long format, parsed)
    nwm_s2.csv          NWM forecasts for Station 2 (long format, parsed)
    usgs_s1_hourly.csv  USGS hourly flow for Station 1
    usgs_s2_hourly.csv  USGS hourly flow for Station 2
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Station metadata
# ---------------------------------------------------------------------------
STATIONS = {
    "S1": {"nwm_id": "20380357", "usgs_id": "09520500", "label": "Rillito Creek, AZ"},
    "S2": {"nwm_id": "21609641", "usgs_id": "11266500", "label": "Kings River, CA"},
}

DATA_DIR = Path(".")
OUT_DIR  = DATA_DIR / "intermediate"

LEAD_MIN, LEAD_MAX = 1, 18   # valid lead-time window (hours)


# ---------------------------------------------------------------------------
# NWM loading & parsing
# ---------------------------------------------------------------------------

def load_nwm(nwm_id: str) -> pd.DataFrame:
    """
    Concatenate all monthly NWM CSVs for one station into a single DataFrame.
    Raw columns: NWM_version_number, model_initialization_time,
                 model_output_valid_time, streamflow_value, streamID
    """
    pattern = str(DATA_DIR / nwm_id / f"streamflow_{nwm_id}_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        sys.exit(f"ERROR: No NWM files found matching {pattern}")

    print(f"  Loading {len(files)} monthly files for NWM stream {nwm_id} ...")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def parse_nwm(raw: pd.DataFrame) -> pd.DataFrame:
    """
    - Parse init/valid timestamps (format '2021-04-21_00:00:00') as UTC
    - Compute lead_hours = valid_time - init_time (integer hours)
    - Keep only rows where lead_hours is in [LEAD_MIN, LEAD_MAX]
    - Drop columns not needed downstream
    """
    df = raw.copy()

    # Timestamps use underscore between date and time — replace before parsing
    df["init_time"]  = pd.to_datetime(
        df["model_initialization_time"].str.replace("_", " ", n=1),
        utc=True,
    )
    df["valid_time"] = pd.to_datetime(
        df["model_output_valid_time"].str.replace("_", " ", n=1),
        utc=True,
    )

    df["lead_hours"] = (
        (df["valid_time"] - df["init_time"]).dt.total_seconds() / 3600
    ).round().astype(int)

    before = len(df)
    df = df[(df["lead_hours"] >= LEAD_MIN) & (df["lead_hours"] <= LEAD_MAX)].copy()
    print(f"    Rows kept after lead-time filter ({LEAD_MIN}–{LEAD_MAX} h): "
          f"{len(df):,} / {before:,}")

    df = df.rename(columns={"streamflow_value": "nwm_flow"})
    return df[["init_time", "valid_time", "lead_hours", "nwm_flow", "NWM_version_number"]]


# ---------------------------------------------------------------------------
# USGS loading & resampling
# ---------------------------------------------------------------------------

def load_usgs(nwm_id: str, usgs_id: str) -> pd.Series:
    """
    Load the USGS aggregate CSV, parse UTC timestamps, return a 15-min
    Series of flow values.
    The third column (quality code) name differs between stations — we
    standardise and drop it.
    """
    path = DATA_DIR / nwm_id / f"{usgs_id}_Strt_2021-04-20_EndAt_2023-04-21.csv"

    if not path.exists():
        sys.exit(f"ERROR: USGS file not found: {path}")

    df = pd.read_csv(path)
    df.columns = ["DateTime", "USGSFlowValue", "quality_code"]

    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    df = df.set_index("DateTime").sort_index()

    flow = df["USGSFlowValue"].copy()

    # Flag negatives as missing (instrument error)
    n_neg = (flow < 0).sum()
    if n_neg:
        print(f"    WARNING: {n_neg} negative flow values found — set to NaN")
        flow[flow < 0] = np.nan

    print(f"    {len(flow):,} rows at 15-min resolution  |  "
          f"range: [{flow.min():.3f}, {flow.max():.3f}] m³/s  |  "
          f"NaN: {flow.isna().sum()}")

    return flow


def resample_hourly(flow_15min: pd.Series) -> pd.Series:
    """
    Resample 15-min USGS flow to 1-hour means.
    Hours where all four 15-min slots are NaN remain NaN.
    """
    hourly = flow_15min.resample("1h").mean()

    n_nan = hourly.isna().sum()
    print(f"    Hourly series: {len(hourly):,} rows  |  NaN hours: {n_nan}")

    return hourly


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------

def check_alignment(nwm_df: pd.DataFrame, usgs_h: pd.Series, label: str):
    """
    Print a brief overlap report so it's easy to spot if the two datasets
    are on different date ranges before moving to Step 2.
    """
    nwm_start, nwm_end   = nwm_df["init_time"].min(), nwm_df["init_time"].max()
    usgs_start, usgs_end = usgs_h.index.min(),         usgs_h.index.max()

    overlap_start = max(nwm_start, usgs_start)
    overlap_end   = min(nwm_end,   usgs_end)
    overlap_hours = max(0, int((overlap_end - overlap_start).total_seconds() / 3600))

    print(f"    NWM  : {nwm_start}  →  {nwm_end}")
    print(f"    USGS : {usgs_start}  →  {usgs_end}")

    if overlap_hours == 0:
        print(f"    WARNING: No temporal overlap for {label}!")
    else:
        print(f"    Overlap: {overlap_hours:,} hours  ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(exist_ok=True)

    for key, meta in STATIONS.items():
        print(f"\n{'='*55}")
        print(f"  {key}  —  {meta['label']}")
        print(f"{'='*55}")

        # ── NWM ──────────────────────────────────────────────────────────
        print("\n[NWM]")
        raw_nwm = load_nwm(meta["nwm_id"])
        nwm_df  = parse_nwm(raw_nwm)

        # ── USGS ─────────────────────────────────────────────────────────
        print("\n[USGS]")
        flow_15min  = load_usgs(meta["nwm_id"], meta["usgs_id"])
        usgs_hourly = resample_hourly(flow_15min)

        # ── Alignment check ───────────────────────────────────────────────
        print("\n[Alignment]")
        check_alignment(nwm_df, usgs_hourly, key)

        # ── Save ──────────────────────────────────────────────────────────
        nwm_out  = OUT_DIR / f"nwm_{key.lower()}.csv"
        usgs_out = OUT_DIR / f"usgs_{key.lower()}_hourly.csv"

        nwm_df.to_csv(nwm_out, index=False)
        usgs_hourly.to_frame("usgs_flow").to_csv(usgs_out)

        print(f"\n  Saved: {nwm_out.name}  ({len(nwm_df):,} rows)")
        print(f"  Saved: {usgs_out.name}  ({len(usgs_hourly):,} rows)")

    print(f"\nDone. Files written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
