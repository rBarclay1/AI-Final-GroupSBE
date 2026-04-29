import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "processed"


def find_usgs_csv(site_dir: Path) -> Path:
    matches = sorted(site_dir.glob("*_Strt_*_EndAt_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No USGS export (*_Strt_*_EndAt_*.csv) in {site_dir}")
    if len(matches) > 1:
        raise ValueError(f"Expected one USGS file in {site_dir}, found: {matches}")
    return matches[0]


def load_usgs_hourly(usgs_path: Path, quality_codes: set[str] | None = None) -> pd.DataFrame:
    """
    USGS is 15-minute (or irregular); resample to hourly mean discharge.
    quality_codes: if set, drop rows whose 00060_cd is not in this set (default: approved only 'A').
    """
    df = pd.read_csv(usgs_path)
    df["datetime"] = pd.to_datetime(df["DateTime"], utc=True)
    df = df.sort_values("datetime")
    if quality_codes is not None:
        df = df[df["00060_cd"].isin(quality_codes)]
    df = df.set_index("datetime")
    hourly = (
        df["USGSFlowValue"]
        .resample("h")
        .mean()
        .to_frame(name="usgs_flow_cfs")
    )
    return hourly


def load_nwm_streamflow_files(site_dir: Path, stream_id: str) -> pd.DataFrame:
    pattern = f"streamflow_{stream_id}_*.csv"
    paths = sorted(site_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} under {site_dir}")
    frames = [pd.read_csv(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    out["model_initialization_time"] = pd.to_datetime(
        out["model_initialization_time"].str.replace("_", " "), utc=True
    )
    out["model_output_valid_time"] = pd.to_datetime(
        out["model_output_valid_time"].str.replace("_", " "), utc=True
    )
    out["lead_hours"] = (
        out["model_output_valid_time"] - out["model_initialization_time"]
    ).dt.total_seconds() / 3600.0
    return out


def nwm_lead_one_hour(nwm: pd.DataFrame) -> pd.DataFrame:
    """
    One NWM value per valid time: 1-hour-ahead forecast from the run initialized
    one hour before valid time (operational-style short lead).
    """
    sel = nwm[np.isclose(nwm["lead_hours"], 1.0)].copy()
    sel = sel.drop_duplicates(subset=["model_output_valid_time"], keep="last")
    idx = pd.DatetimeIndex(sel["model_output_valid_time"])
    s = sel.set_index(idx)["streamflow_value"].to_frame(name="nwm_flow_cfs")
    s.index.name = "datetime"
    return s.sort_index()


def merge_site(site_dir: Path, stream_id: str) -> pd.DataFrame:
    usgs_path = find_usgs_csv(site_dir)
    usgs = load_usgs_hourly(usgs_path, quality_codes={"A"})
    nwm_raw = load_nwm_streamflow_files(site_dir, stream_id)
    nwm = nwm_lead_one_hour(nwm_raw)
    merged = usgs.join(nwm, how="outer")
    merged.insert(0, "stream_id", stream_id)
    return merged


def discover_sites() -> list[tuple[Path, str]]:
    """Subfolders named with the NWM stream id (digits only)."""
    sites: list[tuple[Path, str]] = []
    for p in sorted(PROJECT_ROOT.iterdir()):
        if p.is_dir() and p.name.isdigit():
            sites.append((p, p.name))
    return sites


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sites = discover_sites()
    if not sites:
        raise SystemExit(f"No site subfolders (numeric names) found under {PROJECT_ROOT}")

    for site_dir, stream_id in sites:
        print(f"Processing stream_id={stream_id} …")
        merged = merge_site(site_dir, stream_id)
        out_path = PROCESSED_DIR / f"{stream_id}_hourly.csv"
        merged.to_csv(out_path)
        n = len(merged)
        both = merged[["usgs_flow_cfs", "nwm_flow_cfs"]].notna().all(axis=1).sum()
        print(f"  wrote {out_path}  rows={n}  rows_with_usgs_and_nwm={both}")


if __name__ == "__main__":
    main()
