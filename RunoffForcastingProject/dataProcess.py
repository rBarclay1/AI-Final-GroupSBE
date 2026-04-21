import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler

STATION_A_NWM_PATH  = "RunoffForcastingProject/20380357/streamflow_20380357_*.csv"
STATION_A_USGS_PATH = "RunoffForcastingProject/20380357/09520500_Strt_2021-04-20_EndAt_2023-04-21.csv"

STATION_B_NWM_PATH  = "RunoffForcastingProject/21609641/streamflow_21609641_*.csv"
STATION_B_USGS_PATH = "RunoffForcastingProject/21609641/11266500_Strt_2021-04-20_EndAt_2023-04-21.csv"

VAL_START   = "2022-08-01 00:00:00"
TRAIN_END   = "2022-07-31 23:00:00"
TEST_START  = "2022-10-01 00:00:00"


def load_nwm(path_pattern):
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"No NWM files found at: {path_pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"  Loaded {len(files)} NWM files → {len(df):,} rows")
    return df


def parse_nwm_times(df):
    df['init_time'] = pd.to_datetime(
        df['model_initialization_time'], format='%Y-%m-%d_%H:%M:%S'
    )
    df['valid_time'] = pd.to_datetime(
        df['model_output_valid_time'], format='%Y-%m-%d_%H:%M:%S'
    )
    df['lead_time_hours'] = (
        (df['valid_time'] - df['init_time']).dt.total_seconds() / 3600
    ).astype(int)
    return df


def pivot_nwm_wide(df):
    df = df[df['lead_time_hours'].between(1, 18)].copy()

    wide = df.pivot_table(
        index='init_time',
        columns='lead_time_hours',
        values='streamflow_value',
        aggfunc='first'
    )

    wide.columns = [f'NWM_lead_{int(c)}h' for c in wide.columns]
    wide = wide.reset_index()
    print(f"  NWM pivoted → {len(wide):,} initialization times x {wide.shape[1]-1} lead columns")
    return wide


def load_usgs_hourly(path):
    df = pd.read_csv(path)

    df['DateTime'] = (
        pd.to_datetime(df['DateTime'], utc=True)
          .dt.tz_localize(None)
    )

    quality_col = [c for c in df.columns if c not in ('DateTime', 'USGSFlowValue')][0]

    n_before = len(df)
    df = df[df[quality_col] == 'A'].copy()
    print(f"  USGS quality filter: {n_before:,} → {len(df):,} rows (kept 'A' only)")

    df = df.set_index('DateTime')

    usgs_hourly = df['USGSFlowValue'].resample('1h', label='left').mean()

    usgs_hourly = usgs_hourly.dropna()
    print(f"  USGS resampled → {len(usgs_hourly):,} hourly observations")
    return usgs_hourly


def align_and_compute_errors(nwm_wide, usgs_hourly):
    result = nwm_wide.copy()

    for lead_h in range(1, 19):
        valid_times = result['init_time'] + pd.Timedelta(hours=lead_h)

        usgs_at_valid = usgs_hourly.reindex(valid_times.values).values

        nwm_col   = f'NWM_lead_{lead_h}h'
        usgs_col  = f'USGS_at_lead_{lead_h}h'
        error_col = f'error_lead_{lead_h}h'

        result[usgs_col]  = usgs_at_valid
        result[error_col] = result[nwm_col] - result[usgs_col]

    result['usgs_obs_t0'] = usgs_hourly.reindex(result['init_time'].values).values

    all_data_cols = (
        [f'NWM_lead_{h}h'     for h in range(1, 19)] +
        [f'USGS_at_lead_{h}h' for h in range(1, 19)] +
        [f'error_lead_{h}h'   for h in range(1, 19)] +
        ['usgs_obs_t0']
    )
    n_before = len(result)
    result = result.dropna(subset=all_data_cols)
    print(f"  Dropped {n_before - len(result):,} rows with any NaN (NWM gaps or missing USGS)")
    print(f"  Aligned dataset: {len(result):,} usable initialization times")
    return result


def split_train_val_test(df):
    train = df[df['init_time'] <  VAL_START].copy()
    val   = df[(df['init_time'] >= VAL_START) & (df['init_time'] <= TRAIN_END)].copy()
    test  = df[df['init_time'] >= TEST_START].copy()
    print(f"  Train: {len(train):,} rows  |  Val: {len(val):,} rows  |  Test: {len(test):,} rows")
    assert train['init_time'].max() < val['init_time'].min(), \
        "Overlap detected between train and val sets — check split dates!"
    assert val['init_time'].max() < test['init_time'].min(), \
        "Overlap detected between val and test sets — check split dates!"
    return train, val, test


def normalize(train_df, val_df, test_df, feature_cols):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_val   = scaler.transform(val_df[feature_cols].values)
    X_test  = scaler.transform(test_df[feature_cols].values)

    return X_train, X_val, X_test, scaler


def process_station(station_name, nwm_path_pattern, usgs_path):
    print(f"\n{'='*60}")
    print(f"Processing station: {station_name}")
    print('='*60)

    print("\n[1] Loading NWM data...")
    nwm_raw = load_nwm(nwm_path_pattern)

    print("[2] Parsing NWM timestamps...")
    nwm_raw = parse_nwm_times(nwm_raw)

    print("[3] Pivoting NWM to wide format...")
    nwm_wide = pivot_nwm_wide(nwm_raw)

    print("[4] Loading USGS observed data...")
    usgs_hourly = load_usgs_hourly(usgs_path)

    print("[5] Aligning NWM forecasts with USGS observations...")
    aligned = align_and_compute_errors(nwm_wide, usgs_hourly)

    print("[6] Splitting train / val / test...")
    train_df, val_df, test_df = split_train_val_test(aligned)

    print("[7] Normalizing features (fit on train only)...")
    feature_cols = [f'NWM_lead_{h}h' for h in range(1, 19)] + ['usgs_obs_t0']
    X_train, X_val, X_test, scaler = normalize(train_df, val_df, test_df, feature_cols)

    error_cols = [f'error_lead_{h}h' for h in range(1, 19)]
    y_train = train_df[error_cols].values
    y_val   = val_df[error_cols].values
    y_test  = test_df[error_cols].values

    print(f"\n  X_train shape : {X_train.shape}  (samples × 19 features)")
    print(f"  y_train shape : {y_train.shape}  (samples × 18 error targets)")
    print(f"  X_val   shape : {X_val.shape}")
    print(f"  X_test  shape : {X_test.shape}")
    print(f"  y_test  shape : {y_test.shape}")

    return {
        'train_df'     : train_df,
        'val_df'       : val_df,
        'test_df'      : test_df,
        'X_train'      : X_train,
        'X_val'        : X_val,
        'X_test'       : X_test,
        'y_train'      : y_train,
        'y_val'        : y_val,
        'y_test'       : y_test,
        'scaler'       : scaler,
        'feature_cols' : feature_cols,
        'error_cols'   : error_cols,
        'aligned'      : aligned,
    }


if __name__ == "__main__":
    station_a = process_station("20380357 (09520500)", STATION_A_NWM_PATH, STATION_A_USGS_PATH)
    station_b = process_station("21609641 (11266500)", STATION_B_NWM_PATH, STATION_B_USGS_PATH)

    print("\n\nPreprocessing complete. Ready for model training.")
    print("  station_a['X_train'] → input features for Station A (19 cols: 18 NWM leads + usgs_obs_t0)")
    print("  station_a['y_train'] → NWM error targets for Station A")
    print("  station_a['X_val']   → validation features for Station A")
    print("  station_b['X_train'] → input features for Station B (19 cols: 18 NWM leads + usgs_obs_t0)")
    print("  station_b['y_train'] → NWM error targets for Station B")
    print("  station_b['X_val']   → validation features for Station B")
