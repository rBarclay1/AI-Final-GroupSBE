"""
diagnose.py — streamflow val-loss diagnostic
Run: python diagnose.py
"""

import os
import sys
import numpy as np
import pandas as pd
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runoff_data_process import (
    process_station,
    load_usgs_hourly,
    STATION_A_NWM_PATH, STATION_A_USGS_PATH,
    STATION_B_NWM_PATH, STATION_B_USGS_PATH,
    VAL_START, VAL_END, TEST_START,
)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

def _label(status, msg):
    markers = {PASS: "✓", WARN: "!", FAIL: "✗"}
    print(f"    [{status}] {markers[status]} {msg}")

def _section(title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")


# ─── Check 1: Dataset sizes ───────────────────────────────────────────────────

def check_dataset_sizes(name, data):
    _section(f"CHECK 1 · Dataset Sizes  —  {name}")
    train_n = len(data['train_df'])
    val_n   = len(data['val_df'])
    test_n  = len(data['test_df'])
    print(f"    Train : {train_n:,} rows")
    print(f"    Val   : {val_n:,} rows")
    print(f"    Test  : {test_n:,} rows")

    if val_n == 0:
        _label(FAIL, "Val set is EMPTY. Val loss is meaningless — check split dates "
               "or whether quality filtering removed all val rows.")
    elif val_n < 100:
        _label(FAIL, f"Val has only {val_n} rows. A near-zero val loss on this few samples "
               "likely reflects noise or an unrepresentative slice, not real model accuracy.")
    elif val_n < 300:
        _label(WARN, f"Val has {val_n} rows — small. Val loss will have high variance.")
    else:
        _label(PASS, f"Val has {val_n} rows — adequate for validation.")


# ─── Check 2: Flow magnitude comparison ──────────────────────────────────────

def check_flow_magnitude(name, data):
    _section(f"CHECK 2 · Flow Magnitude (usgs_obs_t0)  —  {name}")

    rows = []
    for split_name, df in [
        ("train", data['train_df']),
        ("val",   data['val_df']),
        ("test",  data['test_df']),
    ]:
        s = df['usgs_obs_t0']
        rows.append({
            "split":  split_name.upper(),
            "count":  len(s),
            "mean":   s.mean(),
            "std":    s.std(),
            "min":    s.min(),
            "median": s.median(),
            "max":    s.max(),
        })

    col_w = 8
    hdr = f"{'split':6s}  {'count':>7s}  {'mean':>9s}  {'std':>9s}  {'min':>9s}  {'median':>9s}  {'max':>9s}"
    print(f"    {hdr}")
    for r in rows:
        print(f"    {r['split']:6s}  {r['count']:>7,}  {r['mean']:>9.3f}  {r['std']:>9.3f}  "
              f"{r['min']:>9.3f}  {r['median']:>9.3f}  {r['max']:>9.3f}")

    train_mean = data['train_df']['usgs_obs_t0'].mean()
    val_mean   = data['val_df']['usgs_obs_t0'].mean()

    if train_mean == 0 or len(data['val_df']) == 0:
        _label(FAIL, "Cannot compare means — train mean is 0 or val is empty.")
        return

    pct_diff = abs(val_mean - train_mean) / train_mean * 100
    print()
    if pct_diff > 50:
        _label(FAIL,
               f"Val mean ({val_mean:.3f} m³/s) differs from train mean ({train_mean:.3f} m³/s) "
               f"by {pct_diff:.1f}%. The val window covers a very different flow regime "
               "(likely low-flow summer). If NWM is naturally more accurate at low flows, "
               "val loss will be near zero even for an untrained model.")
    elif pct_diff > 20:
        _label(WARN,
               f"Val mean ({val_mean:.3f}) differs from train mean ({train_mean:.3f}) "
               f"by {pct_diff:.1f}%. Moderate regime shift — could bias val loss downward.")
    else:
        _label(PASS,
               f"Val mean ({val_mean:.3f}) is within {pct_diff:.1f}% of train mean "
               f"({train_mean:.3f}). Flow magnitudes look comparable.")


# ─── Check 3: Quality filter impact ──────────────────────────────────────────

def check_quality_filter(name, usgs_path):
    _section(f"CHECK 3 · Quality Filter Impact  —  {name}")

    raw = pd.read_csv(usgs_path)
    quality_col = [c for c in raw.columns if c not in ('DateTime', 'USGSFlowValue')][0]

    total   = len(raw)
    counts  = raw[quality_col].value_counts()
    a_count = int(counts.get('A', 0))
    dropped = total - a_count
    pct_dropped = 100 * dropped / total if total > 0 else 0

    print(f"    Quality code distribution (column: '{quality_col}'):")
    for code, cnt in counts.items():
        print(f"      '{code}':  {cnt:,}  ({100*cnt/total:.1f}%)")
    print(f"\n    Total: {total:,}  |  Kept 'A': {a_count:,}  |  Dropped: {dropped:,} ({pct_dropped:.1f}%)")

    # Per-period coverage
    raw['DateTime'] = pd.to_datetime(raw['DateTime'], utc=True).dt.tz_localize(None)
    val_s  = pd.Timestamp(VAL_START)
    val_e  = pd.Timestamp(VAL_END)
    test_s = pd.Timestamp(TEST_START)

    splits = {
        "train": raw[raw['DateTime'] < val_s],
        "val":   raw[(raw['DateTime'] >= val_s) & (raw['DateTime'] <= val_e)],
        "test":  raw[raw['DateTime'] >= test_s],
    }

    print(f"\n    Fraction 'A' quality by time window:")
    pcts = {}
    for sname, sub in splits.items():
        p = 100 * (sub[quality_col] == 'A').sum() / len(sub) if len(sub) > 0 else float('nan')
        pcts[sname] = p
        print(f"      {sname.upper():6s}: {p:.1f}%  ({len(sub):,} raw rows)")

    print()
    if pct_dropped > 30:
        _label(FAIL,
               f"{pct_dropped:.1f}% of raw USGS rows removed by the 'A' filter. "
               "Large gaps in the USGS series lead to NaN-dropped rows in aligned, "
               "potentially shrinking the val set dramatically.")
    elif pct_dropped > 10:
        _label(WARN, f"{pct_dropped:.1f}% of USGS rows removed. Check val period coverage.")
    else:
        _label(PASS, f"Only {pct_dropped:.1f}% removed by quality filter — minimal impact.")

    train_pct, val_pct = pcts.get('train', float('nan')), pcts.get('val', float('nan'))
    if not np.isnan(val_pct) and not np.isnan(train_pct) and train_pct > 0:
        gap = train_pct - val_pct
        if gap > 15:
            _label(WARN,
                   f"Val window has {val_pct:.1f}% 'A' coverage vs {train_pct:.1f}% in train "
                   f"(gap = {gap:.1f}pp). Fewer clean observations in val window — "
                   "val set is smaller than the calendar window suggests.")
        else:
            _label(PASS, f"'A' coverage is similar across train ({train_pct:.1f}%) "
                   f"and val ({val_pct:.1f}%).")


# ─── Check 4: Error distribution ─────────────────────────────────────────────

def check_error_distribution(name, data):
    _section(f"CHECK 4 · Error Distribution  —  {name}")

    for col in ['error_lead_1h', 'error_lead_18h']:
        print(f"\n    {col}:")
        print(f"      {'split':6s}  {'count':>6s}  {'mean':>9s}  {'std':>9s}  "
              f"{'min':>10s}  {'max':>10s}  {'|mean|':>9s}")

        stats = {}
        for split_name, df in [("train", data['train_df']), ("val", data['val_df'])]:
            s = df[col]
            stats[split_name] = {
                'mean': s.mean(), 'std': s.std(),
                'min': s.min(), 'max': s.max(),
                'abs_mean': s.abs().mean(), 'count': len(s),
            }
            r = stats[split_name]
            print(f"      {split_name.upper():6s}  {r['count']:>6,}  {r['mean']:>9.4f}  "
                  f"{r['std']:>9.4f}  {r['min']:>10.4f}  {r['max']:>10.4f}  {r['abs_mean']:>9.4f}")

        tr  = stats.get('train', {})
        val = stats.get('val',   {})
        if not tr or not val or tr['abs_mean'] == 0:
            continue

        ratio = val['abs_mean'] / tr['abs_mean']
        print()
        if ratio < 0.2:
            _label(FAIL,
                   f"{col}: Val mean absolute error is only {ratio:.2f}x that of train "
                   f"({val['abs_mean']:.4f} vs {tr['abs_mean']:.4f} m³/s). "
                   "Errors are dramatically smaller in val — NWM is nearly perfect in that "
                   "window, so a model predicting ~0 correction achieves near-zero val loss "
                   "regardless of what it learned on train.")
        elif ratio < 0.5:
            _label(WARN,
                   f"{col}: Val mean abs error is {ratio:.2f}x train's "
                   f"({val['abs_mean']:.4f} vs {tr['abs_mean']:.4f} m³/s). "
                   "Noticeably smaller — val loss is likely easier to minimize than train loss.")
        else:
            _label(PASS,
                   f"{col}: Val mean abs error is {ratio:.2f}x train's — magnitudes are comparable.")

        if tr['std'] > 0 and val['std'] / tr['std'] < 0.3:
            _label(WARN,
                   f"{col}: Val error std ({val['std']:.4f}) is only "
                   f"{val['std']/tr['std']:.2f}x train std ({tr['std']:.4f}). "
                   "Low target variance in val means the model can score well by predicting "
                   "the mean — not a test of generalization.")


# ─── Check 5: Flow time series plot ──────────────────────────────────────────

def plot_flow_timeseries(data_a, data_b,
                         usgs_path_a, usgs_path_b):
    _section("CHECK 5 · Flow Time Series Plot")

    val_start_dt = pd.Timestamp(VAL_START)
    val_end_dt   = pd.Timestamp(VAL_END)
    test_start_dt = pd.Timestamp(TEST_START)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
    fig.suptitle(
        "USGS Observed Flow — Full Date Range\n"
        "Shaded: Train (blue) · Val (orange) · Test (red)",
        fontsize=12, y=0.98
    )

    pairs = [
        (axes[0], "Station A  (NWM 20380357 / USGS 09520500)", data_a, usgs_path_a),
        (axes[1], "Station B  (NWM 21609641 / USGS 11266500)", data_b, usgs_path_b),
    ]

    for ax, title, data, usgs_path in pairs:
        # Load full hourly USGS series for a dense plot
        try:
            usgs_hourly = load_usgs_hourly(usgs_path)
            ax.plot(usgs_hourly.index, usgs_hourly.values,
                    linewidth=0.6, color='steelblue', alpha=0.85, label='USGS hourly obs')
            x_min = usgs_hourly.index.min()
            x_max = usgs_hourly.index.max()
        except Exception:
            # Fallback: use aligned usgs_obs_t0
            aligned = data['aligned']
            ax.plot(aligned['init_time'], aligned['usgs_obs_t0'],
                    linewidth=0.6, color='steelblue', alpha=0.85, label='usgs_obs_t0 (aligned)')
            x_min = aligned['init_time'].min()
            x_max = aligned['init_time'].max()

        ax.axvspan(x_min,         val_start_dt,  alpha=0.07, color='royalblue', zorder=0)
        ax.axvspan(val_start_dt,  val_end_dt,    alpha=0.20, color='orange',    zorder=0)
        ax.axvspan(test_start_dt, x_max,         alpha=0.12, color='red',       zorder=0)

        ax.axvline(val_start_dt,  color='darkorange', linestyle='--', linewidth=1.4,
                   label=f"Val start  {VAL_START[:10]}")
        ax.axvline(val_end_dt,    color='darkorange', linestyle=':',  linewidth=1.2,
                   label=f"Val end    {VAL_END[:10]}")
        ax.axvline(test_start_dt, color='firebrick',  linestyle='--', linewidth=1.4,
                   label=f"Test start {TEST_START[:10]}")

        # Annotate split sizes
        train_n = len(data['train_df'])
        val_n   = len(data['val_df'])
        test_n  = len(data['test_df'])
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 1
        ax.set_title(f"{title}    [train={train_n:,} · val={val_n:,} · test={test_n:,}]",
                     fontsize=10)
        ax.set_ylabel("Flow (m³/s)", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagnostic_plot.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved → {out_path}")
    _label(PASS, "Time series plot generated successfully.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*62)
    print("  STREAMFLOW DIAGNOSTIC")
    print("  Question: why is station B val loss ≈ 0 while train loss stays high?")
    print("="*62)

    print("\n[Running preprocessing pipeline — output from runoff_data_process.py]\n")
    data_a = process_station("20380357 (09520500)", STATION_A_NWM_PATH, STATION_A_USGS_PATH)
    data_b = process_station("21609641 (11266500)", STATION_B_NWM_PATH, STATION_B_USGS_PATH)

    print("\n\n" + "="*62)
    print("  DIAGNOSTIC RESULTS")
    print("="*62)

    for label_name, data, usgs_path in [
        ("Station A  (20380357 / 09520500)", data_a, STATION_A_USGS_PATH),
        ("Station B  (21609641 / 11266500)", data_b, STATION_B_USGS_PATH),
    ]:
        check_dataset_sizes(label_name, data)
        check_flow_magnitude(label_name, data)
        check_quality_filter(label_name, usgs_path)
        check_error_distribution(label_name, data)

    plot_flow_timeseries(data_a, data_b, STATION_A_USGS_PATH, STATION_B_USGS_PATH)

    print("\n" + "="*62)
    print("  DIAGNOSTIC COMPLETE  —  see diagnostic_plot.png")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
