"""
feature_engineering.py

Ports Parts 2-4 of LLM_EKF_Based_Predictor1.m to Python/numpy, building the
exact (X, Y) training tensors the BiLSTM needs:
  - X: (num_samples, lookback=10, num_features=10)  -- causally normalized
  - Y: (num_samples, horizonAhead=45)               -- multi-step forward returns

This is a faithful, from-scratch Python re-derivation of the SAME formulas
already verified in gnc_forecast_core.cpp/.hpp (which is the production
C++ inference reference) -- kept separate and self-contained here so the
training pipeline has no build/runtime dependency on the C++ side. See the
chat notes on cross-checking the two against each other.

Inputs are LOCAL Tiingo cache CSVs (date,open,high,low,close,volume --
same schema your MATLAB script and demo_main.cpp both already read/write),
NOT a live fetch -- so this script needs no network access at all, only
files you already have on disk from prior MATLAB/C++ runs.

Feature order (must match featureList in the .m file exactly):
  ['Returns', 'RSI', 'LLM_Sentiment', 'ParkinsonVol', 'SectorRS',
   'VPMRatio', 'RangeExp', 'EKF_TruePrice', 'EKF_Velocity', 'EKF_Residual']

NOT YET WIRED UP: LLM_Sentiment is filled with zeros (neutral) unless you
pass --sentiment-csv pointing at a (date, llm_sentiment_score) file, e.g.
one of your <TICKER>_LLM_Sentiment_cache.csv files. This mirrors the .m
file's own synthetic/neutral fallback path -- it's not silently dropped,
just flagged the same way the .m file itself flags it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# CSV loading (main ticker + benchmark tickers)
# =============================================================================
@dataclass
class OhlcvSeries:
    dates: list  # datetime.date, ascending
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


def load_tiingo_cache_csv(path: str) -> OhlcvSeries:
    dates, o, h, l, c, v = [], [], [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        headers_lower = {k.lower(): k for k in (reader.fieldnames or [])}
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [r for r in required if r not in headers_lower]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}")
        for row in reader:
            try:
                date_str = row[headers_lower["date"]][:10]  # keep YYYY-MM-DD, drop time/zone
                d = dt.date.fromisoformat(date_str)
                o.append(float(row[headers_lower["open"]]))
                h.append(float(row[headers_lower["high"]]))
                l.append(float(row[headers_lower["low"]]))
                c.append(float(row[headers_lower["close"]]))
                v.append(float(row[headers_lower["volume"]]))
                dates.append(d)
            except (ValueError, KeyError):
                continue  # skip malformed row, mirrors the .m file's NaN-mask filtering

    order = np.argsort(dates)
    dates = [dates[i] for i in order]
    return OhlcvSeries(
        dates=dates,
        open=np.array(o)[order],
        high=np.array(h)[order],
        low=np.array(l)[order],
        close=np.array(c)[order],
        volume=np.array(v)[order],
    )


def align_prior_close(benchmark: OhlcvSeries, target_dates: list) -> np.ndarray:
    """Reindexes benchmark.close onto target_dates using MATLAB's 'previous'
    retime semantics (last known value on/before each target date). Mirrors
    `retime(..., marketTT.Time, 'previous')` in the .m file's SPY/XLV blocks.
    """
    bench_map = dict(zip(benchmark.dates, benchmark.close))
    bench_sorted_dates = benchmark.dates
    out = np.zeros(len(target_dates))
    last_val = benchmark.close[0] if len(benchmark.close) else 0.0
    bi = 0
    for i, d in enumerate(target_dates):
        while bi < len(bench_sorted_dates) and bench_sorted_dates[bi] <= d:
            last_val = bench_map[bench_sorted_dates[bi]]
            bi += 1
        out[i] = last_val
    return out


# =============================================================================
# Indicators / EKF (faithful port of gnc_forecast_core.cpp + the .m file's
# Part 2 formulas not previously ported: Returns, SectorRS, VPMRatio,
# RangeExp)
# =============================================================================
def trailing_mov_mean(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.zeros(n)
    running = 0.0
    for i in range(n):
        running += x[i]
        start = max(0, i - window + 1)
        if i >= window:
            running -= x[i - window]
        out[i] = running / (i - start + 1)
    return out


def daily_true_range(high, low, close) -> np.ndarray:
    prev_close = np.concatenate(([close[0]], close[:-1]))
    a = high - low
    b = np.abs(high - prev_close)
    c = np.abs(low - prev_close)
    return np.maximum.reduce([a, b, c])


def rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    gains = np.maximum(d, 0.0)
    losses = np.maximum(-d, 0.0)
    avg_gain = trailing_mov_mean(gains, window)
    avg_loss = trailing_mov_mean(losses, window)
    return 100.0 - 100.0 / (1.0 + avg_gain / np.maximum(avg_loss, 1e-6))


def parkinson_vol(high, low, window: int = 20) -> np.ndarray:
    raw = (np.log(np.maximum(high, 1e-6) / np.maximum(low, 1e-6))) ** 2 / (4.0 * np.log(2.0))
    smoothed = trailing_mov_mean(raw, window)
    return np.sqrt(np.maximum(smoothed, 0.0))


def adaptive_ekf_smoother(close, volume, daily_tr, lookback_range_days=252):
    """Faithful re-derivation of runAdaptiveEkfSmoother() from
    gnc_forecast_core.cpp -- same forward EKF pass + backward RTS sweep,
    same value-zone velocity damping. See that file for the line-by-line
    equivalent this was translated from.
    """
    n = len(close)
    atr_lookback = min(14, n)
    recent_atr = np.mean(daily_tr[n - atr_lookback:n])
    recent_atr = max(recent_atr, 1e-3 * max(close[-1], 1.0))

    lookback = min(lookback_range_days, n)
    rolling_low = np.array([np.min(close[max(0, i - lookback + 1):i + 1]) for i in range(n)])
    rolling_high = np.array([np.max(close[max(0, i - lookback + 1):i + 1]) for i in range(n)])
    rolling_span = np.maximum(rolling_high - rolling_low, 1e-6)

    x_est = np.zeros((n, 2))
    p_est = np.zeros((n, 2, 2))
    x_est[0] = [close[0], 0.0]
    P = np.array([[recent_atr ** 2, 0.0], [0.0, 0.1 * recent_atr ** 2]])
    p_est[0] = P

    dt_ = 1.0
    Q = np.array([[0.02 * recent_atr ** 2, 0.01 * recent_atr ** 2],
                  [0.01 * recent_atr ** 2, 0.05 * recent_atr ** 2]])
    R_meas = 0.25 * recent_atr ** 2

    for t in range(1, n):
        value_zone_ceiling = rolling_low[t - 1] + 0.25 * rolling_span[t - 1]
        damping_span = max(0.10 * rolling_span[t - 1], 1e-6)
        vel_damping = 1.0
        if x_est[t - 1, 1] < 0.0 and close[t - 1] < value_zone_ceiling:
            vel_damping = max(0.40, 1.0 - (value_zone_ceiling - close[t - 1]) / damping_span)
        F = np.array([[1.0, dt_], [0.0, vel_damping]])

        x_pred = F @ x_est[t - 1]
        P_pred = F @ P @ F.T + Q

        vol_start = max(0, t - 20)
        vol_mean = np.mean(volume[vol_start:t + 1])
        vol_factor = max(volume[t] / max(vol_mean, 1e-6), 0.1)
        R_adaptive = R_meas * (1.0 / vol_factor)

        y_innov = close[t] - x_pred[0]
        S = P_pred[0, 0] + R_adaptive
        K = P_pred[:, 0] / S

        x_est[t] = x_pred + K * y_innov
        KH = np.zeros((2, 2))
        KH[:, 0] = K
        P = (np.eye(2) - KH) @ P_pred
        p_est[t] = P

    x_smooth = np.zeros((n, 2))
    x_smooth[-1] = x_est[-1]
    p_smooth_last = p_est[-1]
    p_smooth = [None] * n
    p_smooth[-1] = p_smooth_last

    for t in range(n - 2, -1, -1):
        value_zone_ceiling_b = rolling_low[t] + 0.25 * rolling_span[t]
        damping_span_b = max(0.10 * rolling_span[t], 1e-6)
        vel_damping_b = 1.0
        if x_est[t, 1] < 0.0 and close[t] < value_zone_ceiling_b:
            vel_damping_b = max(0.40, 1.0 - (value_zone_ceiling_b - close[t]) / damping_span_b)
        Fb = np.array([[1.0, dt_], [0.0, vel_damping_b]])

        x_pred_next = Fb @ x_est[t]
        P_pred_next = Fb @ p_est[t] @ Fb.T + Q

        PFbT = p_est[t] @ Fb.T
        C = PFbT @ np.linalg.inv(P_pred_next)

        innov = x_smooth[t + 1] - x_pred_next
        x_smooth[t] = x_est[t] + C @ innov
        diffP = p_smooth[t + 1] - P_pred_next
        p_smooth[t] = p_est[t] + C @ diffP @ C.T

    residual = close - x_smooth[:, 0]
    return x_smooth[:, 0], x_smooth[:, 1], residual


# =============================================================================
# Feature matrix construction (Part 2-3 equivalent)
# =============================================================================
FEATURE_LIST = ["Returns", "RSI", "LLM_Sentiment", "ParkinsonVol", "SectorRS",
                "VPMRatio", "RangeExp", "EKF_TruePrice", "EKF_Velocity", "EKF_Residual"]


def build_feature_matrix(main: OhlcvSeries, spy: OhlcvSeries | None, xlv: OhlcvSeries | None,
                          sentiment_by_date: dict | None) -> np.ndarray:
    close, high, low, volume = main.close, main.high, main.low, main.volume
    n = len(close)

    returns = np.concatenate(([0.0], np.diff(close) / np.maximum(close[:-1], 1e-6)))

    if xlv is not None:
        xlv_close_aligned = align_prior_close(xlv, main.dates)
        sector_return = np.concatenate(([0.0], np.diff(xlv_close_aligned) /
                                         np.maximum(xlv_close_aligned[:-1], 1e-6)))
        sector_rs = returns - sector_return
    elif spy is not None:
        spy_close_aligned = align_prior_close(spy, main.dates)
        market_return = np.concatenate(([0.0], np.diff(spy_close_aligned) /
                                         np.maximum(spy_close_aligned[:-1], 1e-6)))
        sector_rs = returns - market_return
    else:
        sector_rs = np.zeros(n)  # no benchmark available -- neutral, flagged not silently averaged in

    down_day_volume = np.where(returns < 0, volume, 0.0)
    vpm_ratio = trailing_mov_mean(down_day_volume, 5) / np.maximum(trailing_mov_mean(volume, 20), 1e-6)

    dtr = daily_true_range(high, low, close)
    range_exp = dtr / np.maximum(trailing_mov_mean(dtr, 20), 1e-6)

    rsi_vals = rsi(close, 14)
    pvol = parkinson_vol(high, low, 20)
    ekf_price, ekf_vel, ekf_resid = adaptive_ekf_smoother(close, volume, dtr, lookback_range_days=252)

    if sentiment_by_date is not None:
        llm_sentiment = np.array([sentiment_by_date.get(d, 0.0) for d in main.dates])
    else:
        llm_sentiment = np.zeros(n)  # neutral fallback, same as the .m file's own fallback path
    # The .m file feeds the BiLSTM LLM_Sentiment_Smoothed = movmean(x,[2 0]),
    # i.e. a 3-day trailing mean -- not the raw daily scores.
    llm_sentiment = trailing_mov_mean(llm_sentiment, 3)

    feature_matrix = np.column_stack([
        returns, rsi_vals, llm_sentiment, pvol, sector_rs,
        vpm_ratio, range_exp, ekf_price, ekf_vel, ekf_resid,
    ])
    assert feature_matrix.shape == (n, len(FEATURE_LIST))
    return feature_matrix


def rolling_zscore_normalize(feature_matrix: np.ndarray, base_lookback_window: int = 60) -> np.ndarray:
    """Causal (non-leaking) rolling z-score normalization, matching the .m
    file's Part 4 loop EXACTLY: for r <= 60, normalize against the fixed
    first-60-row block; for r > 60, normalize against the trailing 60 rows
    strictly before r (never including row r itself).
    """
    n, f = feature_matrix.shape
    out = np.zeros_like(feature_matrix)
    for r in range(n):
        if r < base_lookback_window:
            sub = feature_matrix[0:base_lookback_window, :]
        else:
            sub = feature_matrix[r - base_lookback_window:r, :]
        mu = sub.mean(axis=0)
        sigma = sub.std(axis=0, ddof=1)  # MATLAB's std() divides by (N-1) by default -- match that, not numpy's default ddof=0
        out[r, :] = (feature_matrix[r, :] - mu) / np.maximum(sigma, 1e-6)
    return out


def build_multistep_targets(close: np.ndarray, horizon_ahead: int = 45) -> np.ndarray:
    n = len(close)
    target = np.zeros((n, horizon_ahead))
    for t in range(n - horizon_ahead):
        for h in range(1, horizon_ahead + 1):
            target[t, h - 1] = (close[t + h] - close[t]) / max(close[t], 1e-6)
    # tail padding: repeat the last fully-known row's FINAL value across the
    # whole horizon for the padded rows, matching the .m file's own (slightly
    # unusual) tail-padding formula exactly.
    if n - horizon_ahead - 1 >= 0:
        pad_value = target[n - horizon_ahead - 1, -1]
    else:
        pad_value = 0.0
    for t in range(max(0, n - horizon_ahead), n):
        target[t, :] = pad_value
    return target


def build_windows(feature_norm: np.ndarray, target_matrix: np.ndarray,
                   lookback: int = 10, horizon_ahead: int = 45):
    n = feature_norm.shape[0]
    total_samples = n - lookback - horizon_ahead + 1
    if total_samples <= 0:
        raise ValueError(f"Not enough rows ({n}) for lookback={lookback} + horizon={horizon_ahead}")
    num_features = feature_norm.shape[1]
    X = np.zeros((total_samples, lookback, num_features), dtype=np.float32)
    Y = np.zeros((total_samples, horizon_ahead), dtype=np.float32)
    for i in range(total_samples):
        X[i] = feature_norm[i:i + lookback, :]
        Y[i] = target_matrix[i + lookback - 1, :]
    return X, Y


def load_sentiment_csv(path: str) -> dict:
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = dt.date.fromisoformat(row["date"][:10])
                out[d] = float(row["llm_sentiment_score"])
            except (KeyError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker-csv", required=True, help="Main ticker's Tiingo cache CSV")
    ap.add_argument("--spy-csv", default=None, help="SPY_Tiingo_cache.csv (market fallback benchmark)")
    ap.add_argument("--xlv-csv", default=None, help="XLV_Tiingo_cache.csv (sector benchmark, preferred)")
    ap.add_argument("--sentiment-csv", default=None,
                     help="Optional <TICKER>_LLM_Sentiment_cache.csv; omit for neutral (0) sentiment")
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=45)
    ap.add_argument("--out", required=True, help="Output .npz path (X, Y arrays)")
    args = ap.parse_args()

    main_series = load_tiingo_cache_csv(args.ticker_csv)
    spy_series = load_tiingo_cache_csv(args.spy_csv) if args.spy_csv else None
    xlv_series = load_tiingo_cache_csv(args.xlv_csv) if args.xlv_csv else None
    sentiment_map = load_sentiment_csv(args.sentiment_csv) if args.sentiment_csv else None

    feature_matrix = build_feature_matrix(main_series, spy_series, xlv_series, sentiment_map)
    feature_norm = rolling_zscore_normalize(feature_matrix, base_lookback_window=60)
    target_matrix = build_multistep_targets(main_series.close, horizon_ahead=args.horizon)
    X, Y = build_windows(feature_norm, target_matrix, lookback=args.lookback, horizon_ahead=args.horizon)

    latest_block = feature_norm[-args.lookback:, :].astype(np.float32)

    np.savez(args.out, X=X, Y=Y, latest_block=latest_block, feature_list=np.array(FEATURE_LIST))
    print(f"Saved {args.out}: X{X.shape}, Y{Y.shape}, latest_block{latest_block.shape}")
    if sentiment_map is None:
        print("NOTE: LLM_Sentiment feature is all-zero (neutral) -- no --sentiment-csv given.")
    if xlv_series is None and spy_series is None:
        print("NOTE: SectorRS feature is all-zero (neutral) -- no --spy-csv/--xlv-csv given.")


if __name__ == "__main__":
    main()
