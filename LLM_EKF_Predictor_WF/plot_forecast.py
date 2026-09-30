"""
plot_forecast.py

Reads a <SYMBOL>_forecast.json file written by gnc_forecast_full and
renders it as a PNG, in the same visual style as the MATLAB dashboard
(history + forecast path, resistance/support/stop-loss reference lines,
a small metrics panel).

Usage:
    python3 python/plot_forecast.py output/INTU_forecast.json
    python3 python/plot_forecast.py output/INTU_forecast.json --out my_plot.png
    python3 python/plot_forecast.py output/*_forecast.json      # one PNG per file

Requires: matplotlib (pip install matplotlib if not already present).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe for Codespaces/VS Code (no display needed)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_one(path: str, out_path: str | None) -> str:
    r = load_report(path)

    hist_dates = [dt.date.fromisoformat(d) for d in r["history"]["dates"]]
    hist_close = r["history"]["close"]
    last_date = hist_dates[-1]
    fc_dates = [last_date + dt.timedelta(days=i + 1) for i in range(len(r["forecast_prices"]))]
    fc_prices = r["forecast_prices"]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(hist_dates, hist_close, color="#333333", linewidth=1.3, label="History (last 90 bars)")
    ax.plot([last_date] + fc_dates, [r["base_price"]] + fc_prices,
            "--o", color="#1f5fd1", linewidth=1.6, markersize=3.5, label="BiLSTM forecast path")

    res = r["resistance"]["level"]
    sup = r["support"]["level"]
    stop = r["stop_loss"]
    ax.axhline(res, linestyle="--", color="#cc1a1a", linewidth=1.2,
               label=f"Resistance ${res:.2f} ({r['resistance']['touches']} touches)")
    ax.axhline(sup, linestyle="--", color="#178a26", linewidth=1.2,
               label=f"Support ${sup:.2f} ({r['support']['touches']} touches)")
    ax.axhline(stop, linestyle=":", color="#8a8a8a", linewidth=1.1,
               label=f"Stop-loss ${stop:.2f}")

    title_suffix = "" if r["models_trained"] else "  \u26a0 UNTRAINED WEIGHTS \u2014 forecast not meaningful"
    ax.set_title(f"{r['symbol']} \u2014 45-day BiLSTM forecast{title_suffix}",
                 fontsize=12, fontweight="bold",
                 color="#b30000" if not r["models_trained"] else "black")
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    metrics = (
        f"Current: ${r['base_price']:.2f}\n"
        f"Target (45d): ${r['target_price_45']:.2f}  ({r['expected_return_45']*100:+.2f}%)\n"
        f"Path floor: ${r['path_floor']:.2f} (day {r['path_floor_day']})\n"
        f"RSI(14): {r['rsi']:.1f}   ATR(14): ${r['atr']:.3f}\n"
        f"Ensemble: {r['num_runs']} runs, cross-run std {r['cross_run_std_last']:.4f}\n"
        f"Sentiment: {r['sentiment_value']:+.2f} [{r['sentiment_source']}]\n"
        f"Benchmark (SectorRS): {r['benchmark']}\n"
        f"Generated: {r['generated_utc']}"
    )
    fig.text(0.985, 0.5, metrics, fontsize=8, family="monospace", va="center", ha="right",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc"))
    fig.subplots_adjust(right=0.72)

    if out_path is None:
        out_path = os.path.splitext(path)[0] + "_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_paths", nargs="+", help="One or more *_forecast.json files")
    ap.add_argument("--out", default=None, help="Output PNG path (only valid with a single input file)")
    args = ap.parse_args()

    if args.out and len(args.json_paths) > 1:
        print("--out can only be used with a single input file", file=sys.stderr)
        sys.exit(1)

    for p in args.json_paths:
        out = plot_one(p, args.out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
