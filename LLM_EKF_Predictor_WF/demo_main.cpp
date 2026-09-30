// demo_main.cpp
//
// Standalone demo: reads the SAME <TICKER>_Tiingo_cache.csv file your
// MATLAB script already writes (date,open,high,low,close,volume columns,
// any order/case) and runs the ported EKF smoother + indicators +
// support/resistance detector, printing a summary comparable to Part 6/7
// of LLM_EKF_Based_Predictor1.m.
//
// Usage:
//   ./gnc_forecast_demo path/to/TICKER_Tiingo_cache.csv
//
// With no argument, falls back to a small synthetic OHLCV series so the
// binary is runnable/testable with zero external files.

#include "gnc_forecast_core.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return std::tolower(c); });
    return s;
}

struct OhlcvSeries {
    std::vector<double> open, high, low, close, volume;
};

bool loadTiingoCsv(const std::string& path, OhlcvSeries& out) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string headerLine;
    if (!std::getline(file, headerLine)) return false;

    std::vector<std::string> headers;
    {
        std::stringstream ss(headerLine);
        std::string col;
        while (std::getline(ss, col, ',')) headers.push_back(toLower(col));
    }

    auto colIndex = [&](const std::string& name) -> int {
        for (size_t i = 0; i < headers.size(); ++i)
            if (headers[i] == name) return static_cast<int>(i);
        return -1;
    };

    const int idxOpen = colIndex("open");
    const int idxHigh = colIndex("high");
    const int idxLow = colIndex("low");
    const int idxClose = colIndex("close");
    const int idxVolume = colIndex("volume");
    if (idxOpen < 0 || idxHigh < 0 || idxLow < 0 || idxClose < 0 || idxVolume < 0) {
        std::cerr << "CSV missing required OHLCV columns.\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) fields.push_back(field);
        if (fields.size() <= static_cast<size_t>(std::max({idxOpen, idxHigh, idxLow, idxClose, idxVolume})))
            continue;
        try {
            out.open.push_back(std::stod(fields[idxOpen]));
            out.high.push_back(std::stod(fields[idxHigh]));
            out.low.push_back(std::stod(fields[idxLow]));
            out.close.push_back(std::stod(fields[idxClose]));
            out.volume.push_back(std::stod(fields[idxVolume]));
        } catch (const std::exception&) {
            continue;  // skip malformed row, mirrors the .m file's NaN-mask filtering
        }
    }
    return !out.close.empty();
}

OhlcvSeries makeSyntheticSeries(int n) {
    OhlcvSeries s;
    double price = 20.0;
    for (int t = 0; t < n; ++t) {
        double drift = 0.02 * std::sin(t / 15.0);
        double noise = 0.15 * std::sin(t * 1.7) - 0.05;
        price = std::max(1.0, price + drift + noise);
        double dayHigh = price + 0.4 + 0.2 * std::fabs(std::sin(t * 0.9));
        double dayLow  = std::max(0.5, price - 0.4 - 0.2 * std::fabs(std::cos(t * 0.7)));
        s.open.push_back(price - 0.1);
        s.high.push_back(dayHigh);
        s.low.push_back(dayLow);
        s.close.push_back(price);
        s.volume.push_back(1.0e6 * (1.0 + 0.3 * std::sin(t * 0.3)));
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace gnc_forecast;

    OhlcvSeries series;
    bool loaded = false;
    if (argc > 1) {
        loaded = loadTiingoCsv(argv[1], series);
        if (!loaded) {
            std::cerr << "Could not load '" << argv[1] << "'; falling back to synthetic data.\n";
        }
    }
    if (!loaded) {
        series = makeSyntheticSeries(300);
        std::cout << "Using synthetic 300-bar OHLCV series (pass a Tiingo cache CSV path to use real data).\n\n";
    }

    const auto tr = dailyTrueRange(series.high, series.low, series.close);
    const auto atr = averageTrueRange(series.high, series.low, series.close, 14);
    const auto rsi = relativeStrengthIndex(series.close, 14);
    const auto macd = macdSimple(series.close, 12, 26, 9);
    const auto pvol = parkinsonVolatility(series.high, series.low, 20);

    const auto ekf = runAdaptiveEkfSmoother(series.close, series.volume, tr, /*lookbackRangeDays=*/252);

    const double currentPrice = series.close.back();
    const double recentAtr = atr.back();
    const auto sr = computeSupportResistance(series.high, series.low, currentPrice, recentAtr,
                                              /*lookbackRangeDays=*/252, /*fractalArm=*/3);

    std::printf("Bars loaded:            %zu\n", series.close.size());
    std::printf("Current price:          $%.2f\n", currentPrice);
    std::printf("Recent ATR (~14d):      $%.4f\n", recentAtr);
    std::printf("EKF smoothed price:     $%.2f (velocity %.5f, residual %.4f)\n",
                ekf.smoothedPrice.back(), ekf.smoothedVelocity.back(), ekf.residual.back());
    std::printf("RSI (14):               %.1f\n", rsi.back());
    std::printf("MACD / signal:          %.4f / %.4f\n", macd.macdLine.back(), macd.signalLine.back());
    std::printf("Parkinson vol (20d):    %.5f\n", pvol.back());
    std::printf("Resistance level:       $%.2f  (%d touches)\n", sr.resistanceLevel, sr.resistanceStrength);
    std::printf("Support level:          $%.2f  (%d touches)\n", sr.supportLevel, sr.supportStrength);
    std::printf("Suggested stop-loss:    $%.2f  (support - 0.5x ATR)\n", sr.supportLevel - 0.5 * recentAtr);

    return 0;
}
