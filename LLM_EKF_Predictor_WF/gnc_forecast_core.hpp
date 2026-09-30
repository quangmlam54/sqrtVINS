// gnc_forecast_core.hpp
//
// C++ port of the numeric core of LLM_EKF_Based_Predictor1.m
// (Part 2-3 technical extractors, the adaptive EKF + RTS smoother, and
// the fractal swing-pivot support/resistance detector).
//
// Deliberately EXCLUDED from this port (see accompanying notes / chat):
//   - Tiingo / news ingestion (websave, webread) -> networking, not codegen-able
//   - LLM sentiment scoring via the Anthropic API   -> networking, not codegen-able
//   - readtable / timetable / retime                -> MATLAB-only data containers
//   - trainNetwork / bilstmLayer / predict           -> training must stay in MATLAB;
//                                                       inference-only CAN be
//                                                       code-generated separately via
//                                                       MATLAB Coder's Deep Learning
//                                                       Toolbox support, but that's a
//                                                       distinct pipeline from this file
//   - figure / subplot / yline / uitable             -> plotting/UI, not codegen-able
//
// Everything below is pure numeric std::vector<double> in, std::vector<double>
// (or small structs) out -- no MATLAB-only constructs -- so it is a faithful,
// drop-in-testable port of the parts of the pipeline that MATLAB Coder (codegen)
// could also auto-generate from the original .m functions almost unchanged.

#pragma once
#include <vector>
#include <string>

namespace gnc_forecast {

// ---------------------------------------------------------------------------
// Basic rolling-window helpers (mirror MATLAB movmean/movmin/movmax with the
// default 'Endpoints','shrink' behavior: window truncates near index 0
// instead of looking ahead or padding with NaN).
// ---------------------------------------------------------------------------
std::vector<double> trailingMovMean(const std::vector<double>& x, int window);
std::vector<double> trailingMovMin(const std::vector<double>& x, int window);
std::vector<double> trailingMovMax(const std::vector<double>& x, int window);

// ---------------------------------------------------------------------------
// Technical extractors (Part 2 in the .m file)
// ---------------------------------------------------------------------------
std::vector<double> dailyTrueRange(const std::vector<double>& high,
                                    const std::vector<double>& low,
                                    const std::vector<double>& close);

// 14-bar ATR-style rolling mean of true range (trailing window, size atrWindow)
std::vector<double> averageTrueRange(const std::vector<double>& high,
                                      const std::vector<double>& low,
                                      const std::vector<double>& close,
                                      int atrWindow = 14);

// Wilder-style RSI using trailing simple moving averages of gains/losses,
// matching the .m file's movmean-based approximation (not true Wilder EMA).
std::vector<double> relativeStrengthIndex(const std::vector<double>& close,
                                           int window = 14);

// NOTE: the .m file's "ema12"/"ema26" are misnamed -- they are actually
// trailing simple moving averages (movmean), not exponential moving
// averages. This port reproduces the ACTUAL behavior for numerical parity;
// see macdSimple() below.
struct MacdResult {
    std::vector<double> macdLine;
    std::vector<double> signalLine;
};
MacdResult macdSimple(const std::vector<double>& close,
                      int fastWindow = 12, int slowWindow = 26, int signalWindow = 9);

// Parkinson (high-low) volatility estimator, trailing 20-bar window.
std::vector<double> parkinsonVolatility(const std::vector<double>& high,
                                        const std::vector<double>& low,
                                        int window = 20);

// ---------------------------------------------------------------------------
// Adaptive asymmetric EKF + backward RTS smoother (Part 3 in the .m file)
// State = [price; velocity]. Velocity is damped (momentum-exhaustion) when
// price is falling inside the bottom 25% of its own trailing 1yr range.
//
// IMPORTANT -- causality (fixed 2026-09): `smoothedPrice`/`smoothedVelocity`
// come from a BACKWARD RTS pass, which lets day t's value be influenced by
// every day AFTER it in whatever series was passed in. At live inference
// this is harmless (the last day IS "today" -- there is no real future
// beyond it). At TRAINING time, when many overlapping windows are cut from
// years of history, every window except the very last one has a "today"
// sitting in the middle of the series, and the backward pass has already
// let it see real future prices -- confirmed via a random-walk test
// (spurious correlation with next-day returns up to 0.6, vs ~0.01-0.03 for
// the forward-only fields below).
//
// USE `filteredPrice`/`filteredVelocity`/`filteredResidual` (the forward
// EKF pass ONLY, no backward smoothing) for ANY feature that feeds a model
// -- training or inference. `smoothedPrice`/`smoothedVelocity`/`residual`
// are kept only for cases that genuinely want the RTS-smoothed estimate
// (e.g. a debug/diagnostic plot of the full history); never wire them into
// a feature matrix.
// ---------------------------------------------------------------------------
struct EkfSmoothResult {
    std::vector<double> smoothedPrice;     // RTS-smoothed -- NOT causal, do not use as a feature
    std::vector<double> smoothedVelocity;  // RTS-smoothed -- NOT causal, do not use as a feature
    std::vector<double> residual;          // close - smoothedPrice -- NOT causal, do not use as a feature

    std::vector<double> filteredPrice;     // forward EKF pass only -- CAUSAL, use this as EKF_TruePrice
    std::vector<double> filteredVelocity;  // forward EKF pass only -- CAUSAL, use this as EKF_Velocity
    std::vector<double> filteredResidual;  // close - filteredPrice -- CAUSAL, use this as EKF_Residual
};

EkfSmoothResult runAdaptiveEkfSmoother(const std::vector<double>& close,
                                       const std::vector<double>& volume,
                                       const std::vector<double>& dailyTR,
                                       int lookbackRangeDays = 252);

// ---------------------------------------------------------------------------
// Fractal swing-pivot support/resistance detector (the enhancement added on
// top of the original .m file).
// ---------------------------------------------------------------------------
struct SupportResistanceResult {
    double resistanceLevel;
    double supportLevel;
    int    resistanceStrength;   // touch count in the merged cluster
    int    supportStrength;
};

SupportResistanceResult computeSupportResistance(const std::vector<double>& high,
                                                 const std::vector<double>& low,
                                                 double currentPrice,
                                                 double recentAtr,
                                                 int lookbackRangeDays = 252,
                                                 int fractalArm = 3);

// ---------------------------------------------------------------------------
// BiLSTM feature-matrix inputs not previously ported (Part 2 of the .m file):
// Returns, SectorRS (vs XLV, falling back to SPY), VPMRatio, RangeExp. Kept
// separate from the indicators above since they feed the BiLSTM's 10-column
// feature matrix specifically, in the exact order the .m file's
// `featureList` uses:
//   {Returns, RSI, LLM_Sentiment, ParkinsonVol, SectorRS,
//    VPMRatio, RangeExp, EKF_TruePrice, EKF_Velocity, EKF_Residual}
// ---------------------------------------------------------------------------

// Daily simple return: (close[t] - close[t-1]) / close[t-1], with return[0] = 0.
std::vector<double> dailyReturns(const std::vector<double>& close);

// Ticker's own return minus a benchmark's daily return over the SAME dates
// (benchmarkClose must already be date-aligned 1:1 to `close` -- the
// caller is responsible for that alignment, e.g. via alignPriorClose()).
std::vector<double> sectorRelativeStrength(const std::vector<double>& close,
                                           const std::vector<double>& benchmarkClose);

// Down-day-volume momentum: 5-day trailing mean of (volume on down days,
// else 0) divided by 20-day trailing mean of total volume.
std::vector<double> vpmRatio(const std::vector<double>& close, const std::vector<double>& volume);

// Today's true range relative to its own 20-day trailing average --
// >1 means today's range is wider than usual (volatility expansion).
std::vector<double> rangeExpansion(const std::vector<double>& dailyTR);

// Aligns benchmarkSeries (dates + close) onto mainDates using "previous
// value" semantics (last known benchmark close on/before each main date) --
// the C++ equivalent of the .m file's `retime(..., 'previous')` calls for
// the SPY/XLV benchmark series.
std::vector<double> alignPriorClose(const std::vector<long long>& benchmarkDatesEpochDays,
                                    const std::vector<double>& benchmarkClose,
                                    const std::vector<long long>& mainDatesEpochDays);

// ---------------------------------------------------------------------------
// Causal (non-leaking) rolling z-score normalization -- Part 4 of the .m
// file. For row r < baseLookbackWindow, normalizes against the FIXED first
// baseLookbackWindow rows; for r >= baseLookbackWindow, normalizes against
// the trailing baseLookbackWindow rows strictly BEFORE r. Must be applied
// with the exact same baseLookbackWindow (60 in the .m file) at both
// training-data-generation time and inference time, or the BiLSTM will see
// out-of-distribution inputs.
// ---------------------------------------------------------------------------
std::vector<std::vector<double>> rollingZScoreNormalize(
    const std::vector<std::vector<double>>& featureMatrix,  // [numRows][numFeatures]
    int baseLookbackWindow = 60);

}  // namespace gnc_forecast
