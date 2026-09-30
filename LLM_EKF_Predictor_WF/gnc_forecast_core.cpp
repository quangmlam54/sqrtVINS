// gnc_forecast_core.cpp
#include "gnc_forecast_core.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace gnc_forecast {

// ============================================================================
// Rolling-window helpers
// ============================================================================
std::vector<double> trailingMovMean(const std::vector<double>& x, int window) {
    const int n = static_cast<int>(x.size());
    std::vector<double> out(n, 0.0);
    double runningSum = 0.0;
    for (int i = 0; i < n; ++i) {
        runningSum += x[i];
        const int startIdx = std::max(0, i - window + 1);
        if (i >= window) {
            runningSum -= x[i - window];
        }
        const int count = i - startIdx + 1;
        out[i] = runningSum / static_cast<double>(count);
    }
    return out;
}

std::vector<double> trailingMovMin(const std::vector<double>& x, int window) {
    const int n = static_cast<int>(x.size());
    std::vector<double> out(n, 0.0);
    for (int i = 0; i < n; ++i) {
        const int startIdx = std::max(0, i - window + 1);
        out[i] = *std::min_element(x.begin() + startIdx, x.begin() + i + 1);
    }
    return out;
}

std::vector<double> trailingMovMax(const std::vector<double>& x, int window) {
    const int n = static_cast<int>(x.size());
    std::vector<double> out(n, 0.0);
    for (int i = 0; i < n; ++i) {
        const int startIdx = std::max(0, i - window + 1);
        out[i] = *std::max_element(x.begin() + startIdx, x.begin() + i + 1);
    }
    return out;
}

// ============================================================================
// Technical extractors
// ============================================================================
std::vector<double> dailyTrueRange(const std::vector<double>& high,
                                    const std::vector<double>& low,
                                    const std::vector<double>& close) {
    const int n = static_cast<int>(close.size());
    std::vector<double> tr(n, 0.0);
    for (int t = 0; t < n; ++t) {
        const double prevClose = (t == 0) ? close[0] : close[t - 1];
        const double a = high[t] - low[t];
        const double b = std::fabs(high[t] - prevClose);
        const double c = std::fabs(low[t] - prevClose);
        tr[t] = std::max({a, b, c});
    }
    return tr;
}

std::vector<double> averageTrueRange(const std::vector<double>& high,
                                      const std::vector<double>& low,
                                      const std::vector<double>& close,
                                      int atrWindow) {
    return trailingMovMean(dailyTrueRange(high, low, close), atrWindow);
}

std::vector<double> relativeStrengthIndex(const std::vector<double>& close,
                                           int window) {
    const int n = static_cast<int>(close.size());
    std::vector<double> gains(n, 0.0), losses(n, 0.0);
    for (int t = 1; t < n; ++t) {
        const double d = close[t] - close[t - 1];
        gains[t] = std::max(d, 0.0);
        losses[t] = std::max(-d, 0.0);
    }
    const auto avgGain = trailingMovMean(gains, window);
    const auto avgLoss = trailingMovMean(losses, window);

    std::vector<double> rsi(n, 0.0);
    for (int t = 0; t < n; ++t) {
        const double denom = std::max(avgLoss[t], 1e-6);
        rsi[t] = 100.0 - (100.0 / (1.0 + avgGain[t] / denom));
    }
    return rsi;
}

MacdResult macdSimple(const std::vector<double>& close,
                      int fastWindow, int slowWindow, int signalWindow) {
    MacdResult result;
    const auto fastMA = trailingMovMean(close, fastWindow);
    const auto slowMA = trailingMovMean(close, slowWindow);

    const int n = static_cast<int>(close.size());
    result.macdLine.resize(n);
    for (int t = 0; t < n; ++t) {
        result.macdLine[t] = fastMA[t] - slowMA[t];
    }
    result.signalLine = trailingMovMean(result.macdLine, signalWindow);
    return result;
}

std::vector<double> parkinsonVolatility(const std::vector<double>& high,
                                        const std::vector<double>& low,
                                        int window) {
    const int n = static_cast<int>(high.size());
    std::vector<double> raw(n, 0.0);
    const double fourLn2 = 4.0 * std::log(2.0);
    for (int t = 0; t < n; ++t) {
        const double h = std::max(high[t], 1e-6);
        const double l = std::max(low[t], 1e-6);
        const double lr = std::log(h / l);
        raw[t] = (lr * lr) / fourLn2;
    }
    auto smoothed = trailingMovMean(raw, window);
    for (auto& v : smoothed) v = std::sqrt(std::max(v, 0.0));
    return smoothed;
}

// ============================================================================
// Adaptive asymmetric EKF + backward RTS smoother
// ============================================================================
namespace {

// Minimal fixed 2x2 matrix helper -- keeps the port free of any linear
// algebra library dependency while mirroring the MATLAB 2-state math
// exactly (F*P*F' + Q, Kalman gain, RTS gain, etc).
struct Mat2 {
    double a11 = 0, a12 = 0, a21 = 0, a22 = 0;
};

Mat2 matMul(const Mat2& A, const Mat2& B) {
    return {A.a11 * B.a11 + A.a12 * B.a21, A.a11 * B.a12 + A.a12 * B.a22,
            A.a21 * B.a11 + A.a22 * B.a21, A.a21 * B.a12 + A.a22 * B.a22};
}
Mat2 transpose(const Mat2& A) { return {A.a11, A.a21, A.a12, A.a22}; }
Mat2 addMat(const Mat2& A, const Mat2& B) {
    return {A.a11 + B.a11, A.a12 + B.a12, A.a21 + B.a21, A.a22 + B.a22};
}

struct Vec2 { double x1 = 0, x2 = 0; };
Vec2 matVec(const Mat2& A, const Vec2& v) {
    return {A.a11 * v.x1 + A.a12 * v.x2, A.a21 * v.x1 + A.a22 * v.x2};
}

}  // namespace

EkfSmoothResult runAdaptiveEkfSmoother(const std::vector<double>& close,
                                       const std::vector<double>& volume,
                                       const std::vector<double>& dailyTR,
                                       int lookbackRangeDays) {
    const int n = static_cast<int>(close.size());
    if (n < 2) throw std::invalid_argument("runAdaptiveEkfSmoother: need >= 2 bars");

    // ~14-day ATR in $, floored for very quiet names (mirrors recentATR in .m)
    const int atrLookback = std::min(14, n);
    double recentAtr = 0.0;
    for (int i = n - atrLookback; i < n; ++i) recentAtr += dailyTR[i];
    recentAtr /= static_cast<double>(atrLookback);
    recentAtr = std::max(recentAtr, 1e-3 * std::max(close[n - 1], 1.0));

    const int lookback = std::min(lookbackRangeDays, n);
    const auto rollingLow  = trailingMovMin(close, lookback);
    const auto rollingHigh = trailingMovMax(close, lookback);
    std::vector<double> rollingSpan(n);
    for (int i = 0; i < n; ++i) rollingSpan[i] = std::max(rollingHigh[i] - rollingLow[i], 1e-6);

    std::vector<Vec2> xEst(n);
    std::vector<Mat2> pEst(n);

    xEst[0] = {close[0], 0.0};
    Mat2 P = {recentAtr * recentAtr, 0.0, 0.0, 0.1 * recentAtr * recentAtr};
    pEst[0] = P;

    const double dt = 1.0;
    const Mat2 Q = {0.02 * recentAtr * recentAtr, 0.01 * recentAtr * recentAtr,
                     0.01 * recentAtr * recentAtr, 0.05 * recentAtr * recentAtr};
    const double Rmeas = 0.25 * recentAtr * recentAtr;

    // ---- Forward pass ----
    for (int t = 1; t < n; ++t) {
        const double valueZoneCeiling = rollingLow[t - 1] + 0.25 * rollingSpan[t - 1];
        const double dampingSpan = std::max(0.10 * rollingSpan[t - 1], 1e-6);

        double velDamping = 1.0;
        if (xEst[t - 1].x2 < 0.0 && close[t - 1] < valueZoneCeiling) {
            velDamping = std::max(0.40, 1.0 - (valueZoneCeiling - close[t - 1]) / dampingSpan);
        }
        const Mat2 F = {1.0, dt, 0.0, velDamping};

        const Vec2 xPred = matVec(F, xEst[t - 1]);
        const Mat2 Ppred = addMat(matMul(matMul(F, P), transpose(F)), Q);

        // Trailing 20-bar mean volume for the adaptive-R volume factor
        int volStart = std::max(0, t - 20);
        double volMean = 0.0;
        for (int k = volStart; k <= t; ++k) volMean += volume[k];
        volMean /= static_cast<double>(t - volStart + 1);
        const double volFactor = std::max(volume[t] / std::max(volMean, 1e-6), 0.1);
        const double Radaptive = Rmeas * (1.0 / volFactor);

        const double yInnov = close[t] - xPred.x1;
        const double S = Ppred.a11 + Radaptive;  // H = [1 0]
        const double k1 = Ppred.a11 / S;
        const double k2 = Ppred.a21 / S;

        xEst[t] = {xPred.x1 + k1 * yInnov, xPred.x2 + k2 * yInnov};

        // P = (I - K*H) * Ppred, with H = [1 0]
        Mat2 Pnew;
        Pnew.a11 = (1.0 - k1) * Ppred.a11;
        Pnew.a12 = (1.0 - k1) * Ppred.a12;
        Pnew.a21 = Ppred.a21 - k2 * Ppred.a11;
        Pnew.a22 = Ppred.a22 - k2 * Ppred.a12;
        P = Pnew;
        pEst[t] = P;
    }

    // ---- Backward RTS smoothing sweep ----
    std::vector<Vec2> xSmooth(n);
    std::vector<Mat2> pSmooth(n);
    xSmooth[n - 1] = xEst[n - 1];
    pSmooth[n - 1] = pEst[n - 1];

    for (int t = n - 2; t >= 0; --t) {
        const double valueZoneCeilingB = rollingLow[t] + 0.25 * rollingSpan[t];
        const double dampingSpanB = std::max(0.10 * rollingSpan[t], 1e-6);

        double velDampingB = 1.0;
        if (xEst[t].x2 < 0.0 && close[t] < valueZoneCeilingB) {
            velDampingB = std::max(0.40, 1.0 - (valueZoneCeilingB - close[t]) / dampingSpanB);
        }
        const Mat2 Fb = {1.0, dt, 0.0, velDampingB};

        const Vec2 xPredNext = matVec(Fb, xEst[t]);
        const Mat2 PpredNext = addMat(matMul(matMul(Fb, pEst[t]), transpose(Fb)), Q);

        // C = P_t * Fb' * inv(PpredNext)   (2x2 inverse, closed form)
        const Mat2 PFbT = matMul(pEst[t], transpose(Fb));
        const double det = PpredNext.a11 * PpredNext.a22 - PpredNext.a12 * PpredNext.a21;
        const double invDet = 1.0 / (std::fabs(det) > 1e-15 ? det : 1e-15);
        const Mat2 PpredInv = {PpredNext.a22 * invDet, -PpredNext.a12 * invDet,
                                -PpredNext.a21 * invDet, PpredNext.a11 * invDet};
        const Mat2 C = matMul(PFbT, PpredInv);

        const Vec2 innov = {xSmooth[t + 1].x1 - xPredNext.x1, xSmooth[t + 1].x2 - xPredNext.x2};
        const Vec2 corr = matVec(C, innov);
        xSmooth[t] = {xEst[t].x1 + corr.x1, xEst[t].x2 + corr.x2};

        const Mat2 diffP = {pSmooth[t + 1].a11 - PpredNext.a11, pSmooth[t + 1].a12 - PpredNext.a12,
                             pSmooth[t + 1].a21 - PpredNext.a21, pSmooth[t + 1].a22 - PpredNext.a22};
        pSmooth[t] = addMat(pEst[t], matMul(matMul(C, diffP), transpose(C)));
    }

    EkfSmoothResult result;
    result.smoothedPrice.resize(n);
    result.smoothedVelocity.resize(n);
    result.residual.resize(n);
    result.filteredPrice.resize(n);
    result.filteredVelocity.resize(n);
    result.filteredResidual.resize(n);
    for (int t = 0; t < n; ++t) {
        result.smoothedPrice[t] = xSmooth[t].x1;
        result.smoothedVelocity[t] = xSmooth[t].x2;
        result.residual[t] = close[t] - xSmooth[t].x1;
        // Forward pass only -- xEst was never touched by the backward RTS
        // sweep above, so this is exactly what a live query ending at day t
        // would have produced. Causal by construction. See the header
        // comment for why this, not smoothedPrice/smoothedVelocity/residual,
        // is what must feed a feature matrix.
        result.filteredPrice[t] = xEst[t].x1;
        result.filteredVelocity[t] = xEst[t].x2;
        result.filteredResidual[t] = close[t] - xEst[t].x1;
    }
    return result;
}

// ============================================================================
// Fractal swing-pivot support/resistance detector
// ============================================================================
namespace {

struct ClusterResult { double level; int strength; bool valid; };

ClusterResult nearestClusteredLevel(const std::vector<double>& pivotPrices,
                                     double currentPrice, double tol, bool above) {
    std::vector<double> candidates;
    for (double p : pivotPrices) {
        if (above ? (p > currentPrice) : (p < currentPrice)) candidates.push_back(p);
    }
    if (candidates.empty()) return {0.0, 0, false};

    std::sort(candidates.begin(), candidates.end());
    const double seed = above ? candidates.front() : candidates.back();

    double sum = 0.0;
    int count = 0;
    for (double p : candidates) {
        if (std::fabs(p - seed) <= tol) { sum += p; ++count; }
    }
    return {sum / count, count, true};
}

}  // namespace

SupportResistanceResult computeSupportResistance(const std::vector<double>& high,
                                                 const std::vector<double>& low,
                                                 double currentPrice,
                                                 double recentAtr,
                                                 int lookbackRangeDays,
                                                 int fractalArm) {
    const int n = static_cast<int>(high.size());
    const int idxStart = std::max(0, n - lookbackRangeDays);
    const int numBars = n - idxStart;

    std::vector<double> swingHighPrices, swingLowPrices;
    for (int k = idxStart + fractalArm; k < idxStart + numBars - fractalArm; ++k) {
        double windowHiMax = high[k - fractalArm];
        double windowLoMin = low[k - fractalArm];
        int hiMaxCount = 0, loMinCount = 0;
        for (int j = k - fractalArm; j <= k + fractalArm; ++j) {
            windowHiMax = std::max(windowHiMax, high[j]);
            windowLoMin = std::min(windowLoMin, low[j]);
        }
        for (int j = k - fractalArm; j <= k + fractalArm; ++j) {
            if (high[j] == windowHiMax) ++hiMaxCount;
            if (low[j] == windowLoMin) ++loMinCount;
        }
        if (high[k] == windowHiMax && hiMaxCount == 1) swingHighPrices.push_back(high[k]);
        if (low[k] == windowLoMin && loMinCount == 1) swingLowPrices.push_back(low[k]);
    }

    const double clusterTol = std::max(0.5 * recentAtr, 1e-3 * std::max(currentPrice, 1.0));

    SupportResistanceResult result{};

    const auto resClu = nearestClusteredLevel(swingHighPrices, currentPrice, clusterTol, true);
    if (resClu.valid) {
        result.resistanceLevel = resClu.level;
        result.resistanceStrength = resClu.strength;
    } else {
        result.resistanceLevel = *std::max_element(high.begin() + idxStart, high.end());
        result.resistanceStrength = 1;
    }

    const auto supClu = nearestClusteredLevel(swingLowPrices, currentPrice, clusterTol, false);
    if (supClu.valid) {
        result.supportLevel = supClu.level;
        result.supportStrength = supClu.strength;
    } else {
        result.supportLevel = *std::min_element(low.begin() + idxStart, low.end());
        result.supportStrength = 1;
    }

    return result;
}

// ============================================================================
// BiLSTM feature-matrix inputs (Returns / SectorRS / VPMRatio / RangeExp)
// ============================================================================
std::vector<double> dailyReturns(const std::vector<double>& close) {
    const int n = static_cast<int>(close.size());
    std::vector<double> ret(n, 0.0);
    for (int t = 1; t < n; ++t) {
        ret[t] = (close[t] - close[t - 1]) / std::max(close[t - 1], 1e-6);
    }
    return ret;
}

std::vector<double> sectorRelativeStrength(const std::vector<double>& close,
                                           const std::vector<double>& benchmarkClose) {
    const auto ownReturns = dailyReturns(close);
    const auto benchReturns = dailyReturns(benchmarkClose);
    const int n = static_cast<int>(close.size());
    std::vector<double> sectorRS(n);
    for (int t = 0; t < n; ++t) sectorRS[t] = ownReturns[t] - benchReturns[t];
    return sectorRS;
}

std::vector<double> vpmRatio(const std::vector<double>& close, const std::vector<double>& volume) {
    const auto returns = dailyReturns(close);
    const int n = static_cast<int>(close.size());
    std::vector<double> downDayVolume(n, 0.0);
    for (int t = 0; t < n; ++t) {
        if (returns[t] < 0.0) downDayVolume[t] = volume[t];
    }
    const auto downMean5 = trailingMovMean(downDayVolume, 5);
    const auto totalMean20 = trailingMovMean(volume, 20);
    std::vector<double> ratio(n);
    for (int t = 0; t < n; ++t) ratio[t] = downMean5[t] / std::max(totalMean20[t], 1e-6);
    return ratio;
}

std::vector<double> rangeExpansion(const std::vector<double>& dailyTR) {
    const auto trMean20 = trailingMovMean(dailyTR, 20);
    const int n = static_cast<int>(dailyTR.size());
    std::vector<double> ratio(n);
    for (int t = 0; t < n; ++t) ratio[t] = dailyTR[t] / std::max(trMean20[t], 1e-6);
    return ratio;
}

std::vector<double> alignPriorClose(const std::vector<long long>& benchmarkDatesEpochDays,
                                    const std::vector<double>& benchmarkClose,
                                    const std::vector<long long>& mainDatesEpochDays) {
    std::vector<double> out(mainDatesEpochDays.size());
    double lastVal = benchmarkClose.empty() ? 0.0 : benchmarkClose[0];
    size_t bi = 0;
    for (size_t i = 0; i < mainDatesEpochDays.size(); ++i) {
        while (bi < benchmarkDatesEpochDays.size() && benchmarkDatesEpochDays[bi] <= mainDatesEpochDays[i]) {
            lastVal = benchmarkClose[bi];
            ++bi;
        }
        out[i] = lastVal;
    }
    return out;
}

// ============================================================================
// Causal rolling z-score normalization
// ============================================================================
std::vector<std::vector<double>> rollingZScoreNormalize(
    const std::vector<std::vector<double>>& featureMatrix, int baseLookbackWindow) {
    const int n = static_cast<int>(featureMatrix.size());
    if (n == 0) return {};
    const int numFeat = static_cast<int>(featureMatrix[0].size());

    std::vector<std::vector<double>> out(n, std::vector<double>(numFeat, 0.0));

    for (int r = 0; r < n; ++r) {
        const int windowStart = (r < baseLookbackWindow) ? 0 : (r - baseLookbackWindow);
        const int windowEnd = (r < baseLookbackWindow) ? baseLookbackWindow : r;  // exclusive
        const int windowLen = windowEnd - windowStart;

        for (int f = 0; f < numFeat; ++f) {
            double mean = 0.0;
            for (int i = windowStart; i < windowEnd; ++i) mean += featureMatrix[i][f];
            mean /= windowLen;

            double variance = 0.0;
            for (int i = windowStart; i < windowEnd; ++i) {
                const double d = featureMatrix[i][f] - mean;
                variance += d * d;
            }
            // MATLAB's std() divides by (N-1); match that exactly for parity.
            const double sigma = (windowLen > 1) ? std::sqrt(variance / (windowLen - 1)) : 0.0;

            out[r][f] = (featureMatrix[r][f] - mean) / std::max(sigma, 1e-6);
        }
    }
    return out;
}

}  // namespace gnc_forecast
