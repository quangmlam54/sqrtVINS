// full_pipeline_main.cpp
//
// Self-contained C++ counterpart of LLM_EKF_Based_Predictor1.m (no MATLAB):
//
//   Tiingo prices (ticker + XLV/SPY benchmark)      -> tiingo_client
//   10-feature matrix + causal 60-day z-score        -> gnc_forecast_core
//   BiLSTM ensemble inference (ONNX Runtime)         -> bilstm_onnx
//   optional live news sentiment via Claude          -> llm_sentiment
//   support / resistance / stop-loss                 -> gnc_forecast_core
//
// USAGE
//   export TIINGO_API_KEY="..."            (required for live mode)
//   export ANTHROPIC_API_KEY="..."         (optional; enables live sentiment)
//   ./gnc_forecast_full [--models DIR] [--no-llm]      interactive ticker prompt
//   ./gnc_forecast_full --selftest [--models DIR]      no network, no keys
//
// IMPORTANT -- forecast validity:
//   The .onnx files in DIR must be TRAINED models (Phase 2). Untrained
//   proof models (random weights) run fine mechanically but produce
//   meaningless numbers. When DIR does not contain a file named
//   TRAINED_MODEL.marker, this program prints a loud warning on every
//   forecast. The Phase 2 training/export script is what creates that marker.

#include "bilstm_onnx.hpp"
#include "gnc_forecast_core.hpp"
#include "llm_sentiment.hpp"
#include "tiingo_client.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace gnc_forecast;
using json = nlohmann::json;

namespace {

constexpr int kLookback = 10;       // BiLSTM input window (days)
constexpr int kNumFeatures = 10;    // must match featureList order in the .m file
constexpr int kMinBars = 80;        // need > 60 (normalization) + window
constexpr int kLookbackCalendarDays = 500;

std::string trim(const std::string& s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
std::string toUpper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    return s;
}
bool fileExists(const std::string& p) { return std::ifstream(p).good(); }

std::vector<std::string> discoverModels(const std::string& dir) {
    std::vector<std::string> paths;
    for (int i = 1; i <= 9; ++i) {
        const std::string p = dir + "/bilstm_run" + std::to_string(i) + ".onnx";
        if (fileExists(p)) paths.push_back(p);
    }
    return paths;
}

// Mirrors retime(..., 'previous') for sentiment: value persists forward from
// the last dated score; dates BEFORE the first score are 0 (fillmissing 0).
std::vector<double> alignSentiment(const std::vector<SentimentScore>& scores,
                                   const std::vector<long long>& tradingEpochDays) {
    std::map<long long, std::pair<double, int>> perDay;  // epochDay -> (sum, count)
    for (const auto& s : scores) {
        const long long d = isoDateToEpochDays(s.date);
        if (d < 0) continue;
        perDay[d].first += s.score;
        perDay[d].second += 1;
    }
    std::vector<double> out(tradingEpochDays.size(), 0.0);
    auto it = perDay.begin();
    double last = 0.0;
    bool have = false;
    for (size_t i = 0; i < tradingEpochDays.size(); ++i) {
        while (it != perDay.end() && it->first <= tradingEpochDays[i]) {
            last = it->second.first / it->second.second;
            have = true;
            ++it;
        }
        out[i] = have ? last : 0.0;
    }
    return out;
}

struct ForecastReport {
    double basePrice = 0;
    std::vector<double> returns45;
    std::vector<double> pricePath;
    double expectedReturn = 0;
    double targetPrice = 0;
    double pathMin = 0;
    int pathMinDay = 0;
    double currentSentimentSmoothed = 0;
    double crossRunSpreadAtEnd = 0;  // sqrt of cross-run variance at the final frame
    int numRuns = 0;
    SupportResistanceResult sr{};
    double atr = 0;
    double stopLoss = 0;
    double rsi = 0;
};

// benchClose may be empty (SectorRS -> 0). sentimentDaily must match main length.
ForecastReport runForecast(const PriceSeries& main, const std::vector<double>& benchAligned,
                           const std::vector<double>& sentimentDaily,
                           const BiLstmOnnxEnsemble& ensemble) {
    const int n = static_cast<int>(main.close.size());

    const auto ret = dailyReturns(main.close);
    const auto rsiV = relativeStrengthIndex(main.close, 14);
    const auto sentSmooth = trailingMovMean(sentimentDaily, 3);  // movmean(x,[2 0])
    const auto pvol = parkinsonVolatility(main.high, main.low, 20);
    const auto sectorRS = benchAligned.empty() ? std::vector<double>(n, 0.0)
                                               : sectorRelativeStrength(main.close, benchAligned);
    const auto vpm = vpmRatio(main.close, main.volume);
    const auto dtr = dailyTrueRange(main.high, main.low, main.close);
    const auto rangeExp = rangeExpansion(dtr);
    const auto ekf = runAdaptiveEkfSmoother(main.close, main.volume, dtr, std::min(252, n));

    // featureList = {Returns, RSI, LLM_Sentiment, ParkinsonVol, SectorRS,
    //                VPMRatio, RangeExp, EKF_TruePrice, EKF_Velocity, EKF_Residual}
    std::vector<std::vector<double>> fm(n, std::vector<double>(kNumFeatures));
    for (int t = 0; t < n; ++t) {
        fm[t] = {ret[t], rsiV[t], sentSmooth[t], pvol[t], sectorRS[t],
                 vpm[t], rangeExp[t], ekf.filteredPrice[t], ekf.filteredVelocity[t], ekf.filteredResidual[t]};
        // FIX (2026-09): was ekf.smoothedPrice/smoothedVelocity/residual --
        // those come from a backward RTS pass and leak future data into
        // every row except the very last. filteredPrice/Velocity/Residual
        // (forward-pass-only) are causal. See gnc_forecast_core.hpp's
        // EkfSmoothResult comment for the full explanation and the
        // random-walk test that caught this.
    }
    const auto fn = rollingZScoreNormalize(fm, 60);

    std::vector<double> window;
    window.reserve(kLookback * kNumFeatures);
    for (int t = n - kLookback; t < n; ++t)
        for (int f = 0; f < kNumFeatures; ++f) window.push_back(fn[t][f]);

    const auto ens = ensemble.predict(window, kLookback, kNumFeatures);

    ForecastReport r;
    r.basePrice = main.close.back();
    r.returns45 = ens.finalReturns;
    r.numRuns = ens.numRuns;
    r.pricePath.resize(ens.horizonAhead);
    for (int h = 0; h < ens.horizonAhead; ++h) r.pricePath[h] = r.basePrice * (1.0 + ens.finalReturns[h]);
    r.expectedReturn = ens.finalReturns.back();
    r.targetPrice = r.pricePath.back();
    const auto mn = std::min_element(r.pricePath.begin(), r.pricePath.end());
    r.pathMin = *mn;
    r.pathMinDay = static_cast<int>(mn - r.pricePath.begin()) + 1;
    r.currentSentimentSmoothed = sentSmooth.back();
    r.crossRunSpreadAtEnd = std::sqrt(std::max(ens.crossRunVariance.back(), 0.0));

    const auto atrV = averageTrueRange(main.high, main.low, main.close, 14);
    r.atr = atrV.back();
    r.rsi = rsiV.back();
    r.sr = computeSupportResistance(main.high, main.low, r.basePrice, r.atr, std::min(252, n), 3);
    r.stopLoss = r.sr.supportLevel - 0.5 * r.atr;
    return r;
}

void printReport(const std::string& symbol, const ForecastReport& r, const std::string& sentimentLabel,
                 bool modelsTrained, const std::string& modelDir) {
    std::printf("\n==================== %s ====================\n", symbol.c_str());
    if (!modelsTrained) {
        std::printf("!! WARNING: no TRAINED_MODEL.marker in '%s'.\n"
                    "!! These BiLSTM weights are NOT trained (proof/random models) -- the forecast\n"
                    "!! numbers below are mechanically valid but MEANINGLESS. Do not act on them.\n\n",
                    modelDir.c_str());
    }
    std::printf("Current price:            $%.2f\n", r.basePrice);
    std::printf("Expected return @ 45d:    %+.2f%%\n", 100.0 * r.expectedReturn);
    std::printf("Target price (frame 45):  $%.2f\n", r.targetPrice);
    std::printf("Projected path floor:     $%.2f (day %d)\n", r.pathMin, r.pathMinDay);
    std::printf("Ensemble runs / spread:   %d runs, cross-run std at frame 45 = %.4f\n", r.numRuns, r.crossRunSpreadAtEnd);
    std::printf("LLM sentiment (smoothed): %+.2f   [%s]\n", r.currentSentimentSmoothed, sentimentLabel.c_str());
    std::printf("RSI(14) / ATR(14):        %.1f / $%.3f\n", r.rsi, r.atr);
    std::printf("Resistance (awareness):   $%.2f (%d touches)\n", r.sr.resistanceLevel, r.sr.resistanceStrength);
    std::printf("Support (stop gate):      $%.2f (%d touches)\n", r.sr.supportLevel, r.sr.supportStrength);
    std::printf("Suggested stop-loss:      $%.2f (support - 0.5x ATR)\n", r.stopLoss);
    std::printf("Forecast path (every 5th frame): ");
    for (size_t h = 4; h < r.pricePath.size(); h += 5) std::printf("$%.2f ", r.pricePath[h]);
    std::printf("\n=====================================================\n\n");
}


// Saves everything the plotting script needs. One file per ticker:
//   <outDir>/<SYMBOL>_forecast.json   (overwritten on every run)
std::string writeReportJson(const std::string& outDir, const std::string& symbol, const PriceSeries& main,
                            const ForecastReport& r, const std::string& sentimentLabel,
                            const std::string& benchName, bool trained, const std::string& modelDir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(outDir, ec);

    char stamp[32];
    const std::time_t now = std::time(nullptr);
    std::tm tmv{};
    gmtime_r(&now, &tmv);
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", &tmv);

    const int histN = std::min<int>(90, static_cast<int>(main.close.size()));
    json hd = json::array(), hc = json::array();
    for (int i = static_cast<int>(main.close.size()) - histN; i < static_cast<int>(main.close.size()); ++i) {
        hd.push_back(main.isoDates[i]);
        hc.push_back(main.close[i]);
    }

    json j;
    j["symbol"] = symbol;
    j["generated_utc"] = stamp;
    j["last_bar_date"] = main.isoDates.back();
    j["base_price"] = r.basePrice;
    j["models_trained"] = trained;
    j["model_dir"] = modelDir;
    j["num_runs"] = r.numRuns;
    j["expected_return_45"] = r.expectedReturn;
    j["target_price_45"] = r.targetPrice;
    j["path_floor"] = r.pathMin;
    j["path_floor_day"] = r.pathMinDay;
    j["cross_run_std_last"] = r.crossRunSpreadAtEnd;
    j["forecast_returns"] = r.returns45;
    j["forecast_prices"] = r.pricePath;
    j["sentiment_value"] = r.currentSentimentSmoothed;
    j["sentiment_source"] = sentimentLabel;
    j["benchmark"] = benchName;
    j["rsi"] = r.rsi;
    j["atr"] = r.atr;
    j["resistance"] = {{"level", r.sr.resistanceLevel}, {"touches", r.sr.resistanceStrength}};
    j["support"] = {{"level", r.sr.supportLevel}, {"touches", r.sr.supportStrength}};
    j["stop_loss"] = r.stopLoss;
    j["history"] = {{"dates", hd}, {"close", hc}};

    const std::string path = outDir + "/" + symbol + "_forecast.json";
    std::ofstream(path) << j.dump(2) << "\n";
    return path;
}

// ---------------------------------------------------------------- selftest
PriceSeries syntheticSeries(int n, double startPrice, double drift, unsigned seed) {
    PriceSeries s;
    double price = startPrice;
    unsigned x = seed;
    auto rnd = [&]() { x = x * 1664525u + 1013904223u; return (x >> 8) / double(1u << 24) - 0.5; };
    const long long day0 = isoDateToEpochDays("2025-01-02");
    for (int t = 0; t < n; ++t) {
        price = std::max(1.0, price * (1.0 + drift + 0.012 * rnd()));
        s.epochDays.push_back(day0 + t);
        s.isoDates.push_back(epochDaysToIso(day0 + t));
        s.open.push_back(price * (1 + 0.003 * rnd()));
        s.high.push_back(price * (1 + 0.010 + 0.004 * std::fabs(rnd())));
        s.low.push_back(price * (1 - 0.010 - 0.004 * std::fabs(rnd())));
        s.close.push_back(price);
        s.volume.push_back(1.0e6 * (1.0 + 0.4 * rnd()));
    }
    return s;
}

int runSelfTest(const std::string& modelDir, const std::string& outDir) {
    std::printf("Self-test (no network, no API keys)\n");
    int failures = 0;

    // 1. Tiingo JSON parsing on a literal payload.
    const char* lit = R"([{"date":"2026-09-21T00:00:00.000Z","open":10.0,"high":11.0,"low":9.5,"close":10.5,"volume":1000},
                          {"date":"2026-09-22T00:00:00.000Z","open":10.5,"high":11.5,"low":10.0,"close":11.0,"volume":1200},
                          {"date":"2026-09-23T00:00:00.000Z","open":null,"high":11.5,"low":10.0,"close":11.0,"volume":1200}])";
    std::string err;
    const auto ps = parseTiingoDailyJson(lit, err);
    if (!err.empty() || ps.close.size() != 2 || ps.isoDates[1] != "2026-09-22") {
        std::printf("  FAIL  Tiingo JSON parse (expected 2 good rows, 1 skipped null row)\n");
        ++failures;
    } else {
        std::printf("  PASS  Tiingo JSON parse (2 rows kept, null row skipped)\n");
    }

    // 2. Full feature + ONNX ensemble pipeline on synthetic data.
    const auto models = discoverModels(modelDir);
    if (models.empty()) {
        std::printf("  FAIL  no bilstm_run*.onnx found in '%s'\n", modelDir.c_str());
        return 1;
    }
    BiLstmOnnxEnsemble ensemble(models);
    const PriceSeries main = syntheticSeries(400, 50.0, 0.0004, 7);
    const PriceSeries bench = syntheticSeries(400, 100.0, 0.0003, 11);
    const auto benchAligned = alignPriorClose(bench.epochDays, bench.close, main.epochDays);
    const std::vector<double> sentiment(main.close.size(), 0.0);
    const ForecastReport rep = runForecast(main, benchAligned, sentiment, ensemble);

    bool finite = std::isfinite(rep.expectedReturn) && std::isfinite(rep.targetPrice) &&
                  static_cast<int>(rep.pricePath.size()) == 45;
    for (double v : rep.pricePath) finite = finite && std::isfinite(v);
    if (!finite) {
        std::printf("  FAIL  pipeline produced non-finite values or wrong horizon length\n");
        ++failures;
    } else {
        std::printf("  PASS  features -> normalization -> %d-model ONNX ensemble -> 45-frame path\n", rep.numRuns);
    }

    const bool srOrdered = rep.sr.supportLevel < rep.basePrice + 1e-9 && rep.sr.resistanceLevel > rep.basePrice - 1e-9;
    std::printf("  %s  support (%.2f) <= price (%.2f) <= resistance (%.2f)\n", srOrdered ? "PASS" : "FAIL",
                rep.sr.supportLevel, rep.basePrice, rep.sr.resistanceLevel);
    if (!srOrdered) ++failures;

    const bool selfTrained = fileExists(modelDir + "/TRAINED_MODEL.marker");
    printReport("SYNTHETIC", rep, "neutral (selftest)", selfTrained, modelDir);
    const std::string jp = writeReportJson(outDir, "SYNTHETIC", main, rep, "neutral (selftest)", "synthetic", selfTrained, modelDir);
    std::printf("Saved %s (plot it with: python3 python/plot_forecast.py %s)\n", jp.c_str(), jp.c_str());
    std::printf(failures == 0 ? "Self-test PASSED.\n" : "Self-test FAILED (%d).\n", failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string modelDir = "onnx_models";
    std::string outDir = "output";
    bool selftest = false, noLlm = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--no-llm") noLlm = true;
        else if (a == "--models" && i + 1 < argc) modelDir = argv[++i];
        else if (a == "--out-dir" && i + 1 < argc) outDir = argv[++i];
        else { std::fprintf(stderr, "Unknown argument: %s\n", a.c_str()); return 2; }
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);
    struct Cleanup { ~Cleanup() { curl_global_cleanup(); } } cleanup;

    if (selftest) return runSelfTest(modelDir, outDir);

    const char* tk = std::getenv("TIINGO_API_KEY");
    if (!tk || trim(tk).empty()) {
        std::fprintf(stderr, "TIINGO_API_KEY is not set. Run: export TIINGO_API_KEY=\"<your real key>\"\n");
        return 1;
    }
    const std::string tiingoKey = trim(tk);
    const char* ak = std::getenv("ANTHROPIC_API_KEY");
    const std::string claudeKey = (ak && !noLlm) ? trim(ak) : "";

    const auto models = discoverModels(modelDir);
    if (models.empty()) {
        std::fprintf(stderr, "No bilstm_run*.onnx files in '%s'. Use --models DIR.\n", modelDir.c_str());
        return 1;
    }
    BiLstmOnnxEnsemble ensemble(models);
    const bool trained = fileExists(modelDir + "/TRAINED_MODEL.marker");
    std::printf("GNC full pipeline -- %zu ONNX model(s) from '%s' (%s)\n", models.size(), modelDir.c_str(),
                trained ? "trained" : "UNTRAINED proof weights");
    std::printf("Live LLM sentiment: %s\n", claudeKey.empty() ? "OFF (neutral 0 used)" : "ON (Claude)");
    std::printf("Enter a ticker (e.g. INTU), or 'quit'.\n\n");

    std::string line;
    while (true) {
        std::printf("Symbol> ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        const std::string symbol = toUpper(trim(line));
        if (symbol.empty()) continue;
        if (symbol == "QUIT" || symbol == "EXIT") break;

        std::string err;
        std::printf("Fetching %s prices...\n", symbol.c_str());
        PriceSeries main = fetchTiingoDaily(symbol, tiingoKey, kLookbackCalendarDays, err);
        if (!err.empty()) { std::fprintf(stderr, "  %s\n\n", err.c_str()); continue; }
        if (static_cast<int>(main.close.size()) < kMinBars) {
            std::fprintf(stderr, "  Only %zu bars for %s; need >= %d.\n\n", main.close.size(), symbol.c_str(), kMinBars);
            continue;
        }

        // Benchmark: XLV preferred, SPY fallback, else SectorRS = 0 (flagged).
        std::vector<double> benchAligned;
        std::string benchName = "none";
        for (const char* b : {"XLV", "SPY"}) {
            std::string berr;
            PriceSeries bs = fetchTiingoDaily(b, tiingoKey, kLookbackCalendarDays, berr);
            if (berr.empty() && !bs.close.empty()) {
                benchAligned = alignPriorClose(bs.epochDays, bs.close, main.epochDays);
                benchName = b;
                break;
            }
        }
        if (benchAligned.empty()) std::printf("  (no benchmark available -> SectorRS set to 0)\n");
        else std::printf("  benchmark for SectorRS: %s\n", benchName.c_str());

        // Sentiment.
        std::vector<double> sentiment(main.close.size(), 0.0);
        std::string sentimentLabel = "NEUTRAL 0 (live LLM off)";
        if (!claudeKey.empty()) {
            std::string nerr;
            const auto news = fetchTiingoNews(symbol, tiingoKey, 50, nerr);
            if (!nerr.empty() || news.empty()) {
                sentimentLabel = "NEUTRAL 0 (news fetch failed/empty" + (nerr.empty() ? std::string() : ": " + nerr) + ")";
            } else {
                std::string serr;
                const auto scores = scoreNewsSentimentWithClaude(news, claudeKey, serr);
                if (!serr.empty() || scores.empty()) {
                    sentimentLabel = "NEUTRAL 0 (Claude scoring failed" + (serr.empty() ? std::string() : ": " + serr) + ")";
                } else {
                    sentiment = alignSentiment(scores, main.epochDays);
                    sentimentLabel = "LIVE (Claude, " + std::to_string(scores.size()) + " headlines)";
                }
            }
        }

        try {
            const ForecastReport rep = runForecast(main, benchAligned, sentiment, ensemble);
            printReport(symbol, rep, sentimentLabel, trained, modelDir);
            const std::string jp = writeReportJson(outDir, symbol, main, rep, sentimentLabel, benchName, trained, modelDir);
            std::printf("Saved %s\nPlot it:  python3 python/plot_forecast.py %s\n\n", jp.c_str(), jp.c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  Forecast failed: %s\n\n", e.what());
        }
    }
    std::printf("Goodbye.\n");
    return 0;
}
