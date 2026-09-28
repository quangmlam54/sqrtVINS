// interactive_main.cpp
//
// Interactive CLI: prompts for a stock ticker, fetches live daily OHLCV
// history from the Tiingo REST API (the same data source your MATLAB
// script uses), and runs it through the same ported numeric core
// (ATR/RSI/MACD/Parkinson vol, adaptive EKF+RTS smoother, fractal
// support/resistance) as demo_main.cpp -- but with no CSV file needed.
//
// Requires:
//   - libcurl        (apt: libcurl4-openssl-dev)
//   - nlohmann/json  (apt: nlohmann-json3-dev)
//   - a Tiingo API key in the TIINGO_API_KEY environment variable --
//     the SAME key your MATLAB script's `apiKey` variable already uses.
//     Free tier signup: https://www.tiingo.com/
//
// Usage:
//   export TIINGO_API_KEY="your_tiingo_key_here"
//   ./build/gnc_forecast_live
//
// NOT included here (see chat / README for why): LLM news-sentiment
// fetch and BiLSTM inference. Nothing in this numeric-only C++ port
// consumes a sentiment feature today, so wiring up ANTHROPIC_API_KEY
// here wouldn't feed anything yet -- flagged as a possible follow-up,
// not silently skipped.

#include "gnc_forecast_core.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

std::string toUpper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return std::toupper(c); });
    return s;
}

std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

size_t curlWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buffer = static_cast<std::string*>(userdata);
    buffer->append(ptr, size * nmemb);
    return size * nmemb;
}

struct HttpResult { long status; std::string body; };

// Minimal libcurl GET wrapper. Throws only on transport-level failure
// (DNS/TLS/connect refused); HTTP error codes (404/401/429/...) come
// back normally so the caller can report something specific.
HttpResult httpGet(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("libcurl initialization failed");

    std::string responseBody;
    char errorBuf[CURL_ERROR_SIZE] = {0};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "gnc_forecast_live/1.0");
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuf);

    const CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error(std::string("HTTP request failed: ") +
                                  (errorBuf[0] ? errorBuf : curl_easy_strerror(res)));
    }

    long httpStatus = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpStatus);
    curl_easy_cleanup(curl);
    return {httpStatus, responseBody};
}

std::string todayIso() {
    const auto t = std::time(nullptr);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&t));
    return std::string(buf);
}

std::string daysAgoIso(int daysBack) {
    const auto t = std::time(nullptr) - static_cast<std::time_t>(daysBack) * 86400;
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&t));
    return std::string(buf);
}

struct OhlcvSeries {
    std::vector<double> open, high, low, close, volume;
};

// Parses a Tiingo /tiingo/daily/{symbol}/prices JSON array (each element:
// {"date":..., "open":..., "high":..., "low":..., "close":..., "volume":...,
// plus adjusted-price fields we don't need}) into an OhlcvSeries. Split out
// from fetchTiingoDaily so it can be exercised on a literal JSON string
// with no network call -- see the sandbox validation notes in chat.
OhlcvSeries parseTiingoDailyJson(const std::string& body, std::string& errorOut) {
    OhlcvSeries series;
    errorOut.clear();

    json parsed;
    try {
        parsed = json::parse(body);
    } catch (const std::exception& e) {
        errorOut = std::string("Failed to parse Tiingo response as JSON: ") + e.what();
        return series;
    }

    if (!parsed.is_array() || parsed.empty()) {
        errorOut = "Tiingo returned no price rows.";
        return series;
    }

    series.open.reserve(parsed.size());
    series.high.reserve(parsed.size());
    series.low.reserve(parsed.size());
    series.close.reserve(parsed.size());
    series.volume.reserve(parsed.size());

    for (const auto& bar : parsed) {
        if (!bar.contains("open") || !bar.contains("high") || !bar.contains("low") ||
            !bar.contains("close") || !bar.contains("volume") ||
            bar["open"].is_null() || bar["high"].is_null() || bar["low"].is_null() ||
            bar["close"].is_null() || bar["volume"].is_null()) {
            continue;  // skip malformed/half-populated rows, mirrors the .m file's NaN-mask filtering
        }
        series.open.push_back(bar["open"].get<double>());
        series.high.push_back(bar["high"].get<double>());
        series.low.push_back(bar["low"].get<double>());
        series.close.push_back(bar["close"].get<double>());
        series.volume.push_back(bar["volume"].get<double>());
    }
    return series;
}

// Fetches daily OHLCV bars for `symbol` from Tiingo's REST API, covering
// roughly the trailing `lookbackDays` calendar days (comfortably more
// than 252 trading days for the default 1yr lookbackRangeDays).
OhlcvSeries fetchTiingoDaily(const std::string& symbol, const std::string& apiKey,
                            int lookbackDays, std::string& errorOut) {
    errorOut.clear();
    const std::string url =
        "https://api.tiingo.com/tiingo/daily/" + symbol + "/prices"
        "?startDate=" + daysAgoIso(lookbackDays) +
        "&endDate=" + todayIso() +
        "&format=json&resampleFreq=daily&token=" + apiKey;

    HttpResult resp;
    try {
        resp = httpGet(url);
    } catch (const std::exception& e) {
        errorOut = std::string("Network error: ") + e.what();
        return {};
    }

    if (resp.status == 404) {
        errorOut = "Symbol '" + symbol + "' not found on Tiingo (404). Check the ticker.";
        return {};
    }
    if (resp.status == 401 || resp.status == 403) {
        errorOut = "Tiingo rejected the API key (HTTP " + std::to_string(resp.status) +
                   "). Check TIINGO_API_KEY.";
        return {};
    }
    if (resp.status != 200) {
        errorOut = "Tiingo returned HTTP " + std::to_string(resp.status) + ": " + resp.body;
        return {};
    }

    OhlcvSeries series = parseTiingoDailyJson(resp.body, errorOut);
    if (!errorOut.empty()) return {};

    if (series.close.size() < 30) {
        errorOut = "Only " + std::to_string(series.close.size()) +
                   " bars returned for '" + symbol + "' -- too few for a reliable EKF/indicator run.";
        return {};
    }
    return series;
}

void printForecastSummary(const std::string& symbol, const OhlcvSeries& series) {
    using namespace gnc_forecast;

    const auto tr = dailyTrueRange(series.high, series.low, series.close);
    const auto atr = averageTrueRange(series.high, series.low, series.close, 14);
    const auto rsi = relativeStrengthIndex(series.close, 14);
    const auto macd = macdSimple(series.close, 12, 26, 9);
    const auto pvol = parkinsonVolatility(series.high, series.low, 20);

    const int lookbackRangeDays = std::min(252, static_cast<int>(series.close.size()));
    const auto ekf = runAdaptiveEkfSmoother(series.close, series.volume, tr, lookbackRangeDays);

    const double currentPrice = series.close.back();
    const double recentAtr = atr.back();
    const auto sr = computeSupportResistance(series.high, series.low, currentPrice, recentAtr,
                                              lookbackRangeDays, 3);

    std::cout << "\n=== " << symbol << " -- live Tiingo data, " << series.close.size() << " bars ===\n";
    std::printf("Current price:          $%.2f\n", currentPrice);
    std::printf("Recent ATR (~14d):      $%.4f\n", recentAtr);
    std::printf("EKF smoothed price:     $%.2f (velocity %.5f, residual %.4f)\n",
                ekf.smoothedPrice.back(), ekf.smoothedVelocity.back(), ekf.residual.back());
    std::printf("RSI (14):               %.1f\n", rsi.back());
    std::printf("MACD / signal:          %.4f / %.4f\n", macd.macdLine.back(), macd.signalLine.back());
    std::printf("Parkinson vol (20d):    %.5f\n", pvol.back());
    std::printf("Resistance level:       $%.2f  (%d touches)\n", sr.resistanceLevel, sr.resistanceStrength);
    std::printf("Support level:          $%.2f  (%d touches)\n", sr.supportLevel, sr.supportStrength);
    std::printf("Suggested stop-loss:    $%.2f  (support - 0.5x ATR)\n\n", sr.supportLevel - 0.5 * recentAtr);
}

}  // namespace

// A short literal payload shaped exactly like Tiingo's real
// /tiingo/daily/{symbol}/prices response (subset of fields; Tiingo's
// actual payload also includes adjOpen/adjHigh/.../divCash/splitFactor,
// which parseTiingoDailyJson() correctly ignores). Used by --selftest
// below to exercise the JSON parsing + full numeric core with no
// network access and no API key required.
static const char* kSelfTestJson = R"([
  {"date":"2026-06-01T00:00:00.000Z","open":180.10,"high":182.40,"low":179.55,"close":181.20,"volume":52000000},
  {"date":"2026-06-02T00:00:00.000Z","open":181.30,"high":183.10,"low":180.90,"close":182.75,"volume":48500000},
  {"date":"2026-06-03T00:00:00.000Z","open":182.60,"high":184.00,"low":181.80,"close":183.40,"volume":45200000},
  {"date":"2026-06-04T00:00:00.000Z","open":183.50,"high":185.20,"low":182.95,"close":184.90,"volume":51000000},
  {"date":"2026-06-05T00:00:00.000Z","open":184.80,"high":186.30,"low":183.60,"close":185.10,"volume":49800000},
  {"date":"2026-06-08T00:00:00.000Z","open":185.00,"high":187.00,"low":184.20,"close":186.55,"volume":47300000},
  {"date":"2026-06-09T00:00:00.000Z","open":186.60,"high":188.10,"low":185.90,"close":187.20,"volume":50100000},
  {"date":"2026-06-10T00:00:00.000Z","open":187.10,"high":187.80,"low":184.50,"close":185.30,"volume":53400000},
  {"date":"2026-06-11T00:00:00.000Z","open":185.40,"high":186.20,"low":183.10,"close":184.00,"volume":52900000},
  {"date":"2026-06-12T00:00:00.000Z","open":183.90,"high":185.50,"low":182.60,"close":184.75,"volume":46700000},
  {"date":"2026-06-15T00:00:00.000Z","open":184.80,"high":186.90,"low":184.10,"close":186.40,"volume":48200000},
  {"date":"2026-06-16T00:00:00.000Z","open":186.50,"high":188.60,"low":185.80,"close":188.10,"volume":51600000},
  {"date":"2026-06-17T00:00:00.000Z","open":188.20,"high":190.00,"low":187.50,"close":189.60,"volume":54000000},
  {"date":"2026-06-18T00:00:00.000Z","open":189.70,"high":191.20,"low":188.90,"close":190.85,"volume":49700000},
  {"date":"2026-06-19T00:00:00.000Z","open":190.90,"high":192.50,"low":190.10,"close":191.40,"volume":47100000},
  {"date":"2026-06-22T00:00:00.000Z","open":191.50,"high":193.00,"low":190.60,"close":192.75,"volume":45900000},
  {"date":"2026-06-23T00:00:00.000Z","open":192.80,"high":194.10,"low":191.90,"close":193.30,"volume":48800000},
  {"date":"2026-06-24T00:00:00.000Z","open":193.40,"high":195.00,"low":192.60,"close":194.50,"volume":52300000},
  {"date":"2026-06-25T00:00:00.000Z","open":194.60,"high":196.20,"low":193.80,"close":195.90,"volume":50500000},
  {"date":"2026-06-26T00:00:00.000Z","open":196.00,"high":197.50,"low":195.10,"close":196.60,"volume":49000000},
  {"date":"2026-06-29T00:00:00.000Z","open":196.70,"high":197.20,"low":194.80,"close":195.40,"volume":53100000},
  {"date":"2026-06-30T00:00:00.000Z","open":195.50,"high":196.80,"low":193.90,"close":194.20,"volume":51800000},
  {"date":"2026-07-01T00:00:00.000Z","open":194.30,"high":195.60,"low":192.50,"close":193.10,"volume":50200000},
  {"date":"2026-07-02T00:00:00.000Z","open":193.20,"high":194.50,"low":191.80,"close":192.40,"volume":48600000},
  {"date":"2026-07-06T00:00:00.000Z","open":192.50,"high":193.90,"low":190.60,"close":191.20,"volume":52000000},
  {"date":"2026-07-07T00:00:00.000Z","open":191.30,"high":192.70,"low":189.50,"close":190.10,"volume":51400000},
  {"date":"2026-07-08T00:00:00.000Z","open":190.20,"high":191.60,"low":188.90,"close":189.70,"volume":49300000},
  {"date":"2026-07-09T00:00:00.000Z","open":189.80,"high":191.20,"low":188.40,"close":190.60,"volume":47800000},
  {"date":"2026-07-10T00:00:00.000Z","open":190.70,"high":192.30,"low":189.90,"close":191.85,"volume":50900000},
  {"date":"2026-07-13T00:00:00.000Z","open":191.90,"high":193.40,"low":190.80,"close":192.70,"volume":48400000}
])";

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--selftest") {
        std::cout << "Running self-test on a literal Tiingo-shaped JSON payload "
                     "(no network, no API key needed)...\n";
        std::string parseError;
        OhlcvSeries series = parseTiingoDailyJson(kSelfTestJson, parseError);
        if (!parseError.empty()) {
            std::cerr << "Self-test FAILED: " << parseError << "\n";
            return 1;
        }
        std::cout << "Parsed " << series.close.size() << " bars from JSON successfully.\n";
        printForecastSummary("SELFTEST", series);
        std::cout << "Self-test PASSED.\n";
        return 0;
    }

    if (argc > 2 && std::string(argv[1]) == "--curltest") {
        // Diagnostic-only path: proves the libcurl+TLS plumbing itself works
        // by hitting an arbitrary URL, independent of Tiingo/JSON parsing.
        // Useful if a real Tiingo fetch fails and you want to know whether
        // the problem is "no internet/TLS from this environment" vs.
        // "something specific to Tiingo/the API key".
        curl_global_init(CURL_GLOBAL_DEFAULT);
        try {
            const HttpResult r = httpGet(argv[2]);
            std::cout << "HTTP status: " << r.status << "\n";
            std::cout << "Body (first 300 chars):\n"
                      << r.body.substr(0, 300) << "\n";
        } catch (const std::exception& e) {
            std::cerr << "curltest FAILED: " << e.what() << "\n";
            curl_global_cleanup();
            return 1;
        }
        curl_global_cleanup();
        return 0;
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);

    const char* apiKeyEnv = std::getenv("TIINGO_API_KEY");
    if (!apiKeyEnv || trim(apiKeyEnv).empty()) {
        std::cerr << "TIINGO_API_KEY is not set. Run:\n"
                     "  export TIINGO_API_KEY=\"your_tiingo_key_here\"\n"
                     "and relaunch this program. Get a free key at https://www.tiingo.com/ .\n";
        curl_global_cleanup();
        return 1;
    }
    const std::string apiKey = trim(apiKeyEnv);

    std::cout << "GNC Forecast Core -- live Tiingo mode\n"
                 "Enter a stock ticker (e.g. AAPL), or 'quit' to exit.\n\n";

    std::string line;
    while (true) {
        std::cout << "Symbol> ";
        if (!std::getline(std::cin, line)) break;  // EOF (Ctrl+D)
        const std::string symbol = toUpper(trim(line));
        if (symbol.empty()) continue;
        if (symbol == "QUIT" || symbol == "EXIT") break;

        std::string fetchError;
        std::cout << "Fetching " << symbol << " from Tiingo...\n";
        OhlcvSeries series = fetchTiingoDaily(symbol, apiKey, /*lookbackDays=*/500, fetchError);

        if (!fetchError.empty()) {
            std::cerr << "\xE2\x9A\xA0\xEF\xB8\x8F  " << fetchError << "\n\n";
            continue;
        }
        printForecastSummary(symbol, series);
    }

    curl_global_cleanup();
    std::cout << "Goodbye.\n";
    return 0;
}
