#include "tiingo_client.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <ctime>
#include <stdexcept>

using json = nlohmann::json;

namespace gnc_forecast {
namespace {

size_t writeCb(char* ptr, size_t size, size_t nmemb, void* ud) {
    static_cast<std::string*>(ud)->append(ptr, size * nmemb);
    return size * nmemb;
}

struct Http { long status; std::string body; };

Http httpGet(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("libcurl initialization failed");
    std::string body;
    char err[CURL_ERROR_SIZE] = {0};
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "gnc_forecast_full/1.0");
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, err);
    const CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error(std::string("HTTP request failed: ") + (err[0] ? err : curl_easy_strerror(rc)));
    }
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(curl);
    return {status, body};
}

std::string isoFromOffset(int daysBack) {
    const std::time_t t = std::time(nullptr) - static_cast<std::time_t>(daysBack) * 86400;
    char buf[16];
    std::tm tmv{};
    gmtime_r(&t, &tmv);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tmv);
    return buf;
}

}  // namespace

long long isoDateToEpochDays(const std::string& iso) {
    if (iso.size() < 10) return -1;
    try {
        std::tm t{};
        t.tm_year = std::stoi(iso.substr(0, 4)) - 1900;
        t.tm_mon = std::stoi(iso.substr(5, 2)) - 1;
        t.tm_mday = std::stoi(iso.substr(8, 2));
        t.tm_hour = 12;
        return static_cast<long long>(timegm(&t) / 86400);
    } catch (...) {
        return -1;
    }
}

std::string epochDaysToIso(long long days) {
    const std::time_t t = static_cast<std::time_t>(days) * 86400;
    std::tm tmv{};
    gmtime_r(&t, &tmv);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tmv);
    return buf;
}

PriceSeries parseTiingoDailyJson(const std::string& body, std::string& errorOut) {
    errorOut.clear();
    PriceSeries s;
    json parsed;
    try {
        parsed = json::parse(body);
    } catch (const std::exception& e) {
        errorOut = std::string("Failed to parse Tiingo JSON: ") + e.what();
        return {};
    }
    if (!parsed.is_array() || parsed.empty()) {
        errorOut = "Tiingo returned no price rows.";
        return {};
    }
    for (const auto& bar : parsed) {
        const char* keys[] = {"date", "open", "high", "low", "close", "volume"};
        bool ok = true;
        for (const char* k : keys) if (!bar.contains(k) || bar[k].is_null()) ok = false;
        if (!ok) continue;  // skip malformed rows (mirrors the .m file's NaN-mask filtering)
        const std::string iso = bar["date"].get<std::string>().substr(0, 10);
        const long long ed = isoDateToEpochDays(iso);
        if (ed < 0) continue;
        s.epochDays.push_back(ed);
        s.isoDates.push_back(iso);
        s.open.push_back(bar["open"].get<double>());
        s.high.push_back(bar["high"].get<double>());
        s.low.push_back(bar["low"].get<double>());
        s.close.push_back(bar["close"].get<double>());
        s.volume.push_back(bar["volume"].get<double>());
    }
    if (s.close.empty()) errorOut = "No usable rows in Tiingo response.";
    return s;
}

PriceSeries fetchTiingoDaily(const std::string& symbol, const std::string& apiKey,
                             int lookbackDays, std::string& errorOut) {
    errorOut.clear();
    const std::string url = "https://api.tiingo.com/tiingo/daily/" + symbol + "/prices"
        "?startDate=" + isoFromOffset(lookbackDays) + "&endDate=" + isoFromOffset(0) +
        "&format=json&resampleFreq=daily&token=" + apiKey;
    Http r;
    try {
        r = httpGet(url);
    } catch (const std::exception& e) {
        errorOut = std::string("Network error: ") + e.what();
        return {};
    }
    if (r.status == 404) { errorOut = "Symbol '" + symbol + "' not found on Tiingo (404)."; return {}; }
    if (r.status == 401 || r.status == 403) {
        errorOut = "Tiingo rejected the API key (HTTP " + std::to_string(r.status) + "). Check TIINGO_API_KEY.";
        return {};
    }
    if (r.status != 200) { errorOut = "Tiingo returned HTTP " + std::to_string(r.status); return {}; }
    return parseTiingoDailyJson(r.body, errorOut);
}

}  // namespace gnc_forecast
