// tiingo_client.hpp -- live Tiingo daily-price fetch (shared by the full pipeline).
#pragma once
#include <string>
#include <vector>

namespace gnc_forecast {

struct PriceSeries {
    std::vector<long long> epochDays;  // days since 1970-01-01, ascending
    std::vector<std::string> isoDates; // "YYYY-MM-DD", same order
    std::vector<double> open, high, low, close, volume;
};

// Parses a Tiingo /tiingo/daily/{symbol}/prices JSON array. Split from the
// network call so it can be self-tested on a literal payload.
PriceSeries parseTiingoDailyJson(const std::string& body, std::string& errorOut);

// Fetches ~lookbackDays calendar days of daily bars. On failure returns an
// empty series and sets errorOut to a specific, user-readable message.
PriceSeries fetchTiingoDaily(const std::string& symbol, const std::string& apiKey,
                             int lookbackDays, std::string& errorOut);

// "YYYY-MM-DD" -> days since epoch (UTC). Returns -1 on malformed input.
long long isoDateToEpochDays(const std::string& iso);

// days since epoch (UTC) -> "YYYY-MM-DD".
std::string epochDaysToIso(long long days);

}  // namespace gnc_forecast
