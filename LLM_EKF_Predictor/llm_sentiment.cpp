// llm_sentiment.cpp
#include "llm_sentiment.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <stdexcept>

using json = nlohmann::json;

namespace gnc_forecast {
namespace {

size_t curlWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buffer = static_cast<std::string*>(userdata);
    buffer->append(ptr, size * nmemb);
    return size * nmemb;
}

struct HttpResult { long status; std::string body; };

HttpResult httpRequest(const std::string& url, const std::string& method,
                       const std::vector<std::string>& headers, const std::string& body) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("libcurl initialization failed");

    std::string responseBody;
    char errorBuf[CURL_ERROR_SIZE] = {0};
    struct curl_slist* headerList = nullptr;
    for (const auto& h : headers) headerList = curl_slist_append(headerList, h.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "gnc_forecast_live/1.0");
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuf);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerList);
    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    }

    const CURLcode res = curl_easy_perform(curl);
    if (headerList) curl_slist_free_all(headerList);
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

std::string urlEncode(const std::string& s) {
    CURL* curl = curl_easy_init();
    char* out = curl_easy_escape(curl, s.c_str(), static_cast<int>(s.length()));
    std::string result(out);
    curl_free(out);
    curl_easy_cleanup(curl);
    return result;
}

}  // namespace

std::vector<NewsItem> parseTiingoNewsJson(const std::string& body, std::string& errorOut) {
    errorOut.clear();
    json parsed;
    try {
        parsed = json::parse(body);
    } catch (const std::exception& e) {
        errorOut = std::string("Failed to parse Tiingo News JSON: ") + e.what();
        return {};
    }
    if (!parsed.is_array()) {
        errorOut = "Tiingo News response was not a JSON array.";
        return {};
    }

    std::vector<NewsItem> items;
    items.reserve(parsed.size());
    for (const auto& item : parsed) {
        NewsItem n;
        // Tiingo News uses "publishedDate" (ISO datetime, e.g.
        // "2026-09-20T12:00:00.000Z") -- keep just the date portion,
        // matching the .m file's day-level LLM_Sentiment alignment.
        if (item.contains("publishedDate") && item["publishedDate"].is_string()) {
            const std::string full = item["publishedDate"].get<std::string>();
            n.date = full.substr(0, 10);
        } else {
            continue;  // skip items with no usable date, mirrors NaN-mask filtering elsewhere
        }
        n.title = item.value("title", "");
        n.description = item.value("description", "");
        items.push_back(std::move(n));
    }
    return items;
}

std::vector<SentimentScore> parseClaudeSentimentResponse(const std::string& body, std::string& errorOut) {
    errorOut.clear();
    json parsed;
    try {
        parsed = json::parse(body);
    } catch (const std::exception& e) {
        errorOut = std::string("Failed to parse Claude response JSON: ") + e.what();
        return {};
    }

    if (!parsed.contains("content") || !parsed["content"].is_array() || parsed["content"].empty()) {
        errorOut = "Claude response had no content block.";
        return {};
    }
    std::string text;
    for (const auto& block : parsed["content"]) {
        if (block.value("type", "") == "text") {
            text += block.value("text", "");
        }
    }
    if (text.empty()) {
        errorOut = "Claude response had no text content block.";
        return {};
    }

    json scoresJson;
    try {
        scoresJson = json::parse(text);
    } catch (const std::exception& e) {
        errorOut = std::string("Failed to parse Claude's JSON reply: ") + e.what() +
                   " -- raw text: " + text.substr(0, 200);
        return {};
    }
    if (!scoresJson.is_array()) {
        errorOut = "Claude's reply JSON was not an array.";
        return {};
    }

    std::vector<SentimentScore> scores;
    scores.reserve(scoresJson.size());
    for (const auto& item : scoresJson) {
        SentimentScore s;
        s.date = item.value("date", "");
        s.score = item.value("score", 0.0);
        scores.push_back(s);
    }
    return scores;
}

std::vector<NewsItem> fetchTiingoNews(const std::string& symbol, const std::string& tiingoApiKey,
                                       int limitCount, std::string& errorOut) {
    errorOut.clear();
    const std::string url =
        "https://api.tiingo.com/tiingo/news?tickers=" + urlEncode(symbol) +
        "&limit=" + std::to_string(limitCount) +
        "&token=" + tiingoApiKey;

    HttpResult resp;
    try {
        resp = httpRequest(url, "GET", {}, "");
    } catch (const std::exception& e) {
        errorOut = std::string("Network error fetching Tiingo News: ") + e.what();
        return {};
    }
    if (resp.status == 401 || resp.status == 403) {
        errorOut = "Tiingo rejected the API key for News API (HTTP " + std::to_string(resp.status) + ").";
        return {};
    }
    if (resp.status != 200) {
        errorOut = "Tiingo News returned HTTP " + std::to_string(resp.status) + ": " + resp.body;
        return {};
    }
    return parseTiingoNewsJson(resp.body, errorOut);
}

std::vector<SentimentScore> scoreNewsSentimentWithClaude(const std::vector<NewsItem>& newsItems,
                                                         const std::string& claudeApiKey,
                                                         std::string& errorOut) {
    errorOut.clear();
    if (newsItems.empty()) return {};

    // Build one prompt covering the whole batch, asking Claude to return a
    // strict JSON array so parsing is deterministic -- one score per input
    // item, in the SAME order, matching the .m file's row-per-headline
    // llm_sentiment_score cache schema.
    json newsArray = json::array();
    for (const auto& n : newsItems) {
        newsArray.push_back({{"date", n.date}, {"title", n.title}, {"description", n.description}});
    }

    std::string promptText =
        "You will be given a JSON array of news items, each with a date, title, and description. "
        "For each item, assign a sentiment score in the range -1.0 (very bearish for the stock) to "
        "+1.0 (very bullish), 0.0 being neutral. Respond with ONLY a JSON array of objects, one per "
        "input item IN THE SAME ORDER, each shaped exactly as {\"date\": \"<the input date>\", "
        "\"score\": <number>}. No prose, no markdown fences, just the JSON array.\n\n"
        "Input:\n" + newsArray.dump();

    json requestBody = {
        {"model", "claude-haiku-4-5-20251001"},
        {"max_tokens", 2000},
        {"messages", json::array({json{{"role", "user"}, {"content", promptText}}})},
    };

    const std::vector<std::string> headers = {
        "content-type: application/json",
        "x-api-key: " + claudeApiKey,
        "anthropic-version: 2023-06-01",
    };

    HttpResult resp;
    try {
        resp = httpRequest("https://api.anthropic.com/v1/messages", "POST", headers, requestBody.dump());
    } catch (const std::exception& e) {
        errorOut = std::string("Network error calling Claude: ") + e.what();
        return {};
    }
    if (resp.status == 401) {
        errorOut = "Claude rejected the API key (HTTP 401). Check ANTHROPIC_API_KEY.";
        return {};
    }
    if (resp.status != 200) {
        errorOut = "Claude API returned HTTP " + std::to_string(resp.status) + ": " + resp.body;
        return {};
    }
    return parseClaudeSentimentResponse(resp.body, errorOut);
}

}  // namespace gnc_forecast
