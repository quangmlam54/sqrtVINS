// llm_sentiment.hpp
//
// C++ port of fetchTiingoNews + scoreNewsSentimentWithClaude from the .m
// file (Part 1.5-ish region -- the LLM_Sentiment feature-column source).
// Separate from bilstm_onnx.hpp entirely: sentiment scoring is a live API
// call (Tiingo News, then Claude), not something that gets "compiled into"
// a model the way the BiLSTM does. See the chat/README notes on why LLMs
// aren't exported to ONNX the way the BiLSTM is.
#pragma once
#include <string>
#include <vector>

namespace gnc_forecast {

struct NewsItem {
    std::string date;        // ISO date, e.g. "2026-09-20"
    std::string title;
    std::string description;
};

struct SentimentScore {
    std::string date;
    double score;  // matches the .m file's llm_sentiment_score convention:
                   // roughly [-1, +1], negative=bearish, positive=bullish
};

// Fetches recent news for `symbol` from Tiingo's News API
// (https://api.tiingo.com/tiingo/news). `apiKey` is the SAME Tiingo key
// used for price data. Throws std::runtime_error on transport failure;
// returns an empty vector (not an exception) for "no news found" so the
// caller can fall back gracefully, matching the .m file's own behavior.
std::vector<NewsItem> fetchTiingoNews(const std::string& symbol, const std::string& tiingoApiKey,
                                       int limitCount, std::string& errorOut);

// Sends `newsItems` to Claude for sentiment scoring (one score per
// headline+description). `claudeApiKey` should come from the
// ANTHROPIC_API_KEY environment variable, same as the .m file's
// getenv('ANTHROPIC_API_KEY') convention -- never hardcode it. Returns
// one SentimentScore per input NewsItem, in the same order.
std::vector<NewsItem> fetchTiingoNews(const std::string& symbol, const std::string& tiingoApiKey,
                                       int limitCount, std::string& errorOut);

// Parses a raw Tiingo News API JSON array (as returned in the response
// body of fetchTiingoNews's HTTP call) with no network access -- split
// out for self-testing against a literal payload.
std::vector<NewsItem> parseTiingoNewsJson(const std::string& body, std::string& errorOut);

std::vector<SentimentScore> scoreNewsSentimentWithClaude(const std::vector<NewsItem>& newsItems,
                                                         const std::string& claudeApiKey,
                                                         std::string& errorOut);

// Parses a raw Claude /v1/messages API JSON response body -- split out
// for self-testing against a literal payload with no network access.
std::vector<SentimentScore> parseClaudeSentimentResponse(const std::string& body, std::string& errorOut);

}  // namespace gnc_forecast
