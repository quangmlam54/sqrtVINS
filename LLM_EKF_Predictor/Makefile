CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
ONNXRUNTIME_ROOT ?= $(HOME)/onnxruntime-linux-x64-1.18.0
CURL_CFLAGS := $(shell pkg-config --cflags libcurl 2>/dev/null)
CURL_LIBS   := $(shell pkg-config --libs libcurl 2>/dev/null)

all: gnc_forecast_demo gnc_forecast_live gnc_forecast_full

gnc_forecast_demo: gnc_forecast_core.cpp demo_main.cpp gnc_forecast_core.hpp
	$(CXX) $(CXXFLAGS) -o $@ gnc_forecast_core.cpp demo_main.cpp

gnc_forecast_live: gnc_forecast_core.cpp interactive_main.cpp gnc_forecast_core.hpp
	$(CXX) $(CXXFLAGS) $(CURL_CFLAGS) -o $@ gnc_forecast_core.cpp interactive_main.cpp $(CURL_LIBS)

# Full pipeline: needs libcurl, nlohmann-json, and the ONNX Runtime SDK.
gnc_forecast_full: gnc_forecast_core.cpp tiingo_client.cpp llm_sentiment.cpp bilstm_onnx.cpp full_pipeline_main.cpp \
                   gnc_forecast_core.hpp tiingo_client.hpp llm_sentiment.hpp bilstm_onnx.hpp
	$(CXX) $(CXXFLAGS) $(CURL_CFLAGS) -I$(ONNXRUNTIME_ROOT)/include -o $@ \
		gnc_forecast_core.cpp tiingo_client.cpp llm_sentiment.cpp bilstm_onnx.cpp full_pipeline_main.cpp \
		-L$(ONNXRUNTIME_ROOT)/lib -Wl,-rpath,$(ONNXRUNTIME_ROOT)/lib -lonnxruntime $(CURL_LIBS)

clean:
	rm -f gnc_forecast_demo gnc_forecast_live gnc_forecast_full

.PHONY: all clean
