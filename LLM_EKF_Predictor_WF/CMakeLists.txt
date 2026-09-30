cmake_minimum_required(VERSION 3.16)
project(gnc_forecast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- numeric core (no external deps) -----------------------------------------
add_library(gnc_forecast_core STATIC gnc_forecast_core.cpp)
target_include_directories(gnc_forecast_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(gnc_forecast_demo demo_main.cpp)
target_link_libraries(gnc_forecast_demo PRIVATE gnc_forecast_core)

# --- optional: libcurl + nlohmann_json ---------------------------------------
find_package(CURL)
find_package(nlohmann_json QUIET)

if(CURL_FOUND AND nlohmann_json_FOUND)
  add_executable(gnc_forecast_live interactive_main.cpp)
  target_link_libraries(gnc_forecast_live PRIVATE gnc_forecast_core CURL::libcurl nlohmann_json::nlohmann_json)

  add_library(gnc_net STATIC tiingo_client.cpp llm_sentiment.cpp)
  target_include_directories(gnc_net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(gnc_net PUBLIC CURL::libcurl nlohmann_json::nlohmann_json)
else()
  message(STATUS "Skipping gnc_forecast_live / gnc_net -- need libcurl + nlohmann_json dev packages:")
  message(STATUS "  sudo apt-get install -y libcurl4-openssl-dev nlohmann-json3-dev")
endif()

# --- optional: ONNX Runtime (BiLSTM inference) --------------------------------
# Point ONNXRUNTIME_ROOT at an unpacked onnxruntime-linux-x64-<ver> folder:
#   cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.18.0
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Unpacked ONNX Runtime SDK folder")

if(TARGET gnc_net AND ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  find_library(ORT_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
  if(ORT_LIB)
    add_library(gnc_bilstm STATIC bilstm_onnx.cpp)
    target_include_directories(gnc_bilstm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "${ONNXRUNTIME_ROOT}/include")
    target_link_libraries(gnc_bilstm PUBLIC "${ORT_LIB}")

    add_executable(gnc_forecast_full full_pipeline_main.cpp)
    target_link_libraries(gnc_forecast_full PRIVATE gnc_forecast_core gnc_net gnc_bilstm)
    set_target_properties(gnc_forecast_full PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")

    # Make the proof models available next to the binary for --selftest.
    add_custom_command(TARGET gnc_forecast_full POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              ${CMAKE_CURRENT_SOURCE_DIR}/onnx_models $<TARGET_FILE_DIR:gnc_forecast_full>/onnx_models)
  else()
    message(STATUS "ONNXRUNTIME_ROOT set but libonnxruntime not found in ${ONNXRUNTIME_ROOT}/lib")
  endif()
else()
  message(STATUS "Skipping gnc_forecast_full -- set ONNXRUNTIME_ROOT (see README, 'ONNX Runtime setup').")
endif()

# Usage (from this folder):
#   cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.18.0
#   cmake --build build
#   ./build/gnc_forecast_full --selftest
#   export TIINGO_API_KEY="..." ; ./build/gnc_forecast_full
