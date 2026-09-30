# GNC forecast — C++ package (no MATLAB at runtime)

## What each binary does

| Binary | Needs | What it does |
|---|---|---|
| `gnc_forecast_demo` | nothing | Indicators/EKF/support-resistance on a CSV or synthetic data |
| `gnc_forecast_live` | libcurl, nlohmann-json, `TIINGO_API_KEY` | Prompts for a ticker, fetches live prices, prints the same indicator summary |
| **`gnc_forecast_full`** | above + **ONNX Runtime SDK** + `.onnx` models | Full pipeline: prices → 10 features → causal normalization → BiLSTM ensemble → 45-day path, optional live Claude sentiment, support/resistance/stop |

## Files

```
gnc_forecast_core.{hpp,cpp}   indicators, EKF+RTS smoother, support/resistance,
                              Returns/SectorRS/VPMRatio/RangeExp, rolling z-score
tiingo_client.{hpp,cpp}       Tiingo daily prices (libcurl + JSON)
llm_sentiment.{hpp,cpp}       Tiingo News fetch + Claude sentiment scoring
bilstm_onnx.{hpp,cpp}         ONNX Runtime ensemble inference (mean across runs)
full_pipeline_main.cpp        the integrated program
onnx_models/bilstm_run{1,2,3}.onnx   UNTRAINED proof models (see warning below)
python/feature_engineering.py        builds (X, Y) training tensors from Tiingo cache CSVs
python/build_bilstm_ensemble_onnx.py builds the untrained proof models
```

## Setup in the Codespace

```bash
# 1. libraries (needs sudo in Codespaces)
sudo apt-get install -y libcurl4-openssl-dev nlohmann-json3-dev

# 2. ONNX Runtime SDK (prebuilt; x86_64 shown -- check `uname -m`)
cd ~ && curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar xzf ort.tgz
export ONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.18.0

# 3. build (from the LLM_EKF_Predictor folder)
cmake -B build          # picks up ONNXRUNTIME_ROOT from the environment
cmake --build build
# or, without CMake:  make
```

If `cmake` prints "Skipping gnc_forecast_full", it tells you which piece it could not find.
If your machine is `aarch64`, download `onnxruntime-linux-aarch64-1.18.0.tgz` instead.

## Run

```bash
./build/gnc_forecast_full --selftest          # no network, no keys: proves the whole chain works
export TIINGO_API_KEY="<your real key>"
export ANTHROPIC_API_KEY="<optional, enables live sentiment>"
./build/gnc_forecast_full                     # interactive: type a ticker
./build/gnc_forecast_full --no-llm            # skip Claude even if the key is set
./build/gnc_forecast_full --models path/to/dir
```
Run from the project folder so the default `onnx_models/` path resolves.

## READ THIS: forecast validity

The bundled `.onnx` files have **random weights**. They prove the plumbing (architecture,
shapes, feature order, ensembling, C++/Python parity) — they do **not** forecast anything.
`gnc_forecast_full` prints a warning on every report until the model directory contains a file
named `TRAINED_MODEL.marker`. That marker should be written only by the Phase 2 training/export
step, once real trained weights are in place. **Phase 2 (PyTorch training on your historical data)
has not been built or run yet.**

## Ensemble math (bug fixed)

Part 5 of the original `.m` file computed `weights = (1./varRuns) ./ sum(1./varRuns, 1)` on an
already single-row vector. `sum(X,1)` on one row is a no-op, so every weight was exactly 1 and
`final45FrameReturns` was the raw **sum** of the 3 runs — exactly 3x the mean, on every run of the
script until this fix. Both the `.m` file and `bilstm_onnx.cpp` now use a plain mean across runs.
`BiLstmEnsembleResult::crossRunVariance` is kept as a diagnostic only.

## What was verified, and what was not

Verified in a Linux sandbox: clean builds (CMake and Makefile, `-Wall -Wextra`, zero warnings);
`--selftest` passes; Python and C++ feature pipelines agree to ~5e-8 on synthetic data
(including the MATLAB-style N-1 standard deviation); C++ and Python ONNX Runtime give matching
outputs on the same model; JSON parsing for Tiingo prices, Tiingo News, and the Claude response
against literal payloads shaped like the real APIs.

**Not verified:** any live call to `api.tiingo.com` from `gnc_forecast_full` or to the Claude API
(the sandbox cannot reach them and has no keys). `gnc_forecast_live` did work against live Tiingo
in your Codespace, and `gnc_forecast_full` reuses the same request pattern. The Claude request
shape (model id, prompt, response parsing) is untested against the real service, so treat the first
live sentiment run as a test and check the `LLM sentiment` label in the report. Sentiment is
not cached between queries yet (the `.m` file caches to `<TICKER>_LLM_Sentiment_cache.csv`).

## Differences from the MATLAB script (deliberate or pending)

- No sentiment cache file; each query calls Claude if `ANTHROPIC_API_KEY` is set.
- Without a key, sentiment is neutral 0, not MATLAB's synthetic sine+noise fallback (0 is what
  the Python training pipeline also uses, so train and inference agree).
- No plotting/table UI; results print to the terminal.
- No training: the C++ side is inference-only by design.
