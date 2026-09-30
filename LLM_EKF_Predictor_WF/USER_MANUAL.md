# GNC Forecast Pipeline — User Manual

A self-contained C++ rebuild of `LLM_EKF_Based_Predictor1.m`, running in a
GitHub Codespace / VS Code, with no MATLAB required at runtime.

---

## 1. What's in this folder

| File / folder | What it is |
|---|---|
| `gnc_forecast_core.{hpp,cpp}` | Indicators, EKF+RTS smoother, support/resistance, feature formulas |
| `tiingo_client.{hpp,cpp}` | Live Tiingo price fetch |
| `llm_sentiment.{hpp,cpp}` | Tiingo News fetch + Claude sentiment scoring |
| `bilstm_onnx.{hpp,cpp}` | ONNX Runtime ensemble inference (BiLSTM) |
| `full_pipeline_main.cpp` | The integrated program — builds `gnc_forecast_full` |
| `demo_main.cpp` / `interactive_main.cpp` | Earlier, simpler binaries (`gnc_forecast_demo`, `gnc_forecast_live`) — still work, kept for comparison |
| `onnx_models/bilstm_run{1,2,3}.onnx` | The BiLSTM ensemble — **currently UNTRAINED, random weights** (see §5) |
| `python/feature_engineering.py` | Builds training data from Tiingo cache CSVs (for Phase 2 training, not yet run) |
| `python/build_bilstm_ensemble_onnx.py` | Builds the untrained proof `.onnx` files above |
| `python/plot_forecast.py` | **New** — turns a saved forecast into a PNG chart |
| `output/` | Where forecasts land: one `<SYMBOL>_forecast.json` + `<SYMBOL>_forecast_plot.png` per run |
| `CMakeLists.txt`, `Makefile` | Two ways to build; use whichever you prefer |

Three binaries come out of a build:
- **`gnc_forecast_demo`** — no network, no keys. Indicators on a CSV or synthetic data.
- **`gnc_forecast_live`** — live Tiingo prices only, no BiLSTM.
- **`gnc_forecast_full`** — the real thing: live prices → features → BiLSTM ensemble → forecast → plot data.

---

## 2. One-time setup

```bash
# Libraries
sudo apt-get install -y libcurl4-openssl-dev nlohmann-json3-dev

# ONNX Runtime SDK (check `uname -m` first — use aarch64 build if that's what you get)
cd ~
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar xzf ort.tgz
export ONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.18.0
echo 'export ONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.18.0' >> ~/.bashrc

# Python plotting dependency
pip install --user matplotlib
```

Then, from this project folder:

```bash
cmake -B build -DONNXRUNTIME_ROOT=$HOME/onnxruntime-linux-x64-1.18.0
cmake --build build
```

If the configure step ends with a line saying "Skipping gnc_forecast_full",
the next line tells you what's missing — usually one of the two steps above.

---

## 3. Day-to-day use

**Every new terminal**, set your keys first (they don't persist across
sessions unless you use Codespaces secrets):

```bash
export TIINGO_API_KEY="<your real key>"
export ANTHROPIC_API_KEY="<your real key>"   # optional — enables live sentiment
```

**Run it:**

```bash
./build/gnc_forecast_full
```

At the `Symbol>` prompt, type a ticker (e.g. `INTU`) and press Enter.
Type `quit` to exit. Useful flags:

| Flag | Effect |
|---|---|
| `--selftest` | Runs on synthetic data — no network, no keys needed. Do this first after any rebuild. |
| `--no-llm` | Skips the Claude sentiment call even if the key is set |
| `--models DIR` | Use a different folder of `.onnx` files |
| `--out-dir DIR` | Where to save the forecast JSON/plot (default: `output/`) |

Every run prints a report to the terminal **and** saves
`output/<SYMBOL>_forecast.json`, which is what the plotter reads.

---

## 4. Generating a plot

```bash
python3 python/plot_forecast.py output/INTU_forecast.json
```

This writes `output/INTU_forecast_plot.png` — history, the 45-day forecast
path, resistance/support/stop-loss lines, and a metrics panel, in the same
spirit as the MATLAB chart.

**To view it in VS Code:** click the file in the Explorer sidebar
(`output/INTU_forecast_plot.png`) — it opens as an image tab, no extra
extension needed, works the same in a Codespace as on desktop VS Code.

**Plot several at once:**
```bash
python3 python/plot_forecast.py output/*_forecast.json
```

**Custom output path:**
```bash
python3 python/plot_forecast.py output/INTU_forecast.json --out ~/Desktop/intu.png
```

---

## 5. Read this before trusting any forecast number

**The `.onnx` models are untrained (random weights).** They prove the
pipeline works end-to-end — real features, real normalization, real ONNX
inference, real ensembling — but the forecast values themselves are
meaningless. Every report and every plot says so with a visible warning
until a file named `TRAINED_MODEL.marker` exists in the model folder.
That marker gets written only by the (not-yet-built) training step, and
only once a trained model beats a naive baseline in validation.

**What's already fixed and verified:**
- The original MATLAB ensemble-weighting formula had a bug that inflated
  every 45-day forecast by exactly 3× (see the `.m` file's Part 5 comment
  and this project's chat history). Fixed in both the `.m` file and here.
- Feature math (Returns, RSI, SectorRS, VPMRatio, RangeExp, EKF, Parkinson
  vol) is cross-checked between the Python and C++ implementations to
  ~5e-8 agreement.
- The BiLSTM architecture (stacked 128→64 bidirectional LSTM, 45-frame
  output) is confirmed correct via ONNX Runtime, matching Python
  reference output exactly.

**What's still open, found while planning the training step:**
- The EKF's backward RTS smoothing pass currently runs over the *entire*
  price history before training windows are cut — meaning the smoothed
  values at any given day have technically "seen" future days, including
  the 45-day window the model is supposed to predict. A random-walk test
  showed strong spurious correlation (up to 0.6) between these features
  and next-day returns purely from this leakage, not from any real
  predictive signal. This needs to be fixed in the training-data pipeline
  before training runs, or the model would validate far better than it
  can ever perform live.
- Live sentiment (Claude) can't be trained on — Tiingo's News API isn't
  available on the current plan (confirmed via a live 403), so there's no
  historical sentiment to train against. Plan: train with sentiment fixed
  at neutral, and treat live sentiment as informational only (run with
  `--no-llm`, or read the sentiment line as a side note) until a real
  historical sentiment source exists.
- Training basket agreed so far: **INTU, BSX, ADBE**.

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `cmake: command not found` | `sudo apt-get install cmake`, or just use `make` instead |
| Configure says "Skipping gnc_forecast_full" | `ONNXRUNTIME_ROOT` isn't set/correct, or `libcurl4-openssl-dev`/`nlohmann-json3-dev` aren't installed — the very next line names which |
| `TIINGO_API_KEY is not set` | You need to `export TIINGO_API_KEY="..."` in *this* terminal — it doesn't carry over between terminal tabs/sessions |
| `Tiingo rejected the API key (HTTP 401/403)` on **prices** | The key itself is wrong — check for stray text (placeholder wording, quotes, brackets) actually inside the `export` line |
| `Tiingo rejected the API key for News API (HTTP 403)` | Prices still work, but your Tiingo plan doesn't include News — this is expected right now, use `--no-llm` |
| Claude sentiment says `NEUTRAL 0 (...)` with a reason | Read the reason in that same line — it tells you exactly which step failed |
| Segfault right after rebuilding one file | Stale `.o` files compiled against an old header — `rm -rf build` and rebuild fully |
| `No such file or directory` running `./build/gnc_forecast_full` | The build didn't actually produce it — re-check the `cmake --build build` output for the skip message above |
| Plot script: `ModuleNotFoundError: matplotlib` | `pip install --user matplotlib` |

**Golden rule when pasting a command with a placeholder** (`<your key
here>`, `abc123examplekey`, etc.): replace the *entire* placeholder,
brackets included, with your real value. Several earlier sessions were
lost to a placeholder being copied in literally instead of replaced.

**Never paste a real API key into this chat.** If one is ever exposed
here (even partially, even in a screenshot), regenerate it from that
service's console before using it again.

---

## 7. What's next

1. Fix the look-ahead leakage in the EKF feature pipeline (§5) before any
   training run — this is the current blocker.
2. Build and run the PyTorch training script for INTU/BSX/ADBE, with a
   validation gate against a naive baseline before writing
   `TRAINED_MODEL.marker`.
3. Re-evaluate live sentiment once a usable historical source exists.
