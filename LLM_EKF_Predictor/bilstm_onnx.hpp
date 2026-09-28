// bilstm_onnx.hpp
//
// C++ inference engine for the BiLSTM ensemble, consuming .onnx files
// produced by Phase 2 training (or, for now, the architecturally-correct
// but untrained proof models from build_bilstm_ensemble_onnx.py -- see
// the README for what's proven vs. still pending).
//
// Mirrors Part 5 of the .m file EXACTLY:
//   - N independently trained models (numTrainRuns = 3 in the .m file)
//   - each produces one (horizonAhead=45)-length return-vector prediction
//     on the SAME latest lookback window
//   - per-frame (per-of-the-45-positions) inverse-variance weighting
//     across the N runs, NOT a simple average
//   - final45FrameReturns = weighted sum across runs, per frame
//
// Requires the ONNX Runtime C++ SDK (onnxruntime_cxx_api.h + libonnxruntime)
// -- same one used for the Phase 0 proof.
#pragma once
#include <string>
#include <vector>

namespace gnc_forecast {

struct BiLstmEnsembleResult {
    std::vector<double> finalReturns;      // final45FrameReturns, length horizonAhead
    std::vector<double> perRunPredictions_flat;  // numRuns * horizonAhead, row-major, for diagnostics
    std::vector<double> crossRunVariance;  // per-frame variance across runs, length horizonAhead --
                                           // diagnostic only (how much the runs disagree at each
                                           // frame); does NOT drive the weighting (see .cpp comment
                                           // on the ensemble-weighting bug fix).
    int numRuns;
    int horizonAhead;
};

class BiLstmOnnxEnsemble {
public:
    // modelPaths: one .onnx file per ensemble run (e.g. 3 paths, matching
    // the .m file's numTrainRuns = 3). All must share the same input shape
    // [seqLen, 1, numFeatures] and output shape [1, horizonAhead].
    explicit BiLstmOnnxEnsemble(const std::vector<std::string>& modelPaths);
    ~BiLstmOnnxEnsemble();

    // latestWindowRowMajor: the most recent `seqLen` rows of the CAUSALLY
    // NORMALIZED feature matrix (rollingZScoreNormalize's output), each row
    // `numFeatures` long, oldest day first -- i.e. exactly
    // featureNorm(end-lookback+1:end, :) from the .m file, flattened
    // row-major (row 0 = oldest day in the window).
    BiLstmEnsembleResult predict(const std::vector<double>& latestWindowRowMajor,
                                 int seqLen, int numFeatures) const;

private:
    struct Impl;
    Impl* impl_;  // pimpl to keep onnxruntime_cxx_api.h out of this public header
};

}  // namespace gnc_forecast
