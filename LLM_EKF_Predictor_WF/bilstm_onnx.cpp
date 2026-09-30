// bilstm_onnx.cpp
#include "bilstm_onnx.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace gnc_forecast {

struct BiLstmOnnxEnsemble::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "bilstm_ensemble"};
    std::vector<Ort::Session> sessions;

    explicit Impl(const std::vector<std::string>& modelPaths) {
        Ort::SessionOptions opts;
        for (const auto& path : modelPaths) {
            sessions.emplace_back(env, path.c_str(), opts);
        }
    }
};

BiLstmOnnxEnsemble::BiLstmOnnxEnsemble(const std::vector<std::string>& modelPaths)
    : impl_(new Impl(modelPaths)) {
    if (modelPaths.empty()) {
        throw std::invalid_argument("BiLstmOnnxEnsemble: need at least one model path");
    }
}

BiLstmOnnxEnsemble::~BiLstmOnnxEnsemble() { delete impl_; }

BiLstmEnsembleResult BiLstmOnnxEnsemble::predict(const std::vector<double>& latestWindowRowMajor,
                                                 int seqLen, int numFeatures) const {
    if (static_cast<int>(latestWindowRowMajor.size()) != seqLen * numFeatures) {
        throw std::invalid_argument("BiLstmOnnxEnsemble::predict: window size mismatch");
    }

    // ONNX Runtime wants float32, shape [seqLen, batch=1, numFeatures] --
    // matches the .m file's {latestBlock} cell input to predict().
    std::vector<float> inputData(latestWindowRowMajor.begin(), latestWindowRowMajor.end());
    std::vector<int64_t> inputShape = {seqLen, 1, numFeatures};

    const int numRuns = static_cast<int>(impl_->sessions.size());
    int horizonAhead = -1;
    std::vector<std::vector<double>> perRun(numRuns);

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* inputNames[] = {"X"};
    const char* outputNames[] = {"Y"};

    for (int r = 0; r < numRuns; ++r) {
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());

        auto outputs = impl_->sessions[r].Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                                              outputNames, 1);
        const auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const int thisHorizon = static_cast<int>(outShape.back());
        if (horizonAhead == -1) {
            horizonAhead = thisHorizon;
        } else if (thisHorizon != horizonAhead) {
            throw std::runtime_error("BiLstmOnnxEnsemble::predict: ensemble members disagree on horizonAhead");
        }

        const float* outData = outputs[0].GetTensorData<float>();
        perRun[r].assign(outData, outData + thisHorizon);
    }

    // Per-frame mean across runs, plus cross-run variance kept ONLY as a
    // diagnostic (how much the ensemble members disagree at each frame).
    //
    // FIX: this used to attempt inverse-variance weighting matching the
    // .m file's Part 5 formula literally -- but that MATLAB formula had a
    // dimension bug (sum(1./varRuns,1) is a no-op on an already-1-row
    // vector), which collapsed the weights to 1.0 for every frame and
    // made the "weighted" result a raw SUM across runs (~numRuns-x too
    // large), not a real average. Per the 2026-09 discussion, both this
    // C++ path and the corrected .m file now use a straightforward mean
    // across runs instead.
    std::vector<double> meanPred(horizonAhead, 0.0);
    for (int h = 0; h < horizonAhead; ++h) {
        double sum = 0.0;
        for (int r = 0; r < numRuns; ++r) sum += perRun[r][h];
        meanPred[h] = sum / numRuns;
    }

    std::vector<double> crossRunVariance(horizonAhead, 0.0);
    for (int h = 0; h < horizonAhead; ++h) {
        double sumSq = 0.0;
        for (int r = 0; r < numRuns; ++r) {
            const double d = perRun[r][h] - meanPred[h];
            sumSq += d * d;
        }
        crossRunVariance[h] = sumSq / std::max(numRuns - 1, 1);  // sample variance, diagnostic only
    }

    const std::vector<double>& finalReturns = meanPred;

    BiLstmEnsembleResult result;
    result.finalReturns = finalReturns;
    result.crossRunVariance = std::move(crossRunVariance);
    result.numRuns = numRuns;
    result.horizonAhead = horizonAhead;
    result.perRunPredictions_flat.reserve(static_cast<size_t>(numRuns) * horizonAhead);
    for (int r = 0; r < numRuns; ++r) {
        for (int h = 0; h < horizonAhead; ++h) {
            result.perRunPredictions_flat.push_back(perRun[r][h]);
        }
    }
    return result;
}

}  // namespace gnc_forecast
