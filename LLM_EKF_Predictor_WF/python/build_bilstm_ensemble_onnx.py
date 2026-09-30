"""
Builds ensemble .onnx files matching the .m file's netArch EXACTLY:

    sequenceInputLayer(10)
    bilstmLayer(128, OutputMode='sequence')   -- layer 1: full-sequence output
    dropoutLayer(0.30)                        -- inference no-op, omitted
    bilstmLayer(64,  OutputMode='last')       -- layer 2: last-step output only
    dropoutLayer(0.20)                        -- inference no-op, omitted
    fullyConnectedLayer(45)
    regressionLayer

Weights are RANDOM (different seed per ensemble member) -- this is the
Phase-0-style plumbing proof for the REAL two-stacked-BiLSTM architecture,
not a trained model. Phase 2 (actual PyTorch training on real historical
data) is what replaces these random weights with learned ones; nothing
else in the C++ inference code needs to change when that happens -- it
just needs to be pointed at the real .onnx files this same builder shape
would produce.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

SEQ_LEN = 10
NUM_FEATURES = 10
HIDDEN_1 = 128
HIDDEN_2 = 64
HORIZON = 45
BATCH = 1


def build_one(seed: int, out_path: str):
    rng = np.random.default_rng(seed)

    def lstm_weights(hidden, input_size):
        W = (rng.standard_normal((2, 4 * hidden, input_size)).astype(np.float32) * 0.08)
        R = (rng.standard_normal((2, 4 * hidden, hidden)).astype(np.float32) * 0.08)
        B = (rng.standard_normal((2, 8 * hidden)).astype(np.float32) * 0.08)
        return W, R, B

    W1, R1, B1 = lstm_weights(HIDDEN_1, NUM_FEATURES)
    W2, R2, B2 = lstm_weights(HIDDEN_2, 2 * HIDDEN_1)  # layer 2 consumes layer 1's bidirectional output
    Wfc = (rng.standard_normal((2 * HIDDEN_2, HORIZON)).astype(np.float32) * 0.08)
    Bfc = rng.standard_normal((HORIZON,)).astype(np.float32) * 0.08

    initializers = [
        numpy_helper.from_array(W1, name="W1"),
        numpy_helper.from_array(R1, name="R1"),
        numpy_helper.from_array(B1, name="B1"),
        numpy_helper.from_array(W2, name="W2"),
        numpy_helper.from_array(R2, name="R2"),
        numpy_helper.from_array(B2, name="B2"),
        numpy_helper.from_array(Wfc, name="Wfc"),
        numpy_helper.from_array(Bfc, name="Bfc"),
        numpy_helper.from_array(np.array([SEQ_LEN, BATCH, 2 * HIDDEN_1], dtype=np.int64), name="reshape1_shape"),
        numpy_helper.from_array(np.array([BATCH, 2 * HIDDEN_2], dtype=np.int64), name="reshape2_shape"),
    ]

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [SEQ_LEN, BATCH, NUM_FEATURES])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [BATCH, HORIZON])

    # Layer 1: bidirectional LSTM, full sequence output (Y1: [seq_len, 2, batch, hidden1])
    lstm1 = helper.make_node("LSTM", inputs=["X", "W1", "R1", "B1"],
                              outputs=["lstm1_Y", "lstm1_h", "lstm1_c"],
                              hidden_size=HIDDEN_1, direction="bidirectional")
    # [seq_len, 2, batch, hidden1] -> [seq_len, batch, 2, hidden1] -> reshape -> [seq_len, batch, 2*hidden1]
    transpose1 = helper.make_node("Transpose", inputs=["lstm1_Y"], outputs=["lstm1_Y_t"], perm=[0, 2, 1, 3])
    reshape1 = helper.make_node("Reshape", inputs=["lstm1_Y_t", "reshape1_shape"], outputs=["layer1_out"])

    # Layer 2: bidirectional LSTM, LAST-step output only (lstm2_h: [2, batch, hidden2])
    lstm2 = helper.make_node("LSTM", inputs=["layer1_out", "W2", "R2", "B2"],
                              outputs=["lstm2_Y", "lstm2_h", "lstm2_c"],
                              hidden_size=HIDDEN_2, direction="bidirectional")
    transpose2 = helper.make_node("Transpose", inputs=["lstm2_h"], outputs=["lstm2_h_t"], perm=[1, 0, 2])
    reshape2 = helper.make_node("Reshape", inputs=["lstm2_h_t", "reshape2_shape"], outputs=["layer2_out"])

    matmul = helper.make_node("MatMul", inputs=["layer2_out", "Wfc"], outputs=["fc_out"])
    add = helper.make_node("Add", inputs=["fc_out", "Bfc"], outputs=["Y"])

    graph = helper.make_graph(
        [lstm1, transpose1, reshape1, lstm2, transpose2, reshape2, matmul, add],
        "bilstm_multistep_ensemble_member",
        [X], [Y], initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="gnc_bilstm_onnx",
                               opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    for i, seed in enumerate([101, 202, 303], start=1):
        build_one(seed, f"onnx_models/bilstm_run{i}.onnx")
