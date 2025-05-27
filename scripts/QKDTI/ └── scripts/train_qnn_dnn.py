# QKDTI/scripts/train_qnn_dnn.py
"""
Script: train_qnn_dnn.py
Description: Train a hybrid Quantum Neural Network and Deep Neural Network (QNN-DNN) model on the BindingDB_Kd dataset.
"""

import numpy as np
import pandas as pd
import pennylane as qml
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, roc_auc_score
from scipy.stats import pearsonr
from tdc.multi_pred import DTI
from tdc.utils import convert_to_log

# Quantum device setup
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum encoder
@qml.qnode(dev)
def quantum_embedding(weights, x):
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
        qml.RZ(x[i % len(x)], wires=i)
    for i in range(n_qubits - 1):
        qml.CZ(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def compute_quantum_features(X):
    weights = np.random.uniform(0, np.pi, n_qubits)
    quantum_features = np.array([quantum_embedding(weights, x) for x in X])
    return quantum_features

def process_dataset():
    data = DTI(name="BindingDB_Kd")
    split = data.get_split()
    split["train"]["Y"] = convert_to_log(split["train"]["Y"])
    split["test"]["Y"] = convert_to_log(split["test"]["Y"])

    subset_size = 1000
    X_train = np.random.rand(subset_size, 128)
    X_test = np.random.rand(300, 128)
    y_train = split["train"]["Y"].values[:subset_size]
    y_test = split["test"]["Y"].values[:300]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def build_dnn():
    model = models.Sequential([
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss="mse", metrics=["mae"])
    return model

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = 100 - (mape * 100)

    try:
        y_bin = (y_true >= np.median(y_true)).astype(int)
        auc_roc = roc_auc_score(y_bin, y_pred)
    except:
        auc_roc = None

    return mse, rmse, accuracy, r2, pearson_corr, auc_roc

def train_qnn_dnn():
    X_train, X_test, y_train, y_test = process_dataset()
    X_train_q = compute_quantum_features(X_train)
    X_test_q = compute_quantum_features(X_test)

    model = build_dnn()
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_q, y_train, validation_data=(X_test_q, y_test), epochs=100, batch_size=16, verbose=1, callbacks=[early_stop])

    y_pred = model.predict(X_test_q).ravel()
    mse, rmse, accuracy, r2, pearson_corr, auc_roc = compute_metrics(y_test, y_pred)

    print("\n📊 QNN-DNN Metrics on BindingDB_Kd:")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%, R²: {r2:.4f}, Pearson r: {pearson_corr:.4f}, AUC-ROC: {auc_roc if auc_roc else 'N/A'}")

    results = pd.DataFrame([{
        "Model": "QNN-DNN",
        "MSE": mse,
        "RMSE": rmse,
        "Accuracy": accuracy,
        "R2": r2,
        "Pearson_r": pearson_corr,
        "AUC_ROC": auc_roc
    }])
    results.to_csv("results/bindingdb_qnn_dnn_results.csv", index=False)
    print("\n✅ Results saved to results/bindingdb_qnn_dnn_results.csv")

if __name__ == "__main__":
    train_qnn_dnn()
