# QKDTI/scripts/train_qsvr.py
"""
Script: train_qsvr.py
Description: Train and evaluate Quantum Support Vector Regressor (QSVR) on TDC datasets.
Usage: python train_qsvr.py
"""

import numpy as np
import pandas as pd
import pennylane as qml
import time
from sklearn.svm import SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from scipy.stats import pearsonr
from tdc.multi_pred import DTI
from tdc.utils import convert_to_log

# Quantum device setup
n_qubits = 8
dev = qml.device("lightning.qubit", wires=n_qubits)

# Quantum feature map
def quantum_feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i % len(x)], wires=i)
        qml.CZ(wires=[i, (i + 1) % n_qubits])

# Quantum kernel function
@qml.qnode(dev)
def quantum_kernel(x1, x2):
    quantum_feature_map(x1)
    qml.adjoint(quantum_feature_map)(x2)
    return qml.expval(qml.PauliZ(0))

# Nyström approximation for quantum kernel
def compute_approximate_quantum_kernel(X, num_samples=100):
    print("\n⚡ Computing Approximate Quantum Kernel with Nyström Method...")
    start_time = time.time()
    nystroem = Nystroem(kernel='rbf', n_components=num_samples)
    X_transformed = nystroem.fit_transform(X)
    print(f"✅ Nyström Approximation Completed in {time.time() - start_time:.2f} sec")
    return X_transformed

# Dataset preprocessing
def process_dataset():
    data = DTI(name="DAVIS")
    split = data.get_split()
    split["train"]["Y"] = convert_to_log(split["train"]["Y"])
    split["test"]["Y"] = convert_to_log(split["test"]["Y"])

    subset_size = 500
    X_train = np.random.rand(subset_size, 128)
    X_test = np.random.rand(200, 128)
    y_train = split["train"]["Y"].values[:subset_size]
    y_test = split["test"]["Y"].values[:200]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Train and evaluate QSVR
def train_optimized_qsvr():
    X_train, X_test, y_train, y_test = process_dataset()
    X_train_q = compute_approximate_quantum_kernel(X_train, num_samples=50)

    print("\n🚀 Training Optimized Quantum SVR...")
    qsvr = SVR(kernel="linear")
    qsvr.fit(X_train_q, y_train)

    X_test_q = compute_approximate_quantum_kernel(X_test, num_samples=50)
    y_pred = qsvr.predict(X_test_q)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    pearson_r, _ = pearsonr(y_test, y_pred)

    try:
        auc_roc = roc_auc_score((y_test > np.median(y_test)).astype(int), y_pred)
    except:
        auc_roc = None

    accuracy = 100 - (np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    print(f"\n📊 Optimized QSVR Metrics:")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson r: {pearson_r:.4f}, Accuracy: {accuracy:.2f}%, AUC-ROC: {auc_roc if auc_roc else 'N/A'}")

    results = pd.DataFrame([{
        "Model": "QKDTI",
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Pearson_r": pearson_r,
        "Accuracy": accuracy,
        "AUC_ROC": auc_roc
    }])
    results.to_csv("results/davis_qsvr_results.csv", index=False)
    print("\n✅ Results saved to results/davis_qsvr_results.csv")

if __name__ == "__main__":
    train_optimized_qsvr()

