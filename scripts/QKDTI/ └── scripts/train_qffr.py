import numpy as np
import pandas as pd
import pennylane as qml
import time
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, mean_absolute_percentage_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tdc.multi_pred import DTI
from tdc.utils import convert_to_log

# Quantum setup
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qffr_circuit(weights, features):
    for i in range(n_qubits):
        qml.RY(features[i % len(features)], wires=i)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def compute_quantum_features(X):
    weights = np.random.uniform(0, np.pi, (2, n_qubits, 3))
    return np.array([qffr_circuit(weights, x) for x in X])

def process_dataset():
    data = DTI(name="DAVIS")
    split = data.get_split()
    split["train"]["Y"] = convert_to_log(split["train"]["Y"])
    split["test"]["Y"] = convert_to_log(split["test"]["Y"])

    X_train = np.random.rand(500, 128)
    X_test = np.random.rand(200, 128)
    y_train = split["train"]["Y"].values[:500]
    y_test = split["test"]["Y"].values[:200]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_qffr():
    X_train, X_test, y_train, y_test = process_dataset()

    print("\n🚀 Training Quantum Feature Fusion Regression (QFFR)...")
    start_time = time.time()

    X_train_q = compute_quantum_features(X_train)
    X_test_q = compute_quantum_features(X_test)

    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train_q, y_train)
    y_pred = model.predict(X_test_q)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    accuracy = 100 - mean_absolute_percentage_error(y_test, y_pred) * 100
    try:
        y_test_bin = (y_test > np.median(y_test)).astype(int)
        auc_roc = roc_auc_score(y_test_bin, y_pred)
    except:
        auc_roc = None

    elapsed_time = time.time() - start_time

    results = pd.DataFrame([{
        "Model": "Quantum FFR",
        "MSE": mse,
        "RMSE": rmse,
        "Accuracy": accuracy,
        "R2": r2,
        "Pearson_r": pearson_corr,
        "AUC_ROC": auc_roc if auc_roc else "N/A",
        "Time (sec)": elapsed_time
    }])

    results.to_csv("results/davis_qffr_results.csv", index=False)
    print("\n✅ QFFR Results saved to results/davis_qffr_results.csv")
    print(results)

if __name__ == "__main__":
    train_qffr()
