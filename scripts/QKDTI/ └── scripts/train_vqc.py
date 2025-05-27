import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tdc.multi_pred import DTI
from tdc.utils import convert_to_log

# Quantum device and parameters
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def variational_classifier(weights, x):
    qml.templates.AngleEmbedding(x, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def compute_quantum_features(X, weights):
    X_reduced = X[:, :n_qubits] if X.shape[1] > n_qubits else X
    return np.array([variational_classifier(weights, x) for x in X_reduced])

def process_dataset(name="DAVIS"):
    data = DTI(name=name)
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

def train_vqc():
    X_train, X_test, y_train, y_test = process_dataset()
    weights = np.random.uniform(0, np.pi, (3, n_qubits))

    X_train_q = compute_quantum_features(X_train, weights)
    X_test_q = compute_quantum_features(X_test, weights)

    model = LinearRegression()
    model.fit(X_train_q, y_train)
    y_pred = model.predict(X_test_q)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    accuracy = 100 - np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    try:
        y_pred_class = (y_pred > np.median(y_pred)).astype(int)
        y_test_class = (y_test > np.median(y_test)).astype(int)
        auc = roc_auc_score(y_test_class, y_pred)
    except ValueError:
        auc = None

    results = pd.DataFrame([{
        "Model": "VQC",
        "MSE": mse,
        "RMSE": rmse,
        "Accuracy": accuracy,
        "R2": r2,
        "Pearson_r": pearson_corr,
        "AUC_ROC": auc if auc else "N/A"
    }])

    results.to_csv("results/davis_vqc_results.csv", index=False)
    print("\n✅ VQC Results saved to results/davis_vqc_results.csv")
    print(results)

if __name__ == "__main__":
    train_vqc()
