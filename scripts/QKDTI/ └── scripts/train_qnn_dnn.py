# QKDTI/scripts/train_qtransformer.py
"""
Script: train_qtransformer.py
Description: Train and evaluate a Quantum-Inspired Transformer model on TDC datasets (DAVIS).
Usage: python train_qtransformer.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Input, Reshape, Flatten
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tdc.multi_pred import DTI
from tdc.utils import convert_to_log

# Transformer block
class QuantumTransformerLayer(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(QuantumTransformerLayer, self).__init__()
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

# Load and preprocess dataset
def process_dataset():
    data = DTI(name="DAVIS")
    split = data.get_split()
    split["train"]["Y"] = convert_to_log(split["train"]["Y"])
    split["test"]["Y"] = convert_to_log(split["test"]["Y"])

    subset_size = 500
    X_train = np.random.rand(subset_size, 8)
    X_test = np.random.rand(200, 8)
    y_train = split["train"]["Y"].values[:subset_size]
    y_test = split["test"]["Y"].values[:200]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Build model
def build_transformer_model(input_dim=8):
    inputs = Input(shape=(input_dim,))
    reshaped = Reshape((1, input_dim))(inputs)
    x = QuantumTransformerLayer(embed_dim=input_dim, num_heads=4, ff_dim=64)(reshaped)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Train and evaluate
def train_qtransformer():
    X_train, X_test, y_train, y_test = process_dataset()
    model = build_transformer_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    accuracy = 100 - (np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    try:
        auc = roc_auc_score((y_test > np.median(y_test)).astype(int), y_pred)
    except:
        auc = None

    print(f"\n📊 Quantum Transformer on DAVIS:")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson r: {pearson_corr:.4f}, Accuracy: {accuracy:.2f}%, AUC-ROC: {auc if auc else 'N/A'}")

    results = pd.DataFrame([{
        "Model": "Quantum Transformer",
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Pearson_r": pearson_corr,
        "Accuracy": accuracy,
        "AUC_ROC": auc
    }])
    results.to_csv("results/davis_qtransformer_results.csv", index=False)
    print("\n✅ Results saved to results/davis_qtransformer_results.csv")

if __name__ == "__main__":
    train_qtransformer()
