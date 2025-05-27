QKDTI: Quantum Kernel-Based Machine Learning Model for Drug-Target Interaction Prediction
This repository contains all source code, scripts, data instructions, preprocessed results, and trained models used in the paper titled "QKDTI: A Quantum Kernel-Based Machine Learning Model for Drug-Target Interaction Prediction".
🧪 Overview
QKDTI integrates Nyström-approximated quantum kernels with support vector regression and hybrid quantum-classical architectures (Quantum Transformers, QNN-DNN) to predict drug-target binding affinity. The framework has been evaluated on three benchmark datasets: DAVIS, KIBA, and BindingDB.

📁 Repository Structure
QKDTI/
├── README.md                  # Project overview and usage guide
├── requirements.txt           # Python dependencies
├── scripts/                   # Core training and evaluation scripts
│   ├── train_qsvr.py          # Quantum SVR using Nyström method
│   ├── train_qtransformer.py  # Quantum Transformer model
│   ├── train_qnn_dnn.py       # QNN-based feature extraction with DNN regression
│   └── data_stats.py          # Dataset statistical profiling script
├── data/
│   └── instructions.md        # Guide to accessing datasets via TDC
├── results/                   # Precomputed results for reproducibility
│   ├── davis_qsvr_results.csv
│   ├── kiba_qtransformer_results.csv
│   └── bindingdb_qnn_dnn_results.csv
├── notebooks/
│   └── demo_QKDTI_pipeline.ipynb  # End-to-end demo notebook
└── saved_models/
    └── qkdti_model.pkl       # Serialized trained model (e.g., SVR)
