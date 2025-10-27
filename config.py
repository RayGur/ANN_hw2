"""
配置檔案

集中管理所有實驗參數和設置
"""

# ============================================================
# 資料集配置
# ============================================================

DATASETS = {
    "iris": {
        "name": "Iris",
        "path": "data/iris.txt",
        "input_dim": 4,
        "output_dim": 3,
        "n_samples": 150,
        "class_names": ["Setosa", "Versicolor", "Virginica"],
        "split_method": "kfold",
        "k_folds": 10,
    },
    "wine": {
        "name": "Wine",
        "path": "data/wine.txt",
        "input_dim": 13,
        "output_dim": 3,
        "n_samples": 178,
        "class_names": ["Class 1", "Class 2", "Class 3"],
        "split_method": "kfold",
        "k_folds": 10,
    },
    "breast_cancer": {
        "name": "Breast Cancer Wisconsin",
        "path": "data/breast-cancer-wisconsin.txt",
        "input_dim": 9,
        "output_dim": 2,
        "n_samples": 699,
        "class_names": ["Benign", "Malignant"],
        "split_method": "simple",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    },
}

# ============================================================
# 架構配置
# ============================================================

ARCHITECTURES = {
    "small": {
        "name": "Small Capacity",
        "hidden_layers": (2, 1),
        "description": "Minimal capacity architecture",
    },
    "recommended": {
        "name": "Recommended Capacity",
        "iris": (8, 4),
        "wine": (20, 10),
        "breast_cancer": (15, 8),
        "description": "Adaptive architecture based on input dimension",
    },
}

# ============================================================
# 訓練配置
# ============================================================

TRAINING_CONFIG = {
    # 基礎參數
    "max_epochs": 100000,
    "convergence_threshold": 1e-6,
    "convergence_patience": 50,
    "early_stopping_patience": 10,
    # 學習率
    "learning_rates": [0.05, 0.2, 0.5],
    # 激活函數
    "activations": ["sigmoid", "tanh", "relu"],
    # 權重初始化
    "weight_scale": 1.0,
    "random_seed": 42,
    # 顯示設置
    "verbose": True,
    "verbose_interval": 1000,
}

# ============================================================
# 訓練方法配置
# ============================================================

TRAINING_METHODS = {
    "standard_bp": {
        "name": "Standard Backpropagation",
        "trainer_class": "StandardBPTrainer",
        "params": {},
    },
    "momentum": {
        "name": "Momentum Method",
        "trainer_class": "MomentumTrainer",
        "params": {"momentum_values": [0.5, 0.7, 0.9]},
    },
    "rprop": {
        "name": "Resilient Propagation",
        "trainer_class": "ResilientPropTrainer",
        "params": {
            "eta_plus": 1.2,
            "eta_minus": 0.5,
            "delta_init": 0.1,
            "delta_max": 50.0,
            "delta_min": 1e-6,
        },
    },
    "levenberg_marquardt": {
        "name": "Levenberg-Marquardt",
        "trainer_class": "LevenbergMarquardtTrainer",
        "params": {"mu": 0.01, "mu_increase": 10, "mu_decrease": 0.1},
    },
}

# ============================================================
# 實驗配置
# ============================================================

EXPERIMENT_1_CONFIG = {
    "name": "Architecture Capacity Comparison",
    "description": "Compare small vs recommended architecture",
    "datasets": ["iris", "wine", "breast_cancer"],
    "architectures": ["small", "recommended"],
    "activations": TRAINING_CONFIG["activations"],
    "learning_rates": TRAINING_CONFIG["learning_rates"],
    "n_experiments": None,  # Will be calculated: 3 datasets × 2 arch × 3 act × 3 lr = 54
}

EXPERIMENT_2_CONFIG = {
    "name": "Training Methods Comparison",
    "description": "Compare different optimization methods",
    "use_best_config": True,  # Use best config from Experiment 1
    "training_methods": ["standard_bp", "momentum", "rprop", "levenberg_marquardt"],
    "momentum_values": TRAINING_METHODS["momentum"]["params"]["momentum_values"],
}

# ============================================================
# 輸出配置
# ============================================================

OUTPUT_CONFIG = {
    "results_dir": "results",
    "figures_dir": "results/figures",
    "logs_dir": "results/logs",
    "models_dir": "results/models",
    # 圖表設置
    "figure_format": "png",
    "figure_dpi": 300,
    "save_figures": True,
    "show_figures": False,
    # 日誌設置
    "save_logs": True,
    "log_format": ["json", "csv"],
    # 模型設置
    "save_models": True,
    "save_best_only": True,
}

# ============================================================
# 輔助函數
# ============================================================


def get_architecture(dataset_name, architecture_type):
    """
    獲取指定資料集和架構類型的隱藏層配置

    Parameters:
    -----------
    dataset_name : str
        資料集名稱
    architecture_type : str
        架構類型 ('small' or 'recommended')

    Returns:
    --------
    tuple
        隱藏層神經元數量
    """
    dataset = DATASETS[dataset_name]
    input_dim = dataset["input_dim"]
    output_dim = dataset["output_dim"]

    if architecture_type == "small":
        hidden_layers = ARCHITECTURES["small"]["hidden_layers"]
    elif architecture_type == "recommended":
        hidden_layers = ARCHITECTURES["recommended"][dataset_name]
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")

    # 完整架構: input → hidden1 → hidden2 → output
    full_architecture = [input_dim] + list(hidden_layers) + [output_dim]

    return full_architecture


def print_config_summary():
    """打印配置摘要"""
    print("=" * 70)
    print("Configuration Summary")
    print("=" * 70)

    print("\n[Datasets]")
    for key, dataset in DATASETS.items():
        print(
            f"  {dataset['name']:.<30} {dataset['n_samples']} samples, "
            f"{dataset['input_dim']} features, {dataset['output_dim']} classes"
        )

    print("\n[Architectures]")
    for key, arch in ARCHITECTURES.items():
        print(f"  {arch['name']:.<30} {arch['description']}")

    print("\n[Training Methods]")
    for key, method in TRAINING_METHODS.items():
        print(f"  {method['name']:.<30} {method['trainer_class']}")

    print("\n[Experiment 1]")
    exp1 = EXPERIMENT_1_CONFIG
    n_exp = (
        len(exp1["datasets"])
        * len(exp1["architectures"])
        * len(exp1["activations"])
        * len(exp1["learning_rates"])
    )
    print(f"  Total experiments: {n_exp}")

    print("\n[Experiment 2]")
    exp2 = EXPERIMENT_2_CONFIG
    print(f"  Training methods: {len(exp2['training_methods'])}")
    print(f"  Momentum values: {exp2['momentum_values']}")

    print("\n" + "=" * 70)


# ============================================================
# 測試程式碼
# ============================================================

if __name__ == "__main__":
    """
    測試配置
    """
    print_config_summary()

    print("\n[Test] Get architecture for Iris (recommended)")
    arch = get_architecture("iris", "recommended")
    print(f"  Architecture: {arch}")

    print("\n[Test] Get architecture for Wine (small)")
    arch = get_architecture("wine", "small")
    print(f"  Architecture: {arch}")
