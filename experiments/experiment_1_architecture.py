"""
實驗一: 架構容量對比實驗

目的:
- 比較小容量 (2, 1) vs 推薦容量架構
- 測試不同激活函數 (Sigmoid, Tanh, ReLU)
- 探索學習率影響 (0.05, 0.2, 0.5)

實驗矩陣:
- 3 datasets (Iris, Wine, BC)
- 2 architectures (Small, Recommended)
- 3 activations (Sigmoid, Tanh, ReLU)
- 3 learning rates (0.05, 0.2, 0.5)

總計: 54 組實驗
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from itertools import product
import time

from src.mlp import MLP
from src.trainers import StandardBPTrainer
from src.data_processor import DataProcessor
from src.metrics import Metrics
from utils.visualization import Visualizer
from utils.logger import TrainingLogger
from config import *


class Experiment1:
    """
    實驗一: 架構容量對比
    """

    def __init__(self, save_results=True):
        """
        初始化實驗

        Parameters:
        -----------
        save_results : bool
            是否儲存結果
        """
        self.save_results = save_results
        self.results = []

        # 創建輸出目錄
        if save_results:
            Path(OUTPUT_CONFIG["figures_dir"]).mkdir(parents=True, exist_ok=True)
            Path(OUTPUT_CONFIG["logs_dir"]).mkdir(parents=True, exist_ok=True)
            Path(OUTPUT_CONFIG["models_dir"]).mkdir(parents=True, exist_ok=True)

    def run_single_experiment(
        self, dataset_name, architecture_type, activation, learning_rate, fold_idx=None
    ):
        """
        執行單個實驗

        Parameters:
        -----------
        dataset_name : str
            資料集名稱
        architecture_type : str
            架構類型 ('small' or 'recommended')
        activation : str
            激活函數
        learning_rate : float
            學習率
        fold_idx : int, optional
            Fold 索引 (for K-fold)

        Returns:
        --------
        dict
            實驗結果
        """
        # 獲取資料集配置
        dataset_config = DATASETS[dataset_name]

        # 獲取架構
        architecture = get_architecture(dataset_name, architecture_type)

        # 載入資料
        X, y = DataProcessor.load_data(dataset_config["path"])

        # 準備標籤 (確保從 0 開始)
        y, num_classes, label_mapping = DataProcessor.prepare_classification_labels(y)

        # 資料分割
        if dataset_config["split_method"] == "kfold":
            # K-fold 交叉驗證
            folds = DataProcessor.create_kfold_splits(
                X, y, k=dataset_config["k_folds"], random_seed=42, shuffle=True
            )
            train, val, test = folds[fold_idx] if fold_idx is not None else folds[0]
        else:
            # 簡單分割
            train, val, test = DataProcessor.train_val_test_split(
                X,
                y,
                train_ratio=dataset_config["train_ratio"],
                val_ratio=dataset_config["val_ratio"],
                random_seed=42,
                shuffle=True,
            )

        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test

        # 標準化
        X_train, X_val, X_test, mean, std = DataProcessor.standardize(
            X_train, X_val, X_test
        )

        # 創建 MLP
        mlp = MLP(
            layer_sizes=architecture,
            activation=activation,
            learning_rate=learning_rate,
            weight_scale=TRAINING_CONFIG["weight_scale"],
            random_seed=TRAINING_CONFIG["random_seed"],
        )

        # 創建訓練器
        trainer = StandardBPTrainer(
            mlp,
            max_epochs=TRAINING_CONFIG["max_epochs"],
            convergence_threshold=TRAINING_CONFIG["convergence_threshold"],
            convergence_patience=TRAINING_CONFIG["convergence_patience"],
            early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"],
            verbose=TRAINING_CONFIG["verbose"],
            verbose_interval=TRAINING_CONFIG["verbose_interval"],
        )

        # 訓練
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_config['name']}")
        print(f"Architecture: {architecture_type} {architecture}")
        print(f"Activation: {activation}")
        print(f"Learning Rate: {learning_rate}")
        if fold_idx is not None:
            print(f"Fold: {fold_idx + 1}/{dataset_config['k_folds']}")
        print(f"{'='*70}")

        start_time = time.time()
        history = trainer.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time

        # 測試
        y_pred = mlp.predict(X_test)
        test_acc = Metrics.accuracy(y_test, y_pred)
        test_loss = mlp.compute_loss(X_test, y_test)

        # 混淆矩陣
        cm = Metrics.confusion_matrix(y_test, y_pred, num_classes)

        # 計算 Precision, Recall, F1 (for BC)
        if dataset_name == "breast_cancer":
            precision, recall, f1 = Metrics.precision_recall_f1(
                y_test, y_pred, average="binary"
            )
        else:
            precision, recall, f1 = Metrics.precision_recall_f1(
                y_test, y_pred, average="macro"
            )

        # 記錄結果
        result = {
            "dataset": dataset_config["name"],
            "architecture": architecture_type,
            "architecture_detail": str(architecture),
            "activation": activation,
            "learning_rate": learning_rate,
            "fold": fold_idx if fold_idx is not None else 0,
            "train_acc": history["train_acc"][-1] if history["train_acc"] else None,
            "val_acc": history["val_acc"][-1] if history["val_acc"] else None,
            "test_acc": test_acc,
            "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "test_loss": test_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "epochs": history["epochs"],
            "training_time": training_time,
            "convergence_epoch": history.get("convergence_epoch"),
            "confusion_matrix": cm.tolist(),
            "history": history,
        }

        print(f"\n✓ Test Accuracy: {test_acc:.4f}")
        print(f"✓ Test Loss: {test_loss:.6f}")
        print(f"✓ Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"✓ Training Time: {training_time:.2f}s")
        print(f"✓ Epochs: {history['epochs']}")

        return result

    def run_full_experiment(self):
        """
        執行完整實驗

        遍歷所有資料集、架構、激活函數和學習率的組合
        """
        print("\n" + "=" * 70)
        print("實驗一: 架構容量對比")
        print("=" * 70)

        # 計算總實驗數
        total_experiments = 0
        for dataset_name in EXPERIMENT_1_CONFIG["datasets"]:
            dataset_config = DATASETS[dataset_name]
            if dataset_config["split_method"] == "kfold":
                n_folds = dataset_config["k_folds"]
            else:
                n_folds = 1

            total_experiments += (
                len(EXPERIMENT_1_CONFIG["architectures"])
                * len(EXPERIMENT_1_CONFIG["activations"])
                * len(EXPERIMENT_1_CONFIG["learning_rates"])
                * n_folds
            )

        print(f"Total experiments: {total_experiments}")
        print("=" * 70)

        experiment_count = 0

        # 遍歷所有組合
        for dataset_name in EXPERIMENT_1_CONFIG["datasets"]:
            dataset_config = DATASETS[dataset_name]

            # 判斷分割方法
            if dataset_config["split_method"] == "kfold":
                fold_indices = range(dataset_config["k_folds"])
            else:
                fold_indices = [None]

            for fold_idx in fold_indices:
                for arch, act, lr in product(
                    EXPERIMENT_1_CONFIG["architectures"],
                    EXPERIMENT_1_CONFIG["activations"],
                    EXPERIMENT_1_CONFIG["learning_rates"],
                ):
                    experiment_count += 1
                    print(f"\n{'#'*70}")
                    print(f"Experiment {experiment_count}/{total_experiments}")
                    print(f"{'#'*70}")

                    try:
                        result = self.run_single_experiment(
                            dataset_name, arch, act, lr, fold_idx
                        )
                        self.results.append(result)
                    except Exception as e:
                        print(f"❌ Error in experiment: {e}")
                        import traceback

                        traceback.print_exc()
                        continue

        print(f"\n{'='*70}")
        print(f"✓ All experiments completed!")
        print(f"{'='*70}\n")

    def analyze_results(self):
        """
        分析實驗結果
        """
        if not self.results:
            print("No results to analyze!")
            return

        print("\n" + "=" * 70)
        print("結果分析")
        print("=" * 70)

        # 轉換為 DataFrame
        df = pd.DataFrame(self.results)

        # 1. 整體統計
        print("\n[1] 整體統計")
        print("-" * 70)
        print(f"總實驗數: {len(df)}")
        print(
            f"平均測試準確率: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}"
        )
        print(f"最高測試準確率: {df['test_acc'].max():.4f}")
        print(f"平均訓練時間: {df['training_time'].mean():.2f}s")

        # 2. 各資料集最佳結果
        print("\n[2] 各資料集最佳配置")
        print("-" * 70)
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            best_idx = dataset_df["test_acc"].idxmax()
            best = dataset_df.loc[best_idx]

            print(f"\n{dataset}:")
            print(f"  最佳準確率: {best['test_acc']:.4f}")
            print(f"  架構: {best['architecture']}")
            print(f"  激活函數: {best['activation']}")
            print(f"  學習率: {best['learning_rate']}")
            print(f"  訓練時間: {best['training_time']:.2f}s")

        # 3. 架構對比
        print("\n[3] 架構對比")
        print("-" * 70)
        arch_comparison = df.groupby(["dataset", "architecture"])["test_acc"].agg(
            ["mean", "std", "max"]
        )
        print(arch_comparison)

        # 4. 激活函數對比
        print("\n[4] 激活函數對比")
        print("-" * 70)
        act_comparison = df.groupby(["dataset", "activation"])["test_acc"].agg(
            ["mean", "std", "max"]
        )
        print(act_comparison)

        # 5. 學習率對比
        print("\n[5] 學習率對比")
        print("-" * 70)
        lr_comparison = df.groupby(["dataset", "learning_rate"])["test_acc"].agg(
            ["mean", "std", "max"]
        )
        print(lr_comparison)

        return df

    def visualize_results(self, df):
        """
        視覺化結果

        Parameters:
        -----------
        df : pd.DataFrame
            結果 DataFrame
        """
        print("\n" + "=" * 70)
        print("生成視覺化圖表")
        print("=" * 70)

        # 資料集名稱映射 (顯示名稱 → config key)
        dataset_mapping = {
            "Iris": "iris",
            "Wine": "wine",
            "Breast Cancer Wisconsin": "breast_cancer",
        }

        # 為每個資料集創建視覺化
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            dataset_name_safe = dataset.replace(" ", "_").lower()

            print(f"\n生成 {dataset} 的圖表...")

            # 1. 架構對比熱力圖 (Activation × Learning Rate)
            for arch in dataset_df["architecture"].unique():
                arch_df = dataset_df[dataset_df["architecture"] == arch]

                # 計算平均準確率矩陣
                pivot = arch_df.pivot_table(
                    values="test_acc",
                    index="activation",
                    columns="learning_rate",
                    aggfunc="mean",
                )

                Visualizer.plot_heatmap(
                    pivot.values,
                    x_labels=[f"LR={lr}" for lr in pivot.columns],
                    y_labels=list(pivot.index),
                    title=f"{dataset} - {arch.capitalize()} Architecture\nTest Accuracy",
                    xlabel="Learning Rate",
                    ylabel="Activation Function",
                    fmt=".3f",
                    save_path=f'{OUTPUT_CONFIG["figures_dir"]}/exp1_{dataset_name_safe}_{arch}_heatmap.png',
                    show=False,
                )

            # 2. 最佳配置的訓練曲線
            best_idx = dataset_df["test_acc"].idxmax()
            best = dataset_df.loc[best_idx]

            if "history" in best and best["history"]:
                history = best["history"]

                # Loss 曲線
                if history["train_loss"]:
                    Visualizer.plot_loss_curve(
                        history["train_loss"],
                        history["val_loss"] if history["val_loss"] else None,
                        title=f"{dataset} - Best Config Training Loss",
                        save_path=f'{OUTPUT_CONFIG["figures_dir"]}/exp1_{dataset_name_safe}_best_loss.png',
                        show=False,
                    )

                # Accuracy 曲線
                if history["train_acc"]:
                    Visualizer.plot_accuracy_curve(
                        history["train_acc"],
                        history["val_acc"] if history["val_acc"] else None,
                        title=f"{dataset} - Best Config Training Accuracy",
                        save_path=f'{OUTPUT_CONFIG["figures_dir"]}/exp1_{dataset_name_safe}_best_acc.png',
                        show=False,
                    )

            # 3. 混淆矩陣
            if "confusion_matrix" in best:
                cm = np.array(best["confusion_matrix"])

                # 使用映射獲取 class names
                dataset_key = dataset_mapping.get(dataset)
                if dataset_key and dataset_key in DATASETS:
                    class_names = DATASETS[dataset_key]["class_names"]
                else:
                    # Fallback: 自動生成
                    print(f"⚠️ Warning: Using auto-generated class names for {dataset}")
                    class_names = [f"Class {i}" for i in range(len(cm))]

                Visualizer.plot_confusion_matrix(
                    cm,
                    class_names,
                    title=f"{dataset} - Best Config Confusion Matrix",
                    save_path=f'{OUTPUT_CONFIG["figures_dir"]}/exp1_{dataset_name_safe}_confusion.png',
                    show=False,
                )

        print("\n✓ 所有圖表已生成")

    def save_results_to_file(self, df):
        """
        儲存結果到文件

        Parameters:
        -----------
        df : pd.DataFrame
            結果 DataFrame
        """
        print("\n儲存結果...")

        # 儲存完整結果 (CSV)
        csv_path = f'{OUTPUT_CONFIG["logs_dir"]}/experiment_1_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to {csv_path}")

        # 儲存摘要 (TXT)
        summary_path = f'{OUTPUT_CONFIG["logs_dir"]}/experiment_1_summary.txt'
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("實驗一: 架構容量對比 - 結果摘要\n")
            f.write("=" * 70 + "\n\n")

            # 各資料集最佳結果
            f.write("各資料集最佳配置:\n")
            f.write("-" * 70 + "\n")
            for dataset in df["dataset"].unique():
                dataset_df = df[df["dataset"] == dataset]
                best_idx = dataset_df["test_acc"].idxmax()
                best = dataset_df.loc[best_idx]

                f.write(f"\n{dataset}:\n")
                f.write(f"  最佳準確率: {best['test_acc']:.4f}\n")
                f.write(
                    f"  架構: {best['architecture']} {best['architecture_detail']}\n"
                )
                f.write(f"  激活函數: {best['activation']}\n")
                f.write(f"  學習率: {best['learning_rate']}\n")
                f.write(f"  Precision: {best['precision']:.4f}\n")
                f.write(f"  Recall: {best['recall']:.4f}\n")
                f.write(f"  F1-score: {best['f1_score']:.4f}\n")
                f.write(f"  訓練時間: {best['training_time']:.2f}s\n")
                f.write(f"  Epochs: {best['epochs']}\n")

        print(f"✓ Summary saved to {summary_path}")


# ============================================================
# 主程式
# ============================================================


def main():
    """
    執行實驗一
    """
    # 創建實驗
    exp = Experiment1(save_results=True)

    # 執行完整實驗
    exp.run_full_experiment()

    # 分析結果
    df = exp.analyze_results()

    # 視覺化
    if df is not None:
        exp.visualize_results(df)
        exp.save_results_to_file(df)

    print("\n" + "=" * 70)
    print("✓ 實驗一完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
