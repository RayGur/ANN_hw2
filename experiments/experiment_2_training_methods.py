"""
實驗二: 訓練方法對比實驗

目的:
- 比較不同優化訓練方法
- 分析收斂速度、訓練時間、最終準確率
- 使用實驗一找出的最佳配置

訓練方法:
1. Standard Backpropagation (基準)
2. Momentum Method (α = 0.5, 0.7, 0.9)
3. Resilient Propagation (RProp)
4. Levenberg-Marquardt (簡化版)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import time
import json

from src.mlp import MLP
from src.trainers import (
    StandardBPTrainer,
    MomentumTrainer,
    ResilientPropTrainer,
    LevenbergMarquardtTrainer,
)
from src.data_processor import DataProcessor
from src.metrics import Metrics
from utils.visualization import Visualizer
from utils.logger import TrainingLogger
from config import *


class Experiment2:
    """
    實驗二: 訓練方法對比
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
        self.best_configs = {}  # 儲存各資料集的最佳配置

        # 創建輸出目錄
        if save_results:
            Path(OUTPUT_CONFIG["figures_dir"]).mkdir(parents=True, exist_ok=True)
            Path(OUTPUT_CONFIG["logs_dir"]).mkdir(parents=True, exist_ok=True)

    def load_best_configs(self, exp1_results_path=None):
        """
        載入實驗一的最佳配置

        Parameters:
        -----------
        exp1_results_path : str, optional
            實驗一結果的 CSV 路徑
            如果為 None，則使用預設配置
        """
        if exp1_results_path and Path(exp1_results_path).exists():
            print(f"載入實驗一結果: {exp1_results_path}")
            df = pd.read_csv(exp1_results_path)

            # 為每個資料集找出最佳配置
            for dataset_name in df["dataset"].unique():
                dataset_df = df[df["dataset"] == dataset_name]
                best_idx = dataset_df["test_acc"].idxmax()
                best = dataset_df.loc[best_idx]

                # 從資料集名稱映射到 config key
                dataset_key_map = {
                    "Iris": "iris",
                    "Wine": "wine",
                    "Breast Cancer Wisconsin": "breast_cancer",
                }
                dataset_key = dataset_key_map.get(
                    dataset_name, dataset_name.lower().replace(" ", "_")
                )

                self.best_configs[dataset_key] = {
                    "architecture": best["architecture"],
                    "activation": best["activation"],
                    "learning_rate": best["learning_rate"],
                    "test_acc": best["test_acc"],
                }

                print(f"\n{dataset_name}:")
                print(f"  架構: {best['architecture']}")
                print(f"  激活函數: {best['activation']}")
                print(f"  學習率: {best['learning_rate']}")
                print(f"  測試準確率: {best['test_acc']:.4f}")
        else:
            # 使用預設配置 (基於經驗)
            print("使用預設最佳配置")
            self.best_configs = {
                "iris": {
                    "architecture": "recommended",
                    "activation": "tanh",
                    "learning_rate": 0.2,
                },
                "wine": {
                    "architecture": "recommended",
                    "activation": "tanh",
                    "learning_rate": 0.2,
                },
                "breast_cancer": {
                    "architecture": "recommended",
                    "activation": "tanh",
                    "learning_rate": 0.2,
                },
            }

    def prepare_data(self, dataset_name):
        """
        準備資料集

        Parameters:
        -----------
        dataset_name : str
            資料集名稱 (config key)

        Returns:
        --------
        tuple
            (X_train, y_train, X_val, y_val, X_test, y_test, dataset_config)
        """
        dataset_config = DATASETS[dataset_name]

        # 載入資料
        X, y = DataProcessor.load_data(dataset_config["path"])

        # 準備標籤
        y, num_classes, label_mapping = DataProcessor.prepare_classification_labels(y)

        # 資料分割 (使用第一個 fold 或簡單分割)
        if dataset_config["split_method"] == "kfold":
            folds = DataProcessor.create_kfold_splits(
                X, y, k=dataset_config["k_folds"], random_seed=42, shuffle=True
            )
            train, val, test = folds[0]  # 使用第一個 fold
        else:
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

        return X_train, y_train, X_val, y_val, X_test, y_test, dataset_config

    def run_single_experiment(self, dataset_name, trainer_name, trainer_params=None):
        """
        執行單個訓練方法實驗

        Parameters:
        -----------
        dataset_name : str
            資料集名稱 (config key)
        trainer_name : str
            訓練器名稱
        trainer_params : dict, optional
            訓練器參數

        Returns:
        --------
        dict
            實驗結果
        """
        # 準備資料
        X_train, y_train, X_val, y_val, X_test, y_test, dataset_config = (
            self.prepare_data(dataset_name)
        )

        # 獲取最佳配置
        best_config = self.best_configs[dataset_name]
        architecture = get_architecture(dataset_name, best_config["architecture"])

        # 創建 MLP
        mlp = MLP(
            layer_sizes=architecture,
            activation=best_config["activation"],
            learning_rate=best_config["learning_rate"],
            weight_scale=TRAINING_CONFIG["weight_scale"],
            random_seed=TRAINING_CONFIG["random_seed"],
        )

        # 創建訓練器
        trainer_classes = {
            "standard_bp": StandardBPTrainer,
            "momentum": MomentumTrainer,
            "rprop": ResilientPropTrainer,
            "levenberg_marquardt": LevenbergMarquardtTrainer,
        }

        TrainerClass = trainer_classes[trainer_name]

        # 訓練器參數
        trainer_kwargs = {
            "max_epochs": TRAINING_CONFIG["max_epochs"],
            "convergence_threshold": TRAINING_CONFIG["convergence_threshold"],
            "convergence_patience": TRAINING_CONFIG["convergence_patience"],
            "early_stopping_patience": TRAINING_CONFIG["early_stopping_patience"],
            "verbose": True,
            "verbose_interval": TRAINING_CONFIG["verbose_interval"],
        }

        if trainer_params:
            trainer_kwargs.update(trainer_params)

        trainer = TrainerClass(mlp, **trainer_kwargs)

        # 顯示實驗信息
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_config['name']}")
        print(f"Trainer: {TRAINING_METHODS[trainer_name]['name']}")
        if trainer_params:
            print(f"Params: {trainer_params}")
        print(f"Architecture: {architecture}")
        print(f"Activation: {best_config['activation']}")
        print(f"Learning Rate: {best_config['learning_rate']}")
        print(f"{'='*70}")

        # 訓練
        start_time = time.time()
        history = trainer.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time

        # 測試
        y_pred = mlp.predict(X_test)
        test_acc = Metrics.accuracy(y_test, y_pred)
        test_loss = mlp.compute_loss(X_test, y_test)

        # 混淆矩陣
        cm = Metrics.confusion_matrix(y_test, y_pred, dataset_config["output_dim"])

        # Precision, Recall, F1
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
            "trainer": TRAINING_METHODS[trainer_name]["name"],
            "trainer_key": trainer_name,
            "trainer_params": trainer_params if trainer_params else {},
            "architecture": best_config["architecture"],
            "activation": best_config["activation"],
            "learning_rate": best_config["learning_rate"],
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
            "early_stop_epoch": history.get("early_stop_epoch"),
            "confusion_matrix": cm.tolist(),
            "history": history,
        }

        print(f"\n✓ Test Accuracy: {test_acc:.4f}")
        print(f"✓ Test Loss: {test_loss:.6f}")
        print(f"✓ Training Time: {training_time:.2f}s")
        print(f"✓ Epochs: {history['epochs']}")
        print(f"✓ Convergence Epoch: {history.get('convergence_epoch', 'N/A')}")

        return result

    def run_full_experiment(self):
        """
        執行完整實驗

        測試所有訓練方法在所有資料集上的表現
        """
        print("\n" + "=" * 70)
        print("實驗二: 訓練方法對比")
        print("=" * 70)

        # 載入最佳配置
        exp1_results = f'{OUTPUT_CONFIG["logs_dir"]}/experiment_1_results.csv'
        self.load_best_configs(exp1_results)

        print("\n" + "=" * 70)

        # 訓練方法列表
        experiments = []

        # Standard BP
        experiments.append(("standard_bp", None))

        # Momentum (多個 α 值)
        for momentum in TRAINING_METHODS["momentum"]["params"]["momentum_values"]:
            experiments.append(("momentum", {"momentum": momentum}))

        # RProp
        experiments.append(("rprop", TRAINING_METHODS["rprop"]["params"]))

        # Levenberg-Marquardt
        experiments.append(
            ("levenberg_marquardt", TRAINING_METHODS["levenberg_marquardt"]["params"])
        )

        total_experiments = len(
            EXPERIMENT_2_CONFIG["use_best_config"] and self.best_configs
        ) * len(experiments)
        print(f"Total experiments: {total_experiments}")
        print("=" * 70)

        experiment_count = 0

        # 遍歷所有資料集和訓練方法
        for dataset_name in self.best_configs.keys():
            for trainer_name, trainer_params in experiments:
                experiment_count += 1
                print(f"\n{'#'*70}")
                print(f"Experiment {experiment_count}/{total_experiments}")
                print(f"{'#'*70}")

                try:
                    result = self.run_single_experiment(
                        dataset_name, trainer_name, trainer_params
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
        print(
            f"平均訓練時間: {df['training_time'].mean():.2f}s ± {df['training_time'].std():.2f}s"
        )
        print(f"平均收斂輪數: {df['epochs'].mean():.1f} ± {df['epochs'].std():.1f}")

        # 2. 各資料集的訓練方法對比
        print("\n[2] 各資料集訓練方法對比")
        print("-" * 70)
        for dataset in df["dataset"].unique():
            print(f"\n{dataset}:")
            dataset_df = df[df["dataset"] == dataset]

            comparison = (
                dataset_df.groupby("trainer")
                .agg(
                    {
                        "test_acc": ["mean", "std", "max"],
                        "training_time": ["mean", "std"],
                        "epochs": ["mean", "std"],
                    }
                )
                .round(4)
            )

            print(comparison)

        # 3. 訓練方法排名 (按準確率)
        print("\n[3] 訓練方法排名 (按平均測試準確率)")
        print("-" * 70)
        trainer_ranking = (
            df.groupby("trainer")["test_acc"]
            .agg(["mean", "std", "max"])
            .sort_values("mean", ascending=False)
        )
        print(trainer_ranking)

        # 4. 訓練方法排名 (按訓練時間)
        print("\n[4] 訓練方法排名 (按平均訓練時間)")
        print("-" * 70)
        time_ranking = (
            df.groupby("trainer")["training_time"]
            .agg(["mean", "std", "min"])
            .sort_values("mean")
        )
        print(time_ranking)

        # 5. 訓練方法排名 (按收斂速度)
        print("\n[5] 訓練方法排名 (按平均收斂輪數)")
        print("-" * 70)
        epoch_ranking = (
            df.groupby("trainer")["epochs"]
            .agg(["mean", "std", "min"])
            .sort_values("mean")
        )
        print(epoch_ranking)

        # 6. Momentum 參數對比
        print("\n[6] Momentum 參數對比")
        print("-" * 70)
        momentum_df = df[df["trainer_key"] == "momentum"].copy()
        if not momentum_df.empty:
            # 提取 momentum 值
            momentum_df["momentum_alpha"] = momentum_df["trainer_params"].apply(
                lambda x: x.get("momentum", "N/A")
            )
            momentum_comparison = (
                momentum_df.groupby("momentum_alpha")
                .agg(
                    {
                        "test_acc": ["mean", "std"],
                        "training_time": ["mean", "std"],
                        "epochs": ["mean", "std"],
                    }
                )
                .round(4)
            )
            print(momentum_comparison)
        else:
            print("  No momentum results found")

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

        # 資料集名稱映射
        dataset_mapping = {
            "Iris": "iris",
            "Wine": "wine",
            "Breast Cancer Wisconsin": "breast_cancer",
        }

        # 1. 各資料集的訓練方法對比 (訓練曲線)
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            dataset_name_safe = dataset.replace(" ", "_").lower()

            print(f"\n生成 {dataset} 的訓練方法對比圖...")

            # 準備數據 (只取前幾個主要方法避免圖表過於擁擠)
            results_dict = {}
            for _, row in dataset_df.iterrows():
                trainer_label = row["trainer"]
                if row["trainer_key"] == "momentum":
                    momentum_alpha = row["trainer_params"].get("momentum", "N/A")
                    trainer_label = f"Momentum (α={momentum_alpha})"

                if "history" in row and row["history"]:
                    results_dict[trainer_label] = row["history"]

            # 損失對比
            if results_dict:
                Visualizer.plot_training_comparison(
                    results_dict,
                    metric="loss",
                    title=f"{dataset} - Training Methods Comparison (Loss)",
                    save_path=f'{OUTPUT_CONFIG["figures_dir"]}/exp2_{dataset_name_safe}_loss_comparison.png',
                    show=False,
                )

                # 準確率對比
                Visualizer.plot_training_comparison(
                    results_dict,
                    metric="accuracy",
                    title=f"{dataset} - Training Methods Comparison (Accuracy)",
                    save_path=f'{OUTPUT_CONFIG["figures_dir"]}/exp2_{dataset_name_safe}_acc_comparison.png',
                    show=False,
                )

        # 2. 整體對比條狀圖
        print("\n生成整體對比圖...")

        import matplotlib.pyplot as plt

        # 準確率對比
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics = ["test_acc", "training_time", "epochs"]
        titles = ["Test Accuracy", "Training Time (s)", "Epochs to Converge"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]

            # 按訓練方法分組
            grouped = (
                df.groupby("trainer")[metric]
                .mean()
                .sort_values(ascending=(metric != "test_acc"))
            )

            grouped.plot(kind="barh", ax=ax, color="skyblue", edgecolor="black")
            ax.set_xlabel(title, fontsize=12)
            ax.set_ylabel("Training Method", fontsize=12)
            ax.set_title(f"Average {title}", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(
            f'{OUTPUT_CONFIG["figures_dir"]}/exp2_overall_comparison.png',
            dpi=300,
            bbox_inches="tight",
        )
        print(f"✓ Saved to {OUTPUT_CONFIG['figures_dir']}/exp2_overall_comparison.png")
        plt.close()

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
        csv_path = f'{OUTPUT_CONFIG["logs_dir"]}/experiment_2_results.csv'

        # 創建簡化版 (移除 history 和 confusion_matrix 這些複雜結構)
        df_simple = df.drop(
            columns=["history", "confusion_matrix", "trainer_params"], errors="ignore"
        )
        df_simple.to_csv(csv_path, index=False)
        print(f"✓ Results saved to {csv_path}")

        # 儲存完整結果 (JSON)
        json_path = f'{OUTPUT_CONFIG["logs_dir"]}/experiment_2_results_full.json'
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✓ Full results saved to {json_path}")

        # 儲存摘要
        summary_path = f'{OUTPUT_CONFIG["logs_dir"]}/experiment_2_summary.txt'
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("實驗二: 訓練方法對比 - 結果摘要\n")
            f.write("=" * 70 + "\n\n")

            # 各資料集最佳訓練方法
            f.write("各資料集最佳訓練方法:\n")
            f.write("-" * 70 + "\n")
            for dataset in df["dataset"].unique():
                dataset_df = df[df["dataset"] == dataset]
                best_idx = dataset_df["test_acc"].idxmax()
                best = dataset_df.loc[best_idx]

                f.write(f"\n{dataset}:\n")
                f.write(f"  最佳方法: {best['trainer']}\n")
                f.write(f"  測試準確率: {best['test_acc']:.4f}\n")
                f.write(f"  訓練時間: {best['training_time']:.2f}s\n")
                f.write(f"  收斂輪數: {best['epochs']}\n")
                f.write(f"  Precision: {best['precision']:.4f}\n")
                f.write(f"  Recall: {best['recall']:.4f}\n")
                f.write(f"  F1-score: {best['f1_score']:.4f}\n")

            # 訓練方法整體排名
            f.write("\n\n訓練方法整體排名 (按平均測試準確率):\n")
            f.write("-" * 70 + "\n")
            ranking = (
                df.groupby("trainer")["test_acc"]
                .agg(["mean", "std"])
                .sort_values("mean", ascending=False)
            )
            for idx, (trainer, row) in enumerate(ranking.iterrows(), 1):
                f.write(f"{idx}. {trainer}: {row['mean']:.4f} ± {row['std']:.4f}\n")

        print(f"✓ Summary saved to {summary_path}")


# ============================================================
# 主程式
# ============================================================


def main():
    """
    執行實驗二
    """
    # 創建實驗
    exp = Experiment2(save_results=True)

    # 執行完整實驗
    exp.run_full_experiment()

    # 分析結果
    df = exp.analyze_results()

    # 視覺化
    if df is not None:
        exp.visualize_results(df)
        exp.save_results_to_file(df)

    print("\n" + "=" * 70)
    print("✓ 實驗二完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
