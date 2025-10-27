"""
視覺化工具模組

提供各種圖表繪製功能:
1. 損失曲線 (Loss Curve)
2. 準確率曲線 (Accuracy Curve)
3. 混淆矩陣 (Confusion Matrix)
4. 熱力圖 (Heatmap)
5. 訓練方法對比 (Training Comparison)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class Visualizer:
    """
    視覺化工具類別

    提供靜態方法繪製各種圖表
    """

    # 設置中文字體 (如果需要)
    plt.rcParams["font.sans-serif"] = ["Microsoft YhHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    @staticmethod
    def plot_loss_curve(
        train_losses,
        val_losses=None,
        title="Training Loss Curve",
        xlabel="Epoch",
        ylabel="Loss",
        save_path=None,
        show=True,
    ):
        """
        繪製損失曲線

        Parameters:
        -----------
        train_losses : list
            訓練損失列表
        val_losses : list, optional
            驗證損失列表
        title : str
            圖表標題
        xlabel : str
            X 軸標籤
        ylabel : str
            Y 軸標籤
        save_path : str, optional
            儲存路徑
        show : bool
            是否顯示圖表
        """
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)

        if val_losses is not None:
            plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_accuracy_curve(
        train_accs,
        val_accs=None,
        title="Training Accuracy Curve",
        xlabel="Epoch",
        ylabel="Accuracy",
        save_path=None,
        show=True,
    ):
        """
        繪製準確率曲線
        """
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(train_accs) + 1)
        plt.plot(epochs, train_accs, "b-", label="Train Accuracy", linewidth=2)

        if val_accs is not None:
            plt.plot(epochs, val_accs, "r-", label="Validation Accuracy", linewidth=2)

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])  # 準確率範圍 0-1
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_confusion_matrix(
        cm,
        class_names,
        title="Confusion Matrix",
        cmap="Blues",
        save_path=None,
        show=True,
    ):
        """
        繪製混淆矩陣

        Parameters:
        -----------
        cm : np.ndarray
            混淆矩陣
        class_names : list
            類別名稱
        title : str
            圖表標題
        cmap : str
            顏色映射
        save_path : str, optional
            儲存路徑
        show : bool
            是否顯示圖表
        """
        plt.figure(figsize=(8, 6))

        # 計算百分比
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # 繪製熱力圖
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count"},
        )

        # 在每個格子添加百分比
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(
                    j + 0.5,
                    i + 0.7,
                    f"({cm_percent[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="gray",
                )

        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_heatmap(
        data,
        x_labels,
        y_labels,
        title="Heatmap",
        xlabel="",
        ylabel="",
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        save_path=None,
        show=True,
    ):
        """
        繪製熱力圖 (通用)

        Parameters:
        -----------
        data : np.ndarray or pd.DataFrame
            數據矩陣
        x_labels : list
            X 軸標籤
        y_labels : list
            Y 軸標籤
        title : str
            圖表標題
        xlabel : str
            X 軸標題
        ylabel : str
            Y 軸標題
        cmap : str
            顏色映射
        annot : bool
            是否顯示數值
        fmt : str
            數值格式
        save_path : str, optional
            儲存路徑
        show : bool
            是否顯示圖表
        """
        plt.figure(figsize=(12, 8))

        sns.heatmap(
            data,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cbar_kws={"label": "Value"},
        )

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_training_comparison(
        results_dict,
        metric="accuracy",
        title="Training Methods Comparison",
        save_path=None,
        show=True,
    ):
        """
        繪製訓練方法對比圖

        Parameters:
        -----------
        results_dict : dict
            結果字典，格式:
            {
                'method_name': {
                    'train_loss': [...],
                    'train_acc': [...],
                    'val_loss': [...],
                    'val_acc': [...]
                }
            }
        metric : str
            要比較的指標 ('accuracy' or 'loss')
        title : str
            圖表標題
        save_path : str, optional
            儲存路徑
        show : bool
            是否顯示圖表
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左圖: 訓練指標
        ax1 = axes[0]
        for method_name, history in results_dict.items():
            if metric == "accuracy":
                data = history.get("train_acc", [])
                ylabel = "Accuracy"
            else:
                data = history.get("train_loss", [])
                ylabel = "Loss"

            epochs = range(1, len(data) + 1)
            ax1.plot(epochs, data, label=method_name, linewidth=2)

        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel(f"Training {ylabel}", fontsize=12)
        ax1.set_title(f"Training {ylabel} Comparison", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 右圖: 驗證指標
        ax2 = axes[1]
        for method_name, history in results_dict.items():
            if metric == "accuracy":
                data = history.get("val_acc", [])
                ylabel = "Accuracy"
            else:
                data = history.get("val_loss", [])
                ylabel = "Loss"

            if data:  # 有驗證數據才繪製
                epochs = range(1, len(data) + 1)
                ax2.plot(epochs, data, label=method_name, linewidth=2)

        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel(f"Validation {ylabel}", fontsize=12)
        ax2.set_title(f"Validation {ylabel} Comparison", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_architecture_comparison(
        results_df, dataset_name, save_path=None, show=True
    ):
        """
        繪製架構對比圖

        Parameters:
        -----------
        results_df : pd.DataFrame
            結果 DataFrame，包含 architecture, activation, lr, accuracy 等欄位
        dataset_name : str
            資料集名稱
        save_path : str, optional
            儲存路徑
        show : bool
            是否顯示圖表
        """
        import pandas as pd

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 左圖: 不同架構的準確率分布
        ax1 = axes[0]
        architectures = results_df["architecture"].unique()
        for arch in architectures:
            data = results_df[results_df["architecture"] == arch]["test_acc"]
            ax1.boxplot(
                [data],
                positions=[list(architectures).index(arch)],
                labels=[arch],
                widths=0.6,
            )

        ax1.set_xlabel("Architecture", fontsize=12)
        ax1.set_ylabel("Test Accuracy", fontsize=12)
        ax1.set_title(
            f"{dataset_name}: Architecture Comparison", fontsize=13, fontweight="bold"
        )
        ax1.grid(True, alpha=0.3, axis="y")

        # 右圖: 不同激活函數的準確率分布
        ax2 = axes[1]
        activations = results_df["activation"].unique()
        for act in activations:
            data = results_df[results_df["activation"] == act]["test_acc"]
            ax2.boxplot(
                [data],
                positions=[list(activations).index(act)],
                labels=[act],
                widths=0.6,
            )

        ax2.set_xlabel("Activation Function", fontsize=12)
        ax2.set_ylabel("Test Accuracy", fontsize=12)
        ax2.set_title(
            f"{dataset_name}: Activation Function Comparison",
            fontsize=13,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


# ============================================================
# 測試程式碼
# ============================================================

if __name__ == "__main__":
    """
    測試視覺化工具
    """
    print("=" * 70)
    print("視覺化工具測試")
    print("=" * 70)

    # 創建測試數據
    epochs = 100
    train_losses = [
        0.5 * np.exp(-0.05 * i) + 0.01 * np.random.randn() for i in range(epochs)
    ]
    val_losses = [
        0.5 * np.exp(-0.04 * i) + 0.02 * np.random.randn() for i in range(epochs)
    ]
    train_accs = [
        1 - 0.5 * np.exp(-0.05 * i) + 0.01 * np.random.randn() for i in range(epochs)
    ]
    val_accs = [
        1 - 0.5 * np.exp(-0.04 * i) + 0.02 * np.random.randn() for i in range(epochs)
    ]

    print("\n[1] 測試損失曲線")
    Visualizer.plot_loss_curve(
        train_losses,
        val_losses,
        title="Test Loss Curve",
        save_path="results/figures/test_loss.png",
        show=False,
    )

    print("\n[2] 測試準確率曲線")
    Visualizer.plot_accuracy_curve(
        train_accs,
        val_accs,
        title="Test Accuracy Curve",
        save_path="results/figures/test_accuracy.png",
        show=False,
    )

    print("\n[3] 測試混淆矩陣")
    cm = np.array([[45, 3, 2], [2, 38, 5], [1, 3, 41]])
    class_names = ["Class A", "Class B", "Class C"]
    Visualizer.plot_confusion_matrix(
        cm,
        class_names,
        title="Test Confusion Matrix",
        save_path="results/figures/test_confusion.png",
        show=False,
    )

    print("\n[4] 測試熱力圖")
    data = np.random.rand(3, 4)
    x_labels = ["LR=0.05", "LR=0.1", "LR=0.2", "LR=0.5"]
    y_labels = ["Sigmoid", "Tanh", "ReLU"]
    Visualizer.plot_heatmap(
        data,
        x_labels,
        y_labels,
        title="Test Heatmap",
        xlabel="Learning Rate",
        ylabel="Activation Function",
        save_path="results/figures/test_heatmap.png",
        show=False,
    )

    print("\n[5] 測試訓練方法對比")
    results = {
        "Standard BP": {
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": val_losses,
            "val_acc": val_accs,
        },
        "Momentum": {
            "train_loss": [l * 0.8 for l in train_losses],
            "train_acc": [a * 1.05 for a in train_accs],
            "val_loss": [l * 0.85 for l in val_losses],
            "val_acc": [a * 1.03 for a in val_accs],
        },
    }
    Visualizer.plot_training_comparison(
        results,
        metric="accuracy",
        save_path="results/figures/test_comparison.png",
        show=False,
    )

    print("\n✓ 所有圖表已儲存到 results/figures/")
    print("=" * 70)
