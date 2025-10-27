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
import pandas as pd
from pathlib import Path
from matplotlib.patches import Patch


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

    @staticmethod
    def plot_architecture_comparison_boxplot(df, save_path=None, show=True):
        """
        繪製架構對比箱型圖

        比較 Small vs Recommended 架構在各資料集上的表現分布

        Parameters:
        -----------
        df : pd.DataFrame
            實驗結果 DataFrame,需包含:
            - dataset: 資料集名稱
            - architecture: 架構類型 (small/recommended)
            - test_acc: 測試準確率
        save_path : str, optional
            儲存路徑
        show : bool
            是否顯示圖表
        """

        # 準備資料
        datasets = df["dataset"].unique()
        architectures = ["small", "recommended"]

        # 設定圖表
        fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5))
        if len(datasets) == 1:
            axes = [axes]

        colors = {"small": "#FF6B6B", "recommended": "#4ECDC4"}

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            dataset_df = df[df["dataset"] == dataset]

            # 準備箱型圖資料
            data_to_plot = []
            labels = []
            box_colors = []

            for arch in architectures:
                arch_data = dataset_df[dataset_df["architecture"] == arch][
                    "test_acc"
                ].values
                if len(arch_data) > 0:
                    data_to_plot.append(arch_data)
                    labels.append(arch.capitalize())
                    box_colors.append(colors[arch])

            # 繪製箱型圖
            bp = ax.boxplot(
                data_to_plot,
                labels=labels,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                medianprops=dict(color="red", linewidth=2),
                meanprops=dict(color="blue", linewidth=2, linestyle="--"),
            )

            # 設定顏色
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # 添加數據點 (散點圖)
            for i, (data, arch) in enumerate(zip(data_to_plot, architectures), 1):
                y = data
                x = np.random.normal(i, 0.04, size=len(y))  # 添加隨機抖動
                ax.scatter(
                    x,
                    y,
                    alpha=0.4,
                    s=30,
                    color=colors[arch],
                    edgecolors="black",
                    linewidth=0.5,
                )

            # 設定標籤和標題
            ax.set_ylabel("Test Accuracy", fontsize=12, fontweight="bold")
            ax.set_title(f"{dataset}", fontsize=13, fontweight="bold")
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            ax.set_axisbelow(True)

            # 添加統計資訊
            for i, (data, arch) in enumerate(zip(data_to_plot, architectures), 1):
                mean_val = np.mean(data)
                median_val = np.median(data)
                ax.text(
                    i,
                    1.02,
                    f"μ={mean_val:.3f}\nM={median_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
                )

        # 總標題
        plt.suptitle(
            "Architecture Capacity Comparison: Small vs Recommended",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )

        # 圖例

        legend_elements = [
            Patch(facecolor=colors["small"], alpha=0.6, label="Small (2, 1)"),
            Patch(facecolor=colors["recommended"], alpha=0.6, label="Recommended"),
            plt.Line2D([0], [0], color="red", linewidth=2, label="Median"),
            plt.Line2D(
                [0], [0], color="blue", linewidth=2, linestyle="--", label="Mean"
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=11,
        )

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
    def plot_architecture_comparison_barplot(df, save_path=None, show=True):
        """
        繪製架構對比條狀圖 (含誤差條)

        比較 Small vs Recommended 架構的平均表現

        Parameters:
        -----------
        df : pd.DataFrame
            實驗結果 DataFrame,需包含:
            - dataset: 資料集名稱
            - architecture: 架構類型 (small/recommended)
            - test_acc: 測試準確率
        save_path : str, optional
            儲存路徑
        show : bool
            是否顯示圖表
        """

        # 準備資料
        datasets = df["dataset"].unique()
        architectures = ["small", "recommended"]

        # 計算統計量
        stats = []
        for dataset in datasets:
            dataset_df = df[df["dataset"] == dataset]
            for arch in architectures:
                arch_data = dataset_df[dataset_df["architecture"] == arch]["test_acc"]
                if len(arch_data) > 0:
                    stats.append(
                        {
                            "dataset": dataset,
                            "architecture": arch,
                            "mean": arch_data.mean(),
                            "std": arch_data.std(),
                            "max": arch_data.max(),
                            "min": arch_data.min(),
                        }
                    )

        stats_df = pd.DataFrame(stats)

        # 設定圖表
        fig, ax = plt.subplots(figsize=(12, 6))

        # 設定位置
        x = np.arange(len(datasets))
        width = 0.35

        colors = {"small": "#FF6B6B", "recommended": "#4ECDC4"}

        # 繪製條狀圖
        for i, arch in enumerate(architectures):
            arch_stats = stats_df[stats_df["architecture"] == arch]
            means = [
                (
                    arch_stats[arch_stats["dataset"] == ds]["mean"].values[0]
                    if len(arch_stats[arch_stats["dataset"] == ds]) > 0
                    else 0
                )
                for ds in datasets
            ]
            stds = [
                (
                    arch_stats[arch_stats["dataset"] == ds]["std"].values[0]
                    if len(arch_stats[arch_stats["dataset"] == ds]) > 0
                    else 0
                )
                for ds in datasets
            ]

            offset = width * (i - 0.5)
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                label=arch.capitalize(),
                color=colors[arch],
                alpha=0.8,
                capsize=5,
                error_kw={"linewidth": 2},
            )

            # 添加數值標籤
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + std + 0.02,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        # 設定標籤和標題
        ax.set_xlabel("Dataset", fontsize=13, fontweight="bold")
        ax.set_ylabel("Average Test Accuracy", fontsize=13, fontweight="bold")
        ax.set_title(
            "Architecture Capacity Comparison: Average Performance",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.legend(fontsize=11, loc="lower right")
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.set_axisbelow(True)

        # 添加統計表格
        table_data = []
        for dataset in datasets:
            row = [dataset]
            for arch in architectures:
                arch_stats = stats_df[
                    (stats_df["dataset"] == dataset)
                    & (stats_df["architecture"] == arch)
                ]
                if len(arch_stats) > 0:
                    mean = arch_stats["mean"].values[0]
                    std = arch_stats["std"].values[0]
                    row.append(f"{mean:.3f}±{std:.3f}")
                else:
                    row.append("N/A")
            table_data.append(row)

        # 在圖下方添加表格
        table = ax.table(
            cellText=table_data,
            colLabels=["Dataset", "Small", "Recommended"],
            cellLoc="center",
            loc="bottom",
            bbox=[0.0, -0.35, 1.0, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 設定表格樣式
        for i in range(len(table_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # 標題行
                    cell.set_facecolor("#4ECDC4")
                    cell.set_text_props(weight="bold", color="white")
                else:
                    cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

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

    # 創建測試數據 (修正版)
    epochs = 100
    train_losses = [
        max(0.01, 0.5 * np.exp(-0.05 * i) + 0.01 * np.random.randn())
        for i in range(epochs)
    ]
    val_losses = [
        max(0.01, 0.5 * np.exp(-0.04 * i) + 0.02 * np.random.randn())
        for i in range(epochs)
    ]
    train_accs = [
        np.clip(1 - 0.5 * np.exp(-0.05 * i) + 0.01 * np.random.randn(), 0, 1)
        for i in range(epochs)
    ]
    val_accs = [
        np.clip(1 - 0.5 * np.exp(-0.04 * i) + 0.02 * np.random.randn(), 0, 1)
        for i in range(epochs)
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
