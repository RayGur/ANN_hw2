"""
訓練記錄器模組

提供訓練過程記錄和結果儲存功能
"""

import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np


class TrainingLogger:
    """
    訓練記錄器

    功能:
    1. 記錄每個 epoch 的訓練指標
    2. 儲存實驗配置
    3. 導出訓練歷史 (JSON, CSV)
    4. 生成實驗摘要
    """

    def __init__(self, log_dir="results/logs", experiment_name=None):
        """
        初始化記錄器

        Parameters:
        -----------
        log_dir : str
            日誌目錄
        experiment_name : str, optional
            實驗名稱，如果為 None 則使用時間戳
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"

        # 訓練歷史
        self.history = []

        # 實驗配置
        self.config = {}

        # 開始記錄
        self._write_header()

    def _write_header(self):
        """寫入日誌標頭"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

    def log_config(self, config):
        """
        記錄實驗配置

        Parameters:
        -----------
        config : dict
            配置字典
        """
        self.config = config

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("Configuration:\n")
            f.write("-" * 70 + "\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    def log_epoch(
        self, epoch, train_loss, train_acc, val_loss=None, val_acc=None, **kwargs
    ):
        """
        記錄單個 epoch 的訓練指標

        Parameters:
        -----------
        epoch : int
            Epoch 編號
        train_loss : float
            訓練損失
        train_acc : float
            訓練準確率
        val_loss : float, optional
            驗證損失
        val_acc : float, optional
            驗證準確率
        **kwargs : dict
            其他指標
        """
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "timestamp": datetime.now().isoformat(),
        }
        record.update(kwargs)

        self.history.append(record)

    def log_message(self, message):
        """
        記錄自定義訊息

        Parameters:
        -----------
        message : str
            訊息內容
        """
        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

    def log_results(self, results):
        """
        記錄最終結果

        Parameters:
        -----------
        results : dict
            結果字典
        """
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write("Final Results:\n")
            f.write("-" * 70 + "\n")
            for key, value in results.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("=" * 70 + "\n")

    def save_history(self, format="json"):
        """
        儲存訓練歷史

        Parameters:
        -----------
        format : str
            儲存格式 ('json' or 'csv')
        """
        if format == "json":
            filepath = self.log_dir / f"{self.experiment_name}_history.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    {"config": self.config, "history": self.history},
                    f,
                    indent=2,
                    default=str,
                )
            print(f"✓ History saved to {filepath}")

        elif format == "csv":
            filepath = self.log_dir / f"{self.experiment_name}_history.csv"
            if self.history:
                keys = self.history[0].keys()
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.history)
                print(f"✓ History saved to {filepath}")

        else:
            raise ValueError(f"Unsupported format: {format}")

    def generate_summary(self):
        """
        生成實驗摘要

        Returns:
        --------
        dict
            摘要字典
        """
        if not self.history:
            return {}

        train_losses = [h["train_loss"] for h in self.history]
        train_accs = [h["train_acc"] for h in self.history]

        summary = {
            "experiment_name": self.experiment_name,
            "total_epochs": len(self.history),
            "final_train_loss": train_losses[-1],
            "final_train_acc": train_accs[-1],
            "best_train_loss": min(train_losses),
            "best_train_acc": max(train_accs),
            "mean_train_loss": np.mean(train_losses),
            "mean_train_acc": np.mean(train_accs),
        }

        # 如果有驗證集
        if self.history[0].get("val_loss") is not None:
            val_losses = [
                h["val_loss"] for h in self.history if h["val_loss"] is not None
            ]
            val_accs = [h["val_acc"] for h in self.history if h["val_acc"] is not None]

            if val_losses:
                summary.update(
                    {
                        "final_val_loss": val_losses[-1],
                        "final_val_acc": val_accs[-1],
                        "best_val_loss": min(val_losses),
                        "best_val_acc": max(val_accs),
                        "mean_val_loss": np.mean(val_losses),
                        "mean_val_acc": np.mean(val_accs),
                    }
                )

        return summary

    def print_summary(self):
        """打印實驗摘要"""
        summary = self.generate_summary()

        print("\n" + "=" * 70)
        print(f"Experiment Summary: {self.experiment_name}")
        print("=" * 70)

        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key:.<50} {value:.6f}")
            else:
                print(f"  {key:.<50} {value}")

        print("=" * 70 + "\n")


# ============================================================
# 測試程式碼
# ============================================================

if __name__ == "__main__":
    """
    測試訓練記錄器
    """
    print("=" * 70)
    print("訓練記錄器測試")
    print("=" * 70)

    # 創建記錄器
    logger = TrainingLogger(experiment_name="test_logger")

    # 記錄配置
    config = {
        "dataset": "Iris",
        "architecture": [4, 8, 4, 3],
        "activation": "tanh",
        "learning_rate": 0.2,
        "batch_size": 32,
    }
    logger.log_config(config)

    # 模擬訓練過程
    print("\n模擬訓練過程...")
    for epoch in range(100):
        train_loss = 0.5 * np.exp(-0.05 * epoch) + 0.01 * np.random.randn()
        train_acc = 1 - 0.5 * np.exp(-0.05 * epoch) + 0.01 * np.random.randn()
        val_loss = 0.5 * np.exp(-0.04 * epoch) + 0.02 * np.random.randn()
        val_acc = 1 - 0.5 * np.exp(-0.04 * epoch) + 0.02 * np.random.randn()

        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        if epoch % 20 == 0:
            print(
                f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
            )

    # 記錄訊息
    logger.log_message("Training completed successfully")

    # 記錄最終結果
    results = {"test_accuracy": 0.9567, "test_loss": 0.1234, "training_time": 12.34}
    logger.log_results(results)

    # 儲存歷史
    logger.save_history(format="json")
    logger.save_history(format="csv")

    # 打印摘要
    logger.print_summary()

    print("=" * 70)
