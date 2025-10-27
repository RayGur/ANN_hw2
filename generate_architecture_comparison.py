"""
生成架構對比視覺化圖表

讀取實驗一結果,生成:
1. 箱型圖: 詳細分布比較
2. 條狀圖: 平均表現比較
"""

import pandas as pd
from pathlib import Path
import sys

# 確保可以 import 專案模組
sys.path.append(str(Path(__file__).parent))

from utils.visualization import Visualizer
from config import OUTPUT_CONFIG


def main():
    """
    主函數
    """
    print("=" * 70)
    print("生成架構對比視覺化圖表")
    print("=" * 70)

    # 載入實驗一結果
    results_path = Path(OUTPUT_CONFIG["logs_dir"]) / "experiment_1_results.csv"

    if not results_path.exists():
        print(f"\n❌ 找不到實驗結果: {results_path}")
        print("請先執行實驗一: python experiments/experiment_1_architecture.py")
        return

    print(f"\n載入結果: {results_path}")
    df = pd.read_csv(results_path)

    print(f"✓ 載入 {len(df)} 筆實驗結果")
    print(f"  - 資料集: {df['dataset'].unique()}")
    print(f"  - 架構: {df['architecture'].unique()}")

    # 確認資料完整性
    required_cols = ["dataset", "architecture", "test_acc"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n❌ 缺少必要欄位: {missing_cols}")
        return

    # 生成箱型圖
    print("\n" + "-" * 70)
    print("[1] 生成架構對比箱型圖")
    print("-" * 70)

    boxplot_path = (
        Path(OUTPUT_CONFIG["figures_dir"]) / "architecture_comparison_boxplot.png"
    )
    Visualizer.plot_architecture_comparison_boxplot(
        df, save_path=str(boxplot_path), show=False
    )

    # 生成條狀圖
    print("\n" + "-" * 70)
    print("[2] 生成架構對比條狀圖")
    print("-" * 70)

    barplot_path = (
        Path(OUTPUT_CONFIG["figures_dir"]) / "architecture_comparison_barplot.png"
    )
    Visualizer.plot_architecture_comparison_barplot(
        df, save_path=str(barplot_path), show=False
    )

    # 生成統計摘要
    print("\n" + "-" * 70)
    print("[3] 統計摘要")
    print("-" * 70)

    for dataset in df["dataset"].unique():
        print(f"\n{dataset}:")
        dataset_df = df[df["dataset"] == dataset]

        for arch in ["small", "recommended"]:
            arch_data = dataset_df[dataset_df["architecture"] == arch]["test_acc"]
            if len(arch_data) > 0:
                print(
                    f"  {arch.capitalize():12s}: "
                    f"Mean={arch_data.mean():.4f}, "
                    f"Std={arch_data.std():.4f}, "
                    f"Max={arch_data.max():.4f}, "
                    f"Min={arch_data.min():.4f}, "
                    f"n={len(arch_data)}"
                )

    # 完成
    print("\n" + "=" * 70)
    print("✓ 架構對比圖表生成完成!")
    print("=" * 70)
    print(f"\n圖表位置:")
    print(f"  1. 箱型圖: {boxplot_path}")
    print(f"  2. 條狀圖: {barplot_path}")
    print("\n建議使用:")
    print("  - 箱型圖: 放在報告結果章節 (顯示完整分布)")
    print("  - 條狀圖: 放在摘要或結論 (快速比較)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
