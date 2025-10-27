"""
MLP Classification Project - 主執行檔

統一的實驗執行入口

使用方法:
    python main.py --experiment all          # 執行所有實驗
    python main.py --experiment exp1         # 只執行實驗一
    python main.py --experiment exp2         # 只執行實驗二
    python main.py --config                  # 顯示配置摘要
    python main.py --test                    # 執行快速測試
"""

import argparse
import sys
from pathlib import Path

# 確保可以 import 專案模組
sys.path.append(str(Path(__file__).parent))

from config import print_config_summary
from experiments.experiment_1_architecture import Experiment1
from experiments.experiment_2_training_methods import Experiment2


def print_header():
    """打印專案標題"""
    print("\n" + "=" * 70)
    print(" " * 10 + "MLP Classification Project")
    print(" " * 5 + "Multilayer Perceptron with Backpropagation")
    print("=" * 70 + "\n")


def run_quick_test():
    """
    執行快速測試

    用小資料集和少量 epochs 快速驗證程式正確性
    """
    print("\n" + "=" * 70)
    print("快速測試模式")
    print("=" * 70)
    print("\n使用 XOR 資料集進行快速功能驗證...")

    import numpy as np
    from src.mlp import MLP
    from src.trainers import StandardBPTrainer, MomentumTrainer, ResilientPropTrainer
    from src.metrics import Metrics

    # XOR 資料
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    results = []

    # 測試 1: Standard BP
    print("\n[1] 測試標準 BP")
    print("-" * 70)
    mlp1 = MLP([2, 4, 2], activation="tanh", learning_rate=0.5, random_seed=42)
    trainer1 = StandardBPTrainer(mlp1, max_epochs=3000, verbose=False)
    trainer1.train(X, y)

    pred1 = mlp1.predict(X)
    acc1 = Metrics.accuracy(y, pred1)
    print(f"✓ Standard BP - Accuracy: {acc1:.2%}, Epochs: {trainer1.history['epochs']}")
    results.append(("Standard BP", acc1))

    # 測試 2: Momentum (改進版)
    print("\n[2] 測試 Momentum")
    print("-" * 70)
    mlp2 = MLP([2, 8, 2], activation="tanh", learning_rate=0.3, random_seed=123)
    trainer2 = MomentumTrainer(
        mlp2, momentum=0.9, max_epochs=5000, convergence_patience=100, verbose=False
    )
    trainer2.train(X, y)

    pred2 = mlp2.predict(X)
    acc2 = Metrics.accuracy(y, pred2)
    print(
        f"✓ Momentum (α=0.9) - Accuracy: {acc2:.2%}, Epochs: {trainer2.history['epochs']}"
    )
    results.append(("Momentum", acc2))

    # 測試 3: RProp
    print("\n[3] 測試 RProp")
    print("-" * 70)
    mlp3 = MLP([2, 6, 2], activation="tanh", learning_rate=0.1, random_seed=456)
    trainer3 = ResilientPropTrainer(mlp3, max_epochs=3000, verbose=False)
    trainer3.train(X, y)

    pred3 = mlp3.predict(X)
    acc3 = Metrics.accuracy(y, pred3)
    print(f"✓ RProp - Accuracy: {acc3:.2%}, Epochs: {trainer3.history['epochs']}")
    results.append(("RProp", acc3))

    # 測試 4: 評估指標
    print("\n[4] 測試評估指標")
    print("-" * 70)
    # 使用最佳結果
    best_pred = pred1 if acc1 >= acc2 else pred2
    cm = Metrics.confusion_matrix(y, best_pred, num_classes=2)
    print(f"✓ Confusion Matrix:\n{cm}")

    # 總結
    print("\n" + "=" * 70)
    print("測試結果總結:")
    print("-" * 70)
    for name, acc in results:
        status = "✓" if acc >= 0.75 else "⚠"
        print(f"  {status} {name:20s}: {acc:.2%}")

    success_count = sum(acc >= 0.75 for _, acc in results)
    print("-" * 70)

    if success_count >= 2:
        print(f"✓ 快速測試通過! ({success_count}/{len(results)} 個方法達標)")
        print("  系統運作正常,可以執行完整實驗")
    else:
        print(f"⚠ 快速測試部分失敗 ({success_count}/{len(results)} 個方法達標)")
        print("  建議:")
        print("    1. 檢查是否需要調整超參數")
        print("    2. 這可能是隨機初始化導致,實際實驗會測試多組配置")

    print("=" * 70 + "\n")


def run_experiment_1():
    """執行實驗一: 架構容量對比"""
    print("\n" + "=" * 70)
    print("開始執行實驗一: 架構容量對比")
    print("=" * 70)
    print("\n實驗說明:")
    print("  - 比較小容量 vs 推薦容量架構")
    print("  - 測試 3 種激活函數 (Sigmoid, Tanh, ReLU)")
    print("  - 探索 3 種學習率 (0.05, 0.2, 0.5)")
    print("  - 涵蓋 3 個資料集 (Iris, Wine, BC)")
    print("\n預計執行時間: 10-30 分鐘 (視硬體而定)")

    response = input("\n確定要執行? (y/n): ")
    if response.lower() != "y":
        print("已取消執行")
        return

    # 執行實驗
    exp = Experiment1(save_results=True)
    exp.run_full_experiment()

    # 分析結果
    df = exp.analyze_results()

    # 視覺化
    if df is not None:
        exp.visualize_results(df)
        exp.save_results_to_file(df)

    print("\n✓ 實驗一完成!")
    print(f"✓ 結果已儲存至 results/ 目錄")


def run_experiment_2():
    """執行實驗二: 訓練方法對比"""
    print("\n" + "=" * 70)
    print("開始執行實驗二: 訓練方法對比")
    print("=" * 70)
    print("\n實驗說明:")
    print("  - 使用實驗一找出的最佳配置")
    print("  - 比較 4 種訓練方法:")
    print("    * Standard Backpropagation")
    print("    * Momentum (α=0.5, 0.7, 0.9)")
    print("    * Resilient Propagation (RProp)")
    print("    * Levenberg-Marquardt")
    print("\n預計執行時間: 5-15 分鐘 (視硬體而定)")

    # 檢查實驗一結果是否存在
    exp1_results = Path("results/logs/experiment_1_results.csv")
    if not exp1_results.exists():
        print("\n⚠ 警告: 找不到實驗一的結果檔案")
        print("建議先執行實驗一,或將使用預設配置")
        response = input("\n繼續執行? (y/n): ")
        if response.lower() != "y":
            print("已取消執行")
            return

    response = input("\n確定要執行? (y/n): ")
    if response.lower() != "y":
        print("已取消執行")
        return

    # 執行實驗
    exp = Experiment2(save_results=True)
    exp.run_full_experiment()

    # 分析結果
    df = exp.analyze_results()

    # 視覺化
    if df is not None:
        exp.visualize_results(df)
        exp.save_results_to_file(df)

    print("\n✓ 實驗二完成!")
    print(f"✓ 結果已儲存至 results/ 目錄")


def run_all_experiments():
    """執行所有實驗"""
    print("\n" + "=" * 70)
    print("執行所有實驗")
    print("=" * 70)
    print("\n將依序執行:")
    print("  1. 實驗一: 架構容量對比")
    print("  2. 實驗二: 訓練方法對比")
    print("\n預計總時間: 15-45 分鐘 (視硬體而定)")

    response = input("\n確定要執行? (y/n): ")
    if response.lower() != "y":
        print("已取消執行")
        return

    # 執行實驗一
    print("\n" + "#" * 70)
    print("# 第 1/2 階段: 實驗一")
    print("#" * 70)

    exp1 = Experiment1(save_results=True)
    exp1.run_full_experiment()
    df1 = exp1.analyze_results()
    if df1 is not None:
        exp1.visualize_results(df1)
        exp1.save_results_to_file(df1)

    print("\n✓ 實驗一完成!")

    # 執行實驗二
    print("\n" + "#" * 70)
    print("# 第 2/2 階段: 實驗二")
    print("#" * 70)

    exp2 = Experiment2(save_results=True)
    exp2.run_full_experiment()
    df2 = exp2.analyze_results()
    if df2 is not None:
        exp2.visualize_results(df2)
        exp2.save_results_to_file(df2)

    print("\n✓ 實驗二完成!")

    # 總結
    print("\n" + "=" * 70)
    print("✓ 所有實驗完成!")
    print("=" * 70)
    print("\n結果已儲存至:")
    print(f"  - results/figures/  (圖表)")
    print(f"  - results/logs/     (日誌和數據)")
    print("\n建議下一步:")
    print("  1. 查看 results/logs/experiment_*_summary.txt 了解摘要")
    print("  2. 檢視 results/figures/ 中的視覺化圖表")
    print("  3. 使用 results/logs/experiment_*_results.csv 進行深入分析")
    print("=" * 70 + "\n")


def show_config():
    """顯示配置摘要"""
    print_config_summary()


def main():
    """主函數"""
    # 解析命令列參數
    parser = argparse.ArgumentParser(
        description="MLP Classification Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main.py --experiment all      執行所有實驗
  python main.py --experiment exp1     執行實驗一
  python main.py --experiment exp2     執行實驗二
  python main.py --config              顯示配置摘要
  python main.py --test                執行快速測試
        """,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["all", "exp1", "exp2"],
        help="選擇要執行的實驗",
    )

    parser.add_argument("--config", action="store_true", help="顯示配置摘要")

    parser.add_argument("--test", action="store_true", help="執行快速測試")

    args = parser.parse_args()

    # 打印標題
    print_header()

    # 執行對應功能
    if args.config:
        show_config()

    elif args.test:
        run_quick_test()

    elif args.experiment == "all":
        run_all_experiments()

    elif args.experiment == "exp1":
        run_experiment_1()

    elif args.experiment == "exp2":
        run_experiment_2()

    else:
        # 沒有參數,顯示互動式選單
        print("請選擇要執行的操作:")
        print("  1. 執行所有實驗")
        print("  2. 執行實驗一 (架構容量對比)")
        print("  3. 執行實驗二 (訓練方法對比)")
        print("  4. 快速測試")
        print("  5. 顯示配置摘要")
        print("  0. 退出")

        while True:
            try:
                choice = input("\n請輸入選項 (0-5): ").strip()

                if choice == "0":
                    print("再見!")
                    break
                elif choice == "1":
                    run_all_experiments()
                    break
                elif choice == "2":
                    run_experiment_1()
                    break
                elif choice == "3":
                    run_experiment_2()
                    break
                elif choice == "4":
                    run_quick_test()
                    break
                elif choice == "5":
                    show_config()
                    break
                else:
                    print("無效的選項,請重新輸入")
            except KeyboardInterrupt:
                print("\n\n程式中斷")
                break
            except Exception as e:
                print(f"發生錯誤: {e}")
                break


if __name__ == "__main__":
    main()
