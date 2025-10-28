"""
診斷 BC 資料問題
"""

import pandas as pd

# 載入數據
df = pd.read_csv("results/logs/experiment_1_results.csv")

print("=" * 70)
print("診斷 Breast Cancer Wisconsin 資料")
print("=" * 70)

# 篩選 BC
bc_df = df[df["dataset"] == "Breast Cancer Wisconsin"]

print(f"\n總筆數: {len(bc_df)}")

print("\n各架構的數據量:")
for arch in ["small", "recommended"]:
    arch_data = bc_df[bc_df["architecture"] == arch]
    print(f"  {arch}: {len(arch_data)} 筆")

    if len(arch_data) > 0:
        acc_values = arch_data["test_acc"].values
        print(f"    準確率: {acc_values}")
        print(f"    平均值: {acc_values.mean():.4f}")
        print(f"    標準差: {acc_values.std():.4f}")

print("\n前 20 筆資料:")
print(
    bc_df[["architecture", "activation", "learning_rate", "fold", "test_acc"]].head(20)
)

print("\n檢查是否有 NaN:")
print(bc_df[["architecture", "test_acc"]].isnull().sum())

print("\n架構值的唯一值:")
print(bc_df["architecture"].unique())

print("\n" + "=" * 70)
