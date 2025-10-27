"""
評估指標模組

提供分類任務的各種評估指標,包括:
1. 準確率 (Accuracy)
2. 混淆矩陣 (Confusion Matrix)
3. 精確率、召回率、F1分數 (Precision, Recall, F1-score)
4. 完整分類報告

設計用於多分類和二分類任務
"""

import numpy as np


class Metrics:
    """
    評估指標計算類別

    提供各種分類評估指標的靜態方法
    """

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        計算準確率

        公式: Accuracy = (正確預測數) / (總樣本數)

        Parameters:
        -----------
        y_true : np.ndarray
            真實標籤
        y_pred : np.ndarray
            預測標籤

        Returns:
        --------
        float
            準確率 (0 到 1 之間)

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 2, 1, 1])
        >>> acc = Metrics.accuracy(y_true, y_pred)
        >>> print(f"Accuracy: {acc:.2%}")
        Accuracy: 80.00%
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        correct = np.sum(y_true == y_pred)
        total = len(y_true)

        return correct / total if total > 0 else 0.0

    @staticmethod
    def confusion_matrix(y_true, y_pred, num_classes=None):
        """
        計算混淆矩陣

        混淆矩陣 [i, j] 表示:
        - 真實類別為 i
        - 被預測為類別 j
        的樣本數量

        Parameters:
        -----------
        y_true : np.ndarray
            真實標籤
        y_pred : np.ndarray
            預測標籤
        num_classes : int, optional
            類別數量,如果為 None 則自動檢測

        Returns:
        --------
        np.ndarray
            混淆矩陣 (num_classes × num_classes)

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 2, 1, 1, 2])
        >>> cm = Metrics.confusion_matrix(y_true, y_pred)
        >>> print(cm)
        [[1 1 0]
         [0 2 0]
         [0 0 2]]
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        if num_classes is None:
            num_classes = max(np.max(y_true), np.max(y_pred)) + 1

        # 初始化混淆矩陣
        cm = np.zeros((num_classes, num_classes), dtype=int)

        # 填充混淆矩陣
        for true_label, pred_label in zip(y_true, y_pred):
            cm[true_label, pred_label] += 1

        return cm

    @staticmethod
    def precision_recall_f1(y_true, y_pred, average="macro", zero_division=0):
        """
        計算精確率、召回率和 F1 分數

        公式:
        - Precision = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - F1 = 2 * (Precision * Recall) / (Precision + Recall)

        Parameters:
        -----------
        y_true : np.ndarray
            真實標籤
        y_pred : np.ndarray
            預測標籤
        average : str
            平均方式:
            - 'macro': 計算每個類別的指標,然後取平均
            - 'micro': 全局計算 (對不平衡資料集較公平)
            - 'binary': 二分類 (只返回正類的指標)
            - None: 返回每個類別的指標
        zero_division : float
            當分母為 0 時的返回值 (default: 0)

        Returns:
        --------
        tuple or dict
            如果 average=None: 返回 (precision_array, recall_array, f1_array)
            否則: 返回 (precision, recall, f1)

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> p, r, f1 = Metrics.precision_recall_f1(y_true, y_pred, average='macro')
        >>> print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # 獲取類別
        classes = np.unique(np.concatenate([y_true, y_pred]))
        num_classes = len(classes)

        # 計算每個類別的 precision, recall, f1
        precisions = []
        recalls = []
        f1_scores = []

        for cls in classes:
            # True Positive: 預測為 cls 且真實為 cls
            tp = np.sum((y_pred == cls) & (y_true == cls))

            # False Positive: 預測為 cls 但真實不是 cls
            fp = np.sum((y_pred == cls) & (y_true != cls))

            # False Negative: 預測不是 cls 但真實是 cls
            fn = np.sum((y_pred != cls) & (y_true == cls))

            # 計算 Precision
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = zero_division

            # 計算 Recall
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = zero_division

            # 計算 F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = zero_division

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1_scores = np.array(f1_scores)

        # 根據 average 參數返回結果
        if average is None:
            return precisions, recalls, f1_scores

        elif average == "macro":
            # 宏平均: 每個類別權重相同
            return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

        elif average == "micro":
            # 微平均: 全局計算
            cm = Metrics.confusion_matrix(y_true, y_pred, num_classes)

            tp_total = np.trace(cm)  # 對角線和
            fp_total = np.sum(cm) - tp_total
            fn_total = fp_total  # 對於 micro 平均, FP = FN

            precision = (
                tp_total / (tp_total + fp_total)
                if (tp_total + fp_total) > 0
                else zero_division
            )
            recall = (
                tp_total / (tp_total + fn_total)
                if (tp_total + fn_total) > 0
                else zero_division
            )

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = zero_division

            return precision, recall, f1

        elif average == "binary":
            # 二分類: 只返回正類 (假設正類是 1)
            if num_classes != 2:
                raise ValueError(
                    "'binary' average only works for binary classification"
                )
            return precisions[1], recalls[1], f1_scores[1]

        else:
            raise ValueError(f"Unsupported average type: {average}")

    @staticmethod
    def classification_report(y_true, y_pred, class_names=None, digits=3):
        """
        生成完整的分類報告

        包含每個類別的 precision, recall, f1-score 和 support

        Parameters:
        -----------
        y_true : np.ndarray
            真實標籤
        y_pred : np.ndarray
            預測標籤
        class_names : list, optional
            類別名稱列表
        digits : int
            小數點位數 (default: 3)

        Returns:
        --------
        str
            格式化的分類報告

        Examples:
        ---------
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 1, 0, 1, 2])
        >>> report = Metrics.classification_report(y_true, y_pred)
        >>> print(report)
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # 獲取類別
        classes = np.unique(np.concatenate([y_true, y_pred]))
        num_classes = len(classes)

        # 如果沒有提供類別名稱,使用數字
        if class_names is None:
            class_names = [f"Class {i}" for i in classes]

        # 計算每個類別的指標
        precisions, recalls, f1_scores = Metrics.precision_recall_f1(
            y_true, y_pred, average=None
        )

        # 計算每個類別的樣本數 (support)
        supports = [np.sum(y_true == cls) for cls in classes]

        # 計算整體準確率
        accuracy = Metrics.accuracy(y_true, y_pred)

        # 計算宏平均和微平均
        macro_p, macro_r, macro_f1 = Metrics.precision_recall_f1(
            y_true, y_pred, average="macro"
        )

        # 構建報告字符串
        report_lines = []

        # 標題行
        header = f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
        report_lines.append(header)
        report_lines.append("-" * len(header))

        # 每個類別的指標
        for i, cls in enumerate(classes):
            line = (
                f"{class_names[i]:>15} "
                f"{precisions[i]:>10.{digits}f} "
                f"{recalls[i]:>10.{digits}f} "
                f"{f1_scores[i]:>10.{digits}f} "
                f"{supports[i]:>10}"
            )
            report_lines.append(line)

        report_lines.append("")

        # 準確率
        acc_line = (
            f"{'accuracy':>15} "
            f"{'':>10} "
            f"{'':>10} "
            f"{accuracy:>10.{digits}f} "
            f"{len(y_true):>10}"
        )
        report_lines.append(acc_line)

        # 宏平均
        macro_line = (
            f"{'macro avg':>15} "
            f"{macro_p:>10.{digits}f} "
            f"{macro_r:>10.{digits}f} "
            f"{macro_f1:>10.{digits}f} "
            f"{len(y_true):>10}"
        )
        report_lines.append(macro_line)

        return "\n".join(report_lines)

    @staticmethod
    def print_confusion_matrix(cm, class_names=None):
        """
        美化打印混淆矩陣

        Parameters:
        -----------
        cm : np.ndarray
            混淆矩陣
        class_names : list, optional
            類別名稱列表

        Examples:
        ---------
        >>> cm = np.array([[10, 2, 0], [1, 12, 1], [0, 0, 15]])
        >>> Metrics.print_confusion_matrix(cm, ['Class A', 'Class B', 'Class C'])
        """
        num_classes = cm.shape[0]

        if class_names is None:
            class_names = [f"C{i}" for i in range(num_classes)]

        # 計算列寬
        col_width = max(max(len(name) for name in class_names), 6) + 2

        # 打印標題
        print("\nConfusion Matrix:")
        print("=" * (col_width * (num_classes + 1) + 10))

        # 打印列標題
        header = f"{'True  Pred':>{col_width}}"
        for name in class_names:
            header += f"{name:>{col_width}}"
        print(header)
        print("-" * len(header))

        # 打印每一行
        for i, name in enumerate(class_names):
            row = f"{name:>{col_width}}"
            for j in range(num_classes):
                row += f"{cm[i, j]:>{col_width}}"
            print(row)

        print("=" * (col_width * (num_classes + 1) + 10))


# ============================================================
# 測試程式碼
# ============================================================

if __name__ == "__main__":
    """
    測試評估指標模組的各項功能
    """
    print("=" * 70)
    print("評估指標模組測試")
    print("=" * 70)

    # 創建測試資料 (三分類問題)
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 3, n_samples)
    # 模擬 85% 準確率的預測
    y_pred = y_true.copy()
    noise_indices = np.random.choice(n_samples, size=15, replace=False)
    y_pred[noise_indices] = np.random.randint(0, 3, 15)

    print("\n[1] 測試準確率")
    print("-" * 70)
    acc = Metrics.accuracy(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    print("\n[2] 測試混淆矩陣")
    print("-" * 70)
    cm = Metrics.confusion_matrix(y_true, y_pred, num_classes=3)
    Metrics.print_confusion_matrix(cm, ["Setosa", "Versicolor", "Virginica"])

    print("\n[3] 測試 Precision, Recall, F1-score")
    print("-" * 70)

    # 每個類別的指標
    print("Per-class metrics:")
    precisions, recalls, f1s = Metrics.precision_recall_f1(y_true, y_pred, average=None)
    for i, (p, r, f1) in enumerate(zip(precisions, recalls, f1s)):
        print(f"  Class {i}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")

    # 宏平均
    print("\nMacro average:")
    p_macro, r_macro, f1_macro = Metrics.precision_recall_f1(
        y_true, y_pred, average="macro"
    )
    print(f"  Precision={p_macro:.3f}, Recall={r_macro:.3f}, F1={f1_macro:.3f}")

    # 微平均
    print("\nMicro average:")
    p_micro, r_micro, f1_micro = Metrics.precision_recall_f1(
        y_true, y_pred, average="micro"
    )
    print(f"  Precision={p_micro:.3f}, Recall={r_micro:.3f}, F1={f1_micro:.3f}")

    print("\n[4] 測試完整分類報告")
    print("-" * 70)
    report = Metrics.classification_report(
        y_true, y_pred, class_names=["Setosa", "Versicolor", "Virginica"]
    )
    print(report)

    print("\n[5] 測試二分類場景 (針對 Breast Cancer)")
    print("-" * 70)
    # 模擬二分類
    y_true_binary = np.random.randint(0, 2, 100)
    y_pred_binary = y_true_binary.copy()
    noise_idx = np.random.choice(100, size=10, replace=False)
    y_pred_binary[noise_idx] = 1 - y_pred_binary[noise_idx]

    cm_binary = Metrics.confusion_matrix(y_true_binary, y_pred_binary, num_classes=2)
    Metrics.print_confusion_matrix(cm_binary, ["Benign", "Malignant"])

    p, r, f1 = Metrics.precision_recall_f1(
        y_true_binary, y_pred_binary, average="binary"
    )
    print(f"\nBinary classification metrics (for Malignant class):")
    print(f"  Precision: {p:.3f}")
    print(f"  Recall:    {r:.3f}")
    print(f"  F1-score:  {f1:.3f}")

    print("\n" + "=" * 70)
    print("測試完成!")
    print("=" * 70)
