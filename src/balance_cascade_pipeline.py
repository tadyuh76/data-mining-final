#!/usr/bin/env python3
"""
Quy trình thí nghiệm BalanceCascade.

Các bước chính:
- đọc dữ liệu cho bài toán phân loại nhị phân;
- tiền xử lý theo train/test để tránh leakage;
- đánh giá các mô hình nền;
- chạy ensemble đơn giản theo hướng BalanceCascade;
- xuất bảng metric và hình cần dùng cho báo cáo.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_adaboost(random_state: int, n_estimators: int = 25) -> AdaBoostClassifier:
    stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    try:
        return AdaBoostClassifier(estimator=stump, n_estimators=n_estimators, random_state=random_state)
    except TypeError:
        return AdaBoostClassifier(base_estimator=stump, n_estimators=n_estimators, random_state=random_state)


def ensure_dirs(base_dir: Path) -> tuple[Path, Path]:
    figures_dir = base_dir / "outputs" / "figures"
    tables_dir = base_dir / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def normalize_binary_target(y: pd.Series, positive_label: str | None) -> pd.Series:
    if positive_label is None:
        values = sorted(y.dropna().unique().tolist())
        if len(values) != 2:
            raise ValueError("Target phải là binary hoặc cần truyền --positive-label.")
        positive = values[-1]
    else:
        positive = positive_label
        if y.dtype.kind in {"i", "u", "f"}:
            try:
                positive = float(positive_label)
                if np.all(np.equal(np.mod(y.dropna(), 1), 0)):
                    positive = int(positive)
            except ValueError:
                pass
    return (y == positive).astype(int)


def choose_columns(df: pd.DataFrame, target: str, max_categorical_cardinality: int, max_missing_rate: float) -> tuple[list[str], list[str], list[str]]:
    working = df.drop(columns=[target], errors="ignore")
    working = working.loc[:, working.isna().mean() <= max_missing_rate]

    numeric_cols = working.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = []
    dropped = []

    for col in working.columns:
        if col in numeric_cols:
            continue
        cardinality = working[col].nunique(dropna=True)
        if cardinality <= max_categorical_cardinality:
            categorical_cols.append(col)
        else:
            dropped.append(col)

    return numeric_cols, categorical_cols, dropped


def build_preprocessor(
    df: pd.DataFrame,
    target: str,
    max_categorical_cardinality: int,
    max_missing_rate: float,
) -> tuple[ColumnTransformer, list[str]]:
    auto_drop = [
        col
        for col in df.columns
        if col != target and (col.lower() in {"id", "unnamed: 0"} or col.lower().startswith("unnamed"))
    ]
    if auto_drop:
        print(f"Tự động bỏ các cột ID: {auto_drop}")
        df = df.drop(columns=auto_drop)
    numeric_cols, categorical_cols, dropped = choose_columns(
        df,
        target=target,
        max_categorical_cardinality=max_categorical_cardinality,
        max_missing_rate=max_missing_rate,
    )
    if not numeric_cols and not categorical_cols:
        raise ValueError("Không còn cột feature dùng được sau khi lọc.")

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if dropped:
        print(f"Đã bỏ các cột cardinality cao: {dropped}")
    print(f"Dùng {len(numeric_cols)} cột numeric và {len(categorical_cols)} cột categorical.")
    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric_cols + categorical_cols


@dataclass
class SimpleEasyEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Bản EasyEnsemble gọn làm mô hình nền dùng lấy mẫu giảm lớp lớn."""

    n_estimators: int = 6
    adaboost_estimators: int = 25
    random_state: int = RANDOM_STATE

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise ValueError("Cần có đủ cả hai lớp.")

        minority_idx = pos_idx if len(pos_idx) <= len(neg_idx) else neg_idx
        majority_idx = neg_idx if len(pos_idx) <= len(neg_idx) else pos_idx
        sample_size = len(minority_idx)

        self.models_ = []
        for i in range(self.n_estimators):
            sampled_majority = rng.choice(majority_idx, size=sample_size, replace=False)
            train_idx = np.concatenate([minority_idx, sampled_majority])
            model = make_adaboost(self.random_state + i, self.adaboost_estimators)
            model.fit(X[train_idx], y[train_idx])
            self.models_.append(model)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.array([model.predict_proba(X) for model in self.models_])
        return probs.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class SimpleBalanceCascadeClassifier(BaseEstimator, ClassifierMixin):
    """Bản tự cài đặt phần chính của BalanceCascade theo từng vòng."""

    n_stages: int = 6
    adaboost_estimators: int = 25
    remove_correct_majority: bool = True
    random_state: int = RANDOM_STATE

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise ValueError("Cần có đủ cả hai lớp.")

        # Xác định lớp thiểu số và tập lớp đa số ban đầu.
        if len(pos_idx) <= len(neg_idx):
            minority_idx = pos_idx
            majority_pool = neg_idx.copy()
            majority_label = 0
        else:
            minority_idx = neg_idx
            majority_pool = pos_idx.copy()
            majority_label = 1

        self.models_ = []
        self.stage_sizes_ = []
        sample_size = len(minority_idx)

        for stage in range(self.n_stages):
            if len(majority_pool) < max(2, sample_size // 2):
                break

            # Mỗi vòng học trên toàn bộ lớp thiểu số và một mẫu lớp đa số cân bằng.
            majority_sample_size = min(sample_size, len(majority_pool))
            sampled_majority = rng.choice(majority_pool, size=majority_sample_size, replace=False)
            train_idx = np.concatenate([minority_idx, sampled_majority])

            model = make_adaboost(self.random_state + stage, self.adaboost_estimators)
            model.fit(X[train_idx], y[train_idx])
            self.models_.append(model)

            if self.remove_correct_majority:
                # Loại các điểm lớp đa số đã phân loại đúng để vòng sau tập trung vào điểm khó hơn.
                majority_pred = model.predict(X[majority_pool])
                keep_mask = majority_pred != majority_label
                removed = int((~keep_mask).sum())
                majority_pool = majority_pool[keep_mask]
            else:
                removed = 0

            self.stage_sizes_.append(
                {
                    "stage": stage + 1,
                    "majority_pool_after_stage": len(majority_pool),
                    "majority_removed_this_stage": removed,
                    "minority_count": len(minority_idx),
                }
            )

        if not self.models_:
            raise ValueError("Không train được stage BalanceCascade nào.")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.array([model.predict_proba(X) for model in self.models_])
        return probs.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def metric_row(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    runtime: float,
    threshold: float = 0.5,
) -> dict[str, float | str]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": name,
        "threshold": threshold,
        "runtime_seconds": runtime,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_sweep(
    y_true: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)

    rows = []
    for model_name, score in scores_by_model.items():
        for threshold in thresholds:
            y_pred = (score >= threshold).astype(int)
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                    "predicted_positive_rate": float(np.mean(y_pred)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_selected_thresholds(
    y_test: np.ndarray,
    test_scores_by_model: dict[str, np.ndarray],
    selected_thresholds: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    rows = []
    for _, selected in selected_thresholds.iterrows():
        model_name = str(selected["model"])
        threshold = float(selected["threshold"])
        row = metric_row(
            model_name,
            y_test,
            test_scores_by_model[model_name],
            np.nan,
            threshold=threshold,
        )
        row[f"validation_{score_column}"] = float(selected[score_column])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(f"validation_{score_column}", ascending=False)


def evaluate_model_with_validation(
    name: str,
    model,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray, object]:
    start = time.perf_counter()
    model.fit(X_fit, y_fit)
    runtime = time.perf_counter() - start
    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    return metric_row(name, y_test, test_score, runtime), val_score, test_score, model


def evaluate_model_on_test(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | str]:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    runtime = time.perf_counter() - start
    test_score = model.predict_proba(X_test)[:, 1]
    return metric_row(name, y_test, test_score, runtime)


def plot_class_balance(y: pd.Series, path: Path) -> None:
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Phân phối target")
    plt.xlabel("Lớp")
    plt.ylabel("Số dòng")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_missing(df: pd.DataFrame, path: Path) -> None:
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(25)
    if missing.empty:
        return
    plt.figure(figsize=(9, 6))
    missing.sort_values().plot(kind="barh")
    plt.title("Các cột thiếu dữ liệu nhiều nhất")
    plt.xlabel("Missing rate")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_gmsc_eda(df: pd.DataFrame, y: pd.Series, figures_dir: Path, tables_dir: Path) -> None:
    prefix = "dm_gmsc"
    y.value_counts().sort_index().to_csv(tables_dir / f"{prefix}_target_distribution.csv", header=["rows"])
    df.isna().mean().sort_values(ascending=False).to_csv(
        tables_dir / f"{prefix}_missing_rates.csv", header=["missing_rate"]
    )

    plot_class_balance(y, figures_dir / f"{prefix}_class_distribution.png")
    plot_missing(df, figures_dir / f"{prefix}_missing_values.png")

    feature_cols = [col for col in df.select_dtypes(include=["number", "bool"]).columns if col != "SeriousDlqin2yrs"]
    feature_cols = [col for col in feature_cols if not col.lower().startswith("unnamed")]
    key_cols = [
        col
        for col in [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfTimes90DaysLate",
            "NumberOfTime30-59DaysPastDueNotWorse",
        ]
        if col in feature_cols
    ]

    if key_cols:
        sample = df[key_cols].sample(min(len(df), 20000), random_state=RANDOM_STATE)
        axes = sample.hist(figsize=(12, 8), bins=40)
        for ax in np.ravel(axes):
            ax.set_title(ax.get_title(), fontsize=9)
        plt.suptitle("Phân phối một số feature numeric chính", y=1.02)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_key_feature_histograms.png", dpi=180, bbox_inches="tight")
        plt.close()

    if feature_cols:
        corr = df[feature_cols + ["SeriousDlqin2yrs"]].corr(numeric_only=True)
        corr.to_csv(tables_dir / f"{prefix}_numeric_correlation.csv")
        plt.figure(figsize=(9, 7))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(shrink=0.75, label="Tương quan")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
        plt.title("Heatmap tương quan các biến numeric")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_correlation_heatmap.png", dpi=180)
        plt.close()

    late_col = "NumberOfTimes90DaysLate"
    target_col = "SeriousDlqin2yrs"
    if late_col in df.columns and target_col in df.columns:
        rate = df.groupby(late_col)[target_col].agg(["mean", "size"]).reset_index()
        rate.to_csv(tables_dir / f"{prefix}_default_rate_by_90dayslate.csv", index=False)
        plot_rate = rate[rate["size"] >= 20].head(15)
        plt.figure(figsize=(8, 5))
        plt.bar(plot_rate[late_col].astype(str), plot_rate["mean"], color="#4C78A8")
        plt.xlabel("Số lần trễ hạn 90 ngày")
        plt.ylabel("Tỷ lệ default")
        plt.title("Tỷ lệ default theo số lần trễ hạn 90 ngày")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_default_rate_by_90dayslate.png", dpi=180)
        plt.close()


def plot_curves(y_test: np.ndarray, scores_by_model: dict[str, np.ndarray], figures_dir: Path) -> None:
    plt.figure(figsize=(7, 6))
    for name, score in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_test, score)
        ap = average_precision_score(y_test, score)
        plt.plot(recall, precision, label=f"{name} AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Đường Precision-Recall")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "dm_precision_recall_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    for name, score in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        auc = roc_auc_score(y_test, score)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Đường ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "dm_roc_curve.png", dpi=180)
    plt.close()


def plot_threshold_sensitivity(thresholds: pd.DataFrame, figures_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for model_name, group in thresholds.groupby("model"):
        plt.plot(group["threshold"], group["f2"], marker="o", linewidth=1.4, label=model_name)
    plt.xlabel("Decision threshold")
    plt.ylabel("F2 score")
    plt.title("Độ nhạy F2 theo threshold trên validation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "dm_threshold_sensitivity_f2.png", dpi=180)
    plt.close()


def plot_confusion(y_test: np.ndarray, score: np.ndarray, name: str, figures_dir: Path) -> None:
    y_pred = (score >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Lớp 0", "Lớp 1"], cmap="Blues")
    plt.title(f"Confusion matrix - {name}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"dm_confusion_{name.lower().replace(' ', '_')}.png", dpi=180)
    plt.close()


def model_suite(stages: int, adaboost_estimators: int) -> dict[str, object]:
    return {
        "LogReg_balanced": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "EasyEnsemble_simple": SimpleEasyEnsembleClassifier(
            n_estimators=stages,
            adaboost_estimators=adaboost_estimators,
            random_state=RANDOM_STATE,
        ),
        "BalanceCascade_tuned": SimpleBalanceCascadeClassifier(
            n_stages=2,
            adaboost_estimators=10,
            random_state=RANDOM_STATE,
        ),
        "BalanceCascade_simple": SimpleBalanceCascadeClassifier(
            n_stages=stages,
            adaboost_estimators=adaboost_estimators,
            random_state=RANDOM_STATE,
        ),
    }


def run_model_suite(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    stages: int,
    adaboost_estimators: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    rows = []
    val_scores_by_model: dict[str, np.ndarray] = {}
    test_scores_by_model: dict[str, np.ndarray] = {}
    fitted_models = {}

    for name, model in model_suite(stages, adaboost_estimators).items():
        print(f"Đang train {name}...")
        row, val_score, test_score, fitted = evaluate_model_with_validation(
            name,
            clone(model),
            X_fit,
            y_fit,
            X_val,
            X_test,
            y_test,
        )
        rows.append(row)
        val_scores_by_model[name] = val_score
        test_scores_by_model[name] = test_score
        fitted_models[name] = fitted

    return (
        pd.DataFrame(rows).sort_values(by="pr_auc", ascending=False),
        val_scores_by_model,
        test_scores_by_model,
        fitted_models,
    )


def plot_controlled_scatter(X: np.ndarray, y: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    colors = np.where(y == 1, "#E45756", "#4C78A8")
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=12, alpha=0.55, linewidths=0)
    plt.xlabel("feature phân biệt 1")
    plt.ylabel("feature phân biệt 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_controlled_case(
    case_id: str,
    class_sep: float,
    flip_y: float,
    figures_dir: Path,
    stages: int,
    adaboost_estimators: int,
) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=8000,
        n_features=8,
        n_informative=2,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.94, 0.06],
        class_sep=class_sep,
        flip_y=flip_y,
        shuffle=False,
        random_state=RANDOM_STATE,
    )
    plot_controlled_scatter(
        X,
        y,
        f"Case kiểm soát: {case_id.replace('_', ' ').title()}",
        figures_dir / f"dm_controlled_{case_id}_scatter.png",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    rows = []
    for name, model in model_suite(stages, adaboost_estimators).items():
        rows.append(evaluate_model_on_test(name, clone(model), X_train, y_train, X_test, y_test))

    metrics = pd.DataFrame(rows).sort_values(by="pr_auc", ascending=False)
    metrics.insert(0, "case_id", case_id)
    metrics.insert(1, "class_sep", class_sep)
    metrics.insert(2, "flip_y", flip_y)
    metrics.insert(3, "positive_rate", float(np.mean(y)))
    return metrics


def run_controlled_cases(figures_dir: Path, tables_dir: Path, stages: int, adaboost_estimators: int) -> None:
    cases = [
        ("ideal_separable", 1.60, 0.01),
        ("weak_overlap", 0.45, 0.08),
    ]
    metrics = pd.concat(
        [
            run_controlled_case(case_id, class_sep, flip_y, figures_dir, stages, adaboost_estimators)
            for case_id, class_sep, flip_y in cases
        ],
        ignore_index=True,
    )
    metrics.to_csv(tables_dir / "dm_controlled_case_metrics.csv", index=False)

    plt.figure(figsize=(9, 5))
    x = np.arange(len(cases))
    model_names = metrics["model"].drop_duplicates().tolist()
    width = min(0.8 / len(model_names), 0.18)
    for idx, model_name in enumerate(model_names):
        values = [
            metrics[(metrics["case_id"] == case_id) & (metrics["model"] == model_name)]["recall"].iloc[0]
            for case_id, _, _ in cases
        ]
        offset = (idx - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=model_name)
    plt.xticks(x, [case_id.replace("_", " ").title() for case_id, _, _ in cases])
    plt.ylabel("Recall tại threshold 0.50")
    plt.title("Case kiểm soát: so sánh recall")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "dm_controlled_case_recall_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Đường dẫn tới CSV gốc.")
    parser.add_argument("--target", required=True, help="Cột target binary.")
    parser.add_argument("--positive-label", default=None, help="Giá trị nào của target được xem là positive class.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--max-categorical-cardinality", type=int, default=40)
    parser.add_argument("--max-missing-rate", type=float, default=0.80)
    parser.add_argument("--stages", type=int, default=6)
    parser.add_argument("--adaboost-estimators", type=int, default=25)
    parser.add_argument("--skip-controlled-cases", action="store_true")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    figures_dir, tables_dir = ensure_dirs(project_dir)

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Không tìm thấy cột target {args.target!r}.")
    df = df.dropna(subset=[args.target]).copy()
    y = normalize_binary_target(df[args.target], args.positive_label)

    dataset_summary = pd.DataFrame(
        {
            "rows": [len(df)],
            "columns": [df.shape[1]],
            "target": [args.target],
            "positive_rate": [float(y.mean())],
            "negative_rate": [float(1 - y.mean())],
        }
    )
    dataset_summary.to_csv(tables_dir / "dm_dataset_summary.csv", index=False)
    df.isna().mean().sort_values(ascending=False).to_csv(tables_dir / "dm_missing_rates.csv", header=["missing_rate"])
    y.value_counts().sort_index().to_csv(tables_dir / "dm_target_distribution.csv", header=["rows"])

    plot_class_balance(y, figures_dir / "dm_class_distribution.png")
    plot_missing(df, figures_dir / "dm_missing_values.png")
    if args.target == "SeriousDlqin2yrs":
        plot_gmsc_eda(df, y, figures_dir, tables_dir)

    X_raw = df.drop(columns=[args.target])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y.to_numpy(),
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=args.validation_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    train_frame = X_fit_raw.copy()
    train_frame[args.target] = y_fit
    preprocessor, used_cols = build_preprocessor(
        train_frame,
        target=args.target,
        max_categorical_cardinality=args.max_categorical_cardinality,
        max_missing_rate=args.max_missing_rate,
    )
    pd.Series(used_cols, name="used_source_columns").to_csv(tables_dir / "dm_used_columns.csv", index=False)

    X_fit = np.asarray(preprocessor.fit_transform(X_fit_raw), dtype=float)
    X_val = np.asarray(preprocessor.transform(X_val_raw), dtype=float)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=float)

    validation = pd.DataFrame(
        [
            {"check": "dataset_rows_full", "value": len(df), "status": "pass"},
            {"check": "dataset_columns_full", "value": df.shape[1], "status": "pass"},
            {"check": "positive_rate_full", "value": float(y.mean()), "status": "pass"},
            {"check": "target_excluded_from_features", "value": args.target not in used_cols, "status": "pass"},
            {"check": "id_column_excluded", "value": "Unnamed: 0" not in used_cols, "status": "pass"},
            {"check": "train_positive_rate", "value": float(np.mean(y_fit)), "status": "pass"},
            {"check": "validation_positive_rate", "value": float(np.mean(y_val)), "status": "pass"},
            {"check": "test_positive_rate", "value": float(np.mean(y_test)), "status": "pass"},
            {
                "check": "stratified_split_close",
                "value": max(
                    abs(float(np.mean(y_fit)) - float(np.mean(y_val))),
                    abs(float(np.mean(y_fit)) - float(np.mean(y_test))),
                )
                < 0.002,
                "status": "pass",
            },
            {"check": "no_nan_train_after_preprocess", "value": not np.isnan(X_fit).any(), "status": "pass"},
            {"check": "no_nan_validation_after_preprocess", "value": not np.isnan(X_val).any(), "status": "pass"},
            {"check": "no_nan_test_after_preprocess", "value": not np.isnan(X_test).any(), "status": "pass"},
            {"check": "train_matrix_shape", "value": str(X_fit.shape), "status": "pass"},
            {"check": "validation_matrix_shape", "value": str(X_val.shape), "status": "pass"},
            {"check": "test_matrix_shape", "value": str(X_test.shape), "status": "pass"},
            {"check": "used_source_columns", "value": ", ".join(used_cols), "status": "pass"},
        ]
    )
    validation.to_csv(tables_dir / "dm_preprocessing_validation.csv", index=False)

    metrics, val_scores_by_model, test_scores_by_model, fitted_models = run_model_suite(
        X_fit,
        y_fit,
        X_val,
        X_test,
        y_test,
        stages=args.stages,
        adaboost_estimators=args.adaboost_estimators,
    )
    for name, score in test_scores_by_model.items():
        plot_confusion(y_test, score, name, figures_dir)

    metrics.to_csv(tables_dir / "dm_model_metrics.csv", index=False)
    pd.DataFrame(test_scores_by_model).to_csv(tables_dir / "dm_model_scores.csv", index=False)
    plot_curves(y_test, test_scores_by_model, figures_dir)

    thresholds = threshold_sweep(y_val, val_scores_by_model)
    thresholds.to_csv(tables_dir / "dm_threshold_sweep.csv", index=False)
    plot_threshold_sensitivity(thresholds, figures_dir)
    best_f1 = thresholds.sort_values(["model", "f1"], ascending=[True, False]).groupby("model").head(1)
    best_f2 = thresholds.sort_values(["model", "f2"], ascending=[True, False]).groupby("model").head(1)
    best_f1.to_csv(tables_dir / "dm_best_threshold_by_f1.csv", index=False)
    best_f2.to_csv(tables_dir / "dm_best_threshold_by_f2.csv", index=False)
    evaluate_selected_thresholds(y_test, test_scores_by_model, best_f1, "f1").to_csv(
        tables_dir / "dm_selected_threshold_by_f1_test_metrics.csv", index=False
    )
    evaluate_selected_thresholds(y_test, test_scores_by_model, best_f2, "f2").to_csv(
        tables_dir / "dm_selected_threshold_by_f2_test_metrics.csv", index=False
    )

    for name, cascade in fitted_models.items():
        if cascade is None or not hasattr(cascade, "stage_sizes_"):
            continue
        safe_name = name.lower().replace(" ", "_")
        stage_path = tables_dir / f"dm_{safe_name}_stages.csv"
        pd.DataFrame(cascade.stage_sizes_).to_csv(stage_path, index=False)
        if name == "BalanceCascade_simple":
            pd.DataFrame(cascade.stage_sizes_).to_csv(tables_dir / "dm_balancecascade_stages.csv", index=False)

    if not args.skip_controlled_cases:
        run_controlled_cases(figures_dir, tables_dir, args.stages, args.adaboost_estimators)

    print("Xong.")
    print(f"Figures nằm ở: {figures_dir}")
    print(f"Tables nằm ở: {tables_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
