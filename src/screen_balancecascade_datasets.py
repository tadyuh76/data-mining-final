#!/usr/bin/env python3
"""
Thử nhanh vài dữ liệu nhị phân để so sánh BalanceCascade với mô hình nền.

Phần này chỉ tạo bảng kết quả và một hình so sánh F2 để hỗ trợ chọn dữ liệu.
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from balance_cascade_pipeline import (
    RANDOM_STATE,
    SimpleBalanceCascadeClassifier,
    SimpleEasyEnsembleClassifier,
    build_preprocessor,
    threshold_sweep,
)


def classification_metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (score >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frame: pd.DataFrame
    target: str
    source: str
    note: str


def load_gmsc(project_dir: Path) -> DatasetSpec:
    path = project_dir / "data" / "raw" / "cs-training.csv"
    df = pd.read_csv(path)
    return DatasetSpec(
        name="give_me_some_credit",
        frame=df,
        target="SeriousDlqin2yrs",
        source="local Kaggle Give Me Some Credit",
        note="Credit-risk, imbalance mạnh, target rõ.",
    )


def load_default_credit(project_dir: Path) -> DatasetSpec | None:
    path = project_dir / "data" / "raw" / "default_credit_card_clients.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return DatasetSpec(
        name="uci_default_credit",
        frame=df,
        target="default.payment.next.month",
        source="local UCI Default of Credit Card Clients",
        note="Credit-risk, dễ giải thích, imbalance vừa.",
    )


def load_breast_cancer_dataset() -> DatasetSpec:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["malignant"] = (df["target"] == 0).astype(int)
    df = df.drop(columns=["target"])
    return DatasetSpec(
        name="sklearn_breast_cancer",
        frame=df,
        target="malignant",
        source="sklearn built-in Breast Cancer Wisconsin",
        note="Medical binary case, ít imbalance hơn.",
    )


def load_landslide_large(project_root: Path) -> DatasetSpec | None:
    path = project_root / "machine-learning-final" / "data" / "raw" / "landslide_catalog.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    size = df["landslide_size"].astype(str).str.lower()
    df["is_large_landslide"] = size.isin(["large", "very_large"]).astype(int)
    drop_cols = ["landslide_size", "source_link", "geolocation", "id"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return DatasetSpec(
        name="landslide_large_event",
        frame=df,
        target="is_large_landslide",
        source="local NASA landslide catalog",
        note="Real-world spatial/event data, nhưng target là nhóm tự tạo.",
    )


def unzip_if_needed(zip_path: Path, out_dir: Path) -> None:
    if not zip_path.exists() or out_dir.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def load_bank_marketing(external_dir: Path) -> DatasetSpec | None:
    zip_path = external_dir / "bank_marketing.zip"
    out_dir = external_dir / "bank_marketing"
    unzip_if_needed(zip_path, out_dir)
    nested_zip = out_dir / "bank.zip"
    if nested_zip.exists():
        unzip_if_needed(nested_zip, out_dir / "bank")
    csv_path = out_dir / "bank" / "bank-full.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, sep=";")
    df["subscribed"] = (df["y"] == "yes").astype(int)
    df = df.drop(columns=["y", "duration"], errors="ignore")
    return DatasetSpec(
        name="uci_bank_marketing",
        frame=df,
        target="subscribed",
        source="UCI Bank Marketing",
        note="Marketing response, imbalance rõ, bỏ duration để tránh leakage.",
    )


def load_german_credit(external_dir: Path) -> DatasetSpec | None:
    zip_path = external_dir / "german_credit.zip"
    out_dir = external_dir / "german_credit"
    unzip_if_needed(zip_path, out_dir)
    data_path = out_dir / "german.data"
    if not data_path.exists():
        return None
    columns = [f"A{i}" for i in range(1, 21)] + ["credit_risk"]
    df = pd.read_csv(data_path, sep=r"\s+", header=None, names=columns)
    df["bad_credit"] = (df["credit_risk"] == 2).astype(int)
    df = df.drop(columns=["credit_risk"])
    return DatasetSpec(
        name="uci_german_credit",
        frame=df,
        target="bad_credit",
        source="UCI Statlog German Credit",
        note="Credit-risk nhỏ, dễ giải thích nhưng ít dữ liệu.",
    )


def load_mammographic_mass(external_dir: Path) -> DatasetSpec | None:
    zip_path = external_dir / "mammographic_mass.zip"
    out_dir = external_dir / "mammographic_mass"
    unzip_if_needed(zip_path, out_dir)
    data_path = out_dir / "mammographic_masses.data"
    if not data_path.exists():
        return None
    cols = ["birads", "age", "shape", "margin", "density", "severity"]
    df = pd.read_csv(data_path, header=None, names=cols, na_values="?")
    df["malignant"] = (df["severity"] == 1).astype(int)
    df = df.drop(columns=["severity"])
    return DatasetSpec(
        name="uci_mammographic_mass",
        frame=df,
        target="malignant",
        source="UCI Mammographic Mass",
        note="Medical binary case, dễ hiểu nhưng không imbalance mạnh.",
    )


def load_specs(project_dir: Path, project_root: Path, external_dir: Path) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = [load_gmsc(project_dir), load_breast_cancer_dataset()]
    optional_loaders = [
        lambda: load_default_credit(project_dir),
        lambda: load_landslide_large(project_root),
        lambda: load_bank_marketing(external_dir),
        lambda: load_german_credit(external_dir),
        lambda: load_mammographic_mass(external_dir),
    ]
    for loader in optional_loaders:
        spec = loader()
        if spec is not None:
            specs.append(spec)
    return specs


def model_candidates(stages: int, adaboost_estimators: int) -> dict[str, object]:
    return {
        "LogReg_balanced": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=160,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "EasyEnsemble_s6": SimpleEasyEnsembleClassifier(
            n_estimators=stages,
            adaboost_estimators=adaboost_estimators,
            random_state=RANDOM_STATE,
        ),
        "BalanceCascade_s2_a10": SimpleBalanceCascadeClassifier(
            n_stages=2,
            adaboost_estimators=10,
            random_state=RANDOM_STATE,
        ),
        "BalanceCascade_s6_a25": SimpleBalanceCascadeClassifier(
            n_stages=stages,
            adaboost_estimators=adaboost_estimators,
            random_state=RANDOM_STATE,
        ),
        "BalanceCascade_s8_a25": SimpleBalanceCascadeClassifier(
            n_stages=8,
            adaboost_estimators=adaboost_estimators,
            random_state=RANDOM_STATE,
        ),
    }


def stratified_limit(df: pd.DataFrame, y: pd.Series, sample_limit: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(df) <= sample_limit:
        return df.reset_index(drop=True), y.reset_index(drop=True)
    sample_idx, _ = train_test_split(
        np.arange(len(df)),
        train_size=sample_limit,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return df.iloc[sample_idx].reset_index(drop=True), y.iloc[sample_idx].reset_index(drop=True)


def evaluate_dataset(
    spec: DatasetSpec,
    sample_limit: int,
    test_size: float,
    stages: int,
    adaboost_estimators: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    df = spec.frame.copy()
    df = df.dropna(subset=[spec.target])
    y = df[spec.target].astype(int)
    df, y = stratified_limit(df, y, sample_limit)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df.drop(columns=[spec.target]),
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    training_frame = X_fit_raw.copy()
    training_frame[spec.target] = y_fit.values
    preprocessor, used_cols = build_preprocessor(
        training_frame,
        target=spec.target,
        max_categorical_cardinality=30,
        max_missing_rate=0.80,
    )
    X_fit = np.asarray(preprocessor.fit_transform(X_fit_raw), dtype=float)
    X_val = np.asarray(preprocessor.transform(X_val_raw), dtype=float)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=float)

    rows: list[dict[str, object]] = []
    val_scores_by_model: dict[str, np.ndarray] = {}
    test_scores_by_model: dict[str, np.ndarray] = {}

    for model_name, model in model_candidates(stages, adaboost_estimators).items():
        fitted = clone(model)
        fitted.fit(X_fit, y_fit)
        val_score = fitted.predict_proba(X_val)[:, 1]
        test_score = fitted.predict_proba(X_test)[:, 1]
        val_scores_by_model[model_name] = val_score
        test_scores_by_model[model_name] = test_score
        threshold_0_50 = classification_metrics_at_threshold(y_test, test_score, 0.50)
        rows.append(
            {
                "dataset": spec.name,
                "source": spec.source,
                "note": spec.note,
                "rows": len(df),
                "positive_rate": float(y.mean()),
                "used_source_columns": len(used_cols),
                "model": model_name,
                "threshold_0_50_precision": threshold_0_50["precision"],
                "threshold_0_50_recall": threshold_0_50["recall"],
                "threshold_0_50_f1": threshold_0_50["f1"],
                "threshold_0_50_f2": threshold_0_50["f2"],
                "threshold_0_50_balanced_accuracy": threshold_0_50["balanced_accuracy"],
                "roc_auc": roc_auc_score(y_test, test_score),
                "pr_auc": average_precision_score(y_test, test_score),
            }
        )

    sweep = threshold_sweep(np.asarray(y_val), val_scores_by_model)
    best_f2 = sweep.sort_values(["model", "f2", "recall"], ascending=[True, False, False]).groupby("model").head(1)
    best_f1 = sweep.sort_values(["model", "f1", "recall"], ascending=[True, False, False]).groupby("model").head(1)
    f2_lookup = best_f2.set_index("model").to_dict(orient="index")
    f1_lookup = best_f1.set_index("model").to_dict(orient="index")

    for row in rows:
        model_name = str(row["model"])
        best_f2_threshold = float(f2_lookup[model_name]["threshold"])
        best_f1_threshold = float(f1_lookup[model_name]["threshold"])
        test_f2_metrics = classification_metrics_at_threshold(
            y_test,
            test_scores_by_model[model_name],
            best_f2_threshold,
        )
        test_f1_metrics = classification_metrics_at_threshold(
            y_test,
            test_scores_by_model[model_name],
            best_f1_threshold,
        )
        row.update(
            {
                "best_f2_threshold": best_f2_threshold,
                "validation_best_f2": f2_lookup[model_name]["f2"],
                "best_f2_precision": test_f2_metrics["precision"],
                "best_f2_recall": test_f2_metrics["recall"],
                "best_f2": test_f2_metrics["f2"],
                "best_f2_predicted_positive_rate": test_f2_metrics["predicted_positive_rate"],
                "best_f1_threshold": best_f1_threshold,
                "validation_best_f1": f1_lookup[model_name]["f1"],
                "best_f1": test_f1_metrics["f1"],
            }
        )

    metric_frame = pd.DataFrame(rows)
    bc_rows = metric_frame[metric_frame["model"].str.startswith("BalanceCascade")]
    best_bc = bc_rows.sort_values("best_f2", ascending=False).iloc[0]
    best_any = metric_frame.sort_values("best_f2", ascending=False).iloc[0]
    best_non_bc = metric_frame[~metric_frame["model"].str.startswith("BalanceCascade")].sort_values(
        "best_f2", ascending=False
    ).iloc[0]

    summary = {
        "dataset": spec.name,
        "source": spec.source,
        "note": spec.note,
        "rows": len(df),
        "positive_rate": float(y.mean()),
        "used_source_columns": len(used_cols),
        "best_model_by_f2": best_any["model"],
        "best_model_f2": best_any["best_f2"],
        "best_balancecascade_config": best_bc["model"],
        "balancecascade_best_f2": best_bc["best_f2"],
        "balancecascade_precision": best_bc["best_f2_precision"],
        "balancecascade_recall": best_bc["best_f2_recall"],
        "best_non_balancecascade": best_non_bc["model"],
        "best_non_balancecascade_f2": best_non_bc["best_f2"],
        "bc_f2_gap_vs_best_non_bc": best_bc["best_f2"] - best_non_bc["best_f2"],
    }
    return rows, summary


def plot_f2_summary(summary: pd.DataFrame, figures_dir: Path) -> None:
    ordered = summary.sort_values("balancecascade_best_f2", ascending=True)
    y = np.arange(len(ordered))
    plt.figure(figsize=(10, 5.5))
    plt.barh(y - 0.18, ordered["balancecascade_best_f2"], height=0.36, label="Best BalanceCascade")
    plt.barh(y + 0.18, ordered["best_non_balancecascade_f2"], height=0.36, label="Best baseline")
    plt.yticks(y, ordered["dataset"])
    plt.xlabel("Best F2 sau threshold tuning")
    plt.title("Screening dataset: BalanceCascade và baseline tốt nhất")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "dm_dataset_screening_f2.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-limit", type=int, default=40000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--external-dir", default="/tmp/balancecascade_datasets")
    parser.add_argument("--stages", type=int, default=6)
    parser.add_argument("--adaboost-estimators", type=int, default=25)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    project_root = project_dir.parent
    output_dir = project_dir / "outputs" / "dataset_screening"
    figures_dir = project_dir / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for spec in load_specs(project_dir, project_root, Path(args.external_dir)):
        print(f"Screening {spec.name}...")
        try:
            rows, summary = evaluate_dataset(
                spec,
                sample_limit=args.sample_limit,
                test_size=args.test_size,
                stages=args.stages,
                adaboost_estimators=args.adaboost_estimators,
            )
        except Exception as exc:  # Giữ screening tiếp tục nếu một dataset phụ bị lỗi.
            summaries.append(
                {
                    "dataset": spec.name,
                    "source": spec.source,
                    "note": f"lỗi: {exc}",
                }
            )
            continue
        all_rows.extend(rows)
        summaries.append(summary)

    metrics = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summaries)
    metrics.to_csv(output_dir / "dm_dataset_screening_metrics.csv", index=False)
    summary.to_csv(output_dir / "dm_dataset_screening_summary.csv", index=False)

    valid_summary = summary.dropna(subset=["balancecascade_best_f2"])
    if not valid_summary.empty:
        plot_f2_summary(valid_summary, figures_dir)

    print(f"Xong. Kết quả nằm ở: {output_dir}")


if __name__ == "__main__":
    main()
