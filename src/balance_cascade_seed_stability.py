#!/usr/bin/env python3
"""
Kiểm tra nhanh độ ổn định theo seed cho bài toán mất cân bằng lớp.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from balance_cascade_pipeline import (
    SimpleBalanceCascadeClassifier,
    SimpleEasyEnsembleClassifier,
    build_preprocessor,
    ensure_dirs,
    metric_row,
    normalize_binary_target,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def parse_seeds(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def model_suite(seed: int, stages: int, adaboost_estimators: int) -> dict[str, object]:
    return {
        "LogReg_balanced": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "EasyEnsemble_simple": SimpleEasyEnsembleClassifier(
            n_estimators=stages,
            adaboost_estimators=adaboost_estimators,
            random_state=seed,
        ),
        "BalanceCascade_tuned": SimpleBalanceCascadeClassifier(
            n_stages=4,
            adaboost_estimators=10,
            random_state=seed,
        ),
        "BalanceCascade_simple": SimpleBalanceCascadeClassifier(
            n_stages=stages,
            adaboost_estimators=adaboost_estimators,
            random_state=seed,
        ),
    }


def stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.groupby("model").agg(
        runs=("seed", "count"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f2_mean=("f2", "mean"),
        f2_std=("f2", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
    )
    return summary.reset_index()


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(project_dir / "data" / "raw" / "cs-training.csv"))
    parser.add_argument("--target", default="SeriousDlqin2yrs")
    parser.add_argument("--positive-label", default="1")
    parser.add_argument("--seeds", default="11,23,42,58,91")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-categorical-cardinality", type=int, default=40)
    parser.add_argument("--max-missing-rate", type=float, default=0.80)
    parser.add_argument("--stages", type=int, default=6)
    parser.add_argument("--adaboost-estimators", type=int, default=25)
    args = parser.parse_args()

    _, tables_dir = ensure_dirs(project_dir)
    seeds = parse_seeds(args.seeds)

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Không tìm thấy cột target {args.target!r}.")
    df = df.dropna(subset=[args.target]).copy()
    y = normalize_binary_target(df[args.target], args.positive_label)
    X_raw = df.drop(columns=[args.target])

    rows = []
    for seed in seeds:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y.to_numpy(),
            test_size=args.test_size,
            stratify=y,
            random_state=seed,
        )
        train_frame = X_train_raw.copy()
        train_frame[args.target] = y_train
        preprocessor, _ = build_preprocessor(
            train_frame,
            target=args.target,
            max_categorical_cardinality=args.max_categorical_cardinality,
            max_missing_rate=args.max_missing_rate,
        )
        X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=float)
        X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=float)

        for name, model in model_suite(seed, args.stages, args.adaboost_estimators).items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            score = model.predict_proba(X_test)[:, 1]
            row = metric_row(name, y_test, score, time.perf_counter() - start)
            row["seed"] = seed
            rows.append(row)

    metrics = pd.DataFrame(rows)
    summary = stability_summary(metrics)
    metrics.to_csv(tables_dir / "dm_seed_stability_metrics.csv", index=False)
    summary.to_csv(tables_dir / "dm_seed_stability_summary.csv", index=False)

    print("Xong kiểm tra seed.")
    print(summary[["model", "runs", "recall_mean", "recall_std", "f2_mean", "f2_std"]])


if __name__ == "__main__":
    main()
