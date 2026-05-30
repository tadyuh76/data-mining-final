# Data Mining Final - BalanceCascade

Repository này giữ các phần cần để chạy lại thí nghiệm:

- Dataset: Give Me Some Credit (`data/raw/cs-training.csv`, `cs-test.csv`)
- Code chính: `src/balance_cascade_pipeline.py`
- EDA và kết quả thí nghiệm: `outputs/`

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy pipeline chính

```bash
python src/balance_cascade_pipeline.py \
  --csv data/raw/cs-training.csv \
  --target SeriousDlqin2yrs \
  --positive-label 1 \
  --stages 6 \
  --adaboost-estimators 25
```

Pipeline sẽ tạo lại bảng trong `outputs/tables/` và hình trong `outputs/figures/`.

## Screening dataset

```bash
python src/screen_balancecascade_datasets.py \
  --sample-limit 200000 \
  --test-size 0.25 \
  --external-dir /tmp/balancecascade_datasets
```

## Kiểm tra độ ổn định theo seed

```bash
python src/balance_cascade_seed_stability.py \
  --csv data/raw/cs-training.csv \
  --target SeriousDlqin2yrs \
  --positive-label 1 \
  --stages 6 \
  --adaboost-estimators 25
```

Phần này chạy lại train/test split và undersampling với vài seed khác nhau để xem metric có dao động mạnh không.

## Ghi chú kỹ thuật

- `SeriousDlqin2yrs` là target.
- `Unnamed: 0` là cột ID và không dùng làm feature.
- Train/test split dùng stratify để giữ tỷ lệ positive gần nhau.
- Chọn cột, imputer và scaler đều dựa trên train, sau đó transform test để tránh leakage.
- Undersampling chỉ dùng trong lúc train, không resample test set.

## Output chính

- `outputs/tables/dm_dataset_summary.csv`
- `outputs/tables/dm_preprocessing_validation.csv`
- `outputs/tables/dm_model_metrics.csv`
- `outputs/tables/dm_best_threshold_by_f2.csv`
- `outputs/tables/dm_threshold_sweep.csv`
- `outputs/tables/dm_seed_stability_summary.csv`
- `outputs/tables/dm_balancecascade_stages.csv`
- `outputs/tables/dm_balancecascade_tuned_stages.csv`
- `outputs/tables/dm_controlled_case_metrics.csv`
- `outputs/dataset_screening/dm_dataset_screening_summary.csv`
- `outputs/figures/dm_gmsc_class_distribution.png`
- `outputs/figures/dm_gmsc_missing_values.png`
- `outputs/figures/dm_precision_recall_curve.png`
- `outputs/figures/dm_roc_curve.png`
- `outputs/figures/dm_threshold_sensitivity_f2.png`
- `outputs/figures/dm_controlled_ideal_separable_scatter.png`
- `outputs/figures/dm_controlled_weak_overlap_scatter.png`
- `outputs/figures/dm_controlled_case_recall_comparison.png`
- `outputs/figures/dm_dataset_screening_f2.png`
