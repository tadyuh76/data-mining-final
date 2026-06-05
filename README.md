# Data Mining Final - BalanceCascade

Bài này dùng BalanceCascade cho bài toán phân loại mất cân bằng lớp trên dataset Give Me Some Credit.

BalanceCascade là thuật toán chính của bài. Logistic Regression balanced, RandomForest balanced và EasyEnsemble được dùng để đối chiếu, để thấy rõ trade-off giữa precision và recall.

## Dữ liệu

- File training chính: `data/raw/cs-training.csv`
- File test gốc từ Kaggle: `data/raw/cs-test.csv`
- Target: `SeriousDlqin2yrs`
- Positive class: `1`

`cs-test.csv` không có target nên không dùng để tính metric. Pipeline đánh giá bằng cách tách train, validation và test từ `cs-training.csv`.

## Chạy lại trên macOS

```bash
cd data-mining-final
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Chạy pipeline chính:

```bash
python3 src/balance_cascade_pipeline.py \
  --csv data/raw/cs-training.csv \
  --target SeriousDlqin2yrs \
  --positive-label 1 \
  --stages 6 \
  --adaboost-estimators 25
```

Chạy seed stability:

```bash
python3 src/balance_cascade_seed_stability.py \
  --csv data/raw/cs-training.csv \
  --target SeriousDlqin2yrs \
  --positive-label 1 \
  --stages 6 \
  --adaboost-estimators 25
```

Hoặc chạy một lượt bằng notebook:

```bash
jupyter notebook notebooks/run_all.ipynb
```

Trong VS Code có thể mở trực tiếp `notebooks/run_all.ipynb`, chọn đúng Python environment đã cài `requirements.txt`, rồi bấm Run All. Notebook này dùng để chạy lại toàn bộ pipeline trong một lượt.

## Chạy lại trên Windows PowerShell

```powershell
cd data-mining-final
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate virtual environment, chạy tạm lệnh này trong cửa sổ PowerShell hiện tại rồi activate lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Chạy pipeline chính:

```powershell
py src\balance_cascade_pipeline.py --csv data\raw\cs-training.csv --target SeriousDlqin2yrs --positive-label 1 --stages 6 --adaboost-estimators 25
```

Chạy seed stability:

```powershell
py src\balance_cascade_seed_stability.py --csv data\raw\cs-training.csv --target SeriousDlqin2yrs --positive-label 1 --stages 6 --adaboost-estimators 25
```

## Output cần kiểm tra

Sau khi chạy xong, xem kết quả trong `outputs/`. Một số file chính:

- `outputs/tables/dm_dataset_summary.csv`
- `outputs/tables/dm_preprocessing_validation.csv`
- `outputs/tables/dm_model_metrics.csv`
- `outputs/tables/dm_best_threshold_by_f2.csv`
- `outputs/tables/dm_selected_threshold_by_f2_test_metrics.csv`
- `outputs/tables/dm_seed_stability_summary.csv`
- `outputs/tables/dm_balancecascade_stages.csv`
- `outputs/tables/dm_balancecascade_tuned_stages.csv`
- `outputs/tables/dm_controlled_case_metrics.csv`
- `outputs/figures/dm_precision_recall_curve.png`
- `outputs/figures/dm_roc_curve.png`
- `outputs/figures/dm_threshold_sensitivity_f2.png`

## Ghi chú ngắn

- `Unnamed: 0` là cột ID, không dùng làm feature.
- Imputer và scaler chỉ fit trên train/fit split để tránh leakage.
- Threshold được chọn trên validation split, sau đó mới đánh giá trên test split.
- Undersampling chỉ xảy ra trong lúc train. Validation và test giữ phân phối lớp thật.
- BalanceCascade simple được giữ lại để minh họa cấu hình quá nhạy; bản tuned mới là cấu hình chính để phân tích trên Give Me Some Credit.
- Code chính nằm trong `src/`. Notebook chỉ dùng khi cần demo chạy một lượt.
