# Credit Scoring Pipeline

Dự báo rủi ro kiệt quệ tài chính cho các công ty niêm yết trên HOSE và HNX, xây dựng dựa trên phương pháp của bài báo:

> Tran, K.L., Le, H.A., Nguyen, T.H., & Nguyen, D.T. (2022). *Explainable Machine Learning for Financial Distress Prediction: Evidence from Vietnam.* Data, 7(11), 160.

Điểm khác so với bài báo gốc: dữ liệu tự crawl từ vnstock thay vì dùng dataset có sẵn, và label được gán lại theo cùng 3 tiêu chí của tác giả.

---

## Cấu trúc project

```
credit scoring/
├── src/
│   ├── get_symbols.py      # lấy danh sách mã CK từ vnstock
│   ├── crawl.py            # crawl BS, IS, quote, ratio
│   ├── build_dataset.py    # tính 25 features + gán label
│   ├── data.py             # load data, split, impute
│   ├── model.py            # train LR + XGBoost, evaluate
│   ├── scorecard.py        # chuyển prob → credit score
│   └── map_features.py     # tính features cho công ty mới (inference)
├── notebooks/
│   ├── eda.ipynb           # phân tích dữ liệu
│   └── model_training.ipynb # training, đánh giá, SHAP, scorecard
├── document/
│   ├── data.md             # mô tả dữ liệu và pipeline
│   ├── eda.md              # tóm tắt kết quả EDA
│   └── model_training.md   # tóm tắt kết quả training
├── data/vnstock/
│   ├── symbols.txt         # danh sách mã CK
│   ├── pkl/                # raw data crawl từng công ty
│   └── dataset_v2.csv      # dataset cuối dùng để train
├── models/
│   ├── logistic.pkl
│   ├── xgboost.pkl
│   └── best_model.pkl      # XGBoost (AUC cao nhất)
└── requirements.txt
```

---

## Cài đặt

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

Tạo file `.env` ở thư mục gốc:

```
VNSTOCK_API_KEY=your_api_key_here
```

API key lấy tại [vnstocks.com](https://vnstocks.com/account) sau khi đăng nhập.

---

## Cách chạy

Nếu muốn chạy từ đầu:

```bash
# 1. Lấy danh sách mã
python src/get_symbols.py

# 2. Crawl dữ liệu (mất khoảng 1-2 tiếng tùy số mã)
#    Mã nào đã có pkl sẽ bị skip
python src/crawl.py

# 3. Build dataset
python src/build_dataset.py
```

Sau đó mở notebook để EDA và train:

```
notebooks/eda.ipynb
notebooks/model_training.ipynb
```

Nếu đã có `dataset_v2.csv` thì bỏ qua bước 1-3, mở notebook chạy thẳng.

---

## Kết quả

| Model | AUC | KS | Brier |
|---|---|---|---|
| Logistic Regression | 0.8877 | 0.6574 | 0.156 |
| XGBoost | **0.9214** | **0.7424** | **0.080** |

Dataset: 266 công ty niêm yết VN, 2018–2024, crawl từ vnstock.

Top SHAP features (XGBoost): `ev_to_ebit`, `eps_diluted`, `price_to_cfo`, `current_ratio`, `ebitda_coverage`

### Risk grade

| Grade | Score | Distress Rate |
|---|---|---|
| High Risk | < 500 | 48.2% |
| Medium Risk | 500–650 | 22.5% |
| Low Risk | 650–750 | 3.3% |
| Very Low Risk | ≥ 750 | 0.0%* |

*Trên test set, không đảm bảo với data mới.

---

## Hạn chế

- Chỉ áp dụng được cho công ty niêm yết (có giá cổ phiếu và báo cáo tài chính công khai), không dùng được cho SME
- Label gán bằng rule-based nên có thể không chính xác 100% với một số công ty đặc thù
