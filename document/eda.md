# EDA Summary — Credit Scoring Dataset

---

## 1. Tổng quan dataset

Dataset gồm **1847 dòng × 28 cột**, tương ứng với **266 công ty niêm yết** trên HOSE và HNX, giai đoạn **2018–2024**.

Mỗi dòng đại diện cho một cặp (công ty, năm) với 25 features tài chính và 1 nhãn `Financial_Distress`. Đây là **panel data** — cùng một công ty xuất hiện ở nhiều năm, nên phải dùng GroupShuffleSplit theo Company khi chia train/test để tránh data leakage.

**Phân bố theo công ty và năm:**

| Số năm có dữ liệu | Số công ty |
|---|---|
| 4 năm | 3 |
| 5 năm | 2 |
| 6 năm | 2 |
| 7 năm (đủ) | 259 |

259/266 công ty có đủ 7 năm. 7 công ty còn lại niêm yết muộn nên thiếu dữ liệu những năm đầu. Số quan sát tăng dần từ 260 (2018–2019) lên 266 (2021–2024) vì các công ty mới được bổ sung dần.

---

## 2. Missing values

Tổng cộng **1573 giá trị missing (~3.4% trung bình)**. Phân bố không đều — tập trung ở một số feature cụ thể.

| Feature | Missing |
|---|---|
| `pe_diluted` | 48.73% |
| `ebitda_coverage` | 10.61% |
| `pe_basic` | 3.68% |
| Cụm market features (`pb_ratio`, `ps_ratio`, `market_cap`, `ev`, `ev_to_*`) | ~2.44% |
| Liquidity cơ bản (`cash_ratio`, `quick_ratio`, `current_ratio`, `assets_to_liab`) | ~0.05% |

**Nguyên nhân của từng nhóm:**

**`pe_diluted` (900 dòng null):** 832/900 dòng vẫn có `pe_basic`, chỉ 68 dòng mất hoàn toàn thông tin định giá. Nguyên nhân là nhiều doanh nghiệp không phát hành cổ phiếu pha loãng nên `eps_diluted = 0`, dẫn đến không tính được P/E diluted.

**`ebitda_coverage` (196 dòng null):** Mẫu số (nợ ngắn hạn + lãi vay) = 0, tức là công ty không có nợ và không có chi phí lãi vay. Distress rate của nhóm này chỉ **2.6%** so với **14.4%** khi có giá trị — đây là tín hiệu lành mạnh, không phải vấn đề chất lượng data.

**Cụm market features (45 dòng null đồng thời):** Các công ty niêm yết muộn, năm đầu chưa có dữ liệu giá cổ phiếu nên không tính được market cap và các chỉ số phái sinh.

**Xử lý:** Giữ nguyên NaN trong file CSV. Impute bằng **median của tập train** trong `data.py` sau khi split, tránh leakage.

---

## 3. EDA biến target (Financial_Distress)

### Phân phối tổng thể

| Nhãn | Số quan sát | Tỷ lệ |
|---|---|---|
| 0 (healthy) | 1604 | 86.84% |
| 1 (distress) | 243 | 13.16% |

Mất cân bằng rõ rệt. Xử lý bằng `class_weight='balanced'` (LR) và `scale_pos_weight` (XGBoost), không dùng SMOTE vì panel data dễ gây leakage.

### Distress rate theo năm

| Năm | Distress | Tổng | Rate |
|---|---|---|---|
| 2018 | 0 | 260 | 0.0% |
| 2019 | 30 | 260 | 11.5% |
| 2020 | 35 | 263 | 13.3% |
| 2021 | 35 | 266 | 13.2% |
| 2022 | 28 | 266 | 10.5% |
| 2023 | 50 | 266 | 18.8% |
| 2024 | 65 | 266 | 24.4% |

Năm 2018 luôn = 0% vì label dùng lookback 2–3 năm, thiếu dữ liệu 2016–2017 để tính. Rate tăng mạnh ở 2023–2024, phản ánh hệ quả của giai đoạn lãi suất cao và thị trường BĐS suy thoái 2022–2023.

### Phân tích theo công ty

- **97/266 công ty (36.5%)** bị distress ít nhất 1 năm
- Phần lớn bị ngắn (59 công ty chỉ bị 1–2 năm), nhưng **11 công ty bị cả 6 năm** — nhóm suy thoái cấu trúc
- **82 công ty distress liên tục**, chỉ 15 công ty gián đoạn (phục hồi rồi tái phát)

Distress phần lớn là trạng thái kéo dài, không phải sự cố tạm thời.

---

## 4. EDA từng feature đơn lẻ

### Phân phối

**23/25 features có |skew| > 2**, và **20/25 có |skew| > 5** — phân phối lệch cực mạnh, điển hình của financial ratios khi mẫu số gần 0 hoặc âm.

Top 5 features lệch nhất: `ev_to_ebit` (40.1), `assets_to_liab` (37.9), `st_debt_to_equity` (34.2), `ev_to_ebitda` (33.6), `pb_tangible` (31.7) — đều bị kéo đuôi bởi outlier cực đoan.

Hệ quả: XGBoost không bị ảnh hưởng bởi skew. Logistic Regression cần StandardScaler trước khi train.

### Tương quan giữa các features (multicollinearity)

**13 cặp features có |corr| > 0.7**, chủ yếu nằm trong cùng nhóm khái niệm:

| Nhóm | Cặp tiêu biểu | Corr |
|---|---|---|
| Valuation | `pe_basic` ↔ `pe_diluted` | 0.997 |
| Liquidity | `quick_ratio` ↔ `current_ratio` | 0.982 |
| Market size | `market_cap` ↔ `ev` | 0.967 |
| Profitability/valuation | `price_to_cfo` ↔ `ev_to_ebit` | 0.946 |
| Leverage | `liab_to_equity` ↔ `ap_to_equity` | 0.921 |
| Leverage | `lt_debt_to_equity` ↔ `liab_to_equity` | 0.901 |

Đa cộng tuyến làm hệ số Logistic Regression khó diễn giải, nhưng không ảnh hưởng đến AUC. XGBoost ít bị ảnh hưởng hơn vì tree-based model không dùng hệ số tuyến tính.

---

## 5. EDA biến target vs từng feature

### So sánh median healthy vs distressed

Sắp xếp theo mức chênh lệch giảm dần:

| Feature | Healthy median | Distress median | Chênh lệch |
|---|---|---|---|
| `lt_debt_to_equity` | 0.009 | 0.104 | +1056% |
| `lt_debt_to_assets` | 0.005 | 0.045 | +800% |
| `ev_to_ebit` | 8.48 | 33.94 | +300% |
| `st_debt_to_equity` | 0.156 | 0.558 | +258% |
| `liab_to_equity` | 0.718 | 1.759 | +145% |
| `ebitda_coverage` | 0.612 | 0.062 | −90% |
| `eps_diluted` | 1553 | 215 | −86% |
| `cash_ratio` | 0.193 | 0.070 | −64% |
| `current_ratio` | 1.818 | 1.206 | −34% |
| `ap_to_assets` | 0.061 | 0.062 | +1.6% (không phân biệt) |

Tất cả chiều chênh lệch đúng kỳ vọng: distress = nợ nhiều hơn, lợi nhuận thấp hơn, thanh khoản kém hơn.

### Tương quan Pearson với target

Top 5 features tương quan với `Financial_Distress`:

| Feature | Pearson | Chiều |
|---|---|---|
| `liab_to_assets` | +0.27 | Nợ/tài sản cao → distress nhiều hơn |
| `st_debt_to_assets` | +0.26 | Nợ ngắn hạn/tài sản cao → distress nhiều hơn |
| `eps_diluted` | −0.24 | EPS cao → distress ít hơn |
| `lt_debt_to_assets` | +0.22 | Nợ dài hạn/tài sản cao → distress nhiều hơn |
| `lt_debt_to_equity` | +0.16 | Đòn bẩy dài hạn cao → distress nhiều hơn |

Tương quan tuyến tính cao nhất chỉ ~0.27 — không feature nào đủ mạnh để chẩn đoán đơn lẻ. Tín hiệu distress phân tán trên nhiều ratio, đây là lý do cần model thay vì rule đơn giản.

### Box plot phân tách theo nhóm

**Phân tách tốt** (median và IQR ít overlap):
- Nhóm leverage: `liab_to_assets`, `st_debt_to_assets`, `lt_debt_to_equity` — distress đẩy lên cao rõ ràng
- Nhóm thanh khoản/sinh lời: `ebitda_coverage`, `eps_diluted`, `current_ratio`, `cash_ratio` — distress kéo xuống thấp rõ ràng

**Phân tách kém** (hai nhóm gần như chồng nhau):
- `market_cap`, `pb_ratio`, `ps_ratio`, `ap_to_assets`, `ev`, `ev_to_ebitda` — quy mô và một số valuation ratio không phân biệt được healthy vs distressed

---

## Kết luận

**Quy mô và cấu trúc:** Panel data 1847 quan sát, 266 công ty, 2018–2024. Bắt buộc split theo Company.

**Nhãn target:** Mất cân bằng 13.16%/86.84%. Distress tăng dần qua các năm, đỉnh 24.4% năm 2024. 97 công ty bị distress, phần lớn kéo dài liên tục nhiều năm.

**Missing values:** 3.4% tổng thể, có nguyên nhân rõ ràng, không phải lỗi dữ liệu. Xử lý bằng impute median sau split.

**Đặc điểm features:** Lệch mạnh (23/25 |skew| > 2), đa cộng tuyến trong cùng nhóm (13 cặp |corr| > 0.7).

**Tín hiệu phân biệt:** Rõ nhất ở nhóm đòn bẩy (`lt_debt_to_equity` chênh +1056%) và thanh khoản/sinh lời (`ebitda_coverage`, `eps_diluted` giảm mạnh). Tương quan đơn biến thấp (max 0.27) cho thấy distress là tổ hợp đa biến phi tuyến — cần model để nắm bắt được pattern đầy đủ.
