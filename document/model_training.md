# Model Training Summary — Credit Scoring Pipeline

---

## 1. Dữ liệu đầu vào

Dữ liệu sử dụng là `dataset_v2.csv` đã được xây dựng ở bước trước, gồm **1847 quan sát, 266 công ty, giai đoạn 2018–2024**. Mỗi quan sát là một cặp (công ty, năm) với 25 features tài chính và nhãn `Financial_Distress` (0/1).

**Nhắc lại đặc điểm quan trọng từ EDA:**

- Dữ liệu **mất cân bằng**: 13.16% distress (243/1847), 86.84% healthy — cần xử lý khi train
- **Panel data**: cùng một công ty xuất hiện ở nhiều năm — cần split đặc biệt để tránh leakage
- **23/25 features lệch mạnh** (|skew| > 2) — Logistic Regression cần chuẩn hóa trước khi train
- **13 cặp features tương quan cao** (|corr| > 0.7) — đa cộng tuyến ảnh hưởng đến hệ số LR nhưng không ảnh hưởng XGBoost
- Tương quan Pearson đơn biến thấp (max 0.27) — tín hiệu distress là phi tuyến, cần model phức tạp hơn rule đơn giản

---

## 2. Chia dữ liệu

### Phương pháp: GroupShuffleSplit theo Company

Không dùng random split thông thường vì đây là panel data. Nếu chia ngẫu nhiên, cùng một công ty có thể xuất hiện ở cả train và test — model sẽ "nhớ" công ty đó thay vì học pattern thật, dẫn đến AUC bị inflate khi đánh giá.

**GroupShuffleSplit** đảm bảo toàn bộ các năm của một công ty chỉ nằm ở một trong hai tập — hoặc train, hoặc test, không bao giờ cả hai. Đây là cách duy nhất tránh data leakage với panel data.

**Kết quả split 70/30:**

| Tập | Số dòng | Distress rate |
|---|---|---|
| Train | 1297 | 13.88% (180/1297) |
| Test | 550 | 11.45% (63/550) |

Chênh lệch distress rate giữa train và test là 2.4 điểm phần trăm — chấp nhận được. Với GroupShuffleSplit theo company group, không thể đảm bảo hai tập hoàn toàn cân bằng về label, nhưng điều quan trọng hơn là không có công ty nào bị leak.

### Xử lý missing values sau khi split

Sau khi split, tính **median của tập train** rồi dùng để fill NaN cho cả train và test. Không tính median trước khi split vì sẽ bị ảnh hưởng bởi thông tin của test set — đây là một dạng data leakage.

---

## 3. Xử lý mất cân bằng nhãn

Tập train có 1117 healthy và 180 distress — tỉ lệ gần 6:1. Không dùng SMOTE vì SMOTE tạo mẫu tổng hợp bằng cách nội suy giữa các quan sát — với panel data, điều này có thể vô tình tạo ra mẫu của một công ty "ảo" kết hợp giữa hai công ty thật, gây nhiễu.

Thay vào đó, xử lý bằng **trọng số trong loss function**:

- **Logistic Regression:** `class_weight='balanced'` — tự động nhân trọng số nghịch đảo tỉ lệ class vào loss
- **XGBoost:** `scale_pos_weight = 1117/180 = 6.2` — mỗi mẫu distress được tính nặng gấp 6.2 lần mẫu healthy trong quá trình train

---

## 4. Các model sử dụng

### Logistic Regression

Model tuyến tính cơ bản, dùng làm baseline để so sánh. Cần chuẩn hóa features trước bằng `StandardScaler` vì LR nhạy cảm với scale — feature có giá trị lớn sẽ thống trị gradient nếu không chuẩn hóa.

**Cấu hình:** `class_weight='balanced'`, `max_iter=1000`, `random_state=42`

### XGBoost (Extreme Gradient Boosting)

Ensemble model dựa trên decision tree, xây dựng nhiều cây tuần tự, mỗi cây học từ sai số của cây trước. Không cần chuẩn hóa features vì tree-based model chỉ quan tâm đến thứ tự giá trị, không bị ảnh hưởng bởi scale. Phù hợp hơn với dữ liệu tài chính vì có thể nắm bắt quan hệ phi tuyến và tương tác giữa các features mà LR không làm được.

**Cấu hình:** `scale_pos_weight=6.2`, `max_depth=5`, `n_estimators=200`, `learning_rate=0.05`, `random_state=42`

---

## 5. Kết quả đánh giá

### Các chỉ số đánh giá

- **AUC (Area Under ROC Curve):** Xác suất model xếp đúng thứ tự một cặp (distress, healthy) ngẫu nhiên. Không phụ thuộc vào ngưỡng quyết định.
- **KS (Kolmogorov-Smirnov):** Khoảng cách tích lũy tối đa giữa phân phối score của distress và healthy. Càng cao càng tốt, từ 0.6 trở lên là mạnh trong thực tế tín dụng.
- **Brier Score:** Sai số trung bình bình phương giữa xác suất dự đoán và nhãn thật. Càng thấp càng tốt — đo chất lượng calibration của model.

### Kết quả

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| AUC | 0.8877 | **0.9214** |
| KS | 0.6574 | **0.7424** |
| Brier | 0.1560 | **0.0798** |

XGBoost thắng cả 3 metrics. Khoảng cách rõ nhất ở Brier score — XGB calibrate tốt hơn gần 2 lần, tức là xác suất dự đoán của XGB sát với tỉ lệ thực tế hơn. Điều này quan trọng khi dùng xác suất để tính credit score.

### Calibration

LR có mean calibration error 0.3016, XGB là 0.1792. Cả hai còn lệch khá xa đường perfect calibration do dataset chỉ có 266 công ty và label được gán bằng rule-based (có noise). Trong thực tế triển khai nếu cần xác suất chính xác, cần thêm bước post-calibration (Platt scaling hoặc isotonic regression).

### Threshold và Confusion Matrix (XGBoost)

Threshold tối ưu theo **Youden's J** (tối đa hóa TPR − FPR) = **0.0676**.

|  | Predicted Healthy | Predicted Distress |
|---|---|---|
| **Actual Healthy** | 377 (TN) | 110 (FP) |
| **Actual Distress** | 2 (FN) | 61 (TP) |

| Metric | Giá trị |
|---|---|
| Precision | 35.67% |
| Recall | 96.83% |
| F1 | 0.5214 |

Threshold thấp (0.0676) vì distress rate trong test set là 11.45% — model cần bias về phía dự đoán positive nhiều hơn để đạt recall cao.

**Recall 96.83%** — model bắt được 61/63 trường hợp distress thực sự, chỉ bỏ sót 2. Đây là điểm mạnh quan trọng trong bài toán tín dụng vì hệ quả của bỏ sót distressed (cho vay công ty sắp mất khả năng trả nợ) tốn kém hơn nhiều so với báo sai (từ chối hoặc review thêm công ty lành mạnh).

**Precision 35.67%** — 110 công ty lành mạnh bị dự báo sai là distress (False Positive). Trade-off này chấp nhận được trong bài toán tín dụng.

---

## 6. Feature Importance — SHAP

Dùng **TreeSHAP** để giải thích đóng góp của từng feature vào dự đoán của XGBoost. SHAP tốt hơn permutation importance vì tính chính xác hơn và có thể giải thích theo từng quan sát.

**Top 10 features theo mean |SHAP|:**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `ev_to_ebit` | 1.3602 |
| 2 | `eps_diluted` | 1.2637 |
| 3 | `price_to_cfo` | 0.6202 |
| 4 | `current_ratio` | 0.5836 |
| 5 | `ebitda_coverage` | 0.3942 |
| 6 | `cash_ratio` | 0.2364 |
| 7 | `lt_debt_to_assets` | 0.2340 |
| 8 | `ev_to_ebitda` | 0.2196 |
| 9 | `liab_to_assets` | 0.1932 |
| 10 | `lt_debt_to_equity` | 0.1841 |

**Nhận xét:**

`ev_to_ebit` đứng đầu với SHAP = 1.3602, cách biệt rõ so với số 2. Đây là chỉ số định giá (EV/EBIT) — đứng đầu không phải vì giá cổ phiếu cao mà vì công ty distress thường có EBIT âm hoặc rất thấp, khiến EV/EBIT ra giá trị bất thường (âm hoặc cực lớn), dễ phân biệt với công ty lành mạnh.

`eps_diluted` đứng thứ 2 — EPS âm hoặc rất thấp là tín hiệu trực tiếp của suy yếu lợi nhuận.

Top 5 SHAP thiên về nhóm định giá thị trường (`ev_to_ebit`, `price_to_cfo`, `eps_diluted`) kết hợp thanh khoản (`current_ratio`, `ebitda_coverage`). Trong khi đó, từ EDA, top 5 Pearson correlation lại là nhóm đòn bẩy (`liab_to_assets`, `st_debt_to_assets`...). Điều này cho thấy XGBoost học được pattern phức tạp hơn: đòn bẩy cao chưa đủ để predict distress, nhưng kết hợp với lợi nhuận thấp và định giá bất thường thì mới thực sự nguy hiểm.

---

## 7. Scorecard

Xác suất từ XGBoost được chuyển sang thang điểm tín dụng theo công thức:

```
Score = 500 + 50 × log((1 − prob) / prob)
```

Thiết kế này đảm bảo prob càng cao (rủi ro cao) thì score càng thấp, và ngược lại. Offset 500 là điểm chuẩn, factor 50 điều chỉnh độ phân tán.

**Kết quả trên tập test:**

| Nhóm | Score trung bình |
|---|---|
| Distressed (y=1) | 474.7 |
| Healthy (y=0) | 737.3 |
| Toàn bộ | 707.2 (std = 154.9) |

Chênh lệch 262.6 điểm giữa hai nhóm — phân tách rõ ràng.

**Phân tầng rủi ro:**

| Grade | Ngưỡng score | Số quan sát | Distress Rate |
|---|---|---|---|
| High Risk | < 500 | 81 | 48.15% |
| Medium Risk | 500 – 650 | 98 | 22.45% |
| Low Risk | 650 – 750 | 60 | 3.33% |
| Very Low Risk | ≥ 750 | 311 | 0.00% |

Distress rate giảm monotonic từ High → Very Low Risk — scorecard phân tầng đúng chiều. Very Low Risk = 0% trên test set lần này, nhưng không đảm bảo giữ nguyên với data mới.

---

## 8. Kết luận

**Model tốt hơn:** XGBoost thắng cả 3 metrics (AUC 0.9214, KS 0.7424, Brier 0.0798), được chọn làm model chính cho scoring.

**Đặc điểm nổi bật:** Recall 96.83% — bắt được hầu hết các trường hợp distress. Scorecard phân tầng đúng chiều, phân tách rõ giữa healthy và distressed (chênh 262 điểm).

**Hạn chế cần lưu ý:**

1. **AUC 0.9214 thấp hơn paper (0.97):** Dataset chỉ có 266 công ty × 7 năm, label gán bằng rule-based nên có noise — một số công ty bị gán nhãn sai hoặc không đầy đủ thông tin.

2. **High Risk distress rate chỉ 48.15%:** Chưa đủ để tự động từ chối mà không cần review thêm. Lý tưởng cần >80% để tự động hóa quyết định. Cần kết hợp thêm biến định tính hoặc thông tin ngoài báo cáo tài chính.

3. **Threshold thấp (0.0676):** Nhạy cảm với nhiễu. Trong triển khai thực tế, nên dùng xác suất trực tiếp để quyết định thay vì ngưỡng cứng, hoặc thiết kế ngưỡng khác nhau cho từng use case (auto-reject vs manual review).

4. **Calibration còn kém:** Cả hai model vẫn lệch khá xa perfect calibration. Nếu cần xác suất chính xác để tính expected loss, cần thêm bước post-calibration.
