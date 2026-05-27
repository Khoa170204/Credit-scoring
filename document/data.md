# Tài liệu dữ liệu — Credit Scoring Pipeline

---

## 1. Thu thập dữ liệu

### Nguồn
Crawl data và build bộ dataset được thực hiện và tham khảo dựa trên bộ dữ liệu của bài báo 'Explainable Machine Learning for Financial Distress Prediction:
Evidence from Vietnam'

Dữ liệu được thu thập từ **vnstock API** (nguồn VCI), bao gồm các công ty niêm yết trên HOSE và HNX. Toàn bộ quá trình crawl được thực hiện tự động qua 3 bước:

**Bước 1 — khai báo API của vnstock (`.env`)**
Trên trang vnstock:
- Đăng nhập để lấy API
- Dán API vào VNSTOCK_API_KEY (API này là cá nhân nên không share)

**Bước 2 — Lấy danh sách mã chứng khoán (`src/get_symbols.py`)**

Lấy toàn bộ mã từ vnstock, sau đó lọc theo 3 tiêu chí:
- Chỉ lấy sàn HOSE và HNX 
- Chỉ lấy loại `stock` (bỏ ETF, chứng chỉ quỹ)
- Loại bỏ các công ty tài chính đặc thù: ngân hàng, chứng khoán, bảo hiểm, công ty tài chính tiêu dùng
Lí do loại bỏ: Vì bộ 25 features được thiết kế cho công ty sản xuất/thương mại, không áp dụng được cho nhóm này

mã được lưu vào `data/vnstock/symbols.txt`.

**Bước 3 — Crawl dữ liệu (`src/crawl.py`)**

Với mỗi mã, crawl 4 loại dữ liệu và lưu vào file `.pkl`:

| Key trong pkl | Mô tả | Format |
|---|---|---|
| `bs` | Bảng cân đối kế toán | Wide: dòng = chỉ tiêu, cột = năm |
| `is` | Báo cáo kết quả kinh doanh | Wide: dòng = chỉ tiêu, cột = năm |
| `quote` | Lịch sử giá cổ phiếu (OHLCV) | Long: mỗi dòng = 1 ngày giao dịch |
| `ratio` | Các tỷ số tài chính tổng hợp | Wide: dòng = tỷ số, cột = kỳ |

Mỗi file pkl tương ứng với 1 công ty, lưu tại `data/vnstock/pkl/<symbol>.pkl`. 
Khoảng cách giữa mỗi request là 2 giây để tránh rate limit.

---

## 2. Tiền xử lý và xây dựng dataset

### Lọc công ty hợp lệ (`src/build_dataset.py`)

Một công ty bị loại khỏi dataset nếu thoả bất kỳ điều kiện nào sau:
- Thiếu bảng cân đối hoặc báo cáo KQKD
- Không có đủ 3 chỉ tiêu bắt buộc: `Total Assets`, `Owner's Equity`, `Liabilities`
- Có ít hơn 5 năm dữ liệu
- Có hơn 180 chỉ tiêu trong bảng cân đối (đây là schema của công ty chứng khoán, khác cấu trúc thông thường)

### Xây dựng nhãn (label)

Với mỗi cặp (công ty, năm), nhãn `Financial_Distress = 1` nếu thoả **ít nhất 1** trong 3 tiêu chí sau:

| Tiêu chí | Định nghĩa |
|---|---|
| 1 | Vốn chủ sở hữu âm trong năm đó |
| 2 | Operating profit / \|Interest expenses\| < 1 trong 2 năm liên tiếp |
| 3 | Operating profit âm trong 3 năm liên tiếp |

Ngược lại, nhãn = 0.

Lưu ý: Năm 2018 luôn có nhãn = 0 vì tiêu chí 2 và 3 cần nhìn lại 1–2 năm trước, mà dữ liệu chỉ bắt đầu từ 2018 nên thiếu data 2016–2017.

### Tính features

Với mỗi cặp (công ty, năm), tính 25 features từ 3 nguồn:
- **Bảng cân đối** (BS): các chỉ tiêu tại thời điểm cuối năm
- **Báo cáo KQKD** (IS): các chỉ tiêu trong năm
- **Giá cổ phiếu** (Quote): giá đóng cửa cuối năm để tính market cap

**Biến trung gian từ bảng cân đối (tại thời điểm cuối năm):**

```
cash             = Cash and cash equivalents
inventory        = Inventories, Net
current_assets   = CURRENT ASSETS
current_liab     = Current liabilities
lt_debt          = Long-term borrowings         (mặc định 0 nếu không có)
st_debt          = Short-term borrowings         (mặc định 0 nếu không có)
ap               = Trade accounts payable        (mặc định 0 nếu không có)
total_assets     = Total Assets
total_liab       = Liabilities
equity           = Owner's Equity
intangible       = Intangible fixed assets       (mặc định 0 nếu không có)
goodwill         = Goodwill                      (mặc định 0 nếu không có)
```

**Biến trung gian từ báo cáo KQKD (trong năm):**

```
net_sales        = Net sales
interest         = Interest expenses
op_profit        = Operating profit/(loss)
eps_diluted      = EPS diluted (VND)
eps_basic        = EPS basic (VND)
eps              = eps_diluted nếu có và khác 0, ngược lại dùng eps_basic
```

**Biến trung gian từ giá cổ phiếu:**

```
close_price      = Giá đóng cửa ngày giao dịch cuối cùng trong năm
shares           = Paid-in capital / 10.000      (mệnh giá cổ phiếu VN = 10.000 VND/CP)
market_cap       = close_price × 1.000 × shares  (close_price đơn vị nghìn VND, nhân 1.000 ra VND)
```

**Biến trung gian để tính EBITDA:**

Vnstock không có chỉ tiêu khấu hao riêng trong báo cáo KQKD, nên khấu hao được ước tính từ bảng cân đối:

```
dep_cur          = Tổng Accumulated depreciation cuối năm Y
dep_prev         = Tổng Accumulated depreciation cuối năm Y-1
da               = dep_cur - dep_prev     (chênh lệch = chi phí khấu hao phát sinh trong năm)
ebitda           = op_profit + da
```

**Biến trung gian để tính Enterprise Value và P/B Tangible:**

```
ev               = market_cap + lt_debt + st_debt - cash
tangible_equity  = equity - intangible - goodwill
```

Năm 2025 bị bỏ qua vì dữ liệu chưa đầy đủ.

### Xử lý missing values

Missing values được giữ nguyên trong file CSV và chỉ được impute tại thời điểm training bằng **median của tập train** (fit trên train, apply cho cả train và test). Cách này tránh data leakage từ test set vào train.

---

## 3. Dataset cuối cùng

**File:** `data/vnstock/dataset_v2.csv`

| Thuộc tính | Giá trị |
|---|---|
| Số dòng | 1.847 |
| Số công ty | 266 |
| Giai đoạn | 2018 – 2024 |
| Số features | 25 |
| Tỷ lệ distress | 13,16% (243/1.847) |

---

## 4. Data dictionary

### Thông tin định danh

| Feature | Type | Meaning |
|---|---|---|
| `Company` | string | Mã chứng khoán của công ty. Ví dụ: FPT, HPG, VNM. |
| `year` | int | Năm tài chính tương ứng với dòng dữ liệu. |
| `Financial_Distress` | int (0/1) | Nhãn dự báo. 1 = công ty đang có dấu hiệu kiệt quệ tài chính, 0 = hoạt động bình thường. |

---

### Nhóm 1 — Thanh khoản (Liquidity)

Đo khả năng công ty thanh toán các nghĩa vụ ngắn hạn.

| Feature | Type | Meaning |
|---|---|---|
| `cash_ratio` | float | Tiền mặt / Nợ ngắn hạn. Đo lượng tiền mặt sẵn có để trả nợ ngắn hạn ngay lập tức. Ví dụ: 0.16 nghĩa là cứ 1 đồng nợ ngắn hạn chỉ có 0.16 đồng tiền mặt. |
| `quick_ratio` | float | (Tài sản ngắn hạn − Hàng tồn kho) / Nợ ngắn hạn. Khả năng thanh toán nhanh, loại bỏ hàng tồn kho vì khó chuyển thành tiền ngay. Dưới 1 là cảnh báo. |
| `current_ratio` | float | Tài sản ngắn hạn / Nợ ngắn hạn. Khả năng thanh toán ngắn hạn tổng thể, tính cả hàng tồn kho. Dưới 1 nghĩa là nợ ngắn hạn vượt quá tài sản ngắn hạn. |

---

### Nhóm 2 — Rủi ro tài chính (Financial Risk)

Đo mức độ sử dụng đòn bẩy tài chính.

| Feature | Type | Meaning |
|---|---|---|
| `lt_debt_to_equity` | float | Nợ dài hạn / Vốn chủ sở hữu. Mức độ dùng nợ dài hạn để tài trợ cho công ty. Càng cao thì áp lực trả nợ trong tương lai càng lớn. |
| `lt_debt_to_assets` | float | Nợ dài hạn / Tổng tài sản. Tỷ trọng nợ dài hạn trong toàn bộ tài sản của công ty. |
| `liab_to_equity` | float | Tổng nợ / Vốn chủ sở hữu. Đòn bẩy tổng thể. Giá trị âm khi vốn chủ sở hữu âm — đây là tình huống cực kỳ nguy hiểm. |
| `liab_to_assets` | float | Tổng nợ / Tổng tài sản. Tỷ lệ tài sản được tài trợ bằng nợ. Ví dụ: 0.45 nghĩa là 45% tài sản đến từ nợ, 55% từ vốn chủ. |
| `st_debt_to_equity` | float | Nợ vay ngắn hạn / Vốn chủ sở hữu. Gánh nặng nợ vay ngắn hạn so với vốn tự có của công ty. |
| `st_debt_to_assets` | float | Nợ vay ngắn hạn / Tổng tài sản. Tỷ trọng nợ vay ngắn hạn trong cơ cấu nguồn vốn. |

---

### Nhóm 3 — Rủi ro kinh doanh (Business Risk)

Đo hiệu quả hoạt động và khả năng trả nợ từ kết quả kinh doanh.

| Feature | Type | Meaning |
|---|---|---|
| `ap_to_equity` | float | Phải trả người bán / Vốn chủ sở hữu. Áp lực nợ thương mại so với vốn tự có. Cao nghĩa là công ty đang phụ thuộc nhiều vào tín dụng từ nhà cung cấp. |
| `ap_to_assets` | float | Phải trả người bán / Tổng tài sản. Tỷ trọng nợ thương mại trong tổng tài sản. |
| `assets_to_liab` | float | Tổng tài sản / Tổng nợ. Khả năng dùng tài sản để bù đắp toàn bộ nợ. Dưới 1 nghĩa là nợ lớn hơn tài sản — mất khả năng thanh toán. |
| `ebitda_coverage` | float | EBITDA / (Nợ vay ngắn hạn + \|Lãi vay\|). Khả năng trả nợ từ lợi nhuận hoạt động. EBITDA được ước tính bằng lợi nhuận kinh doanh cộng thêm khấu hao. Missing 10,6% khi công ty không có nợ vay và không có chi phí lãi vay (mẫu số = 0) — nhóm này thực ra chủ yếu là công ty lành mạnh. |

---

### Nhóm 4 — Yếu tố thị trường (Market Factors)

Đo định giá công ty dựa trên giá cổ phiếu. Missing 2,4% là các công ty niêm yết sau 2018, chưa có giá cổ phiếu cho năm đầu tiên.

| Feature | Type | Meaning |
|---|---|---|
| `pe_basic` | float | Giá cổ phiếu / EPS cơ bản. Nhà đầu tư sẵn sàng trả bao nhiêu đồng cho 1 đồng lợi nhuận. Âm khi công ty thua lỗ. |
| `pe_diluted` | float | Giá cổ phiếu / EPS pha loãng. Tương tự pe_basic nhưng tính thêm cổ phiếu tiềm năng từ trái phiếu chuyển đổi, quyền chọn. Missing 48,7% vì nhiều công ty không phát hành thêm cổ phiếu pha loãng nên EPS diluted = 0. |
| `pb_ratio` | float | Vốn hoá / Vốn chủ sở hữu. Giá thị trường so với giá trị sổ sách. Dưới 1 nghĩa là thị trường định giá công ty thấp hơn giá trị tài sản ròng trên sổ sách. |
| `ps_ratio` | float | Vốn hoá / Doanh thu thuần. Giá thị trường so với doanh thu. Dùng khi công ty chưa có lợi nhuận nên không tính được P/E. |
| `pb_tangible` | float | Vốn hoá / (Vốn chủ − Tài sản vô hình − Goodwill). Tương tự pb_ratio nhưng loại trừ tài sản vô hình và goodwill, phản ánh giá trị tài sản hữu hình thực tế hơn. |
| `market_cap` | float | Giá đóng cửa cuối năm × Số cổ phiếu lưu hành. Tổng giá trị vốn hoá thị trường tính bằng VND. Ví dụ: FPT năm 2023 khoảng 80 nghìn tỷ đồng. |
| `price_to_cfo` | float | Vốn hoá / Lợi nhuận kinh doanh. Giá thị trường so với dòng tiền từ hoạt động kinh doanh. Âm khi lợi nhuận kinh doanh âm. |

---

### Nhóm 5 — Định giá doanh nghiệp (Enterprise Value)

| Feature | Type | Meaning |
|---|---|---|
| `ev` | float | Vốn hoá + Nợ dài hạn + Nợ ngắn hạn − Tiền mặt. Giá trị toàn bộ doanh nghiệp nếu mua lại, tính cả nợ và trừ tiền mặt sẵn có. Đơn vị VND. |
| `ev_to_revenue` | float | EV / Doanh thu thuần. Bội số doanh thu. Dùng để so sánh định giá giữa các công ty trong cùng ngành, đặc biệt với công ty chưa có lợi nhuận. |
| `ev_to_ebitda` | float | EV / EBITDA. Bội số EV/EBITDA, chỉ số định giá phổ biến nhất trong M&A. Âm khi EBITDA âm — thường là dấu hiệu rủi ro cao. |
| `ev_to_ebit` | float | EV / EBIT. Bội số EV/EBIT. **Feature quan trọng nhất theo SHAP.** Công ty distress thường có EBIT âm hoặc rất thấp, khiến chỉ số này ra giá trị bất thường (âm hoặc cực lớn), dễ phân biệt với công ty lành mạnh. |
| `eps_diluted` | float | Lợi nhuận sau thuế / Số cổ phiếu pha loãng, đơn vị VND. Thu nhập trên mỗi cổ phiếu. Âm khi công ty thua lỗ. Nếu không có dữ liệu EPS pha loãng thì dùng EPS cơ bản thay thế. |

---

