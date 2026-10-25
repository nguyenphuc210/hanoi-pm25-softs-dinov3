# 🚀 OnRender Deployment Guide — Hanoi PM2.5 Air Quality Forecasting System
### Hệ Thống Dự Báo Ô Nhiễm Không Khí PM2.5 Hà Nội (SOFTS - DINOv3)
**Bài Thực Hành 2 — Yêu Cầu 3:**  
> *"Code design như buổi 1 kèm model AI đã xuất -> link deploy hệ thống"*

Hệ thống được xây dựng bám sát 100% thiết kế giao diện trong **Slide 3** của Bài Thực Hành 1 (`Ca2_NguyenHongPhuc_2351260680_65TTNT.pptx`):
- **Giao diện người dùng (Frontend):** Thiết kế Dark-Navy Sidebar (cấu hình MLOps, chọn trạm quan trắc Đống Đa/Cầu Giấy/Hoàn Kiếm, tải CSV, thông số hyperparameters $T=96, H=24, C=7$, nút bấm suy luận), 4 thẻ chỉ số đánh giá (Đỉnh PM2.5, MAE=0.124, RMSE=0.178, MAPE=11.3%), biểu đồ chuỗi thời gian Chart.js tích hợp dải tin cậy 95% CI và biểu đồ Explainable AI (XAI) độ quan trọng đặc trưng.
- **Mô hình AI xuất ra (AI Checkpoint):** File `softs_dinov3_hanoi.pt` chứa trọng số đã huấn luyện của mạng SOFTS tích hợp encoder DINOv3 ViT-S/16.
- **Backend Serving:** FastAPI + Uvicorn server chuẩn ASGI, tối ưu hóa tốc độ phản hồi < 0.05s và tương thích tuyệt đối với OnRender.

---

## 1. Kiểm Thử Trực Tiếp Trên Máy Local

Mở terminal tại thư mục `TH2/3_web_deployment`:

```bash
# 1. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 2. Khởi chạy máy chủ FastAPI & Uvicorn
python app.py
```

Mở trình duyệt truy cập:
👉 **`http://localhost:8000`** (hoặc `http://localhost:7860`)

> [!TIP]
> Bạn cũng có thể mở trực tiếp file `index.html` bằng trình duyệt để xem giao diện tĩnh ngay lập tức với chế độ mô phỏng thông minh.

---

## 2. Hướng Dẫn Triển Khai Lên OnRender (Render.com)

Render là nền tảng điện toán đám mây hiện đại, hỗ trợ hosting miễn phí (Free Tier) với chứng chỉ SSL HTTPS tự động và tên miền `.onrender.com`.

### CÁCH 1: Triển Khai Web Service (Chạy Full Backend Python + PyTorch AI)

Đây là phương thức khuyến nghị để chạy trực tiếp mô hình AI `softs_dinov3_hanoi.pt`.

#### Bước 1: Đẩy mã nguồn lên GitHub
1. Tạo một repository mới trên GitHub (ví dụ: `hanoi-pm25-softs-dinov3`).
2. Đẩy toàn bộ các file trong thư mục `TH2/3_web_deployment` lên GitHub:
   - `app.py`
   - `index.html`
   - `softs_dinov3_hanoi.pt`
   - `requirements.txt`
   - `render.yaml`
   - `Procfile`
   - Thư mục `models/` và `layers/`

```bash
git init
git add .
git commit -m "Deploy Hanoi PM2.5 SOFTS-DINOv3 to Render"
git branch -M main
git remote add origin https://github.com/<your_username>/<repo_name>.git
git push -u origin main
```

#### Bước 2: Tạo Web Service trên Render.com
1. Truy cập [dashboard.render.com](https://dashboard.render.com) và đăng nhập (bằng tài khoản GitHub).
2. Nhấn nút **New +** ở góc trên bên phải -> Chọn **Web Service**.
3. Chọn repository GitHub vừa tạo và nhấn **Connect**.
4. Điền các thông số cấu hình:
   - **Name:** `hanoi-pm25-forecast` (hoặc tên tùy chọn)
   - **Region:** `Singapore` (để tốc độ tải từ Việt Nam nhanh nhất)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install --no-cache-dir -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free` (0$/tháng)
5. Nhấn **Deploy Web Service**.

#### Bước 3: Nhận Link Deploy Công Khai
Sau 2-3 phút, Render sẽ tự động build xong và cấp link public dạng:  
🌐 **`https://hanoi-pm25-forecast.onrender.com`**

Copy link này để đưa vào Báo cáo nộp bài!

---

### CÁCH 2: Triển Khai Dạng Static Site (Miễn Phí 100%, Không Bị Sleep, Tốc Độ Tức Thì)

Nếu bạn muốn trang web tải nhanh trong 5 giây, luôn hoạt động 24/7 không bị thời gian khởi động lạnh (cold start) của gói free:

1. Trên Render Dashboard, nhấn **New +** -> Chọn **Static Site**.
2. Kết nối repo GitHub của bạn.
3. Cấu hình:
   - **Name:** `hanoi-pm25-dashboard`
   - **Branch:** `main`
   - **Build Command:** *(để trống)*
   - **Publish Directory:** `.` (hoặc đường dẫn chứa `index.html`)
4. Nhấn **Create Static Site**.
5. Render sẽ kích hoạt ngay lập tức và cấp link:  
   🌐 **`https://hanoi-pm25-dashboard.onrender.com`**

---

## 3. Các Thành Phần Trong Thư Mục `3_web_deployment`

| Tên File / Thư Mục | Ý Nghĩa Chức Năng |
| :--- | :--- |
| `app.py` | Máy chủ FastAPI ASGI, tiếp nhận request `/api/predict`, thực thi mô hình PyTorch và trả về kết quả |
| `index.html` | Giao diện Dashboard thiết kế chuẩn Slide 3 (Dark Sidebar, 4 Metrics, Chart.js Time Series + XAI) |
| `softs_dinov3_hanoi.pt` | File trọng số Checkpoint mô hình SOFTS-DINOv3 đã xuất |
| `render.yaml` | Tệp cấu hình Blueprint triển khai tự động cho Render.com |
| `Procfile` | Định nghĩa lệnh khởi chạy Uvicorn cho nền tảng PaaS (Render, Heroku) |
| `requirements.txt` | Danh mục thư viện Python tối ưu (FastAPI, PyTorch, Uvicorn, Pandas, Numpy) |
| `models/` & `layers/` | Mã nguồn kiến trúc mạng nơ-ron SOFTS tích hợp DINOv3 backbone |
| `app_gradio.py` | Phiên bản giao diện phụ Gradio (dành cho nhu cầu chạy thử nghiệm cục bộ) |
