# ♻️ EcoSort AI - Hệ Thống Phân Loại Rác Thải Hỗ Trợ Tái Chế Tại Nguồn

> **Môn học:** Trí tuệ nhân tạo với Python  
> **Mã đề tài:** `CV-03` · Phân loại rác thải hỗ trợ tái chế (⭐⭐)  
> **Công nghệ chính:** Python 3.11/3.12 · Ultralytics YOLO · FastAPI · Streamlit · OpenCV · SQLite · Plotly

---

## 📌 1. Giới thiệu bài toán & Mục tiêu

- **Đối tượng người dùng:** Hộ gia đình, trường học, khu dân cư đang thực hiện phân loại rác thải tại nguồn.
- **Vấn đề thực tế:** Người dân thường khó phân biệt các loại rác tái chế, hữu cơ, rác nguy hại; danh mục quá lớn để tra cứu thủ công.
- **Giải pháp AI:** 
  - Chụp ảnh hoặc quét trực tiếp qua camera/webcam.
  - Mô hình YOLO tự động phát hiện vật thể và phân loại chính xác vào 4 nhóm: **Hữu cơ (xanh lá)**, **Tái chế (xanh dương)**, **Nguy hại (đỏ/vàng)**, **Còn lại (xám)**.
  - Cung cấp hướng dẫn xử lý và mẹo phân loại tương ứng ngay tại chỗ.

---

## 🏛️ 2. Kiến trúc hệ thống (System Architecture)

```
[ Người dùng / Camera / Client ]
               │
       ┌───────┴────────────────┐
       ▼                        ▼
[ Giao diện Streamlit ]   [ FastAPI REST API ] (Port 8000)
 (Port 8501 - Dashboard)        │
       │                        │
       └──────────────┬─────────┘
                      ▼
            [ YOLO AI Engine ]
            - YOLO Object Detection
            - Label to Category Mapping
                      │
                      ▼
            [ SQLite Database ]
            - waste_types & instructions
            - detection_history & logs
```

---

## 🚀 3. Hướng dẫn cài đặt & Khởi chạy

### Yêu cầu hệ thống:
- Python 3.10, 3.11 hoặc 3.12
- Hệ điều hành: Windows, macOS hoặc Linux

### Bước 1: Cài đặt thư viện
```powershell
pip install -r requirements.txt
```

### Bước 2: Chạy ứng dụng giao diện Web (Streamlit)
```powershell
python main.py
# hoặc bấm đúp vào file run.bat
```
👉 Truy cập: `http://localhost:8501`

### Bước 3: Chạy máy chủ API riêng biệt (FastAPI)
```powershell
python api.py
# hoặc bấm đúp vào file run_api.bat
```
👉 API Server: `http://127.0.0.1:8000`  
👉 Tài liệu tương tác Swagger UI: `http://127.0.0.1:8000/docs`

---

## 📡 4. Tài liệu API (FastAPI Endpoints)

| Phương thức | Endpoint | Mô tả |
|:---:|---|---|
| `GET` | `/` | Kiểm tra trạng thái hoạt động của API (Health check) |
| `GET` | `/categories` | Danh sách các nhóm rác và hướng dẫn xử lý |
| `POST` | `/predict` | Nhận diện vật thể rác từ file ảnh tải lên |
| `GET` | `/docs` | Giao diện kiểm thử trực tiếp Swagger UI |

### Ví dụ gọi API bằng cURL:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_images/sample_plastic_bottle.jpg"
```

### Dữ liệu trả về (JSON):
```json
{
  "success": true,
  "filename": "sample_plastic_bottle.jpg",
  "total_detected": 1,
  "categories": ["Tái chế"],
  "detections": [
    {
      "class_name": "Chai / Đồ nhựa",
      "raw_class_name": "plastic",
      "category": "Tái chế",
      "confidence": 0.885,
      "bounding_box": {"x1": 45.0, "y1": 60.0, "x2": 280.0, "y2": 510.0},
      "instructions": "Tráng sạch nước bẩn, tháo nắp chai riêng, ép bẹp thân chai để tiết kiệm không gian và bỏ vào thùng tái chế (xanh dương)."
    }
  ]
}
```

---

## 📊 5. Kiểm thử & Đánh giá mô hình (Metrics & Reports)

Dự án cung cấp sẵn 2 kịch bản kiểm thử:

1. **Đánh giá trên tập test chuẩn (Precision, Recall, mAP50, Confusion Matrix):**
   ```powershell
   python evaluate.py data/dataset.yaml
   ```
   Kết quả và biểu đồ Confusion Matrix được lưu vào thư mục `reports/`.

2. **Kiểm thử trên tập ảnh rác tự chụp thực tế (Việt Nam):**
   - Chép các ảnh chụp từ điện thoại vào thư mục `data/test_vietnam/`.
   - Chạy lệnh kiểm thử:
     ```powershell
     python test_vietnam.py
     ```
   - Kết quả ảnh khoanh vùng và bảng báo cáo được lưu tại `reports/vietnam_predictions/`.

---

## 🌐 6. Hướng dẫn triển khai truy cập từ ngoài (Public Access)

### Cách 1: Dùng Ngrok (Demo nhanh tại buổi bảo vệ)
1. Tải [ngrok](https://ngrok.com/) và mở terminal:
```powershell
ngrok http 8501
```
2. Copy đường link dạng `https://xxxx.ngrok-free.app` chia sẻ cho ban giám khảo hoặc người dùng truy cập trực tiếp từ điện thoại.

### Cách 2: Triển khai qua Streamlit Community Cloud (Miễn phí 24/7)
1. Push mã nguồn lên GitHub cá nhân:
   ```powershell
   git remote add origin https://github.com/<tai-khoan>/<ten-repo>.git
   git branch -M main
   git push -u origin main
   ```
2. Đăng nhập [share.streamlit.io](https://share.streamlit.io/) bằng GitHub.
3. Chọn Repository vừa tạo, chọn nhánh `main`, file chính là `app.py` và bấm **Deploy**.
