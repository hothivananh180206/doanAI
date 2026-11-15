# Phân loại rác bằng Python, Streamlit và YOLO

Ứng dụng web chạy cục bộ trên máy. Giao diện được dựng bằng Streamlit trong
Python, có một lớp CSS gọn trong `app.py` để đồng bộ màu sắc, thẻ nội dung và
thanh điều hướng; dự án không dùng framework frontend riêng. YOLO xử lý ảnh,
OpenCV/NumPy đọc và vẽ khung, SQLite lưu tài khoản và lịch sử, Streamlit hiển
thị biểu đồ thống kê.

## Chạy ứng dụng

Máy Windows đã cài Python 3.12:

```powershell
Set-Location D:\doanAI
$python = "$env:LocalAppData\Programs\Python\Python312\python.exe"
& $python -m pip install -r .\requirements.txt
& $python .\main.py
```

Hoặc chạy trực tiếp:

```powershell
& $python -m streamlit run .\app.py
```

Mở địa chỉ `http://127.0.0.1:8501`. Dừng máy chủ bằng `Ctrl+C` trong PowerShell.
Lần đầu chạy, Streamlit có thể cần Internet để tải model `yolo11n.pt` và
`yolo11n-cls.pt`.

## Đăng nhập và tải lại trang

Phiên đăng nhập được lưu bằng cookie trình duyệt và token ngẫu nhiên đã băm trong
SQLite. Tải lại trang (F5) hoặc mở lại URL trang sẽ giữ đăng nhập và trang hiện
tại; token bị thu hồi khi người dùng đăng xuất. Cookie có thời hạn tối đa một
năm. Tài khoản bị khóa hoặc xóa không thể dùng token để tiếp tục truy cập.
Không chia sẻ cookie hoặc hồ sơ trình duyệt đã đăng nhập cho người khác.

Ở lần đầu mở ứng dụng, tạo tài khoản quản trị viên. Mật khẩu lưu dạng PBKDF2
trong `database/waste.db`. Sau đó quản trị viên có thể tạo/khóa tài khoản, quản
lý danh mục và hướng dẫn, cấu hình model, xem lịch sử.

## Phân quyền

- **Người dùng:** nhận diện ảnh, gửi góp ý, tra cứu danh mục/hướng dẫn; chỉ xem
  thống kê và xem/xóa lịch sử của tài khoản mình.
- **Quản trị viên:** quản lý tài khoản và vai trò, danh mục, hướng dẫn, cấu hình
  và huấn luyện model; xem lịch sử và thống kê toàn hệ thống.
- Tài khoản bị khóa hoặc bị xóa sẽ bị đăng xuất ở lần ứng dụng kiểm tra phiên
  tiếp theo. Tài khoản tự đăng ký luôn mang vai trò người dùng.

## Nhận diện ảnh

1. Đăng nhập và mở trang **Nhận diện rác**.
2. Tải ảnh JPG/PNG/BMP/WEBP hoặc chụp một ảnh từ webcam.
3. YOLO phát hiện vật thể, vẽ bounding box, đếm vật thể và lưu kết quả vào SQLite.
4. Quản trị viên có thể cấu hình model classification đã huấn luyện để phân loại
   vật liệu trong từng vùng cắt. Lịch sử ảnh gốc và ảnh có khung lưu trong
   `data/history`.

Camera dùng `st.camera_input` để chụp ảnh đơn; không phải video stream thời gian
thực. Việc phát video liên tục trong trình duyệt thường cần thành phần frontend
riêng, nằm ngoài phạm vi giao diện Streamlit hiện tại.

## Model và độ chính xác

Mặc định `yolo11n.pt` là model YOLO detect trên các nhãn COCO phổ thông, không
phải detector rác chuyên biệt. Nó có thể tìm thấy vật như chai hoặc cốc nhưng
không luôn biết đó là nhựa hay thủy tinh. Dùng checkpoint `detect` được huấn
luyện bằng bounding box rác để nhận diện đúng các lớp rác và vẽ khung chuyên
dụng.

Checkpoint `runs/classify/.../best.pt` chỉ phân loại ảnh/vùng cắt, không tạo
bounding box. Dataset phân loại hiện có khoảng một ảnh mỗi lớp cho train và
validation nên ứng dụng từ chối huấn luyện classifier với số lượng này để tránh
tạo model có vẻ hoạt động nhưng dự đoán không đáng tin. Cần ít nhất 50 ảnh train
và 10 ảnh validation khác nhau cho mỗi lớp; đây chỉ là mức khởi đầu.

Để có kết quả nhiều khung như chai nhựa và thùng carton trong cùng ảnh, cần
dataset **YOLO detect** đã gắn bounding box cho từng vật thể. Trang
**Quản trị hệ thống → Cài đặt AI** có chức năng huấn luyện detector từ `data.yaml`.
Cấu trúc thường dùng:

```text
dataset/detect/
  data.yaml
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Mỗi ảnh cần một tệp `.txt` cùng tên trong thư mục `labels`; mỗi dòng theo dạng
`class_id x_center y_center width height` với tọa độ chuẩn hóa từ 0 đến 1.
Ví dụ `dataset/detect/data.yaml`:

```yaml
path: .
train: images/train
val: images/val
names:
  0: plastic bottle
  1: cardboard
```

`data.yaml` khai báo đường dẫn train/val và tên lớp. Ảnh mẫu trong giao diện
không đủ làm dataset huấn luyện: hiện dự án chưa có ảnh detect đã gắn khung,
nên chưa thể huấn luyện đúng model cho ví dụ đó. Cần bổ sung ảnh đa dạng và
nhãn bounding box chính xác trước khi bấm huấn luyện. Tên các lớp cần khớp với
tên loại trong danh mục (ví dụ `plastic bottle` được gán nhóm **Tái chế**).
Không thể bảo đảm độ chính xác cụ thể chỉ từ số epoch.
Để giảm RAM khi huấn luyện trên CPU, ứng dụng dùng ảnh 320 px, batch 1, không
nạp sẵn dataset vào bộ nhớ và không tạo worker phụ; huấn luyện có thể chậm hơn.

Nếu model nhận diện sai, dùng các ảnh có nhãn rõ và huấn luyện lại; không thể
bảo đảm nhận diện chính xác mọi ảnh Google. Tôn trọng giấy phép ảnh và kiểm tra
nguồn trước khi đưa ảnh vào dataset.

## Cấu trúc dự án

```text
app.py                  # Điều hướng Streamlit, Python
main.py                 # Khởi chạy Streamlit
config.py               # Đường dẫn và cấu hình
ai/
  detector.py           # YOLO detect, classifier crop, OpenCV
  trainer.py            # Huấn luyện classifier
database/
  db.py                 # SQLite, tài khoản, danh mục, lịch sử
pages/
  home.py
  recognition.py
  catalog.py
  instructions.py
  history.py
  statistics.py
  admin.py
services/
  auth.py
dataset/
data/history/
ai/model/
requirements.txt
```

## SQLite

File cơ sở dữ liệu là `database/waste.db`. Các bảng gồm:

- `users`: tài khoản, vai trò, trạng thái.
- `waste_types`, `instructions`: danh mục rác và hướng dẫn xử lý.
- `detection_history`, `detection_details`: ảnh, thời điểm, model và từng vật thể.
- `feedback`: góp ý người dùng.
- `app_settings`: cấu hình đường dẫn model.

Không xóa file cơ sở dữ liệu khi ứng dụng đang chạy.
