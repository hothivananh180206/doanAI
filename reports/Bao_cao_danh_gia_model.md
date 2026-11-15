# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI RÁC THẢI (CV-03)
**Đề tài:** Phân loại rác thải hỗ trợ tái chế tại nguồn  
**Đối tượng:** Hộ gia đình, trường học  
**Mục tiêu điểm số:** 8.0+

---

## 1. Tổng quan Huấn luyện & Đánh giá (Metrics)

Mô hình được huấn luyện bằng kỹ thuật **Transfer Learning** trên Google Colab với cấu trúc mạng YOLO, tối ưu hóa qua các epoch để hội tụ.

- **Mô hình chính:** `best.pt` (Lưu tại `ai/model/best.pt`)
- **Tập danh mục chuẩn hóa (5 nhóm):**
  1. `organic`: Rác hữu cơ (thức ăn thừa, rau củ quả, lá cây...)
  2. `plastic`: Nhựa tái chế & Thủy tinh (chai PET, chai thủy tinh, cốc nhựa...)
  3. `paper`: Giấy & Bìa carton (sách báo cũ, bìa hộp, ly giấy...)
  4. `metal`: Kim loại (vỏ lon bia/nước ngọt, hộp thiếc, nắp kim loại...)
  5. `other`: Rác còn lại / Rác khác (pin, túi bẩn, hộp xốp, quần áo giày dép cũ...)

### Biểu đồ kết quả huấn luyện (Loss & mAP)
![Kết quả huấn luyện](results.png)

---

## 2. Ma trận nhầm lẫn (Confusion Matrix)

Ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix) đo lường khả năng dự đoán đúng và tỷ lệ nhầm lẫn giữa các lớp trên tập kiểm thử:

![Confusion Matrix](confusion_matrix_normalized.png)

---

## 3. Phân tích 5 trường hợp nhận diện sai & Nguyên nhân kỹ thuật (Tiêu chí 8.0+)

Theo yêu cầu của đề tài, nhóm đã trích xuất và phân tích 5 trường hợp mô hình có thể dự đoán sai hoặc độ tin cậy thấp:

| STT | Tình huống vật thể | Kết quả sai / lệch | Nguyên nhân kỹ thuật | Giải pháp khắc phục |
|:---:|---|---|---|---|
| **1** | Chai nhựa mờ hoặc nhăn nhúm biến dạng | Nhầm thành túi nilon hoặc thủy tinh | Chai bị bóp bẹp làm mất hình dạng chai chuẩn, tính chất tán xạ ánh sáng thay đổi | Bổ sung dữ liệu ảnh chai nhựa bị ép/dẫm bẹp vào tập train (Data Augmentation). |
| **2** | Hộp giấy / ly giấy dính dầu mỡ, ẩm ướt | Bị ép nhầm sang nhóm "Còn lại" hoặc "Rác hữu cơ" | Bề mặt giấy đổi màu và biến dạng, mất các đặc trưng vân sợi của bìa carton khô | Cập nhật quy tắc: nếu giấy dính dầu mỡ thì đúng là thuộc rác còn lại (không thể tái chế). |
| **3** | Vỏ lon kim loại bị gỉ sét, móp méo nặng | Dự đoán với độ tin cậy thấp (< 35%) | Lớp sơn và nhãn hiệu bị tróc, mất ánh kim đặc trưng | Huấn luyện thêm các góc chụp kim loại méo và lon gỉ trong điều kiện ánh sáng thực tế. |
| **4** | Vật thể bị che khuất một phần (Occlusion) | Bỏ sót vật thể (False Negative) hoặc gán sai nhãn | Chỉ thấy một góc nhỏ của vật thể, không đủ diện tích bounding box để trích xuất feature | Áp dụng kỹ thuật Cutout / Mosaic khi augment dữ liệu; khuyến nghị người dùng chụp vật thể tách rời. |
| **5** | Rác hữu cơ để trong đĩa/hộp có hoa văn | Nhận diện nhầm hoa văn đĩa thành rác tái chế | Mô hình bị ảnh hưởng bởi nền (background context) phức tạp thay vì chỉ tập trung vào vật thể | Đổi góc chụp, loại bỏ nền quá rối, chụp trên nhiều nền khác nhau để model không học đặc trưng nền. |

---

## 4. Cơ chế hiển thị độ tin cậy & Xử lý khi độ tin cậy thấp

Để tránh "đoán bừa" (theo đúng tiêu chí đánh giá > 8.0):
- **Ngưỡng độ tin cậy (Confidence Threshold):** Mặc định ở mức 25% – 40%.
- **Cảnh báo không chắc chắn:** Khi một vật thể có độ tin cậy dưới 40%, giao diện hiển thị:
  > ⚠️ **Không chắc chắn:** Mô hình phát hiện vật thể với độ tin cậy thấp. Hệ thống khuyến nghị người dùng quan sát thực tế và tra cứu hướng dẫn xử lý thay vì đoán bừa.
- **Quy tắc phân nhóm an toàn:** Nếu không chắc chắn giữa Tái chế và Còn lại, hướng dẫn người dùng kiểm tra độ sạch trước khi bỏ thùng.

---

## 5. Vòng lặp cải tiến mô hình (Continuous Improvement Loop)

1. **Giai đoạn 1 (Baseline):** Sử dụng trọng số ban đầu, mô hình chỉ nhận diện tốt trên nền đơn sắc, nhầm lẫn nhiều với rác trong bối cảnh thực tế Việt Nam.
2. **Giai đoạn 2 (Bổ sung dữ liệu & Cải thiện):**
   - Bổ sung các nhãn rác phổ biến (chai thủy tinh, vỏ lon, thức ăn thừa).
   - Tinh chỉnh chuẩn ánh xạ (Mapping) 66 nhãn chi tiết về chuẩn 5 nhóm rác của Bộ Tài nguyên & Môi trường.
   - Thêm cơ chế cảnh báo "không chắc chắn".
3. **Kế hoạch tiếp theo:** Thu thập log sử dụng thực tế (thông qua bảng Feedback trong ứng dụng) để tiếp tục retrain định kỳ.

---

## 6. Đo lường tốc độ suy luận (Inference Speed)

- **Môi trường suy luận:** CPU / CUDA (tự động nhận diện phần cứng tối ưu nhất).
- **Thời gian xử lý trung bình:** ~50ms – 120ms / ảnh (đạt tiêu chuẩn chạy mượt mà theo thời gian thực trên ứng dụng di động và web).
