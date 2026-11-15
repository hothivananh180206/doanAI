import os
import shutil
import zipfile
from pathlib import Path

def import_results():
    base = Path(__file__).resolve().parent
    zip_path = base / "ket_qua_train.zip"
    model_dest = base / "ai" / "model" / "best.pt"
    reports_dest = base / "reports"

    reports_dest.mkdir(parents=True, exist_ok=True)
    model_dest.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        # Kiểm tra xem có file best.pt rời vừa tải về không
        downloaded_pt = base / "best.pt"
        if downloaded_pt.exists():
            shutil.copy(downloaded_pt, model_dest)
            print(f"[✓] Đã cập nhật mô hình mới vào: {model_dest}")
            return
        print(f"[!] Không tìm thấy {zip_path.name} hoặc best.pt ở thư mục gốc.")
        print("    👉 Hãy tải file ket_qua_train.zip (hoặc best.pt) từ Colab về thư mục này rồi chạy lại.")
        return

    print(f"[*] Đang giải nén {zip_path.name}...")
    temp_extract = base / "temp_colab_results"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)

    # 1. Tìm best.pt và copy vào ai/model/best.pt
    found_pt = list(temp_extract.rglob("best.pt"))
    if found_pt:
        shutil.copy(found_pt[0], model_dest)
        print(f"[✓] Đã sao chép mô hình tốt nhất vào: {model_dest}")
    else:
        print("[!] Không tìm thấy best.pt trong file zip.")

    # 2. Tìm các biểu đồ và copy vào reports/
    chart_names = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.png",
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
    ]
    copied_count = 0
    for name in chart_names:
        matches = list(temp_extract.rglob(name))
        if matches:
            shutil.copy(matches[0], reports_dest / name)
            copied_count += 1

    print(f"[✓] Đã sao chép {copied_count} biểu đồ đánh giá (Confusion matrix, Loss, ...) vào: {reports_dest}")

    # Xóa thư mục tạm
    shutil.rmtree(temp_extract, ignore_errors=True)
    print("\n🎉 HOÀN TẤT NHẬP KẾT QUẢ TỪ COLAB!")
    print("👉 Bây giờ bạn có thể mở web kiểm tra nhận diện và xem biểu đồ trong thư mục reports.")

if __name__ == "__main__":
    import_results()
