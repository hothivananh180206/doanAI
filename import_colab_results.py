import os
import shutil
import zipfile
from pathlib import Path

def import_results():
    base = Path(__file__).resolve().parent
    model_dest = base / "ai" / "model" / "best.pt"
    reports_dest = base / "reports"

    reports_dest.mkdir(parents=True, exist_ok=True)
    model_dest.parent.mkdir(parents=True, exist_ok=True)

    # Tìm file zip mới nhất từ Colab
    zips = list(base.glob("*.zip"))
    colab_zips = [z for z in zips if "organic" in z.name.lower() or "ket_qua" in z.name.lower() or "train" in z.name.lower()]
    
    if colab_zips:
        # Chọn file zip mới nhất
        colab_zips.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        zip_path = colab_zips[0]
        print(f"[*] Đang giải nén kết quả mới nhất: {zip_path.name}...")
        
        temp_extract = base / "temp_colab_results"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)

        # 1. Tìm best.pt và copy vào ai/model/best.pt
        found_pt = list(temp_extract.rglob("best.pt"))
        if found_pt:
            shutil.copy(found_pt[0], model_dest)
            print(f"[✓] Đã sao chép mô hình mới nhất vào: {model_dest}")
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

        print(f"[✓] Đã sao chép {copied_count} biểu đồ đánh giá vào: {reports_dest}")
        shutil.rmtree(temp_extract, ignore_errors=True)

    else:
        downloaded_pt = base / "best.pt"
        if downloaded_pt.exists():
            shutil.copy(downloaded_pt, model_dest)
            print(f"[✓] Đã cập nhật mô hình mới vào: {model_dest}")
        else:
            print("[!] Không tìm thấy file zip hoặc best.pt ở thư mục gốc.")
            print("    👉 Hãy tải file kết quả từ Colab về thư mục này rồi chạy lại.")
            return

    print("\n🎉 HOÀN TẤT CẬP NHẬT MÔ HÌNH VÀ BÁO CÁO TỪ COLAB!")
    print("👉 Bây giờ hãy mở web kiểm tra lại ảnh vỏ chuối để xem kết quả Rác Hữu Cơ nhé!")

if __name__ == "__main__":
    import_results()
