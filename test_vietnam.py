from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import cv2

from ai.detector import detect_waste
from config import DEFAULT_DETECTION_MODEL
from database.db import get_setting


def test_on_vietnam_images(
    image_dir: str | Path = "data/test_vietnam",
    output_dir: str | Path = "reports/vietnam_predictions",
    confidence: float = 0.25,
) -> None:
    """
    Chạy kiểm thử trên tập ảnh rác tự chụp thực tế tại Việt Nam (theo yêu cầu 40% ảnh thực tế).
    Tự động lưu ảnh có bounding box và lập bảng báo cáo kết quả.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]
    )

    if not image_files:
        print(f"[!] Thư mục {image_dir} chưa có ảnh nào.")
        print(
            "    👉 Hãy chép các ảnh rác tự chụp bằng điện thoại vào thư mục này để chạy kiểm thử!"
        )
        return

    print(
        f"[*] Tìm thấy {len(image_files)} ảnh tự chụp trong {image_dir}. Bắt đầu kiểm thử..."
    )

    detector_model = get_setting("detector_model", DEFAULT_DETECTION_MODEL)
    classifier_model = get_setting("classifier_model", "")

    results_table = []
    total_time = 0.0

    for img_path in image_files:
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        start = time.perf_counter()
        rgb_img, detections, rgb_annotated, classification = detect_waste(
            image_bytes=image_bytes,
            confidence_threshold=confidence,
            detector_path=detector_model,
            classifier_path=classifier_model or None,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        total_time += duration_ms

        # Lưu ảnh kết quả có bounding box
        result_file = output_dir / f"result_{img_path.name}"
        bgr_annotated = cv2.cvtColor(rgb_annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(result_file), bgr_annotated)

        det_names = [f"{d['class_name']} ({d['confidence']:.0%})" for d in detections]
        cats = list({d["category"] for d in detections})

        results_table.append(
            {
                "file": img_path.name,
                "objects": ", ".join(det_names) if det_names else "Không phát hiện",
                "categories": ", ".join(cats) if cats else "Chưa xác định",
                "count": len(detections),
                "speed_ms": round(duration_ms, 1),
            }
        )

    # Xuất báo cáo Markdown
    report_file = output_dir.parent / "Bao_cao_kiem_thu_anh_thuc_te.md"
    avg_speed = total_time / len(image_files) if image_files else 0

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO KIỂM THỬ TRÊN TẬP ẢNH TỰ CHỤP THỰC TẾ (VIỆT NAM)\n\n")
        f.write(f"- **Tổng số ảnh kiểm thử:** {len(image_files)}\n")
        f.write(f"- **Thời gian xử lý trung bình:** {avg_speed:.1f} ms / ảnh\n")
        f.write(f"- **Thư mục ảnh kết quả có khung:** `{output_dir.name}/`\n\n")
        f.write("## Bảng chi tiết kết quả nhận diện\n\n")
        f.write("| STT | Tên ảnh | Vật thể phát hiện | Nhóm phân loại | Số lượng | Thời gian |\n")
        f.write("|:---:|---|---|---|:---:|:---:|\n")
        for i, row in enumerate(results_table, 1):
            f.write(
                f"| {i} | `{row['file']}` | {row['objects']} | {row['categories']} | {row['count']} | {row['speed_ms']} ms |\n"
            )

    print(f"\n[✓] Hoàn thành kiểm thử {len(image_files)} ảnh!")
    print(f"[✓] Ảnh đã vẽ bounding box lưu tại: {output_dir}")
    print(f"[✓] Báo cáo chi tiết đã lưu tại: {report_file}")


if __name__ == "__main__":
    test_on_vietnam_images()
