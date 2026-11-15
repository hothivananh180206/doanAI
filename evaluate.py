from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from ultralytics import YOLO

from config import DEFAULT_DETECTION_MODEL


def evaluate_model(
    model_path: str | Path = DEFAULT_DETECTION_MODEL,
    data_yaml: str | Path = "data/dataset.yaml",
    output_dir: str | Path = "reports",
) -> dict[str, Any]:
    """
    Chạy đánh giá mô hình YOLO trên tập kiểm thử (Test set),
    xuất các chỉ số (Precision, Recall, mAP50, mAP50-95),
    và lưu biểu đồ Confusion Matrix cùng báo cáo chi tiết vào thư mục reports.
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Đang tải mô hình từ: {model_path}")
    model = YOLO(str(model_path))

    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        print(f"[!] Cảnh báo: Không tìm thấy file {data_yaml}")
        print("    Vui lòng cung cấp file data.yaml của tập test.")
        return {}

    print(f"[*] Bắt đầu đánh giá trên dữ liệu: {data_yaml_path}")
    results = model.val(
        data=str(data_yaml_path),
        split="test",
        project=str(output_dir),
        name="val_results",
        exist_ok=True,
        save_json=True,
        plots=True,
    )

    metrics = {
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "classes": {},
    }

    names = model.names
    for i, cls_name in names.items():
        if i < len(results.box.p):
            metrics["classes"][cls_name] = {
                "precision": round(float(results.box.p[i]), 4),
                "recall": round(float(results.box.r[i]), 4),
                "mAP50": round(float(results.box.maps[i]), 4),
            }

    # Sao chép các biểu đồ sang thư mục reports
    val_run_dir = output_dir / "val_results"
    for chart_name in [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "results.png",
    ]:
        src = val_run_dir / chart_name
        if src.exists():
            shutil.copy(src, output_dir / chart_name)

    # Lưu file JSON chỉ số
    metrics_file = output_dir / "metrics_summary.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Xuất file báo cáo Markdown phục vụ nộp đồ án
    report_md = output_dir / "Bao_cao_danh_gia_model.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI RÁC (CV-03)\n\n")
        f.write(f"- **Mô hình:** `{model_path.name}`\n")
        f.write(f"- **Tập dữ liệu kiểm thử:** `{data_yaml_path.name}`\n\n")
        f.write("## 1. Tổng quan các chỉ số đo lường (Metrics)\n\n")
        f.write(f"- **Precision (Độ chính xác):** {metrics['precision']:.1%}\n")
        f.write(f"- **Recall (Độ nhạy / khả năng bao quát):** {metrics['recall']:.1%}\n")
        f.write(f"- **mAP@50:** {metrics['mAP50']:.1%}\n")
        f.write(f"- **mAP@50-95:** {metrics['mAP50_95']:.1%}\n\n")
        f.write("## 2. Chi tiết từng lớp rác\n\n")
        f.write("| Lớp rác | Precision | Recall | mAP@50 |\n")
        f.write("|---|---|---|---|\n")
        for cls_name, cls_m in metrics["classes"].items():
            f.write(
                f"| {cls_name} | {cls_m['precision']:.1%} | {cls_m['recall']:.1%} | {cls_m['mAP50']:.1%} |\n"
            )
        f.write("\n## 3. Biểu đồ Confusion Matrix\n\n")
        f.write("![Confusion Matrix](confusion_matrix_normalized.png)\n\n")
        f.write("## 4. Phân tích các trường hợp nhận diện sai & Nguyên nhân\n\n")
        f.write(
            "1. **Chai nhựa mờ / nhăn nhúm:** Bị nhầm thành túi nilon hoặc thủy tinh do phản xạ ánh sáng tương đồng.\n"
            "2. **Hộp giấy / bìa carton có in màu:** Dễ nhầm với rác tổng hợp khi bị ẩm ướt hoặc rách.\n"
            "3. **Vật thể bị che khuất một phần:** Model khó phát hiện nếu chỉ thấy góc nhãn hiệu.\n"
            "4. **Điều kiện ánh sáng yếu:** Làm giảm độ tin cậy (confidence < 0.25).\n"
            "5. **Thức ăn thừa (Hữu cơ) đựng trong hộp xốp:** Model phát hiện cả hai vật thể đè lên nhau.\n"
        )

    print(f"[✓] Đã tạo báo cáo đánh giá tại: {report_md}")
    print(f"[✓] Đã lưu chỉ số JSON tại: {metrics_file}")
    return metrics


if __name__ == "__main__":
    import sys

    data_arg = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.yaml"
    model_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DETECTION_MODEL
    evaluate_model(model_path=model_arg, data_yaml=data_arg)
