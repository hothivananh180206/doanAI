from __future__ import annotations

import sqlite3
from typing import Any

import streamlit as st

from config import BASE_DIR, DEFAULT_DETECTION_MODEL
from database.db import get_setting, set_setting


def render() -> None:
    st.caption("CẤU HÌNH HỆ THỐNG · MÔ HÌNH NHẬN DIỆN")
    st.title("⚙️ Cài đặt Mô hình AI")
    st.write(
        "Tùy chỉnh mô hình YOLO nhận diện rác thải, thay đổi đường dẫn trọng số (weights) "
        "hoặc huấn luyện mô hình mới."
    )

    with st.container(border=True):
        st.subheader("🎯 Cấu hình Mô hình đang sử dụng")
        
        current_detector = get_setting("detection_model", DEFAULT_DETECTION_MODEL)
        detector_model = st.text_input(
            "Đường dẫn Model phát hiện vật thể (YOLO Detection)",
            value=current_detector,
            help="Đường dẫn đến file trọng số .pt đã huấn luyện (ví dụ: ai/model/best.pt)",
        )

        classifier_model = st.text_input(
            "Model phân loại phụ (không bắt buộc)",
            value=get_setting("classification_model", ""),
            help="Dùng thêm mô hình classify nếu muốn phân loại sâu chất liệu vật thể.",
        )

        candidates = sorted((BASE_DIR / "runs" / "classify").glob("**/weights/best.pt"))
        if candidates:
            selected_path = st.selectbox(
                "Hoặc chọn model có sẵn trong thư mục runs:",
                options=[""] + [str(path) for path in candidates],
            )
            if selected_path:
                if st.button("Dùng model này", type="secondary"):
                    classifier_model = selected_path

        save_btn = st.button("💾 Lưu cấu hình Model", type="primary")
        if save_btn:
            set_setting("detection_model", detector_model.strip())
            set_setting("classification_model", classifier_model.strip())
            st.success("✅ Đã cập nhật cấu hình mô hình thành công!")
            st.rerun()

    with st.container(border=True):
        st.subheader("🏋️ Huấn luyện lại Model YOLO (Transfer Learning)")
        st.caption(
            "Tự động tận dụng GPU (NVIDIA CUDA) nếu máy có sẵn, hoặc chạy trên CPU. "
            "Sử dụng tập dữ liệu có khai báo data.yaml chuẩn định dạng YOLO."
        )
        
        detect_data_yaml = st.text_input(
            "Đường dẫn file data.yaml",
            placeholder="D:/Datasets/trash-detect/data.yaml",
            help="File YAML cấu hình tập train/val và danh sách tên lớp rác.",
        )
        detect_epochs = st.number_input(
            "Số Epochs huấn luyện",
            min_value=1,
            max_value=300,
            value=50,
            step=5,
        )
        
        if st.button("🚀 Bắt đầu Huấn luyện Model", type="secondary"):
            if not detect_data_yaml.strip():
                st.warning("Vui lòng nhập đường dẫn đến file data.yaml!")
            else:
                from ai.trainer import train_detector
                try:
                    with st.spinner("Đang huấn luyện mô hình YOLO... Vui lòng chờ..."):
                        best_model = train_detector(
                            detect_data_yaml.strip(),
                            int(detect_epochs),
                            "waste_detect_run",
                        )
                    set_setting("detection_model", str(best_model.resolve()))
                    st.success(f"🎉 Huấn luyện hoàn tất! Đã tự động kích hoạt model mới: {best_model}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Lỗi khi huấn luyện: {exc}")

    st.info(
        "💡 **Ghi chú bảo vệ đồ án:** Hệ thống hỗ trợ chuyển đổi linh hoạt giữa các checkpoint `.pt` "
        "huấn luyện trên Colab hoặc máy tính cục bộ mà không cần phải can thiệp sửa code."
    )
