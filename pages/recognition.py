from collections import Counter
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from ai.detector import detect_waste, save_history_images
from config import BASE_DIR, DEFAULT_DETECTION_MODEL, IMAGE_SUFFIXES
from database.db import (
    create_history,
    get_setting,
    save_feedback,
    set_setting,
)


def _group_detections_by_category(
    detections: list[dict[str, Any]],
) -> dict[str, Counter[str]]:
    grouped: dict[str, Counter[str]] = {}
    for item in detections:
        category = str(item.get("category") or "").strip() or "Chưa ánh xạ"
        grouped.setdefault(category, Counter())[str(item["class_name"])] += 1
    return grouped


def render() -> None:
    user_id = 1
    st.caption("PHÂN TÍCH ẢNH · NHẬN DIỆN VẬT THỂ · GỢI Ý PHÂN LOẠI")
    st.title("Nhận diện rác")
    st.write("Tải ảnh lên hoặc chụp ảnh bằng camera để bắt đầu phân tích.")

    detector_model = get_setting("detection_model", DEFAULT_DETECTION_MODEL)
    classifier_model = get_setting("classification_model", "")
    detection_weights = detector_model
    classifier_weights = classifier_model

    with st.container(border=True):
        source = st.radio(
            "Chọn cách thêm ảnh",
            ["📁 Tải ảnh lên", "📷 Camera trực tiếp"],
            horizontal=True,
            label_visibility="visible",
        )
        if source == "📁 Tải ảnh lên":
            uploaded = st.file_uploader(
                "Chọn ảnh rác từ thiết bị",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="waste_upload",
                help="Hỗ trợ JPG, JPEG, PNG, BMP và WEBP.",
            )
        else:
            uploaded = st.camera_input(
                "Chụp một ảnh từ webcam",
                key="waste_camera",
            )
            st.caption("Camera chụp một ảnh mỗi lần; không phát video liên tục.")
        if uploaded is not None:
            selected_name = escape(
                str(uploaded.name) if hasattr(uploaded, "name") else "Ảnh camera"
            )
            st.markdown(
                f'<span class="eco-pill">✓ Đã chọn: {selected_name}</span>',
                unsafe_allow_html=True,
            )

        confidence_col, action_col = st.columns([2, 1], vertical_alignment="bottom")
        with confidence_col:
            threshold = st.slider(
                "Ngưỡng tin cậy",
                min_value=0.1,
                max_value=0.9,
                value=0.25,
                step=0.05,
                help="Tăng ngưỡng để chỉ hiển thị những dự đoán YOLO tự tin hơn.",
            )
        with action_col:
            run_detection = st.button(
                "🔍  Nhận diện ngay",
                type="primary",
                disabled=uploaded is None,
                use_container_width=True,
            )

    if not run_detection or uploaded is None:
        return

    image_bytes = uploaded.getvalue()
    source_name = "camera" if source == "📷 Camera trực tiếp" else "upload"
    try:
        with st.spinner("Đang chạy YOLO trên ảnh..."):
            (
                original_rgb,
                detections,
                annotated_rgb,
                image_classification,
            ) = detect_waste(
                image_bytes=image_bytes,
                confidence_threshold=threshold,
                detector_path=detection_weights.strip() or None,
                classifier_path=classifier_weights.strip() or None,
            )
    except Exception as exc:
        st.error(f"Không nhận diện được ảnh: {exc}")
        return

    file_id = uuid4().hex
    try:
        original_path, result_path = save_history_images(
            image_bytes, annotated_rgb, file_id
        )
        history_id = create_history(
            user_id=user_id,
            image_path=original_path,
            result_image_path=result_path,
            model_path=detection_weights.strip() or DEFAULT_DETECTION_MODEL,
            source=source_name,
            detections=detections,
            image_classification=image_classification,
        )
    except Exception as exc:
        for path in (locals().get("original_path"), locals().get("result_path")):
            if isinstance(path, Path):
                path.unlink(missing_ok=True)
        st.error(f"Không lưu được lịch sử nhận diện: {exc}")
        return

    st.divider()
    st.subheader("✅ Kết quả nhận diện")
    st.caption(f"Lượt nhận diện #{history_id} · {len(detections)} vật thể")
    unmapped_detections = [
        item
        for item in detections
        if item.get("waste_type_id") is None
        or str(item["category"]).strip().casefold() == "chưa ánh xạ"
    ]
    detections_by_category = _group_detections_by_category(detections)
    mapped_categories = {
        category
        for category in detections_by_category
        if category.casefold() != "chưa ánh xạ"
    }
    if (
        image_classification is not None
        and str(image_classification["category"]).strip().casefold()
        != "chưa ánh xạ"
    ):
        mapped_categories.add(str(image_classification["category"]))

    image_col, summary_col = st.columns([1.2, 0.8], gap="medium")
    with image_col:
        st.image(
            annotated_rgb,
            caption="YOLO đã vẽ khung quanh vật thể",
            use_container_width=True,
        )
        with st.expander("Xem ảnh gốc"):
            st.image(original_rgb, use_container_width=True)
    with summary_col:
        with st.container(border=True):
            st.markdown("#### Danh sách vật thể")
            if detections_by_category:
                for category, items in sorted(detections_by_category.items()):
                    st.markdown(f"**{category} · {sum(items.values())} vật thể**")
                    for label, count in items.most_common():
                        st.write(f"• {label}: {count}")
            else:
                st.info("Không phát hiện vật thể nên chưa có số lượng để đếm.")
        with st.container(border=True):
            st.markdown("#### Nhóm rác")
            BIN_TIPS = {
                "Hữu cơ": ("🟢 Rác hữu cơ", "Cho vào thùng màu xanh lá (Thức ăn thừa, rau củ quả, lá cây...)"),
                "Tái chế": ("🔵 Rác tái chế", "Cho vào thùng màu xanh dương (Nhựa, kim loại, giấy, thủy tinh, carton...)"),
                "Còn lại": ("⚪ Rác còn lại", "Cho vào thùng màu xám (Quần áo cũ, giày dép, rác không tái chế...)"),
                "Nguy hại": ("⚠️ Rác nguy hại", "Bỏ vào điểm thu gom riêng (Pin, ắc quy, thiết bị điện tử...)"),
            }
            if mapped_categories:
                for category in sorted(mapped_categories):
                    tip = BIN_TIPS.get(category)
                    if tip:
                        title, bin_desc = tip
                        st.success(f"**{title}**\n\n✓ {bin_desc}")
                    else:
                        st.success(f"**{category}** ♻️")
            elif not detections and image_classification is None:
                st.warning(
                    "Chưa phân nhóm được: model không phát hiện vật thể và chưa "
                    "có model phân loại ảnh được cấu hình."
                )
            elif detections:
                st.warning("Vật thể nhận diện được nhưng chưa khớp danh mục rác.")
            if unmapped_detections or (
                image_classification is not None
                and str(image_classification["category"]).strip().casefold()
                == "chưa ánh xạ"
            ):
                st.warning("Một số kết quả chưa ghép được với danh mục rác.")

    if image_classification is not None:
        st.subheader("Kết quả phân loại ảnh · 1 đáp án")
        result_col, category_col, confidence_col = st.columns(3)
        with result_col:
            st.metric("Loại rác dự đoán", image_classification["class_name"].title())
        with category_col:
            st.metric("Nhóm", image_classification["category"])
        with confidence_col:
            st.metric("Độ tin cậy model", f"{image_classification['confidence']:.1%}")
        if image_classification["instructions"]:
            st.info(image_classification["instructions"])
        st.warning(
            "Độ tin cậy là điểm số của model, không phải phần trăm đảm bảo đúng. "
            "Hãy xác minh kết quả, nhất là khi ảnh có nhiều loại rác."
        )

    if detections:
        st.success(f"Đã hoàn tất phân tích {len(detections)} vật thể.")
    elif image_classification is not None:
        st.success("Đã phân loại toàn ảnh; model không tạo được khung vật thể.")
    else:
        st.warning(
            "Ảnh chưa có kết quả nhận diện. Hãy thử ảnh cận cảnh, rõ vật thể hơn "
            "hoặc cấu hình model nhận diện rác phù hợp."
        )

    if unmapped_detections:
        if classifier_model:
            st.warning(
                f"{len(unmapped_detections)} vật thể chưa khớp danh mục. "
                "Hãy kiểm tra tên lớp trong model phân loại có trùng tên loại rác "
                "đang khai báo trong danh mục hay không."
            )
        else:
            st.warning(
                f"{len(unmapped_detections)} vật thể chưa xác định được danh mục. "
                "YOLO mặc định chỉ nhận diện vật thể phổ thông; hãy cấu hình model "
                "phân loại rác trong trang Quản trị để phân loại vật thể chưa khớp."
            )

    if not detections and image_classification is not None:
        if (
            str(image_classification["category"]).strip().casefold()
            == "chưa ánh xạ"
        ):
            st.warning(
                "Tên lớp dự đoán của model chưa khớp tên loại rác nào trong danh mục."
            )
    elif detections:
        st.subheader("Vật thể có khung nhận diện")
        st.dataframe(
            [
                {
                    "Nhãn nhận diện": item["class_name"],
                    "Nhóm": item["category"],
                    "Độ tin cậy YOLO": f"{item['confidence']:.1%}",
                    "Độ tin cậy phân loại": (
                        f"{item['classifier_confidence']:.1%}"
                        if item["classifier_confidence"] is not None
                        else "Chưa dùng classifier"
                    ),
                    "Hướng dẫn": item["instructions"],
                }
                for item in detections
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Gửi góp ý về kết quả nhận diện"):
        with st.form(f"feedback_{history_id}"):
            feedback_text = st.text_area("Góp ý của bạn")
            sent = st.form_submit_button("Gửi góp ý", type="primary")
        if sent:
            if not feedback_text.strip():
                st.error("Hãy nhập nội dung góp ý.")
            else:
                save_feedback(user_id, history_id, feedback_text)
                st.success("Cảm ơn bạn đã gửi góp ý!")
