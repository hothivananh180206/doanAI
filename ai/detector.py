from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch

from config import DEFAULT_DETECTION_MODEL
from database.db import list_waste_types


def _best_device() -> str:
    return "cpu"


# Ánh xạ tên lớp mô hình sang 5 danh mục chuẩn theo đề tài CV-03:
# 1. Hữu cơ (organic)
# 2. Nhựa tái chế (plastic)
# 3. Giấy & Carton (paper)
# 4. Kim loại (metal)
# 5. Rác khác (other)
WASTE_LABEL_ALIASES = {
    # 1. Rác hữu cơ
    "organic": "organic",
    "biological": "organic",
    "food waste": "organic",
    "biodegradable": "organic",
    "vegetable": "organic",
    "fruit": "organic",
    "banana": "organic",

    # 2. Nhựa tái chế
    "plastic": "plastic",
    "plastic bottle": "plastic",
    "clear plastic bottle": "plastic",
    "other plastic bottle": "plastic",
    "plastic bottle cap": "plastic",
    "disposable plastic cup": "plastic",
    "other plastic": "plastic",
    "other plastic container": "plastic",
    "other plastic cup": "plastic",
    "other plastic wrapper": "plastic",
    "plastic film": "plastic",
    "plastic glooves": "plastic",
    "plastic lid": "plastic",
    "plastic straw": "plastic",
    "plastic utensils": "plastic",
    "polypropylene bag": "plastic",
    "single-use carrier bag": "plastic",
    "tupperware": "plastic",
    "plastic bag": "plastic",
    "crisp packet": "plastic",

    # 3. Giấy & Bìa carton
    "paper": "paper",
    "cardboard": "paper",
    "corrugated carton": "paper",
    "drink carton": "paper",
    "egg carton": "paper",
    "meal carton": "paper",
    "other carton": "paper",
    "pizza box": "paper",
    "book": "paper",
    "magazine": "paper",
    "newspaper": "paper",
    "magazine paper": "paper",
    "normal paper": "paper",
    "paper bag": "paper",
    "paper cup": "paper",
    "paper straw": "paper",
    "tissues": "paper",
    "toilet tube": "paper",
    "wrapping paper": "paper",

    # 4. Kim loại
    "metal": "metal",
    "can": "metal",
    "drink can": "metal",
    "food can": "metal",
    "aerosol": "metal",
    "aluminium blister pack": "metal",
    "aluminium foil": "metal",
    "metal bottle cap": "metal",
    "metal lid": "metal",
    "pop tab": "metal",
    "scrap metal": "metal",

    # 5. Rác khác (Còn lại) - Pin, quần áo, giày dép, xốp, v.v.
    "other": "other",
    "trash": "other",
    "battery": "other",
    "electronics": "other",
    "electronic": "other",
    "e-waste": "other",
    "clothes": "other",
    "cloth": "other",
    "shoes": "other",
    "shoe": "other",
    "textile": "other",
    "miscellaneous": "other",
    "misc": "other",
    "foam cup": "other",
    "foam food container": "other",
    "garbage bag": "other",
    "rope - strings": "other",
    "six pack rings": "other",
    "spread tub": "other",
    "squeezable tube": "other",
    "styrofoam piece": "other",

    # Thủy tinh → Tái chế (chai thủy tinh ở VN được thu gom tái chế)
    "glass": "plastic",        # dùng nhóm plastic vì DB không có "glass" riêng
    "brown-glass": "plastic",
    "green-glass": "plastic",
    "white-glass": "plastic",
    "broken glass": "other",   # thủy tinh vỡ → Còn lại (nguy hiểm)
    "glass bottle": "plastic",
    "glass cup": "plastic",
    "glass jar": "plastic",
    "disposable food container": "other",
}

# Tên tiếng Việt hiển thị trên giao diện chi tiết (chuẩn 5 lớp)
VI_LABEL_MAP = {
    "organic": "Rác hữu cơ",
    "biological": "Rác hữu cơ",
    "food waste": "Rác hữu cơ (Thức ăn thừa)",
    "banana": "Rác hữu cơ (Rau củ / Vỏ quả)",
    "vegetable": "Rác hữu cơ (Rau củ quả)",
    "fruit": "Rác hữu cơ (Trái cây)",
    "plastic": "Nhựa / Thủy tinh tái chế",
    "paper": "Giấy & Bìa carton",
    "cardboard": "Giấy & Bìa carton",
    "metal": "Kim loại & Vỏ lon",
    "other": "Rác khác",
    "trash": "Rác khác",
    # Thủy tinh nguyên → Tái chế
    "glass": "Thủy tinh (Tái chế)",
    "brown-glass": "Chai thủy tinh nâu (Tái chế)",
    "green-glass": "Chai thủy tinh xanh (Tái chế)",
    "white-glass": "Chai thủy tinh trắng (Tái chế)",
    "glass bottle": "Chai thủy tinh (Tái chế)",
    "glass cup": "Ly thủy tinh (Tái chế)",
    "glass jar": "Lọ thủy tinh (Tái chế)",
    "broken glass": "Thủy tinh vỡ (Nguy hiểm)",
    # Còn lại
    "battery": "Pin / Thiết bị điện tử",
    "clothes": "Quần áo / Vải",
    "shoes": "Giày dép",
}

# Tên không dấu hiển thị trên bounding box OpenCV
ASCII_LABEL_MAP = {
    "organic": "Huu co",
    "biological": "Huu co",
    "food waste": "Huu co (Thuc an thua)",
    "banana": "Huu co (Rau cu / Vo qua)",
    "vegetable": "Huu co (Rau cu)",
    "fruit": "Huu co (Trai cay)",
    "plastic": "Nhua tai che",
    "paper": "Giay / Carton",
    "cardboard": "Giay / Carton",
    "metal": "Kim loai",
    "other": "Rac khac",
    "trash": "Rac khac",
    "glass": "Thuy tinh (tai che)",
    "brown-glass": "TT nau (tai che)",
    "green-glass": "TT xanh (tai che)",
    "white-glass": "TT trang (tai che)",
    "glass bottle": "Chai thuy tinh",
    "glass cup": "Ly thuy tinh",
    "glass jar": "Lo thuy tinh",
    "broken glass": "TT vo (nguy hiem)",
    "battery": "Pin / Dien tu",
    "clothes": "Quan ao / Vai",
    "shoes": "Giay dep",
}

# Màu khung theo nhóm rác (định dạng BGR cho OpenCV)
CATEGORY_BOX_COLORS = {
    "Hữu cơ": (50, 168, 82),    # Xanh lá cây
    "Tái chế": (220, 120, 40),   # Xanh dương
    "Còn lại": (120, 120, 120),  # Xám
}


def _waste_type_for_label(
    label: str, waste_types: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    normalized_label = label.strip().casefold()
    if normalized_label in waste_types:
        return waste_types[normalized_label]

    alias = WASTE_LABEL_ALIASES.get(normalized_label)
    if alias and alias in waste_types:
        return waste_types[alias]

    # Tìm kiếm theo từ khóa nếu tên lớp là cụm từ
    for key, target in WASTE_LABEL_ALIASES.items():
        if key in normalized_label or normalized_label in key:
            if target in waste_types:
                return waste_types[target]

    for wt_name, wt_data in waste_types.items():
        if wt_name in normalized_label:
            return wt_data

    return None


@st.cache_resource(show_spinner="Đang tải mô hình YOLO...")
def _load_model(model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


def detect_waste(
    image_bytes: bytes,
    confidence_threshold: float,
    detector_path: str | None = None,
    classifier_path: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, dict[str, Any] | None]:
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("OpenCV không đọc được ảnh. Hãy chọn JPG hoặc PNG hợp lệ.")

    detector_name = detector_path or DEFAULT_DETECTION_MODEL
    detector = _load_model(detector_name)
    best_dev = _best_device()
    detected = detector.predict(
        source=frame, conf=confidence_threshold, device=best_dev, verbose=False
    )[0]
    classifier = _load_model(classifier_path) if classifier_path else None
    waste_types = {
        str(item["name"]).strip().casefold(): item for item in list_waste_types()
    }
    image_classification: dict[str, Any] | None = None
    has_detections = detected.boxes is not None and len(detected.boxes) > 0

    if (
        not has_detections
        and classifier is not None
        and getattr(classifier, "task", None) == "classify"
    ):
        classification = classifier.predict(source=frame, verbose=False)[0]
        if classification.probs is not None:
            class_id = int(classification.probs.top1)
            waste_label = str(classification.names[class_id])
            waste_type = _waste_type_for_label(waste_label, waste_types)
            image_classification = {
                "class_name": waste_label,
                "category": waste_type["category"] if waste_type else "Chưa ánh xạ",
                "instructions": waste_type["instructions"] if waste_type else "",
                "confidence": float(classification.probs.top1conf.item()),
            }

    annotated = frame.copy()
    detections: list[dict[str, Any]] = []
    if detected.boxes is not None:
        for box in detected.boxes:
            x1, y1, x2, y2 = (
                int(value) for value in box.xyxy[0].detach().cpu().tolist()
            )
            x1 = max(0, min(x1, frame.shape[1] - 1))
            x2 = max(x1 + 1, min(x2, frame.shape[1]))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            y2 = max(y1 + 1, min(y2, frame.shape[0]))
            object_name = detected.names[int(box.cls[0].item())]
            detector_confidence = float(box.conf[0].item())
            waste_label = object_name
            classifier_confidence: float | None = None
            waste_type = _waste_type_for_label(str(object_name), waste_types)
            crop = frame[y1:y2, x1:x2]

            # Khắc phục nhầm lẫn: khi mô hình đoán Cardboard/Paper/Trash nhưng thực tế là rác hữu cơ
            raw_name_lower = str(object_name).strip().casefold()
            if raw_name_lower in {"cardboard", "paper", "normal paper", "other carton", "trash", "other"} and crop.size:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
                banana_yellow = (h >= 16) & (h <= 36) & (s >= 80) & (v >= 80)
                veg_green = (h >= 36) & (h <= 85) & (s >= 75) & (v >= 70)
                orange_fruit = ((h < 16) | (h > 165)) & (s >= 90) & (v >= 90)
                cooked_food = ((h < 15) | (h > 165)) & (s >= 60) & (v >= 40) & (v <= 180)
                total_px = crop.shape[0] * crop.shape[1]

                # Ưu tiên kiểm tra thức ăn thừa / nấu chín / nước sốt trước
                if np.sum(cooked_food) / total_px > 0.08:
                    waste_label = "food waste"
                    waste_type = _waste_type_for_label("food waste", waste_types)
                elif np.sum(veg_green) / total_px > 0.15:
                    waste_label = "vegetable"
                    waste_type = _waste_type_for_label("vegetable", waste_types)
                elif np.sum(orange_fruit) / total_px > 0.18:
                    waste_label = "fruit"
                    waste_type = _waste_type_for_label("fruit", waste_types)
                elif np.sum(banana_yellow) / total_px > 0.16:
                    waste_label = "banana"
                    waste_type = _waste_type_for_label("banana", waste_types)

            if (
                waste_type is None
                and classifier is not None
                and getattr(classifier, "task", None) == "classify"
            ):
                if crop.size:
                    classification = classifier.predict(source=crop, verbose=False)[0]
                    if classification.probs is not None:
                        class_id = int(classification.probs.top1)
                        predicted_label = str(classification.names[class_id])
                        predicted_type = _waste_type_for_label(
                            predicted_label, waste_types
                        )
                        if predicted_type is not None:
                            waste_label = predicted_label
                            waste_type = predicted_type
                            classifier_confidence = float(
                                classification.probs.top1conf.item()
                            )

            category_name = waste_type["category"] if waste_type else "Chưa ánh xạ"
            box_color = CATEGORY_BOX_COLORS.get(category_name, (53, 134, 83))

            raw_key = str(waste_label).strip().casefold()
            ascii_text = ASCII_LABEL_MAP.get(raw_key, str(waste_label))
            vi_display_name = VI_LABEL_MAP.get(raw_key, str(waste_label).title())

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            label_text = f"{ascii_text} {detector_confidence:.0%}"
            text_y = max(y1 - 8, 20)
            cv2.putText(
                annotated,
                label_text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA,
            )
            detections.append(
                {
                    "class_name": vi_display_name,
                    "raw_class_name": str(waste_label),
                    "object_name": str(object_name),
                    "category": category_name,
                    "instructions": waste_type["instructions"] if waste_type else "",
                    "waste_type_id": waste_type["id"] if waste_type else None,
                    "confidence": detector_confidence,
                    "classifier_confidence": classifier_confidence,
                    "box": (float(x1), float(y1), float(x2), float(y2)),
                }
            )

    # Fallback phân tích toàn ảnh nếu YOLO không bắt được khung
    if not detections:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hf, sf, vf = hsv_frame[:, :, 0], hsv_frame[:, :, 1], hsv_frame[:, :, 2]
        tot = frame.shape[0] * frame.shape[1]
        c_food = ((hf < 15) | (hf > 165)) & (sf >= 60) & (vf >= 40) & (vf <= 180)
        v_green = (hf >= 36) & (hf <= 85) & (sf >= 75) & (vf >= 70)
        b_yellow = (hf >= 16) & (hf <= 36) & (sf >= 80) & (vf >= 80)

        fallback_label = None
        if np.sum(c_food) / tot > 0.08:
            fallback_label = "food waste"
        elif np.sum(v_green) / tot > 0.15:
            fallback_label = "vegetable"
        elif np.sum(b_yellow) / tot > 0.16:
            fallback_label = "banana"

        if fallback_label:
            w_type = _waste_type_for_label(fallback_label, waste_types)
            cat_name = w_type["category"] if w_type else "Hữu cơ"
            b_color = CATEGORY_BOX_COLORS.get(cat_name, (50, 168, 82))

            h_f, w_f = frame.shape[:2]
            fx1, fy1 = int(w_f * 0.05), int(h_f * 0.05)
            fx2, fy2 = int(w_f * 0.95), int(h_f * 0.95)

            ascii_text = ASCII_LABEL_MAP.get(fallback_label, fallback_label)
            vi_name = VI_LABEL_MAP.get(fallback_label, fallback_label)

            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), b_color, 2)
            cv2.putText(
                annotated,
                f"{ascii_text} 75%",
                (fx1, max(fy1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                b_color,
                2,
                cv2.LINE_AA,
            )
            detections.append(
                {
                    "class_name": vi_name,
                    "raw_class_name": fallback_label,
                    "object_name": fallback_label,
                    "category": cat_name,
                    "instructions": w_type["instructions"] if w_type else "",
                    "waste_type_id": w_type["id"] if w_type else 6,
                    "confidence": 0.75,
                    "classifier_confidence": None,
                    "box": (float(fx1), float(fy1), float(fx2), float(fy2)),
                }
            )

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return rgb_image, detections, rgb_annotated, image_classification


def save_history_images(
    image_bytes: bytes, result_rgb: np.ndarray, file_id: str
) -> tuple[Path, Path]:
    from config import HISTORY_DIR

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    image_path = HISTORY_DIR / f"{file_id}_original.jpg"
    result_path = HISTORY_DIR / f"{file_id}_result.jpg"
    original = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original is None or not cv2.imwrite(str(image_path), original):
        raise OSError(f"Không ghi được ảnh gốc: {image_path}")
    if not cv2.imwrite(
        str(result_path), cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    ):
        image_path.unlink(missing_ok=True)
        raise OSError(f"Không ghi được ảnh kết quả: {result_path}")
    return image_path, result_path


def pil_image_to_bytes(image: Image.Image) -> bytes:
    import io

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92)
    return output.getvalue()
