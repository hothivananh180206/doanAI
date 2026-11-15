from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image

from ai.detector import detect_waste
from config import DEFAULT_DETECTION_MODEL
from database.db import get_setting, list_waste_types

app = FastAPI(
    title="EcoSort AI - Waste Detection & Classification API",
    description="API phân loại rác thải và hướng dẫn xử lý tại nguồn theo đề tài CV-03",
    version="1.0.0",
)

# Cho phép gọi API từ frontend web, mobile app hoặc bên ngoài
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Kiểm tra trạng thái API")
def index() -> dict[str, Any]:
    return {
        "status": "online",
        "service": "EcoSort AI API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "description": "API nhận diện vật thể rác qua ảnh và trả về nhóm phân loại + hướng dẫn xử lý",
    }


@app.get("/categories", summary="Danh mục rác và hướng dẫn xử lý")
def get_categories() -> dict[str, Any]:
    """Trả về toàn bộ danh mục rác và hướng dẫn phân loại hiện tại trong hệ thống."""
    types = list_waste_types()
    return {
        "total": len(types),
        "categories": types,
    }


@app.post("/predict", summary="Nhận diện rác từ file ảnh")
async def predict_waste(
    file: UploadFile = File(..., description="File ảnh rác (JPG, PNG)"),
    confidence: float = Query(
        0.25,
        ge=0.05,
        le=1.0,
        description="Ngưỡng độ tin cậy của model YOLO (mặc định: 0.25)",
    ),
) -> dict[str, Any]:
    """
    Nhận file ảnh tải lên, phát hiện các vật thể rác trong ảnh bằng YOLO,
    ánh xạ sang nhóm rác (Hữu cơ, Tái chế, Nguy hại, Còn lại) và trả về hướng dẫn xử lý.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File tải lên phải là định dạng hình ảnh (image/jpeg, image/png...)",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File ảnh rỗng.")

        detector_model = get_setting("detector_model", DEFAULT_DETECTION_MODEL)
        classifier_model = get_setting("classifier_model", "")

        _, detections, _, classification = detect_waste(
            image_bytes=contents,
            confidence_threshold=confidence,
            detector_path=detector_model,
            classifier_path=classifier_model or None,
        )

        categories_found = sorted(
            list(
                {
                    d["category"]
                    for d in detections
                    if d["category"] != "Chưa ánh xạ"
                }
            )
        )

        formatted_detections = []
        for d in detections:
            box = d.get("box", (0, 0, 0, 0))
            formatted_detections.append(
                {
                    "class_name": d.get("class_name"),
                    "raw_class_name": d.get("raw_class_name"),
                    "category": d.get("category"),
                    "confidence": round(float(d.get("confidence", 0)), 4),
                    "bounding_box": {
                        "x1": round(box[0], 1),
                        "y1": round(box[1], 1),
                        "x2": round(box[2], 1),
                        "y2": round(box[3], 1),
                    },
                    "instructions": d.get("instructions", ""),
                }
            )

        return {
            "success": True,
            "filename": file.filename,
            "total_detected": len(formatted_detections),
            "categories": categories_found,
            "detections": formatted_detections,
            "image_classification": classification,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Lỗi khi xử lý nhận diện ảnh: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn

    print("Khởi động EcoSort API tại http://127.0.0.1:8000")
    print("Xem tài liệu API Swagger tại http://127.0.0.1:8000/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
