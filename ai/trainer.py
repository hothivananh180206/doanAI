from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from config import BASE_DIR


MIN_TRAIN_IMAGES_PER_CLASS = 50
MIN_VAL_IMAGES_PER_CLASS = 10


def _image_count(image_dir: Path) -> int:
    if not image_dir.is_dir():
        return 0
    return sum(
        path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in image_dir.iterdir()
    )


def _best_device() -> str | int:
    """Trả về device tốt nhất: GPU nếu có, ngược lại CPU với số thread tối ưu."""
    if torch.cuda.is_available():
        return 0  # GPU đầu tiên
    # Tối ưu số thread CPU (giới hạn 8 để tránh overhead scheduling)
    torch.set_num_threads(min(os.cpu_count() or 4, 8))
    return "cpu"


def _is_gpu() -> bool:
    return torch.cuda.is_available()


def _gpu_vram_mb() -> int:
    """Trả về VRAM (MB) của GPU đầu tiên, 0 nếu không có GPU."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)


def train_classifier(epochs: int, run_name: str) -> Path:
    data_dir = BASE_DIR / "dataset"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("Dataset phải có thư mục train và val.")

    classes = [path for path in train_dir.iterdir() if path.is_dir()]
    val_classes = {path.name for path in val_dir.iterdir() if path.is_dir()}
    if not classes or {path.name for path in classes} != val_classes:
        raise ValueError("Các thư mục lớp trong train và val phải giống nhau.")
    for folder in classes:
        train_count = _image_count(train_dir / folder.name)
        val_count = _image_count(val_dir / folder.name)
        if train_count < MIN_TRAIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Lớp '{folder.name}' chỉ có {train_count} ảnh train; "
                f"cần ít nhất {MIN_TRAIN_IMAGES_PER_CLASS} ảnh."
            )
        if val_count < MIN_VAL_IMAGES_PER_CLASS:
            raise ValueError(
                f"Lớp '{folder.name}' chỉ có {val_count} ảnh validation; "
                f"cần ít nhất {MIN_VAL_IMAGES_PER_CLASS} ảnh."
            )

    from ultralytics import YOLO

    device = _best_device()
    gpu = _is_gpu()
    vram = _gpu_vram_mb()

    # MX550 chỉ có 2GB VRAM → dùng batch nhỏ; GPU mạnh hơn dùng batch lớn hơn
    if gpu:
        batch = 8 if vram >= 4096 else 4   # ≥4GB: batch=8, <4GB (MX550): batch=4
    else:
        batch = 8

    model: Any = YOLO("yolo11n-cls.pt")
    model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=224,
        batch=batch,
        workers=2,
        cache=False,
        project=str(BASE_DIR / "runs" / "classify"),
        name=run_name,
        device=device,
        amp=gpu,       # AMP chỉ bật trên GPU → tăng tốc thêm ~30% trên CUDA
        patience=20,   # Early stopping
        optimizer="SGD",
    )
    best_model = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"Không tìm thấy model sau khi huấn luyện: {best_model}")
    return best_model


def train_detector(data_yaml: str | Path, epochs: int, run_name: str) -> Path:
    dataset_config = Path(data_yaml).expanduser()
    if not dataset_config.is_absolute():
        dataset_config = BASE_DIR / dataset_config
    if not dataset_config.is_file() or dataset_config.suffix.lower() not in {
        ".yaml",
        ".yml",
    }:
        raise FileNotFoundError(
            f"Không tìm thấy tệp cấu hình dataset YOLO (.yaml/.yml): {dataset_config}"
        )

    from ultralytics import YOLO

    device = _best_device()
    gpu = _is_gpu()
    vram = _gpu_vram_mb()

    # MX550 (2GB VRAM) + imgsz=224: batch=8 vừa đủ; tăng tốc ~2x so với imgsz=320
    if gpu:
        batch = 16 if vram >= 4096 else 8  # ≥4GB: batch=16, <4GB (MX550): batch=8
    else:
        batch = 4

    model: Any = YOLO("yolo11n.pt")
    model.train(
        data=str(dataset_config.resolve()),
        epochs=epochs,
        imgsz=224,              # Giảm từ 320→224: nhanh hơn ~2x, tiết kiệm VRAM
        batch=batch,
        workers=2,              # Prefetch data song song → GPU không phải chờ CPU
        cache=False,            # Tắt cache khi RAM hạn chế
        project=str(BASE_DIR / "runs" / "detect"),
        name=run_name,
        device=device,
        amp=gpu,                # AMP chỉ bật trên GPU → ~30% nhanh hơn
        patience=20,            # Early stopping: dừng sớm nếu không cải thiện
        optimizer="SGD",        # SGD nhanh hơn AdamW trên CPU
    )
    best_model = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"Không tìm thấy model sau khi huấn luyện: {best_model}")
    return best_model
