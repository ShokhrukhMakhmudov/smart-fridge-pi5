"""
Обучение YOLOv11n на кастомном датасете.

Скрипт:
  1. Ищет data.yaml в каталоге dataset/ (можно переопределить --data)
  2. Запускает ultralytics YOLO('yolo11n.pt').train(...)
  3. По завершении копирует best.pt в models/best.pt для удобства

Запуск:
    python train/train.py
    python train/train.py --epochs 200 --batch 32 --imgsz 640
    python train/train.py --data /path/to/data.yaml --device 0   # GPU 0

Заметки:
  - На Pi 5 обучать НЕ стоит — это работа для x86/CUDA-машины.
  - Используйте requirements-dev.txt на dev-машине, там есть torch+CUDA.
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # чтобы видеть config.py

from config import DATASET_DIR, MODELS_DIR  # noqa: E402


def find_data_yaml(explicit: str | None) -> Path:
    """Найти data.yaml — явно указанный или автоматически в dataset/."""
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"data.yaml не найден: {path}")
        return path

    # Поиск рекурсивно — Roboflow часто кладёт в dataset/<project>/data.yaml
    candidates: list[Path] = list(DATASET_DIR.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"data.yaml не найден в {DATASET_DIR}. "
            "Скачайте датасет с Roboflow и распакуйте в dataset/."
        )
    if len(candidates) > 1:
        print(f"[train] Найдено несколько data.yaml, использую первый: {candidates[0]}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение YOLOv11n")
    parser.add_argument("--data", default=None,
                        help="Путь к data.yaml (по умолчанию ищется в dataset/)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu",
                        help="cpu / 0 / 0,1 (GPU id)")
    parser.add_argument("--base", default="yolo11n.pt",
                        help="Базовая модель ultralytics (yolo11n.pt по умолчанию)")
    parser.add_argument("--name", default="smart_fridge",
                        help="Имя запуска (папка в runs/detect/)")
    args = parser.parse_args()

    # Импорт ultralytics — тяжёлый, поэтому внутри main()
    from ultralytics import YOLO  # noqa: PLC0415

    data_yaml: Path = find_data_yaml(args.data)
    print(f"[train] data.yaml: {data_yaml}")
    print(f"[train] Базовая модель: {args.base}")
    print(f"[train] Эпох: {args.epochs}, batch: {args.batch}, "
          f"imgsz: {args.imgsz}, device: {args.device}")

    model = YOLO(args.base)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        # Базовые аугментации — подходят для большинства задач
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.0,
        # Сохраняем чекпоинты периодически
        save_period=10,
        patience=30,  # ранняя остановка, если 30 эпох нет прогресса
    )

    # ── Копируем лучший чекпоинт в models/ для удобного запуска ──
    save_dir: Path = Path(results.save_dir)
    best_pt: Path = save_dir / "weights" / "best.pt"
    if best_pt.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        target: Path = MODELS_DIR / "best.pt"
        shutil.copy2(best_pt, target)
        print(f"[train] Лучшая модель скопирована: {target}")
    else:
        print(f"[train] Внимание: best.pt не найден в {save_dir}/weights/")


if __name__ == "__main__":
    main()
