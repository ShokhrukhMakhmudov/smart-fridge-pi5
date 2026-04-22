"""
Визуальный тест YOLO-модели — камера/видео → сырые боксы → окно.

В отличие от main.py здесь НЕТ трекера, линии портала, сервера и подавления
по руке. Это минимальный скрипт "прогнать модель и посмотреть, что она
детектит". Полезен для отладки: если тут модель работает, а в main.py
глючит — проблема в пайплайне, а не в модели.

Запуск:
    python test_camera.py                              # models/best.pt + камера
    python test_camera.py --model yolo11n.pt           # другая модель
    python test_camera.py --video test.mp4 --loop      # видеофайл в цикле
    python test_camera.py --conf 0.25                  # пониженный порог
    python test_camera.py --cam 1                      # другая веб-камера
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from camera import Camera
from config import MODEL_CONF, MODEL_PATH

FONT = cv2.FONT_HERSHEY_SIMPLEX
# Палитра цветов — разные классы рисуются разным цветом.
COLORS = [
    (0, 255, 0),    (0, 0, 255),    (255, 0, 0),    (255, 255, 0),
    (0, 255, 255),  (255, 0, 255),  (128, 255, 0),  (255, 128, 0),
]


def find_latest_model() -> str:
    """
    Найти самый свежий best.pt из runs/detect/. Если ничего нет —
    возвращаем путь из config.MODEL_PATH (обычно models/best.pt).
    """
    runs = Path("runs/detect")
    if runs.exists():
        candidates = sorted(
            runs.glob("*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return str(candidates[-1])
    return MODEL_PATH


def draw_detections(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: list,
    class_ids: list,
    class_names: dict,
) -> None:
    """Нарисовать bounding boxes + подписи поверх кадра."""
    for box, conf, cls_id in zip(boxes_xyxy, confs, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        color = COLORS[int(cls_id) % len(COLORS)]
        label = f"{class_names[int(cls_id)]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Фоновая плашка под текст — читабельнее на пёстром фоне
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), FONT, 0.6, (0, 0, 0), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Быстрый тест YOLO-модели на камере/видео")
    parser.add_argument("--model", default=None,
                        help="Путь к весам (.pt). По умолчанию: models/best.pt или свежий из runs/")
    parser.add_argument("--conf", type=float, default=MODEL_CONF,
                        help=f"Порог уверенности (по умолчанию {MODEL_CONF})")
    parser.add_argument("--video", default=None,
                        help="Путь к видеофайлу — тест на записи вместо камеры")
    parser.add_argument("--loop", action="store_true",
                        help="Зациклить видео (только с --video)")
    parser.add_argument("--cam", type=int, default=None,
                        help="Индекс веб-камеры (для OpenCV-бэкенда)")
    args = parser.parse_args()

    model_path: str = args.model or find_latest_model()
    print(f"[test] Модель: {model_path}")

    if not Path(model_path).exists() and "/" in model_path:
        # Для не-абсолютных имён типа "yolo11n.pt" Ultralytics сам скачает
        print(f"[test] Файл не найден локально — Ultralytics попробует скачать")

    model = YOLO(model_path)
    print(f"[test] Классы модели: {list(model.names.values())}")
    print(f"[test] Порог уверенности: {args.conf}")

    cam_kwargs: dict = {
        "video_path": args.video,
        "loop_video": args.loop,
    }
    if args.cam is not None:
        cam_kwargs["index"] = args.cam

    prev_time: float = time.time()
    fps: float = 0.0
    frame_num: int = 0

    with Camera(**cam_kwargs) as cam:
        source: str = args.video if args.video else f"камера ({cam.resolved_backend})"
        print(f"[test] Источник: {source}, {cam.width}x{cam.height}")
        print("[test] Нажмите 'q' в окне или Ctrl+C для выхода")

        while True:
            frame: Optional[np.ndarray] = cam.read()
            if frame is None:
                print("[test] Поток закончился")
                break
            frame_num += 1

            # 1. Инференс (verbose=False — чтобы Ultralytics не сыпал в лог
            # сообщениями про каждый кадр)
            results = model(frame, conf=args.conf, verbose=False)

            # 2. Рисуем боксы, если есть детекции
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                draw_detections(
                    frame,
                    boxes_xyxy=boxes.xyxy.cpu().numpy(),
                    confs=boxes.conf.cpu().tolist(),
                    class_ids=boxes.cls.int().cpu().tolist(),
                    class_names=results[0].names,
                )

            # 3. FPS (экспоненциальное сглаживание)
            now: float = time.time()
            dt: float = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # 4. Оверлей — FPS, модель, количество детекций
            n_dets: int = len(results[0].boxes) if results and results[0].boxes is not None else 0
            overlay_lines = [
                f"FPS: {fps:.1f}",
                f"Model: {Path(model_path).name}",
                f"Detections: {n_dets}",
                f"Conf: {args.conf:.2f}",
            ]
            for i, text in enumerate(overlay_lines):
                y: int = 25 + i * 22
                cv2.putText(frame, text, (10, y), FONT, 0.6, (0, 0, 0), 3)
                cv2.putText(frame, text, (10, y), FONT, 0.6, (255, 255, 255), 1)

            cv2.imshow("YOLO Test — q to quit", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[test] Остановлено пользователем")
