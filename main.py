"""
Smart Fridge OS — точка входа.

Связывает все компоненты: камера → детектор → трекер → пересечение линии
→ замок + WebSocket-уведомления. Параллельно поднимает FastAPI сервер
в фоновом потоке, чтобы внешние клиенты могли управлять замком и
подписываться на события.

Запуск:
    python main.py                             # CV + сервер с настройками из config.py
    python main.py --no-server                 # только CV-цикл, без HTTP
    python main.py --no-cv                     # только сервер (для отладки API)
    python main.py --mode cpu                  # форсировать конкретный бэкенд
    python main.py --show                      # показывать окно OpenCV
    python main.py --video test.mp4 --show     # прогнать модель на видеофайле
    python main.py --video test.mp4 --loop     # + зациклить видео
"""

import argparse
import asyncio
import threading
import time
from typing import Optional

import cv2
import numpy as np
import uvicorn

from camera import Camera
from config import (
    CROSSING_LINE_Y,
    HAND_SUPPRESSION,
    INFERENCE_MODE,
    MODEL_CONF,
    MODEL_HEF_PATH,
    MODEL_IMGSZ,
    MODEL_PATH,
    PERF_LOG_EVERY,
    SERVER_HOST,
    SERVER_PORT,
    SHOW_WINDOW,
)
from detection import (
    ByteTracker,
    Detector,
    LineCrossingDetector,
)
from detection.crossing import CrossingEvent
from hardware import Lock
from server import WebSocketHub, build_app


# ── Визуализация (используется только при --show) ────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_COLOR = (0, 255, 255)
TAKEN_COLOR = (0, 0, 255)
RETURNED_COLOR = (0, 255, 0)
BOX_COLOR = (255, 180, 0)


def _draw_overlay(
    frame: np.ndarray,
    tracks,
    line_y: int,
    fps: float,
) -> None:
    """Нарисовать линию портала, боксы треков и FPS."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), LINE_COLOR, 2)
    cv2.putText(frame, "PORTAL", (10, line_y - 8), FONT, 0.5, LINE_COLOR, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), FONT, 0.6, (255, 255, 255), 2)

    for tracked in tracks:
        det = tracked.detection
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), BOX_COLOR, 2)
        label: str = f"#{tracked.track_id} {det.class_name} {det.confidence:.2f}"
        cv2.putText(frame, label, (det.x1, max(15, det.y1 - 5)),
                    FONT, 0.5, BOX_COLOR, 1)


# ── Сервер в фоновом потоке ──────────────────────────────────────

class _ServerThread(threading.Thread):
    """Поток с uvicorn-инстансом, чтобы CV-цикл оставался в main thread."""

    def __init__(self, app, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self.config = uvicorn.Config(
            app=app, host=host, port=port, log_level="info",
        )
        self.server = uvicorn.Server(self.config)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready: threading.Event = threading.Event()

    def run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # Сообщаем основному потоку, что loop готов и можно слать события
        self._ready.set()
        self.loop.run_until_complete(self.server.serve())

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout=timeout)

    def stop(self) -> None:
        self.server.should_exit = True


# ── CV-цикл ──────────────────────────────────────────────────────

def run_cv_loop(
    detector: Detector,
    tracker: ByteTracker,
    crossing: LineCrossingDetector,
    show_window: bool,
    video_path: Optional[str] = None,
    loop_video: bool = False,
) -> None:
    """Главный цикл: камера/видео → детектор → трекер → пересечения."""
    with Camera(video_path=video_path, loop_video=loop_video) as cam:
        frame_h: int = cam.height
        line_y: int = int(frame_h * CROSSING_LINE_Y)
        crossing.line_y = line_y
        crossing.buffer.line_y = line_y

        print(f"[main] Камера {cam.width}x{cam.height} ({cam.resolved_backend})")
        print(f"[main] Линия портала: y={line_y}")
        print(f"[main] Бэкенд детектора: {detector.active_mode}")
        print(f"[main] Подавление по руке: {'вкл' if HAND_SUPPRESSION else 'выкл'}")
        print("[main] Жмите Ctrl+C для выхода (или 'q' в окне, если --show)")

        prev_time: float = time.time()
        fps: float = 0.0
        frame_num: int = 0

        while True:
            frame: Optional[np.ndarray] = cam.read()
            if frame is None:
                print("[main] Камера вернула None, выхожу")
                break
            frame_num += 1

            t0: float = time.perf_counter()

            # 1. Детекция объектов на кадре
            detections = detector.detect(frame)

            # 2. Трекинг — сопоставление детекций с persistent ID
            tracks = tracker.update(detections)

            # 3. Пересечение линии (с подавлением по руке внутри)
            crossing.process_frame(frame, tracks)

            frame_ms: float = (time.perf_counter() - t0) * 1000
            now: float = time.time()
            dt: float = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            if PERF_LOG_EVERY > 0 and frame_num % PERF_LOG_EVERY == 0:
                print(f"[perf] кадр {frame_num}: {frame_ms:.1f} мс, "
                      f"FPS≈{fps:.1f}, треков={tracker.active_tracks}")

            if show_window:
                _draw_overlay(frame, tracks, line_y, fps)
                cv2.imshow("Smart Fridge OS", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    if show_window:
        cv2.destroyAllWindows()


# ── Сборка и запуск ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Fridge OS")
    parser.add_argument("--mode", default=INFERENCE_MODE,
                        choices=["auto", "hailo", "ncnn", "cpu"],
                        help="Режим инференса (по умолчанию из config.py)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Путь к .pt модели (для NCNN/CPU)")
    parser.add_argument("--hef", default=MODEL_HEF_PATH,
                        help="Путь к .hef модели (для Hailo)")
    parser.add_argument("--no-server", action="store_true",
                        help="Не поднимать HTTP/WS сервер")
    parser.add_argument("--no-cv", action="store_true",
                        help="Не запускать CV-цикл (только сервер)")
    parser.add_argument("--show", action="store_true", default=SHOW_WINDOW,
                        help="Показывать окно OpenCV с визуализацией")
    parser.add_argument("--video", default=None,
                        help="Путь к видеофайлу — запуск на записи вместо камеры")
    parser.add_argument("--loop", action="store_true",
                        help="Зациклить видеофайл (с --video)")
    parser.add_argument("--host", default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    args = parser.parse_args()

    # ── Инициализация железа ─────────────────────────────────────
    lock: Lock = Lock()

    # ── Инициализация модели и трекера ───────────────────────────
    detector: Optional[Detector] = None
    tracker: Optional[ByteTracker] = None
    crossing: Optional[LineCrossingDetector] = None
    products: list[str] = []
    model_info: dict = {"mode": "none"}

    if not args.no_cv:
        detector = Detector(
            model_path=args.model,
            mode=args.mode,
            confidence=MODEL_CONF,
            imgsz=MODEL_IMGSZ,
            hef_path=args.hef,
        )
        tracker = ByteTracker()
        products = list(detector.class_names.values())
        model_info = {
            "mode": detector.active_mode,
            "model_path": args.model,
            "hef_path": args.hef,
            "imgsz": MODEL_IMGSZ,
            "confidence": MODEL_CONF,
            "classes": products,
        }

    # ── Сервер ───────────────────────────────────────────────────
    hub: WebSocketHub = WebSocketHub()
    server: Optional[_ServerThread] = None

    if not args.no_server:
        app = build_app(lock=lock, hub=hub, model_info=model_info, products=products)
        server = _ServerThread(app=app, host=args.host, port=args.port)
        server.start()
        if server.wait_until_ready(timeout=5.0):
            print(f"[main] Сервер запущен: http://{args.host}:{args.port}")
        else:
            print("[main] Сервер не успел подняться за 5 сек, продолжаю без него")

    # ── Колбэк CV → WebSocket ────────────────────────────────────
    # Когда CV-пайплайн фиксирует пересечение, рассылаем событие всем ws.
    def _on_crossing_event(event: CrossingEvent) -> None:
        msg: dict = {"type": "crossing", **event.to_dict()}
        if server is not None and server.loop is not None:
            hub.broadcast_threadsafe(msg, loop=server.loop)

    # Стартовый line_y перенастроится в run_cv_loop под фактическую высоту кадра
    if not args.no_cv:
        crossing = LineCrossingDetector(
            line_y=240,
            on_event=_on_crossing_event,
            hand_suppression=HAND_SUPPRESSION,
        )

    # ── Основной цикл ────────────────────────────────────────────
    try:
        if args.no_cv:
            print("[main] CV-цикл отключён (--no-cv), жду Ctrl+C")
            while True:
                time.sleep(1.0)
        else:
            assert detector is not None and tracker is not None and crossing is not None
            run_cv_loop(
                detector, tracker, crossing,
                show_window=args.show,
                video_path=args.video,
                loop_video=args.loop,
            )
    except KeyboardInterrupt:
        print("\n[main] Получен Ctrl+C, останавливаюсь")
    finally:
        # Порядок важен: сначала закрыть замок (безопасность), затем медиа
        lock.cleanup()
        if crossing is not None:
            crossing.close()
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
