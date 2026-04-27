"""
Конфигурация Smart Fridge OS.

Все параметры проекта в одном месте. Значения по умолчанию подобраны
под Raspberry Pi 5 + Hailo AI HAT (Hailo-8L) + 2x IMX219 (MIPI CSI) +
электромагнитный замок SRD-05VDC-SL-C через реле.

Переменные окружения (префикс SF_) переопределяют значения по умолчанию,
например: SF_SERVER_PORT=9000 python main.py
"""

import os
from pathlib import Path

# ── Пути проекта ─────────────────────────────────────────────────
# Корень проекта — каталог, в котором лежит этот файл.
BASE_DIR: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = BASE_DIR / "dataset"
MODELS_DIR: Path = BASE_DIR / "models"


def _env_str(name: str, default: str) -> str:
    """Прочитать строковую переменную окружения с префиксом SF_."""
    return os.environ.get(f"SF_{name}", default)


def _env_int(name: str, default: int) -> int:
    """Прочитать целочисленную переменную окружения."""
    try:
        return int(os.environ.get(f"SF_{name}", default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Прочитать вещественную переменную окружения."""
    try:
        return float(os.environ.get(f"SF_{name}", default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Прочитать булеву переменную окружения (1/true/yes = True)."""
    raw: str = os.environ.get(f"SF_{name}", str(default)).lower()
    return raw in ("1", "true", "yes", "on")


# ── Камера ───────────────────────────────────────────────────────
# Индекс веб-камеры для OpenCV-фоллбэка (при отсутствии Picamera2).
CAMERA_INDEX: int = _env_int("CAMERA_INDEX", 0)
# Разрешение кадра — 640x640 совпадает с MODEL_IMGSZ, что экономит ресайз.
CAMERA_WIDTH: int = _env_int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT: int = _env_int("CAMERA_HEIGHT", 640)
CAMERA_FPS: int = _env_int("CAMERA_FPS", 30)
# USE_PICAMERA=True принудительно использует Picamera2 (MIPI CSI),
# False — OpenCV VideoCapture. "auto" — автоматический выбор.
CAMERA_BACKEND: str = _env_str("CAMERA_BACKEND", "auto")  # auto/picamera/opencv


# ── Модель детектора ─────────────────────────────────────────────
# Путь к основной модели. Для CPU/NCNN — .pt или .../best_ncnn_model/,
# для Hailo — .hef. При INFERENCE_MODE="auto" детектор сам выбирает,
# какой файл брать.
MODEL_PATH: str = _env_str("MODEL_PATH", str(MODELS_DIR / "best.pt"))
MODEL_HEF_PATH: str = _env_str("MODEL_HEF_PATH", str(MODELS_DIR / "best.hef"))
# Размер входа YOLO. 640 — стандарт, 320 быстрее на CPU.
MODEL_IMGSZ: int = _env_int("MODEL_IMGSZ", 320)
# Минимальная уверенность детектора.
MODEL_CONF: float = _env_float("MODEL_CONF", 0.5)
# Режим инференса: auto/hailo/ncnn/cpu.
#   auto — пробует Hailo → NCNN → CPU по убыванию скорости.
INFERENCE_MODE: str = _env_str("INFERENCE_MODE", "auto")


# ── Детекция пересечения ─────────────────────────────────────────
# Координата горизонтальной линии портала (0..1 от высоты кадра).
CROSSING_LINE_Y: float = _env_float("CROSSING_LINE_Y", 0.5)
# Сколько подряд кадров с одним направлением нужно, чтобы зафиксировать
# событие. Защита от одиночного ложного срабатывания трекера.
CROSSING_BUFFER_FRAMES: int = _env_int("CROSSING_BUFFER_FRAMES", 5)
# После события объект должен уйти от линии на столько пикселей,
# прежде чем он сможет снова вызвать событие (защита от дребезга).
CROSSING_RESET_DISTANCE: int = _env_int("CROSSING_RESET_DISTANCE", 30)
# Окно дедупликации одинаковых событий (сек).
CROSSING_DEDUP_WINDOW_SEC: float = _env_float("CROSSING_DEDUP_WINDOW_SEC", 2.0)
# Подавлять пересечения, пока в кадре присутствует рука (MediaPipe).
HAND_SUPPRESSION: bool = _env_bool("HAND_SUPPRESSION", True)


# ── GPIO / Электромагнитный замок ────────────────────────────────
# BCM пин, подключённый к IN реле SRD-05VDC-SL-C.
LOCK_GPIO_PIN: int = _env_int("LOCK_GPIO_PIN", 17)
# Автозакрытие через N секунд после open() (0 = не закрывать автоматически).
LOCK_AUTO_CLOSE_SEC: int = _env_int("LOCK_AUTO_CLOSE_SEC", 5)
# Реле активно-низкое (SRD-05VDC-SL-C): LOW = замок открыт, HIGH = закрыт.
LOCK_ACTIVE_LOW: bool = _env_bool("LOCK_ACTIVE_LOW", True)
# True — принудительный мок-режим (для Windows / dev). False — пробовать
# реальный GPIO, при сбое автоматически переходить в мок.
GPIO_MOCK: bool = _env_bool("GPIO_MOCK", False)


# ── Сервер (FastAPI) ─────────────────────────────────────────────
SERVER_HOST: str = _env_str("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = _env_int("SERVER_PORT", 8000)
# API-ключ для защищённых эндпоинтов (None = без авторизации).
SERVER_API_KEY: str | None = os.environ.get("SF_SERVER_API_KEY")


# ── Отладка ──────────────────────────────────────────────────────
# Показывать окно OpenCV с визуализацией (True для dev, на headless-Pi: SF_SHOW_WINDOW=false).
SHOW_WINDOW: bool = _env_bool("SHOW_WINDOW", True)
# Печатать пер-кадровую статистику каждые N кадров.
PERF_LOG_EVERY: int = _env_int("PERF_LOG_EVERY", 30)
