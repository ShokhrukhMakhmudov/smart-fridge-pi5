"""
Унифицированный захват кадров с камеры.

Поддерживает два бэкенда:
  - Picamera2  — нативный драйвер MIPI CSI для IMX219 (рекомендуется на Pi 5)
  - OpenCV     — универсальный VideoCapture (USB-камеры, Windows-разработка)

Режим выбирается автоматически: если доступен picamera2 и CAMERA_BACKEND != "opencv",
используется CSI; иначе — OpenCV. Класс реализует контекст-менеджер, поэтому
ресурсы камеры гарантированно освобождаются при любом исходе цикла.
"""

from typing import Optional

import numpy as np

from config import (
    CAMERA_BACKEND,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
)


class Camera:
    """
    Источник BGR-кадров одинакового формата независимо от бэкенда.

    Пример использования:
        with Camera() as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    break
                ...  # обработка кадра
    """

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
        backend: str = CAMERA_BACKEND,
    ) -> None:
        self.index: int = index
        self.width: int = width
        self.height: int = height
        self.fps: int = fps
        self.backend: str = backend

        self._picam = None   # экземпляр Picamera2 (при наличии)
        self._cap = None     # экземпляр cv2.VideoCapture (фоллбэк)
        self._resolved_backend: str = "none"

        self._open()

    # ── Инициализация ────────────────────────────────────────────

    def _open(self) -> None:
        """Открыть камеру согласно выбранному бэкенду с фоллбэком на OpenCV."""
        if self.backend in ("auto", "picamera"):
            if self._try_open_picamera():
                return
            if self.backend == "picamera":
                raise RuntimeError(
                    "CAMERA_BACKEND=picamera, но Picamera2 недоступна. "
                    "Установите: sudo apt install python3-picamera2"
                )

        # Фоллбэк на OpenCV VideoCapture
        self._open_opencv()

    def _try_open_picamera(self) -> bool:
        """Попытаться открыть CSI-камеру через Picamera2. True при успехе."""
        try:
            from picamera2 import Picamera2  # noqa: PLC0415 — ленивый импорт
        except ImportError:
            print("[camera] Picamera2 не установлена, переключаюсь на OpenCV")
            return False

        try:
            picam = Picamera2(camera_num=self.index)
            config = picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"},
                controls={"FrameRate": float(self.fps)},
            )
            picam.configure(config)
            picam.start()
            self._picam = picam
            self._resolved_backend = "picamera"
            print(f"[camera] Picamera2 запущена: {self.width}x{self.height}@{self.fps}fps")
            return True
        except Exception as e:
            print(f"[camera] Picamera2 не смогла открыться ({e}), пробую OpenCV")
            return False

    def _open_opencv(self) -> None:
        """Открыть камеру через OpenCV VideoCapture."""
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть камеру {self.index} через OpenCV"
            )

        # Запрашиваем желаемые параметры. Реально поддерживаемое разрешение
        # может быть другим — перечитываем фактические значения после.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        self._cap = cap
        self._resolved_backend = "opencv"
        print(f"[camera] OpenCV запущен: {self.width}x{self.height}")

    # ── Публичный API ────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """
        Прочитать следующий кадр (BGR, uint8). None — при ошибке или конце потока.
        """
        if self._picam is not None:
            # Picamera2 возвращает numpy-массив напрямую. Формат BGR888
            # уже совместим с OpenCV/YOLO, дополнительной конверсии не нужно.
            try:
                return self._picam.capture_array("main")
            except Exception as e:
                print(f"[camera] Ошибка чтения Picamera2: {e}")
                return None

        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                return None
            return frame

        return None

    def release(self) -> None:
        """Закрыть камеру и освободить ресурсы. Идемпотентно."""
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as e:
                print(f"[camera] Ошибка закрытия Picamera2: {e}")
            self._picam = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                print(f"[camera] Ошибка закрытия OpenCV: {e}")
            self._cap = None

    # ── Контекст-менеджер ────────────────────────────────────────

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    @property
    def resolved_backend(self) -> str:
        """Какой бэкенд фактически используется: picamera / opencv / none."""
        return self._resolved_backend
