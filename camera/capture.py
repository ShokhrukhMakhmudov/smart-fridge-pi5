"""
Унифицированный захват кадров с камеры.

Поддерживает три источника:
  - Picamera2     — нативный драйвер MIPI CSI для IMX219 (рекомендуется на Pi 5)
  - OpenCV камера — универсальный VideoCapture (USB-камеры, Windows-разработка)
  - Видеофайл    — для тестирования модели без live-источника (передать video_path)

Режим выбирается автоматически: если указан video_path — читаем файл;
иначе если доступен picamera2 и CAMERA_BACKEND != "opencv" — используем CSI;
иначе — OpenCV. Класс реализует контекст-менеджер.
"""

import time
from pathlib import Path
from typing import Optional, Union

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
        video_path: Optional[Union[str, Path]] = None,
        loop_video: bool = False,
        realtime_video: bool = True,
    ) -> None:
        self.index: int = index
        self.width: int = width
        self.height: int = height
        self.fps: int = fps
        self.backend: str = backend
        self.video_path: Optional[Path] = Path(video_path) if video_path else None
        self.loop_video: bool = loop_video
        self.realtime_video: bool = realtime_video

        self._picam = None   # экземпляр Picamera2 (при наличии)
        self._cap = None     # экземпляр cv2.VideoCapture (фоллбэк)
        self._resolved_backend: str = "none"
        self._frame_interval: float = 0.0   # для имитации fps при чтении файла
        self._last_frame_time: float = 0.0

        self._open()

    # ── Инициализация ────────────────────────────────────────────

    def _open(self) -> None:
        """Открыть источник кадров: видеофайл, Picamera2 или OpenCV-камеру."""
        # Приоритет: если передан путь к видео — используем его
        if self.video_path is not None:
            self._open_video()
            return

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

    def _open_video(self) -> None:
        """Открыть видеофайл как источник кадров."""
        import cv2  # noqa: PLC0415

        if not self.video_path.exists():
            raise FileNotFoundError(f"Видеофайл не найден: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть видео: {self.video_path}"
            )

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        file_fps: float = cap.get(cv2.CAP_PROP_FPS) or float(self.fps)
        self.fps = int(file_fps)
        frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._cap = cap
        self._resolved_backend = "video"
        if self.realtime_video and file_fps > 0:
            self._frame_interval = 1.0 / file_fps
        print(
            f"[camera] Видео: {self.video_path.name} — "
            f"{self.width}x{self.height} @ {self.fps}fps, кадров={frame_count}, "
            f"loop={self.loop_video}, realtime={self.realtime_video}"
        )

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
        import platform
        import glob
        import cv2

        is_linux: bool = platform.system() == "Linux"
        backend_flag: int = cv2.CAP_V4L2 if is_linux else cv2.CAP_ANY

        # На Linux перебираем реальные пути /dev/videoN через glob
        if is_linux:
            candidates = sorted(
                glob.glob("/dev/video*"),
                key=lambda p: int(p.replace("/dev/video", "") or -1)
            )
        else:
            candidates = [str(i) for i in range(5)]

        cap = None
        opened_path = ""
        for path in candidates:
            src = path if is_linux else int(path)
            try:
                trial = cv2.VideoCapture(src, backend_flag)
            except Exception as e:
                print(f"[camera] {path}: исключение: {e}")
                continue

            if not trial.isOpened():
                trial.release()
                print(f"[camera] {path}: не открылся")
                continue

            ok, _frame = trial.read()
            if not ok or _frame is None:
                trial.release()
                print(f"[camera] {path}: открыт, но read() вернул None")
                continue

            cap = trial
            opened_path = path
            break

        if cap is None:
            raise RuntimeError(
                "Не удалось открыть USB-камеру ни по одному устройству.\n"
                "Проверьте:\n"
                "  ls /dev/video*\n"
                "  v4l2-ctl --list-devices"
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        actual_fps: float = cap.get(cv2.CAP_PROP_FPS) or float(self.fps)
        self._cap = cap
        self._resolved_backend = "opencv"
        print(
            f"[camera] OpenCV запущен: {opened_path} "
            f"{self.width}x{self.height}@{int(actual_fps)}fps"
        )
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
                # Для видеофайла: зациклить или отдать None (конец потока)
                if self._resolved_backend == "video" and self.loop_video:
                    import cv2  # noqa: PLC0415
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                    if not ok:
                        return None
                else:
                    return None

            # Если читаем файл и хотим играть в реальном темпе —
            # подождать между кадрами, чтобы не пролистать видео за секунду
            if self._frame_interval > 0:
                now: float = time.perf_counter()
                wait: float = self._frame_interval - (now - self._last_frame_time)
                if wait > 0:
                    time.sleep(wait)
                self._last_frame_time = time.perf_counter()

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
