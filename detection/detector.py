"""
Детектор объектов с тремя режимами инференса.

Порядок выбора в режиме 'auto':
  1. Hailo (HEF-модель через hailort) — самый быстрый, требует HAT
  2. NCNN  (экспорт ultralytics → ARM NEON) — быстрый на Pi 5 без HAT
  3. CPU   (ultralytics YOLO на .pt) — универсальный фоллбэк

На выходе: список списков [x1, y1, x2, y2, conf, class_id, class_name]
для каждого кадра. Трекинг сюда не встроен — за это отвечает ByteTracker
в detection/tracker.py, чтобы режим инференса не был связан с бэкендом
ассоциации треков.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Определяем наличие Hailo один раз при импорте. На Windows/dev машинах
# модуля нет — и это нормально, просто недоступен режим Hailo.
try:
    from hailo_platform import HEF, VDevice  # noqa: F401
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


@dataclass
class Detection:
    """Одна детекция в кадре (без trackId)."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_list(self) -> list:
        """Упаковать в список [x1, y1, x2, y2, conf, class_id, class_name]."""
        return [self.x1, self.y1, self.x2, self.y2,
                self.confidence, self.class_id, self.class_name]


class Detector:
    """
    Универсальный детектор с автоматическим выбором бэкенда.

    Пример:
        det = Detector(model_path="models/best.pt", mode="auto")
        for d in det.detect(frame):
            ...

    Параметр mode:
        'auto'  — перебрать Hailo → NCNN → CPU
        'hailo' — только Hailo (ошибка при отсутствии HAT)
        'ncnn'  — только NCNN-папка рядом с .pt
        'cpu'   — ultralytics на .pt
    """

    def __init__(
        self,
        model_path: str,
        mode: str = "auto",
        confidence: float = 0.5,
        imgsz: int = 640,
        hef_path: Optional[str] = None,
    ) -> None:
        self.model_path: Path = Path(model_path)
        self.hef_path: Optional[Path] = Path(hef_path) if hef_path else None
        self.confidence: float = confidence
        self.imgsz: int = imgsz
        self.mode: str = mode

        # Поля, заполняемые инициализаторами конкретного бэкенда.
        self._active_mode: str = "none"
        self._yolo = None       # ultralytics.YOLO (CPU/NCNN)
        self._hailo = None      # внутренняя обёртка Hailo
        self.class_names: dict[int, str] = {}

        self._initialize()

    # ── Инициализация бэкендов ───────────────────────────────────

    def _initialize(self) -> None:
        """Пробуем бэкенды в порядке приоритета согласно self.mode."""
        requested: str = self.mode.lower()

        if requested in ("auto", "hailo"):
            if self._try_init_hailo():
                return
            if requested == "hailo":
                raise RuntimeError(
                    "mode='hailo', но Hailo недоступен (нет hailort или .hef)"
                )

        if requested in ("auto", "ncnn"):
            if self._try_init_ncnn():
                return
            if requested == "ncnn":
                raise RuntimeError(
                    "mode='ncnn', но NCNN-модель не найдена рядом с .pt"
                )

        if requested in ("auto", "cpu"):
            if self._try_init_cpu():
                return

        raise RuntimeError(
            f"Не удалось инициализировать ни один бэкенд детектора "
            f"(mode={self.mode}, model_path={self.model_path})"
        )

    def _try_init_hailo(self) -> bool:
        """Инициализировать Hailo. True — если успех."""
        if not HAILO_AVAILABLE:
            return False

        # Ищем .hef: либо явный hef_path, либо model.hef рядом с model.pt.
        hef: Optional[Path] = self.hef_path
        if hef is None:
            candidate: Path = self.model_path.with_suffix(".hef")
            if candidate.exists():
                hef = candidate

        if hef is None or not hef.exists():
            print("[detector] Hailo доступен, но .hef-модель не найдена")
            return False

        try:
            self._hailo = _HailoRunner(
                hef_path=hef,
                confidence=self.confidence,
                imgsz=self.imgsz,
            )
            self.class_names = self._hailo.class_names
            self._active_mode = "hailo"
            print(f"[detector] Активный режим: Hailo ({hef.name})")
            return True
        except Exception as e:
            print(f"[detector] Инициализация Hailo не удалась: {e}")
            return False

    def _try_init_ncnn(self) -> bool:
        """Инициализировать NCNN через ultralytics. True — если успех."""
        # ultralytics распознаёт NCNN по папке вида best_ncnn_model/.
        ncnn_dir: Path = self.model_path.parent / f"{self.model_path.stem}_ncnn_model"
        if not ncnn_dir.exists() or not ncnn_dir.is_dir():
            return False

        try:
            from ultralytics import YOLO  # noqa: PLC0415

            self._yolo = YOLO(str(ncnn_dir))
            self.class_names = dict(self._yolo.names)
            self._active_mode = "ncnn"
            print(f"[detector] Активный режим: NCNN ({ncnn_dir.name})")
            return True
        except Exception as e:
            print(f"[detector] Инициализация NCNN не удалась: {e}")
            return False

    def _try_init_cpu(self) -> bool:
        """Инициализировать CPU-бэкенд ultralytics. True — если успех."""
        if not self.model_path.exists():
            print(f"[detector] .pt-модель не найдена: {self.model_path}")
            return False

        try:
            from ultralytics import YOLO  # noqa: PLC0415

            self._yolo = YOLO(str(self.model_path))
            self.class_names = dict(self._yolo.names)
            self._active_mode = "cpu"
            print(f"[detector] Активный режим: CPU ({self.model_path.name})")
            return True
        except Exception as e:
            print(f"[detector] Инициализация CPU не удалась: {e}")
            return False

    # ── Публичный API ────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Запустить детектор на кадре. Возвращает список Detection.
        """
        if self._active_mode == "hailo":
            return self._hailo.infer(frame)

        if self._yolo is not None:
            results = self._yolo.predict(
                frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                verbose=False,
            )
            return self._parse_ultralytics(results)

        return []

    def _parse_ultralytics(self, results) -> list[Detection]:
        """Преобразовать вывод ultralytics в список Detection."""
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        xyxy: np.ndarray = boxes.xyxy.cpu().numpy()
        confs: list[float] = boxes.conf.cpu().tolist()
        class_ids: list[int] = boxes.cls.int().cpu().tolist()

        out: list[Detection] = []
        for box, conf, cls_id in zip(xyxy, confs, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            out.append(Detection(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=self.class_names.get(int(cls_id), f"class_{cls_id}"),
            ))
        return out

    @property
    def active_mode(self) -> str:
        """Какой бэкенд фактически используется: hailo / ncnn / cpu / none."""
        return self._active_mode


# ── Hailo-бэкенд ─────────────────────────────────────────────────

class _HailoRunner:
    """
    Обёртка над hailort для запуска YOLO-HEF на Hailo-8L.

    Постпроцессинг намеренно минимальный: HEF-модели обычно компилируются
    с включённым NMS-слоем, поэтому на выходе уже лежат итоговые боксы.
    Если NMS вынесен наружу, добавьте здесь декодирование якорей.
    """

    def __init__(
        self,
        hef_path: Path,
        confidence: float = 0.5,
        imgsz: int = 640,
    ) -> None:
        from hailo_platform import (  # noqa: PLC0415
            HEF,
            VDevice,
            FormatType,
            HailoStreamInterface,
            ConfigureParams,
            InputVStreamParams,
            OutputVStreamParams,
        )

        self.confidence: float = confidence
        self.imgsz: int = imgsz

        # Откроем HEF и захватим виртуальное устройство — оно будет жить
        # до конца процесса (одно на детектор).
        self._hef = HEF(str(hef_path))
        self._device = VDevice()

        configure_params = ConfigureParams.create_from_hef(
            hef=self._hef,
            interface=HailoStreamInterface.PCIe,
        )
        network_groups = self._device.configure(self._hef, configure_params)
        self._network_group = network_groups[0]
        self._network_params = self._network_group.create_params()

        self._input_vstreams_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8,
        )
        self._output_vstreams_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32,
        )

        input_vstream_info = self._hef.get_input_vstream_infos()[0]
        self._input_name: str = input_vstream_info.name
        self._input_shape = input_vstream_info.shape  # (H, W, C)

        # Имена классов в HEF не хранятся — берём из метаданных соседнего JSON
        # (best.json с полем "names"), иначе заполняем автоматическими именами.
        self.class_names: dict[int, str] = self._load_class_names(hef_path)

    def _load_class_names(self, hef_path: Path) -> dict[int, str]:
        """Попытаться прочитать имена классов из best.json рядом с .hef."""
        meta: Path = hef_path.with_suffix(".json")
        if not meta.exists():
            return {}
        try:
            import json  # noqa: PLC0415
            data = json.loads(meta.read_text(encoding="utf-8"))
            names = data.get("names") or {}
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
        except Exception as e:
            print(f"[detector] Не удалось прочитать {meta}: {e}")
        return {}

    # ── Предобработка + инференс ─────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Ресайз и паддинг кадра под вход сети (NHWC uint8)."""
        import cv2  # noqa: PLC0415

        target_h, target_w, _ = self._input_shape
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h))
        # Hailo ожидает RGB, OpenCV читает BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.expand_dims(frame.astype(np.uint8), axis=0)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Запустить инференс и вернуть список Detection."""
        from hailo_platform import InferVStreams  # noqa: PLC0415

        orig_h, orig_w = frame.shape[:2]
        input_data = {self._input_name: self._preprocess(frame)}

        with InferVStreams(
            self._network_group,
            self._input_vstreams_params,
            self._output_vstreams_params,
        ) as infer_pipeline:
            with self._network_group.activate(self._network_params):
                raw_out = infer_pipeline.infer(input_data)

        return self._postprocess(raw_out, orig_w, orig_h)

    def _postprocess(
        self,
        raw_out: dict,
        orig_w: int,
        orig_h: int,
    ) -> list[Detection]:
        """
        Разобрать выход HEF с встроенным NMS.

        Формат выхода YOLOv11-NMS Hailo: словарь, где под каждым классом
        лежит массив [n, 5] с колонками [ymin, xmin, ymax, xmax, score] в
        нормализованных координатах [0..1]. Если модель скомпилирована
        иначе — подправьте этот метод.
        """
        detections: list[Detection] = []
        target_h, target_w, _ = self._input_shape
        sx: float = orig_w / target_w
        sy: float = orig_h / target_h

        for output_name, data in raw_out.items():
            arr = np.asarray(data)
            # Стандартный формат выхода NMS: [batch, classes, max_dets, 5]
            if arr.ndim == 4:
                arr = arr[0]  # убираем batch
                for cls_id, per_class in enumerate(arr):
                    for row in per_class:
                        score: float = float(row[4])
                        if score < self.confidence:
                            continue
                        ymin, xmin, ymax, xmax = row[0:4]
                        x1 = int(xmin * target_w * sx)
                        y1 = int(ymin * target_h * sy)
                        x2 = int(xmax * target_w * sx)
                        y2 = int(ymax * target_h * sy)
                        detections.append(Detection(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=score,
                            class_id=int(cls_id),
                            class_name=self.class_names.get(
                                int(cls_id), f"class_{cls_id}"
                            ),
                        ))
            # Иной формат — придётся доработать под конкретный экспорт
            else:
                print(f"[detector] Неизвестный формат выхода Hailo "
                      f"({output_name}, shape={arr.shape}) — пропускаю")

        return detections
