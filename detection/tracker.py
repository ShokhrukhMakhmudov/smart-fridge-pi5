"""
Трекер объектов поверх детектора.

Используем lapx (быстрый ассайнмент по венгерскому алгоритму) и
встроенный ByteTrack из ultralytics, но в виде standalone-обёртки,
чтобы трекинг не был привязан к конкретному бэкенду инференса
(Hailo / NCNN / CPU).

Реализация — упрощённая версия ByteTrack: ассоциация по IoU между
текущими детекциями и активными треками + временно потерянные треки
держатся max_lost кадров. Это покрывает случай "продукт исчезает за
рукой на 1-2 кадра" без подключения тяжёлой зависимости.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detector import Detection


@dataclass
class TrackedDetection:
    """Детекция с присвоенным persistent track_id."""
    track_id: int
    detection: Detection

    @property
    def center(self) -> tuple[int, int]:
        return self.detection.center


@dataclass
class _Track:
    """Внутреннее состояние одного трека."""
    track_id: int
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2
    class_id: int
    confidence: float
    lost_frames: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU двух bbox в формате (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ByteTracker:
    """
    Лёгкий IoU-трекер с поддержкой временно потерянных треков.

    Параметры:
        iou_threshold: минимальный IoU для ассоциации детекции с треком
        max_lost:      сколько кадров держать трек, не получая обновлений
        match_classes: ассоциировать только детекции того же класса, что и трек
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 30,
        match_classes: bool = True,
    ) -> None:
        self.iou_threshold: float = iou_threshold
        self.max_lost: int = max_lost
        self.match_classes: bool = match_classes
        self._tracks: dict[int, _Track] = {}
        self._next_id: int = 1

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        """
        Сопоставить новые детекции с существующими треками и вернуть
        список TrackedDetection для текущего кадра.
        """
        # Шаг 1: построить матрицу IoU между активными треками и детекциями
        track_ids: list[int] = list(self._tracks.keys())
        if track_ids and detections:
            iou_matrix: np.ndarray = np.zeros(
                (len(track_ids), len(detections)), dtype=np.float32
            )
            for i, tid in enumerate(track_ids):
                track: _Track = self._tracks[tid]
                for j, det in enumerate(detections):
                    if self.match_classes and det.class_id != track.class_id:
                        continue
                    iou_matrix[i, j] = _iou(
                        track.bbox,
                        (det.x1, det.y1, det.x2, det.y2),
                    )
            assignments: dict[int, int] = self._greedy_assign(iou_matrix)
        else:
            assignments = {}

        # Шаг 2: обновить ассоциированные треки
        used_dets: set[int] = set()
        out: list[TrackedDetection] = []
        for i, tid in enumerate(track_ids):
            j: Optional[int] = assignments.get(i)
            if j is None:
                self._tracks[tid].lost_frames += 1
                continue
            det = detections[j]
            track = self._tracks[tid]
            track.bbox = (det.x1, det.y1, det.x2, det.y2)
            track.confidence = det.confidence
            track.lost_frames = 0
            track.history.append(det.center)
            used_dets.add(j)
            out.append(TrackedDetection(track_id=tid, detection=det))

        # Шаг 3: создать треки для не сопоставленных детекций
        for j, det in enumerate(detections):
            if j in used_dets:
                continue
            new_id: int = self._next_id
            self._next_id += 1
            self._tracks[new_id] = _Track(
                track_id=new_id,
                bbox=(det.x1, det.y1, det.x2, det.y2),
                class_id=det.class_id,
                confidence=det.confidence,
                history=[det.center],
            )
            out.append(TrackedDetection(track_id=new_id, detection=det))

        # Шаг 4: удалить треки, потерянные слишком давно
        stale: list[int] = [
            tid for tid, t in self._tracks.items() if t.lost_frames > self.max_lost
        ]
        for tid in stale:
            del self._tracks[tid]

        return out

    def _greedy_assign(self, iou_matrix: np.ndarray) -> dict[int, int]:
        """
        Жадное назначение пар (track, detection) по убыванию IoU.

        Возвращает {row_index: col_index}. Подходит для типичной картинки
        холодильника (≤10 объектов в кадре); для большого числа треков
        стоит заменить на полноценный венгерский алгоритм из lapx.
        """
        assignments: dict[int, int] = {}
        if iou_matrix.size == 0:
            return assignments

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        # Получаем индексы (row, col), отсортированные по убыванию IoU
        flat = np.argsort(-iou_matrix, axis=None)
        for idx in flat:
            row, col = divmod(int(idx), iou_matrix.shape[1])
            if iou_matrix[row, col] < self.iou_threshold:
                break
            if row in used_rows or col in used_cols:
                continue
            assignments[row] = col
            used_rows.add(row)
            used_cols.add(col)
        return assignments

    @property
    def active_tracks(self) -> int:
        """Число треков, обновлённых на последнем кадре (lost_frames == 0)."""
        return sum(1 for t in self._tracks.values() if t.lost_frames == 0)
