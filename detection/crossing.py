"""
Детектор пересечения горизонтальной линии портала.

Линия делит кадр пополам (или на любой относительной координате из конфига).
Когда центр трека последовательно пересекает линию в одном направлении
N кадров подряд, генерируется событие:

  - вверх (cy уменьшается)   → "taken"     — продукт взят из холодильника
  - вниз  (cy увеличивается) → "returned"  — продукт возвращён на полку

Защиты от ложных срабатываний:
  1. CrossingBuffer — событие подтверждается только после N стабильных
     кадров с одним направлением (не реагируем на одиночный шум трекера)
  2. Cooldown по дистанции — после события трек должен уйти от линии
     минимум на CROSSING_RESET_DISTANCE px, прежде чем сможет вызвать
     следующее событие (защита от дребезга у самой линии)
  3. Дедупликация по (продукт, событие) в окне CROSSING_DEDUP_WINDOW_SEC
  4. HandSuppressor — пока MediaPipe видит руку в кадре, события
     отбрасываются (рука = неустойчивые боксы продуктов)
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config import (
    CROSSING_BUFFER_FRAMES,
    CROSSING_DEDUP_WINDOW_SEC,
    CROSSING_RESET_DISTANCE,
)

from .tracker import TrackedDetection


# ── Событие ──────────────────────────────────────────────────────

@dataclass
class CrossingEvent:
    """Событие пересечения линии портала, эмитится наружу как JSON."""
    event: str            # "taken" | "returned"
    track_id: int
    product: str
    timestamp: str        # ISO-8601 без микросекунд

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "track_id": self.track_id,
            "product": self.product,
            "timestamp": self.timestamp,
        }


# ── Буфер подтверждения ──────────────────────────────────────────

@dataclass
class _TrackCrossingState:
    """Состояние одного трека для CrossingBuffer."""
    anchor_zone: Optional[str] = None         # "above" / "below" — где трек "живёт"
    current_zone: Optional[str] = None        # "above" / "below" — где сейчас
    zone_frames: int = 0                      # сколько кадров подряд в current_zone
    cooldown_frames: int = 0                  # сколько кадров ещё кулдаун


def _zone_of(cy: float, line_y: int) -> str:
    """Определить зону центра относительно линии портала."""
    return "above" if cy < line_y else "below"


class CrossingBuffer:
    """
    Подтверждает пересечение линии через смену зон.

    Вместо отслеживания "пересёк ли центр линию за один кадр" (что
    хрупко — реальное пересечение случается ровно в одном кадре),
    запоминаем *в какой половине кадра* трек стабильно находится:
      - "above"  — центр выше линии портала
      - "below"  — центр ниже линии портала

    Когда трек был anchor_zone=above и N кадров подряд сидит в зоне
    below — эмитим "returned" (спустился вниз). И наоборот — "taken".

    Такая логика устойчива к дрожанию бокса на ±несколько пикселей
    и работает, даже если детектор пропустил сам момент пересечения.
    """

    def __init__(
        self,
        line_y: int,
        confirm_frames: int = CROSSING_BUFFER_FRAMES,
        reset_distance: int = CROSSING_RESET_DISTANCE,
    ) -> None:
        self.line_y: int = line_y
        self.confirm_frames: int = confirm_frames
        self.reset_distance: int = reset_distance
        self._states: dict[int, _TrackCrossingState] = {}

    def update(self, track_id: int, cy: float) -> Optional[str]:
        """
        Обновить состояние трека. Возвращает "taken" / "returned" в момент
        подтверждения события, иначе None.
        """
        state: _TrackCrossingState = self._states.setdefault(
            track_id, _TrackCrossingState()
        )
        zone: str = _zone_of(cy, self.line_y)

        # Первый раз видим трек — фиксируем якорную зону, событий не эмитим
        if state.anchor_zone is None:
            state.anchor_zone = zone
            state.current_zone = zone
            state.zone_frames = 1
            return None

        # Кулдаун после события: ждём, пока счётчик истечёт И трек уйдёт
        # от линии, чтобы не выдать повторное событие от дрожания у границы
        if state.cooldown_frames > 0:
            state.cooldown_frames -= 1
            state.current_zone = zone
            state.zone_frames = 1
            if state.cooldown_frames == 0 and abs(cy - self.line_y) > self.reset_distance:
                # Кулдаун закончен, новая якорная зона — текущая
                state.anchor_zone = zone
            return None

        # Накапливаем подтверждения пребывания в одной зоне
        if zone == state.current_zone:
            state.zone_frames += 1
        else:
            state.current_zone = zone
            state.zone_frames = 1

        # Трек всё ещё в своей якорной зоне — событий нет
        if zone == state.anchor_zone:
            return None

        # Трек перешёл в противоположную зону. Ждём подтверждения
        if state.zone_frames < self.confirm_frames:
            return None

        # Подтверждено: смена зоны above→below = returned, below→above = taken
        direction: str = "taken" if zone == "above" else "returned"
        state.anchor_zone = zone
        state.zone_frames = 0
        # Кулдаун в кадрах ≈ confirm_frames, чтобы не словить дребезг
        state.cooldown_frames = self.confirm_frames
        return direction

    def cleanup(self, active_ids: set[int]) -> None:
        """Удалить состояние треков, которых больше нет в кадре."""
        stale: list[int] = [tid for tid in self._states if tid not in active_ids]
        for tid in stale:
            del self._states[tid]


# ── Подавление по руке (MediaPipe Hands) ─────────────────────────

class HandSuppressor:
    """
    Возвращает True, если в кадре сейчас есть рука.

    Когда рука в кадре — детекции продуктов нестабильны (рука загораживает,
    смещает, частично перекрывает), и буфер пересечения может выдать
    ложное событие. Поэтому при is_hand_present() == True детектор
    пересечений просто игнорирует кадр.

    Если mediapipe не установлен или не инициализировался — модуль
    переходит в no-op (всегда возвращает False), и подавление отключается.
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = None  # экземпляр mediapipe.solutions.hands.Hands
        try:
            import mediapipe as mp  # noqa: PLC0415

            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            print("[crossing] MediaPipe Hands активен — рука будет подавлять события")
        except ImportError:
            print("[crossing] MediaPipe не установлен — подавление по руке отключено")
        except Exception as e:
            print(f"[crossing] MediaPipe Hands не инициализирован: {e}")

    def is_hand_present(self, frame_bgr: np.ndarray) -> bool:
        """Проверить, видит ли MediaPipe хотя бы одну руку в кадре."""
        if self._hands is None:
            return False
        try:
            import cv2  # noqa: PLC0415

            # MediaPipe ожидает RGB, OpenCV даёт BGR
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self._hands.process(rgb)
            return bool(result.multi_hand_landmarks)
        except Exception as e:
            print(f"[crossing] Сбой MediaPipe Hands: {e}")
            return False

    def close(self) -> None:
        """Освободить ресурсы MediaPipe (важно при долгой работе)."""
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None


# ── Объединяющий детектор ────────────────────────────────────────

class LineCrossingDetector:
    """
    Высокоуровневый компонент, который связывает буфер, дедупликацию,
    подавление по руке и колбэк события в одно целое.

    Использование:
        detector = LineCrossingDetector(
            line_y=240,
            on_event=lambda e: print(e),
            hand_suppression=True,
        )
        for frame in stream:
            tracks = tracker.update(detector_objs)
            detector.process_frame(frame, tracks)
    """

    def __init__(
        self,
        line_y: int,
        on_event: Optional[Callable[[CrossingEvent], None]] = None,
        confirm_frames: int = CROSSING_BUFFER_FRAMES,
        reset_distance: int = CROSSING_RESET_DISTANCE,
        dedup_window_sec: float = CROSSING_DEDUP_WINDOW_SEC,
        hand_suppression: bool = True,
    ) -> None:
        self.line_y: int = line_y
        self.buffer: CrossingBuffer = CrossingBuffer(
            line_y=line_y,
            confirm_frames=confirm_frames,
            reset_distance=reset_distance,
        )
        self.dedup_window_sec: float = dedup_window_sec
        self._on_event: Optional[Callable[[CrossingEvent], None]] = on_event
        self.events: list[CrossingEvent] = []
        self._recent_events: dict[tuple[str, str], float] = {}
        self.hand_suppressor: Optional[HandSuppressor] = (
            HandSuppressor() if hand_suppression else None
        )

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: list[TrackedDetection],
    ) -> list[CrossingEvent]:
        """
        Обработать один кадр. Возвращает список новых событий (после
        фильтрации/дедупликации) — обычно 0 или 1.
        """
        # 1. Подавление по руке: если рука в кадре, пропускаем фазу
        # подтверждения смены зоны, но обновляем current_zone,
        # чтобы после исчезновения руки решение принималось от
        # актуальной позиции трека.
        hand_in_frame: bool = (
            self.hand_suppressor is not None
            and self.hand_suppressor.is_hand_present(frame)
        )

        new_events: list[CrossingEvent] = []
        active_ids: set[int] = set()

        for tracked in tracks:
            tid: int = tracked.track_id
            _, cy = tracked.center
            active_ids.add(tid)

            if hand_in_frame:
                # Пока рука в кадре — не накапливаем подтверждения смены зоны,
                # но обновляем текущую зону, чтобы после исчезновения руки
                # решение принималось от актуальной позиции трека
                state = self.buffer._states.setdefault(tid, _TrackCrossingState())
                if state.anchor_zone is None:
                    state.anchor_zone = _zone_of(cy, self.buffer.line_y)
                state.current_zone = _zone_of(cy, self.buffer.line_y)
                state.zone_frames = 0
                continue

            direction: Optional[str] = self.buffer.update(tid, cy)
            if direction is None:
                continue

            event: Optional[CrossingEvent] = self._maybe_emit(
                event_type=direction,
                track_id=tid,
                product=tracked.detection.class_name,
            )
            if event is not None:
                new_events.append(event)

        # 2. Удаляем состояние пропавших треков из буфера
        self.buffer.cleanup(active_ids)

        return new_events

    def _maybe_emit(
        self,
        event_type: str,
        track_id: int,
        product: str,
    ) -> Optional[CrossingEvent]:
        """Проверить дедупликацию и эмитнуть событие в колбэк."""
        now: float = time.monotonic()
        key: tuple[str, str] = (product, event_type)
        last: Optional[float] = self._recent_events.get(key)
        if last is not None and (now - last) < self.dedup_window_sec:
            print(f"[crossing] DEDUP {event_type} {product} (трек #{track_id})")
            return None

        self._recent_events[key] = now
        event = CrossingEvent(
            event=event_type,
            track_id=track_id,
            product=product,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.events.append(event)
        print(f"[crossing] {event.to_dict()}")

        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as e:
                print(f"[crossing] Колбэк on_event упал: {e}")
        return event

    def close(self) -> None:
        """Освободить ресурсы MediaPipe Hands."""
        if self.hand_suppressor is not None:
            self.hand_suppressor.close()
