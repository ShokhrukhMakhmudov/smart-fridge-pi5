"""
Управление электромагнитным замком через реле.

Реле SRD-05VDC-SL-C (Songle, активно-низкое):
  GPIO LOW   →  реле включено  →  NO замкнут  →  замок под напряжением → ОТКРЫТ
  GPIO HIGH  →  реле выключено →  NO разомкнут →  замок без питания   → ЗАКРЫТ

Используем gpiozero вместо RPi.GPIO:
  - не требует root (доступ через /dev/gpiochip*)
  - корректно автоопределяет бэкенд (lgpio на Pi 5, RPi.GPIO на Pi 4)
  - проще API для активно-низких реле (active_high=False)

Класс автоматически переходит в мок-режим, если:
  - GPIO_MOCK=True в конфиге (явно)
  - gpiozero не установлен
  - инициализация пина не удалась (например, на Windows)

Поддерживается опциональное автозакрытие через N секунд после open(),
реализованное через threading.Timer — он не блокирует основной цикл CV.
"""

import threading
from typing import Optional

from config import (
    GPIO_MOCK,
    LOCK_ACTIVE_LOW,
    LOCK_AUTO_CLOSE_SEC,
    LOCK_GPIO_PIN,
)


class Lock:
    """
    Электромагнитный замок с авто-закрытием и мок-режимом.

    Пример:
        lock = Lock()
        lock.open(auto_close_sec=5)   # открыть и закрыть через 5 сек
        ...
        lock.close()                  # принудительно закрыть
        print(lock.status())          # {"locked": True, "mock": False}
        lock.cleanup()                # обязательно вызвать перед выходом
    """

    def __init__(
        self,
        gpio_pin: int = LOCK_GPIO_PIN,
        active_low: bool = LOCK_ACTIVE_LOW,
        force_mock: bool = GPIO_MOCK,
    ) -> None:
        self.gpio_pin: int = gpio_pin
        self.active_low: bool = active_low
        self._mock: bool = force_mock
        self._device = None                          # gpiozero.OutputDevice
        self._locked: bool = True                    # инвариант: стартуем закрытым
        self._auto_close_timer: Optional[threading.Timer] = None
        self._lock_mutex: threading.Lock = threading.Lock()

        if self._mock:
            print(f"[lock] Принудительный мок-режим (GPIO_MOCK=True)")
            return

        self._init_gpio()

    # ── Инициализация ────────────────────────────────────────────

    def _init_gpio(self) -> None:
        """Открыть GPIO через gpiozero. При сбое — мок-режим."""
        try:
            from gpiozero import OutputDevice  # noqa: PLC0415

            # active_high=False означает, что .on() выставит LOW (нужно для
            # активно-низкого реле), .off() выставит HIGH. initial_value=False
            # гарантирует, что замок закрыт сразу после конфигурирования пина.
            self._device = OutputDevice(
                pin=self.gpio_pin,
                active_high=not self.active_low,
                initial_value=False,
            )
            print(f"[lock] gpiozero инициализирован: GPIO{self.gpio_pin} "
                  f"(active_low={self.active_low}, замок ЗАКРЫТ)")
        except ImportError:
            print("[lock] gpiozero не установлен — мок-режим. "
                  "Установите: pip install gpiozero")
            self._mock = True
        except Exception as e:
            # Типовые причины: запуск не на Pi, пин занят, нет прав на gpiochip
            print(f"[lock] Сбой инициализации GPIO ({e}) — мок-режим")
            self._mock = True

    # ── Публичный API ────────────────────────────────────────────

    def open(self, auto_close_sec: Optional[int] = LOCK_AUTO_CLOSE_SEC) -> None:
        """
        Открыть замок. Если auto_close_sec > 0 — запланировать закрытие.

        Повторный вызов перезапускает таймер автозакрытия с нуля,
        что удобно: при каждом запросе на открытие "продлеваем" сессию.
        """
        with self._lock_mutex:
            self._cancel_timer()
            if not self._mock and self._device is not None:
                # .on() в gpiozero уже учитывает active_high.
                self._device.on()
            self._locked = False
            print(f"[lock] ОТКРЫТ (mock={self._mock})")

            if auto_close_sec and auto_close_sec > 0:
                self._auto_close_timer = threading.Timer(
                    interval=auto_close_sec,
                    function=self._auto_close,
                )
                self._auto_close_timer.daemon = True
                self._auto_close_timer.start()
                print(f"[lock] автозакрытие через {auto_close_sec} сек")

    def close(self) -> None:
        """Принудительно закрыть замок и отменить автозакрытие."""
        with self._lock_mutex:
            self._cancel_timer()
            self._do_close()

    def status(self) -> dict:
        """Текущее состояние замка для API."""
        return {
            "locked": self._locked,
            "mock": self._mock,
            "gpio_pin": self.gpio_pin,
            "active_low": self.active_low,
            "auto_close_pending": self._auto_close_timer is not None,
        }

    @property
    def is_locked(self) -> bool:
        return self._locked

    def cleanup(self) -> None:
        """
        Гарантированно закрыть замок и освободить GPIO. Идемпотентно.

        Обязательно вызывать в finally при завершении приложения, иначе
        пин может остаться в "открытом" состоянии после краша.
        """
        with self._lock_mutex:
            self._cancel_timer()
            self._do_close()
            if self._device is not None:
                try:
                    self._device.close()
                except Exception as e:
                    print(f"[lock] Сбой close() gpiozero: {e}")
                self._device = None

    # ── Внутренние помощники ─────────────────────────────────────

    def _do_close(self) -> None:
        """Физически закрыть замок (без работы с таймером и мьютексом)."""
        if not self._mock and self._device is not None:
            self._device.off()
        self._locked = True
        print(f"[lock] ЗАКРЫТ (mock={self._mock})")

    def _cancel_timer(self) -> None:
        """Отменить активный таймер автозакрытия, если он есть."""
        if self._auto_close_timer is not None:
            self._auto_close_timer.cancel()
            self._auto_close_timer = None

    def _auto_close(self) -> None:
        """Колбэк таймера автозакрытия."""
        with self._lock_mutex:
            # За время ожидания таймера могли вызвать close() вручную
            if self._locked:
                return
            print("[lock] срабатывание автозакрытия")
            self._do_close()
            self._auto_close_timer = None
