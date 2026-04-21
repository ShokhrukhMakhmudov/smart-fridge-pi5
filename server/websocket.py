"""
Хаб WebSocket-подписчиков для рассылки событий в реальном времени.

Один WebSocketHub держит множество клиентов и в фоне получает события
из CV-пайплайна (продукт взят/возвращён) и от замка (открыт/закрыт).
Каждое событие — это словарь, который сериализуется в JSON и отправляется
всем активным подписчикам.

Соединения, упавшие во время отправки, тихо удаляются — мёртвые клиенты
не должны блокировать живых.
"""

import asyncio
import json
from typing import Optional

from fastapi import WebSocket


class WebSocketHub:
    """Регистр активных WebSocket-клиентов с broadcast()."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._mutex: asyncio.Lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Принять входящее соединение и добавить в реестр."""
        await websocket.accept()
        async with self._mutex:
            self._clients.add(websocket)
        print(f"[ws] клиент подключился, активно: {len(self._clients)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Удалить клиента из реестра. Идемпотентно."""
        async with self._mutex:
            self._clients.discard(websocket)
        print(f"[ws] клиент отключился, активно: {len(self._clients)}")

    async def broadcast(self, message: dict) -> None:
        """Отправить JSON всем подписчикам, выбросив дохлых."""
        text: str = json.dumps(message, ensure_ascii=False)
        dead: list[WebSocket] = []

        async with self._mutex:
            clients = list(self._clients)

        for ws in clients:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._mutex:
                for ws in dead:
                    self._clients.discard(ws)
            print(f"[ws] удалено мёртвых соединений: {len(dead)}")

    def broadcast_threadsafe(
        self,
        message: dict,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """
        Отправить событие из НЕ-asyncio потока (например, из CV-цикла).

        Если loop не передан — пытается взять текущий event loop. Если его
        нет (вызов из произвольного потока) — событие просто пропускается:
        у CV-пайплайна нет смысла блокироваться из-за отсутствующего сервера.
        """
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
        except Exception as e:
            print(f"[ws] broadcast_threadsafe сбой: {e}")

    @property
    def client_count(self) -> int:
        return len(self._clients)
