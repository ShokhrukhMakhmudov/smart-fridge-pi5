"""
HTTP API на FastAPI.

Эндпоинты:
  GET  /health         — простой ping для load-balancer'а / supervisor'а
  GET  /status         — общий статус: замок, сервер, активная модель
  GET  /products       — список классов (имён продуктов) из загруженной модели
  POST /lock/open      — открыть замок (опционально с auto_close_sec)
  POST /lock/close     — закрыть замок принудительно
  WS   /ws             — поток событий (продукт взят/возвращён, статус замка)

Авторизация: если в конфиге задан SF_SERVER_API_KEY, защищённые роуты
требуют заголовок X-API-Key. /health и /ws доступны без ключа.
"""

from typing import Optional

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field

from config import LOCK_AUTO_CLOSE_SEC, SERVER_API_KEY
from hardware.lock import Lock

from .websocket import WebSocketHub


# ── Pydantic-схемы ───────────────────────────────────────────────

class OpenLockRequest(BaseModel):
    """Тело запроса на открытие замка."""
    auto_close_sec: int = Field(
        default=LOCK_AUTO_CLOSE_SEC,
        ge=0,
        le=300,
        description="Через сколько секунд закрыть автоматически (0 = не закрывать)",
    )


class StatusResponse(BaseModel):
    """Общий статус холодильника."""
    lock: dict
    server: dict
    model: dict


# ── Авторизация ──────────────────────────────────────────────────

def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Проверка API-ключа. Если SERVER_API_KEY не задан — авторизация выключена,
    что удобно для локальной разработки. В проде обязательно задайте ключ.
    """
    if SERVER_API_KEY is None:
        return
    if x_api_key != SERVER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный или отсутствующий API-ключ",
        )


# ── Фабрика приложения ──────────────────────────────────────────

def build_app(
    lock: Lock,
    hub: WebSocketHub,
    model_info: dict,
    products: list[str],
) -> FastAPI:
    """
    Собрать FastAPI приложение с прокинутыми зависимостями.

    Аргументы:
        lock:        экземпляр Lock (управление замком)
        hub:         реестр WebSocket-клиентов для broadcast'а
        model_info:  словарь с info о модели (mode, path, imgsz, conf)
        products:    список имён классов из модели
    """
    app = FastAPI(
        title="Smart Fridge OS API",
        description="REST + WebSocket API для умного холодильника на Raspberry Pi 5",
        version="1.0.0",
    )

    # ── Простые роуты ────────────────────────────────────────────

    @app.get("/health")
    def health() -> dict:
        """Лёгкая проверка живости — без авторизации."""
        return {"status": "ok"}

    @app.get("/status", response_model=StatusResponse)
    def status_endpoint(_: None = Depends(verify_api_key)) -> StatusResponse:
        return StatusResponse(
            lock=lock.status(),
            server={
                "ws_clients": hub.client_count,
                "auth_enabled": SERVER_API_KEY is not None,
            },
            model=model_info,
        )

    @app.get("/products")
    def get_products(_: None = Depends(verify_api_key)) -> dict:
        """Список имён классов, на которые натренирована модель."""
        return {"products": products, "count": len(products)}

    # ── Управление замком ────────────────────────────────────────

    @app.post("/lock/open")
    async def open_lock(
        body: OpenLockRequest = OpenLockRequest(),
        _: None = Depends(verify_api_key),
    ) -> dict:
        lock.open(auto_close_sec=body.auto_close_sec)
        result = lock.status()
        # Сразу оповестим всех ws-подписчиков о смене состояния
        await hub.broadcast({"type": "lock", "action": "open", "status": result})
        return result

    @app.post("/lock/close")
    async def close_lock(_: None = Depends(verify_api_key)) -> dict:
        lock.close()
        result = lock.status()
        await hub.broadcast({"type": "lock", "action": "close", "status": result})
        return result

    # ── WebSocket-канал событий ──────────────────────────────────

    @app.websocket("/ws")
    async def ws_events(websocket: WebSocket) -> None:
        """
        Поток событий в реальном времени.

        Сервер пушит сообщения вида:
            {"type": "crossing", "event": "taken", "product": "cola", ...}
            {"type": "lock",     "action": "open",  "status": {...}}

        Клиент может ничего не присылать — на любые входящие сообщения
        мы отвечаем pong'ом, чтобы соединение не разрывалось keep-alive'ом.
        """
        await hub.connect(websocket)
        try:
            while True:
                msg = await websocket.receive_text()
                # Простейший keep-alive: ответим тем же текстом или pong'ом
                await websocket.send_text(msg if msg else "pong")
        except WebSocketDisconnect:
            pass
        finally:
            await hub.disconnect(websocket)

    return app
