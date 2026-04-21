# Smart Fridge OS — контекст для Claude

## О проекте

Прототип умного холодильника на Raspberry Pi 5 + Hailo AI HAT (13 TOPS).
Камера фиксирует продукты, пересекающие "линию портала" (имитация двери),
и эмитит события `taken` / `returned`. Электромагнитный замок управляется
через GPIO + реле; внешние клиенты подключаются по REST + WebSocket.

Это **переписанная с нуля версия** более старых прототипов:
- `../smart-fridge-os/smart-fridge-prototype/` — первый прототип с DINOv2
- `../smart-fridge-os/smart-fridge-v2/` — второй прототип с SSH-туннелем
- `./` — текущая чистая версия для Pi 5 + Hailo

## Железо

| Компонент          | Модель                                                       |
|--------------------|--------------------------------------------------------------|
| SBC                | Raspberry Pi 5 (4/8 GB), ARM Cortex-A76, **aarch64**         |
| AI-ускоритель      | Hailo AI HAT 13 TOPS (Hailo-8L) — **опционально**            |
| Камеры             | 2x IMX219 160° (MIPI CSI), либо USB для тестов               |
| Замок              | Электромагнитный 12 В + реле SRD-05VDC-SL-C (активно-низкое) |
| ОС                 | Raspberry Pi OS Bookworm 64-bit                              |

## Архитектура пайплайна

```
Camera (Picamera2/OpenCV)
    ↓ BGR frame
Detector (Hailo / NCNN / CPU — auto-fallback)
    ↓ list[Detection]
ByteTracker (IoU + lost-frames buffer)
    ↓ list[TrackedDetection] с persistent track_id
LineCrossingDetector
  ├─ HandSuppressor (MediaPipe Hands)  → пропуск кадра, если рука в зоне
  └─ CrossingBuffer (N подтверждений)  → событие taken/returned
    ↓
WebSocketHub.broadcast({type, event, product, ...})
    ↓
HTTP-клиенты подписаны на /ws
```

Параллельно работает `Lock` (gpiozero) — открывает/закрывает замок по
HTTP-запросу или автоматически через `threading.Timer`.

## Ключевая особенность — два режима инференса с автовыбором

```python
# detection/detector.py
try:
    from hailo_platform import HEF, VDevice
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
```

Режимы (`config.INFERENCE_MODE` или `--mode` в CLI):
- `auto` — Hailo (`.hef`) → NCNN (`{name}_ncnn_model/`) → CPU (`.pt`)
- `hailo` — только Hailo, ошибка при отсутствии
- `ncnn` — только NCNN
- `cpu` — только ultralytics на `.pt`

Один и тот же код работает на чистом Pi 5 и на Pi 5 + Hailo HAT.

## Структура проекта

```
smart-fridge-pi5/
├── main.py                  # точка входа: CV-цикл + сервер в фоновом потоке
├── config.py                # все параметры + env-vars с префиксом SF_
├── camera/capture.py        # Picamera2 / OpenCV с автовыбором, контекст-менеджер
├── detection/
│   ├── detector.py          # Hailo / NCNN / CPU, _HailoRunner внутри
│   ├── tracker.py           # IoU-трекер с lost-frames (упрощённый ByteTrack)
│   └── crossing.py          # CrossingBuffer + HandSuppressor + дедупликация
├── hardware/lock.py         # gpiozero, активно-низкое реле, авто-закрытие
├── server/
│   ├── api.py               # FastAPI: /health, /status, /products, /lock/*
│   └── websocket.py         # WebSocketHub — broadcast из CV-потока
├── train/
│   ├── train.py             # ultralytics YOLO('yolo11n.pt').train(...)
│   └── export_hailo.py      # .pt → .onnx → готовая команда docker для HEF
├── dataset/                 # сюда распаковать Roboflow YOLOv11 export
├── models/                  # best.pt, best.hef, best_ncnn_model/
├── requirements-pi.txt      # для Pi 5 (без torch/picamera2/RPi.GPIO/hailort)
├── requirements-dev.txt     # для Windows/Linux x86 разработки
└── install.sh               # установка на Pi 5 (apt + pip --break-system-packages)
```

## Технологический стек

- **Язык**: Python 3.11+
- **Детекция**: Ultralytics YOLOv11n (`.pt` → `.onnx`/`.hef`/`ncnn`)
- **Трекинг**: собственный IoU-трекер (упрощённый ByteTrack, без зависимости)
- **Подавление**: MediaPipe Hands ≥ 0.10.14 (поддерживает aarch64)
- **GPIO**: gpiozero + lgpio (без root, `/dev/gpiochip*`)
- **Камера**: Picamera2 (MIPI CSI), OpenCV (USB, dev)
- **Сервер**: FastAPI + uvicorn, WebSocket для real-time событий
- **Hailo**: hailort (отдельный .deb пакет, не через pip)

## Конвенции

### ARM-совместимость пакетов

Файлы `requirements-pi.txt` содержит **только** пакеты с готовыми wheels
для aarch64. Намеренно исключены:
- `torch`/`torchvision` — ставится отдельно с `--index-url https://download.pytorch.org/whl/cpu`
- `picamera2` — только через `apt install python3-picamera2` (нет в PyPI)
- `RPi.GPIO`/`lgpio` — через apt (системные пакеты)
- `hailort` — отдельный `.deb` пакет от Hailo

### Конфигурация через env-переменные

Все параметры в [config.py](config.py) переопределяются через переменные
окружения с префиксом `SF_`:

```bash
SF_SERVER_PORT=9000 SF_GPIO_MOCK=true python main.py
```

### Mock-режимы для разработки на Windows

- `Lock` автоматически переходит в мок при отсутствии gpiozero/lgpio
- `Camera` падает с Picamera2 на OpenCV
- `Detector` в режиме `auto` падает с Hailo на NCNN/CPU
- `HandSuppressor` отключается, если mediapipe не установлен

Это позволяет запускать `python main.py` на Windows для отладки сервера/API.

### Реле SRD-05VDC-SL-C — активно-низкое

```
GPIO LOW   → реле включено  → замок под напряжением → ОТКРЫТ
GPIO HIGH  → реле выключено → замок без питания    → ЗАКРЫТ
```

В коде это `LOCK_ACTIVE_LOW=True` и `gpiozero.OutputDevice(active_high=False, initial_value=False)` —
гарантирует, что замок закрыт сразу после конфигурирования пина (нет
"щелчка" при загрузке).

## Запуск

### На Pi 5

```bash
./install.sh                    # установка (один раз)
python3 main.py                 # CV + сервер
python3 main.py --show          # + окно OpenCV
python3 main.py --mode hailo    # принудительно Hailo
```

### На dev-машине

```bash
pip install -r requirements-dev.txt
pip install torch torchvision   # отдельно

python main.py --no-cv          # только сервер для отладки API
python train/train.py           # обучение модели
python train/export_hailo.py    # экспорт .pt → .onnx + инструкция HEF
```

## API

| Метод | Путь          | Описание                                           |
|-------|---------------|----------------------------------------------------|
| GET   | /health       | Ping без авторизации                               |
| GET   | /status       | Замок, активная модель, число WS-клиентов          |
| GET   | /products     | Имена классов из загруженной модели                |
| POST  | /lock/open    | Открыть замок (тело: `{"auto_close_sec": 5}`)      |
| POST  | /lock/close   | Принудительно закрыть                              |
| WS    | /ws           | Поток событий: crossing (taken/returned) + lock    |

Авторизация через `X-API-Key` включается заданием `SF_SERVER_API_KEY`.
`/health` и `/ws` всегда доступны без ключа.

## Датасет

YOLOv8 и YOLOv11 используют **одинаковый формат** — экспорт Roboflow
"YOLOv8" / "YOLOv11" взаимозаменяемы. Базовая модель в [train/train.py](train/train.py)
по умолчанию `yolo11n.pt`.

Структура после распаковки:
```
dataset/
├── train/{images,labels}/
├── valid/{images,labels}/
├── test/{images,labels}/   # опционально
└── data.yaml
```

`train/train.py` ищет `data.yaml` рекурсивно в `dataset/` через `rglob`.

## Что НЕ перенесено из старых прототипов

- DINOv2-классификатор (`product_classifier.py` из v1) — заменён на
  кастомное обучение YOLOv11n на нужных классах
- Ollama LLaVA фоллбэк — не нужен при кастомной модели
- Reed switch door sensor — не было в требованиях, можно добавить как
  опциональный класс по аналогии с `Lock`
- WebClient + SSE для удалённого сервера — не было полного описания
  целевого web-API, можно добавить позже
- Sound player (pygame) — не было в требованиях

## Полезные команды

```bash
# Проверить, видит ли система Hailo HAT
hailortcli scan

# Проверить камеру MIPI CSI
libcamera-hello -t 2000

# Тест GPIO без подключения железа
SF_GPIO_MOCK=true python main.py --no-cv

# Экспорт в NCNN для ускорения на Pi без Hailo
yolo export model=models/best.pt format=ncnn

# Отладка WebSocket
websocat ws://pi.local:8000/ws
```

## Troubleshooting

- **Picamera2 не открывается** → проверьте `libcamera-hello`, шлейф CSI,
  включение камеры в `raspi-config`
- **gpiozero ошибка backend** → на Pi 5 нужен `python3-lgpio` (apt)
- **Hailo не определяется** → `hailortcli scan`, переустановите драйвер,
  перезагрузитесь
- **MediaPipe ModuleNotFoundError на Pi** → нужна версия ≥ 0.10.14
- **Низкий FPS на CPU** → экспорт в NCNN или `MODEL_IMGSZ=320`
- **Замок не реагирует** → проверьте перемычку JD-VCC и `LOCK_ACTIVE_LOW`
