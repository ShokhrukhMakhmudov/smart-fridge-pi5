# Smart Fridge OS

Прототип умного холодильника на Raspberry Pi 5 с компьютерным зрением.
Камера фиксирует продукты, пересекающие "линию портала" (имитация двери),
и эмитит события `taken` / `returned`. Электромагнитный замок управляется
через GPIO + реле; внешние клиенты подключаются по REST + WebSocket.

## Возможности

- **Два режима инференса с автовыбором** — Hailo HAT 13 TOPS (.hef через hailort)
  или NCNN/CPU (через ultralytics). Один и тот же код работает на чистом Pi 5
  и на Pi 5 + Hailo HAT.
- **YOLOv11n + ByteTrack-подобный трекер** — устойчивые ID между кадрами.
- **CrossingBuffer + MediaPipe Hands** — события подтверждаются за N кадров,
  кадры с рукой в зоне игнорируются → меньше ложных срабатываний.
- **Электромагнитный замок** через реле SRD-05VDC-SL-C (gpiozero, без root)
  с автозакрытием по таймеру и мок-режимом для разработки.
- **FastAPI + WebSocket** — статус, открытие/закрытие двери, поток событий.
- **Готовые скрипты обучения и экспорта** — `train/train.py` и
  `train/export_hailo.py` (ONNX → инструкции для Docker'а Hailo SDK).

---

## 1. Требования к железу

| Компонент          | Модель / характеристики                                       |
|--------------------|---------------------------------------------------------------|
| SBC                | Raspberry Pi 5 (4/8 GB), ARM Cortex-A76, aarch64              |
| AI-ускоритель      | Hailo AI HAT 13 TOPS (Hailo-8L) — **опционально**             |
| Камеры             | 2x IMX219 160° (MIPI CSI), либо USB-камера для тестов         |
| Замок              | Электромагнитный 12 В + реле SRD-05VDC-SL-C (активно-низкое)  |
| Питание            | 5 В / 5 A для Pi 5, отдельный 12 В для замка                  |
| ОС                 | Raspberry Pi OS Bookworm 64-bit                               |

**Подключение реле к замку** см. в комментариях [hardware/lock.py](hardware/lock.py).

---

## 2. Установка на Raspberry Pi 5

```bash
git clone <repo-url> smart-fridge-os
cd smart-fridge-os
chmod +x install.sh
./install.sh
```

Скрипт:
1. ставит системные пакеты через `apt` (`picamera2`, `libcamera`, `python3-rpi.gpio`);
2. ставит `torch`/`torchvision` с CPU-индекса PyTorch (нет ARM-wheels в основном PyPI);
3. ставит остальное из `requirements-pi.txt` (с флагом `--break-system-packages`,
   как требует Bookworm + PEP 668).

### Установка Hailo SDK (опционально, для HAT)

```bash
# Скачайте архив с https://hailo.ai/developer-zone/ и установите .deb пакеты
sudo apt install ./hailort_*_arm64.deb
sudo apt install ./hailort-pcie-driver_*_all.deb
sudo reboot
```

После перезагрузки `python -c "from hailo_platform import VDevice"` должен
работать без ошибок.

---

## 3. Установка для разработки (Windows / Linux x86)

```bash
git clone <repo-url> smart-fridge-os
cd smart-fridge-os
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements-dev.txt
# Установите torch отдельно — выберите вариант под ваше железо:
pip install torch torchvision                                                # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA
```

На dev-машине:
- GPIO работает в мок-режиме автоматически (`gpiozero` отсутствует или нет пинов).
- Камера читается через OpenCV `VideoCapture(0)` (USB-камера).
- Hailo недоступен → `--mode auto` упадёт на NCNN или CPU.

---

## 4. Загрузка датасета из Roboflow

См. [dataset/README.md](dataset/README.md). Если коротко:
1. Roboflow → Versions → Download Dataset → формат **YOLOv11**.
2. Распаковать в `dataset/`. После должна появиться `dataset/.../data.yaml`.

---

## 5. Обучение модели

Запускайте на dev-машине (на Pi 5 обучать не стоит — слишком долго):

```bash
python train/train.py                          # 100 эпох, batch 16, imgsz 640
python train/train.py --epochs 200 --batch 32  # больше эпох/батч
python train/train.py --device 0               # GPU 0 (CUDA)
```

Лучший чекпоинт автоматически копируется в `models/best.pt`.

---

## 6. Конвертация в Hailo HEF

Шаг 1 (на dev-машине) — экспорт в ONNX:

```bash
python train/export_hailo.py
```

Шаг 2 — компиляция ONNX → HEF в Docker'е Hailo SDK. Скрипт распечатает
готовую `docker run …` команду, которую нужно выполнить на машине с
установленным `hailo_ai_sw_suite`. После компиляции скопируйте `.hef` на Pi
в `models/best.hef`.

Альтернатива для Pi без HAT — экспорт в NCNN:

```bash
yolo export model=models/best.pt format=ncnn
```

В результате рядом с `best.pt` появится папка `best_ncnn_model/`, которую
детектор подхватит автоматически.

---

## 7. Запуск проекта

```bash
python main.py                  # CV + REST/WS сервер на 0.0.0.0:8000
python main.py --show           # + окно OpenCV с визуализацией
python main.py --mode hailo     # принудительно Hailo (ошибка, если HAT нет)
python main.py --mode ncnn      # принудительно NCNN
python main.py --mode cpu       # принудительно CPU (медленно)
python main.py --no-server      # только CV-цикл
python main.py --no-cv          # только сервер (для отладки API)
```

Полезные переменные окружения (см. [config.py](config.py)):

```bash
SF_SERVER_PORT=9000 \
SF_GPIO_MOCK=true \
SF_SERVER_API_KEY=secret \
python main.py
```

---

## 8. API

| Метод | Путь            | Описание                                            |
|-------|-----------------|-----------------------------------------------------|
| GET   | `/health`       | Простой ping (без авторизации)                      |
| GET   | `/status`       | Замок, активная модель, число WS-клиентов           |
| GET   | `/products`     | Имена классов из загруженной модели                 |
| POST  | `/lock/open`    | Открыть замок (тело: `{"auto_close_sec": 5}`)       |
| POST  | `/lock/close`   | Закрыть замок                                       |
| WS    | `/ws`           | Поток событий (crossing/lock)                       |

Если в окружении задан `SF_SERVER_API_KEY`, защищённые роуты требуют
заголовок `X-API-Key: <ключ>`. `/health` и `/ws` всегда доступны без ключа.

Пример работы с API:

```bash
curl http://pi.local:8000/status
curl -X POST http://pi.local:8000/lock/open \
     -H "Content-Type: application/json" \
     -d '{"auto_close_sec": 10}'

# WebSocket — например, через websocat
websocat ws://pi.local:8000/ws
# {"type": "crossing", "event": "taken", "product": "cola", "track_id": 7, ...}
```

---

## 9. Troubleshooting

**`Picamera2 не смогла открыться`** — убедитесь, что `libcamera-hello`
работает (`libcamera-hello -t 2000`). Если нет — проверьте подключение
шлейфа CSI и включён ли камерный интерфейс в `raspi-config`.

**`gpiozero`: ошибка `pigpio` / `lgpio`** — на Pi 5 используется `lgpio`
backend. Установите `python3-lgpio` через apt и перезапустите процесс.

**Hailo не определяется** — проверьте `hailortcli scan`. Если устройство
не найдено, переустановите `hailort-pcie-driver` и перезагрузитесь.

**`mediapipe`: `ModuleNotFoundError`** — на Pi 5 нужна версия ≥ 0.10.14
(она поддерживает aarch64). Старые версии падают с ошибкой архитектуры.

**Низкий FPS на CPU** — экспортируйте модель в NCNN или уменьшите
`MODEL_IMGSZ` в `config.py` (320 вместо 640 ускоряет в 2-4 раза).

**Замок не реагирует** — проверьте перемычку JD-VCC на реле и опцию
`LOCK_ACTIVE_LOW` в `config.py`. Активно-низкие реле (как SRD-05VDC-SL-C)
требуют `True`.

---

## Структура проекта

```
smart-fridge-os/
├── main.py                  # точка входа: CV + сервер
├── config.py                # все настройки в одном месте (+ env-vars SF_*)
├── camera/
│   └── capture.py           # Picamera2 / OpenCV с автовыбором
├── detection/
│   ├── detector.py          # Hailo / NCNN / CPU инференс
│   ├── tracker.py           # IoU-трекер с потерянными треками
│   └── crossing.py          # CrossingBuffer + MediaPipe Hands
├── hardware/
│   └── lock.py              # gpiozero + автозакрытие + мок
├── server/
│   ├── api.py               # FastAPI: /status, /lock/*, /products
│   └── websocket.py         # WebSocketHub для broadcast
├── train/
│   ├── train.py             # ultralytics YOLO trainer
│   └── export_hailo.py      # .pt → .onnx → инструкция для Hailo SDK
├── dataset/                 # сюда распаковать датасет из Roboflow
├── models/                  # сюда положить best.pt / best.hef
├── requirements-pi.txt      # для Raspberry Pi 5 (aarch64)
├── requirements-dev.txt     # для dev-машины (Windows / Linux x86)
└── install.sh               # установочный скрипт для Pi 5
```
