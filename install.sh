#!/usr/bin/env bash
# Установка Smart Fridge OS на Raspberry Pi 5 (Raspberry Pi OS Bookworm 64-bit).
#
# Что делает скрипт:
#   1. Обновляет apt-индекс
#   2. Ставит системные зависимости: libcamera, picamera2, RPi.GPIO, инструменты сборки
#   3. Ставит PyTorch отдельно с CPU-индекса (нет ARM-wheel в основном PyPI)
#   4. Ставит остальные Python-пакеты из requirements-pi.txt
#   5. Печатает следующие шаги (Hailo SDK + загрузка модели)
#
# Запуск:
#     chmod +x install.sh
#     ./install.sh
#
# По умолчанию используется флаг --break-system-packages, потому что
# Raspberry Pi OS Bookworm запрещает глобальный pip без него (PEP 668).
# Если предпочитаете venv — раскомментируйте блок VENV ниже.

set -e  # любая ошибка прерывает скрипт
set -u  # не ссылаться на неопределённые переменные

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[install]${NC} $*"; }
warn() { echo -e "${YELLOW}[install]${NC} $*"; }
err()  { echo -e "${RED}[install]${NC} $*" >&2; }

# ── Проверка платформы ───────────────────────────────────────────
ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    warn "Архитектура $ARCH не aarch64. Скрипт рассчитан на Pi 5 — продолжаю на свой страх и риск."
fi

# ── Шаг 1. apt update ────────────────────────────────────────────
log "Обновляю apt-индекс…"
sudo apt update

# ── Шаг 2. Системные пакеты ──────────────────────────────────────
log "Ставлю системные зависимости (libcamera, picamera2, gpio, сборка)…"
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-picamera2 \
    python3-libcamera \
    python3-rpi.gpio \
    python3-lgpio \
    libcamera-apps \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    libtiff-dev \
    git \
    build-essential \
    cmake

# ── (опционально) virtualenv ─────────────────────────────────────
# Если хотите изолированное окружение — раскомментируйте и уберите
# флаг --break-system-packages из вызовов pip ниже.
#
# VENV_DIR=".venv"
# if [[ ! -d "$VENV_DIR" ]]; then
#     log "Создаю venv в $VENV_DIR…"
#     python3 -m venv --system-site-packages "$VENV_DIR"
# fi
# # shellcheck disable=SC1091
# source "$VENV_DIR/bin/activate"
# PIP_FLAGS=""
PIP_FLAGS="--break-system-packages"

# ── Шаг 3. PyTorch (отдельный индекс для CPU/aarch64) ────────────
log "Ставлю PyTorch (CPU-сборка для aarch64)…"
pip install $PIP_FLAGS torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ── Шаг 4. Остальные пакеты ──────────────────────────────────────
log "Ставлю Python-зависимости из requirements-pi.txt…"
pip install $PIP_FLAGS -r requirements-pi.txt

# ── Шаг 5. Проверка ──────────────────────────────────────────────
log "Проверка установленных компонентов:"
python3 - <<'PY'
import importlib

modules = [
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("ultralytics", "Ultralytics"),
    ("mediapipe", "MediaPipe"),
    ("gpiozero", "gpiozero"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("torch", "PyTorch"),
    ("picamera2", "Picamera2"),
]
for mod, label in modules:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  ✓ {label:12} {ver}")
    except Exception as e:
        print(f"  ✗ {label:12} НЕ установлен ({e})")

try:
    from hailo_platform import VDevice
    print("  ✓ hailo_platform доступен (HAT обнаружен)")
except ImportError:
    print("  · hailo_platform НЕ установлен — будет работать NCNN/CPU фоллбэк")
PY

cat <<'EOF'

══════════════════════════════════════════════════════════════════════
  Установка завершена.

  Дальнейшие шаги:

  1) (опционально) Установить Hailo SDK для использования HAT:
       Скачайте hailort .deb с https://hailo.ai/developer-zone/
       sudo apt install ./hailort_*_arm64.deb
       sudo apt install ./hailort-pcie-driver_*_all.deb
       sudo reboot

  2) Положить обученную модель в models/best.pt
     (или models/best.hef для Hailo). См. README.md.

  3) Запустить:
       python3 main.py --show         # с окном превью
       python3 main.py --no-server    # только CV без HTTP
══════════════════════════════════════════════════════════════════════
EOF
