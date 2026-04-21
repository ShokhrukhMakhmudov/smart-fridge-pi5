"""
Экспорт обученной модели в формат Hailo HEF.

Процесс состоит из двух шагов:

  Шаг 1 (скрипт делает сам): best.pt → best.onnx
         Используем ultralytics export() с opset=13, simplify=True.

  Шаг 2 (выполняется в Docker'е Hailo SDK): best.onnx → best.hef
         Hailo Dataflow Compiler (DFC) принимает ONNX, оптимизирует,
         квантизирует и собирает .hef для устройства Hailo-8L.
         DFC — отдельная программа, ставится через .deb пакеты или
         запускается в готовом Docker'е hailo_ai_sw_suite.

Запуск:
    python train/export_hailo.py
    python train/export_hailo.py --weights models/best.pt --imgsz 640

После выполнения скрипт печатает готовую docker-команду для шага 2.
Запускать её следует на машине, где установлен Hailo SDK
(не обязательно на Pi, обычно на dev-машине с Linux + Docker).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS_DIR  # noqa: E402

# Калибровочный набор — нужен компилятору для квантизации int8.
# Достаточно ~64-128 типичных кадров; обычно берут случайную выборку из train/.
CALIBRATION_HINT: str = """\
# В Docker'е Hailo нужен калибровочный набор для квантизации.
# Подготовьте папку с 64-128 типичными изображениями (640x640) и смонтируйте
# её в контейнер. Чем разнообразнее набор, тем выше точность int8-модели.
"""


def export_to_onnx(weights: Path, imgsz: int) -> Path:
    """Шаг 1: экспортировать best.pt в ONNX."""
    from ultralytics import YOLO  # noqa: PLC0415

    print(f"[export] Загружаю модель: {weights}")
    model = YOLO(str(weights))

    print(f"[export] Экспортирую в ONNX (imgsz={imgsz}, opset=13, simplify=True)")
    onnx_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=13,
        simplify=True,
        dynamic=False,
    )
    onnx_path: Path = Path(onnx_path_str)
    print(f"[export] ONNX готов: {onnx_path}")
    return onnx_path


def print_hailo_instructions(onnx_path: Path, imgsz: int) -> None:
    """Шаг 2: вывести готовую docker-команду для компиляции в HEF."""
    onnx_abs: Path = onnx_path.resolve()
    out_dir: Path = onnx_abs.parent
    hef_name: str = onnx_abs.stem + ".hef"

    print("\n" + "=" * 70)
    print("  ШАГ 2: компиляция ONNX → HEF в Docker'е Hailo SDK")
    print("=" * 70)
    print(CALIBRATION_HINT)

    print("Подготовьте каталог с калибровочными PNG/JPG (64-128 шт), например:")
    print(f"  mkdir -p {out_dir}/calib && cp /path/to/train/*.jpg {out_dir}/calib/\n")

    print("Затем запустите Docker (на dev-машине с установленным Hailo SDK):\n")
    print("  docker run --rm -it \\")
    print(f"    -v {out_dir}:/workspace \\")
    print("    hailo_ai_sw_suite:latest \\")
    print("    bash -c '\\")
    print(f"      hailomz compile yolov11n \\")
    print(f"        --ckpt /workspace/{onnx_abs.name} \\")
    print(f"        --calib-path /workspace/calib \\")
    print(f"        --hw-arch hailo8l \\")
    print(f"        --classes <NUM_CLASSES> \\")
    print(f"        --output-dir /workspace \\")
    print("    '")
    print()
    print(f"После успешной компиляции файл будет здесь:")
    print(f"  {out_dir / hef_name}")
    print()
    print("Скопируйте его на Pi 5 в models/ и укажите путь в config.py:")
    print(f"  MODEL_HEF_PATH = \"{MODELS_DIR / hef_name}\"")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт YOLOv11n в ONNX + инструкция по HEF")
    parser.add_argument("--weights", default=str(MODELS_DIR / "best.pt"),
                        help="Путь к обученным весам .pt")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Размер входа модели")
    args = parser.parse_args()

    weights: Path = Path(args.weights)
    if not weights.exists():
        print(f"[export] Файл весов не найден: {weights}")
        print(f"[export] Сначала обучите модель: python train/train.py")
        sys.exit(1)

    onnx_path: Path = export_to_onnx(weights, args.imgsz)
    print_hailo_instructions(onnx_path, args.imgsz)


if __name__ == "__main__":
    main()
