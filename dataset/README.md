# Датасет

В этой папке должен лежать YOLO-датасет для обучения.

## Шаги подготовки

1. Зайдите на [roboflow.com](https://roboflow.com) и откройте свой проект.
2. Перейдите в раздел **Versions** → нужная версия → **Download Dataset**.
3. В диалоге выгрузки выберите формат **YOLOv11** (он совместим с ultralytics).
4. Скачайте zip-архив и распакуйте его прямо в эту папку (`dataset/`).
5. Проверьте, что в результате получилась структура:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/                 # (опционально)
│   ├── images/
│   └── labels/
└── data.yaml             # описание классов и путей
```

Содержимое `data.yaml` должно выглядеть примерно так:

```yaml
train: ../train/images
val:   ../valid/images
test:  ../test/images

nc: 3                          # число классов
names: ['cola', 'fanta', 'tea']
```

## Запуск обучения

```bash
python train/train.py                       # параметры по умолчанию
python train/train.py --epochs 200 --batch 32
```

Скрипт автоматически найдёт `data.yaml` рекурсивно и сохранит лучший
чекпоинт в `models/best.pt`.

## Что делать дальше

- Хотите ускорить инференс на Pi 5 без HAT — экспортируйте в NCNN:
  ```bash
  yolo export model=models/best.pt format=ncnn
  ```
- Хотите запустить на Hailo HAT — конвертируйте в HEF:
  ```bash
  python train/export_hailo.py
  ```
  Скрипт распечатает готовую docker-команду для финального шага.
