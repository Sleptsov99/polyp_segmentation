# Polyp Segmentation with U-Net

Бинарная сегментация полипов на колоноскопических снимках с помощью U-Net + ResNet34.

![Results](results.png)

---

## Результаты обучения

| Метрика | Значение |
|---|---|
| Best Val Dice | **0.9026** |
| Best Val Loss | **0.0942** |
| Эпох | 30 |

---

## Датасет

[Kvasir-SEG](https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation) — 1000 колоноскопических снимков с масками полипов, размеченными врачами.

- 800 изображений — обучение
- 200 изображений — валидация

---

## Архитектура

```
Входное изображение [3, 256, 256]
        ↓
   Энкодер: ResNet34 (предобучен на ImageNet)
   256×256 → 128×128 → 64×64 → 32×32
        ↓
   Декодер: U-Net decoder со skip connections
   32×32 → 64×64 → 128×128 → 256×256
        ↓
   Выходная маска [1, 256, 256]
```

**Почему U-Net + ResNet34:**
- ResNet34 уже умеет извлекать признаки (края, текстуры, формы) — не нужно обучать с нуля
- Skip connections передают детали из энкодера в декодер — маска получается чёткой

---

## Структура проекта

```
polyp-segmentation/
├── config.py       # гиперпараметры
├── dataset.py      # загрузка данных и аугментации
├── model.py        # архитектура U-Net
├── train.py        # цикл обучения
├── evaluate.py     # визуализация результатов
├── best_model.pth  # лучшие веса модели
└── results.png     # примеры предсказаний
```

---

## Гиперпараметры

| Параметр | Значение |
|---|---|
| Image size | 256×256 |
| Batch size | 8 |
| Epochs | 30 |
| Learning rate | 1e-4 |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |
| Val split | 20% |

---

## Аугментации (train)

- Horizontal Flip (p=0.5)
- Vertical Flip (p=0.5)
- Random Rotate 90° (p=0.5)
- Normalize (ImageNet mean/std)

---

## Окружение

| | |
|---|---|
| GPU | 8 ядер |
| CPU | 8 ядер |
| Encoder weights | ImageNet |

---

## Запуск

**Установка зависимостей:**
```bash
pip install torch torchvision opencv-python albumentations scikit-learn segmentation-models-pytorch matplotlib tqdm
```

**Обучение:**
```bash
python train.py
```

**Визуализация результатов:**
```bash
python evaluate.py
```
