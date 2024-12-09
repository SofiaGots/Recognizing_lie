import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import os

# Библиотеки, взятые из интернета (вся работа с ними также взята из интернета)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.emotion_net import EmotionNet
from lib import encoder


DATASET_DIR = os.environ['DATASET_DIR'] # Путь к исходному набору данных
MODELS_DIR = os.environ['MODELS_DIR'] # Путь к директории, где будут храниться результаты обучения
DATASET_RES = os.path.join(MODELS_DIR, os.environ['DATASET_RES']) # CSV файл с предварительно обработанными данными (эмоции + признаки лица)

MODEL_FILE = os.path.join(MODELS_DIR, os.environ['MODEL_FILE']) # Путь для сохранения обученной модели
SCALER_MEAN = os.path.join(MODELS_DIR, os.environ['SCALER_MEAN']) # Сохранения параметров данных (среднее по трем каналам)
SCALER_SCALE = os.path.join(MODELS_DIR, os.environ['SCALER_SCALE']) # Сохранения параметров масштабирования данных

RANDOM_STATE = int(os.environ['RANDOM_STATE']) # начальный параметр для повторяемости экспериментов (чтобы люди, бравшие данную модель, имели такие же результаты обучения)


# Загрузка и подготовка данных
data = pd.read_csv(DATASET_RES, header=None)
landmarks = data.iloc[:, 1:].values  # Признаки
emotion = data.iloc[:, 0].values   # Эмоции

# Масштабирование данных
# Данный код скопирован из интернета, так как я не знала функцию масштабирования
scaler = StandardScaler() # Масштабирует данные так, чтобы они имели среднее 0 и стандартное отклонение 1
landmarks_normalized = scaler.fit_transform(landmarks)
np.save(SCALER_MEAN, scaler.mean_)
np.save(SCALER_SCALE, scaler.scale_)


# Разделение данных (на тестовые и обучающие)
X_train, X_test, y_train, y_test = train_test_split(landmarks_normalized, emotion, test_size=0.2, random_state=RANDOM_STATE)

# Преобразование в PyTorch тензоры
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Инициализация модели
input_size = X_train.shape[1]
num_classes = len(encoder.category)
model = EmotionNet(input_size, num_classes)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # Прогон тренировочных данных через модель
    outputs = model(X_train)

    # Вычисление ошибки предсказания
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Оценка работы модели
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = torch.argmax(y_pred, axis=1)
    # Точность правильных предсказываний
    accuracy = (y_pred_classes == y_test).sum().item() / y_test.size(0)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # # Матрица ошибок (можно раскоментировать, чтобы посмотреть на результат работы)
    # cm = confusion_matrix(y_test.numpy(), y_pred_classes.numpy())
    # sns.heatmap(cm, annot=True, fmt='d')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.show()

# Сохранение модели
torch.save(model.state_dict(), MODEL_FILE)
