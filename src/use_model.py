import torch
import numpy as np
import cv2
import mediapipe as mp
import os

# Библиотеки, взятые из интернета (вся работа с ними также взята из интернета)
from loguru import logger as log
from lib.emotion_net import EmotionNet

from lib.dataset import get_data
from lib.landmark import preprocess_image
from lib import encoder

DATATEST_DIR = os.environ['DATATEST_DIR']  # Директория с тестовыми изображениями

MODELS_DIR = os.environ['MODELS_DIR']  # Директория с моделью
MODEL_FILE = os.path.join(MODELS_DIR, os.environ['MODEL_FILE'])  # Файл модели
SCALER_MEAN = os.path.join(MODELS_DIR, os.environ['SCALER_MEAN'])  # Сохранения параметров данных (среднее по трем каналам)
SCALER_SCALE = os.path.join(MODELS_DIR, os.environ['SCALER_SCALE'])  # Сохранения параметров масштабирования данных
LABEL_CLASSES = os.path.join(MODELS_DIR, os.environ['LABEL_CLASSES'])  # Файл с индексами эмоций


input_size = 1918  # Количество точек для распознавания
num_classes = 4  # Количество категорий (эмоций)

# Инициализация модели
model = EmotionNet(input_size, num_classes)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
model.eval()

# Масштабироание данных
scaler_mean = np.load(SCALER_MEAN)
scaler_scale = np.load(SCALER_SCALE)


# Предсказание эмоций с потока изображение при помощи заранее обученой модели
def predict_emotion(landmarks):
    try:
        # Нормализация данных
        landmarks_normalized = (landmarks - scaler_mean) / scaler_scale
        input_tensor = torch.tensor(landmarks_normalized, dtype=torch.float32).unsqueeze(0)

        # Определение вероятности итоговой эмоции
        with torch.no_grad():
            outputs = model(input_tensor)
            # Преобразует выводы модели в вероятность
            probabilities = torch.softmax(outputs, dim=1).numpy().flatten()
            predicted_class_idx = np.argmax(probabilities)
            # Итоговая вероятность предсказанной эмоции
            confidence = probabilities[predicted_class_idx]

        # Определение эмоции
        predicted_emotion = encoder.decode(predicted_class_idx)

        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
        }

    except ValueError as e:
        log.error(f'Error processing the image: {e}')
        return {
            'error': str(e),
            'predicted_emotion': None,
            'confidence': 0,
            'ranked_classes': []
        }


def main():

    for emotion, files in get_data(DATATEST_DIR):

        for file in files:
            try:
                # Преобразование изображения в набор координат ключевых точек.
                landmarks = preprocess_image(image=file)
                # Распознание эмоции
                result = predict_emotion(landmarks)

                print('========================RESULTS===================================================')
                print(f'File: {file}')
                print(f'Original Emotion: {emotion}')
                print(f"Predicted Emotion: {result['predicted_emotion']}")
                print(f"Confidence: {result['confidence'] * 100:.2f}%")

            except Exception as e:
                log.error(f"Unexpected error in emotion detection: {e}")


if __name__ == '__main__':
    main()
