import torch
import numpy as np
import cv2
import mediapipe as mp
import os

# Библиотека, взятая из интернета (вся работа с ней также взята из интернета)
from loguru import logger as log

from lib.emotion_net import EmotionNet
from lib.dataset import get_data
from lib.emotion_rec import classify_emotion

from lib.landmark import preprocess_image, get_face_coordinates
from lib import encoder

DATATEST_DIR = os.environ['DATATEST_DIR'] # Директория с тестовыми изображениями

MODELS_DIR = os.environ['MODELS_DIR'] # Директория с моделью
MODEL_FILE = os.path.join(MODELS_DIR, os.environ['MODEL_FILE']) # Файл модели
SCALER_MEAN = os.path.join(MODELS_DIR, os.environ['SCALER_MEAN']) # Сохранения параметров данных (среднее по трем каналам)
SCALER_SCALE = os.path.join(MODELS_DIR, os.environ['SCALER_SCALE']) # Сохранения параметров масштабирования данных
LABEL_CLASSES = os.path.join(MODELS_DIR, os.environ['LABEL_CLASSES']) # Файл с индексами эмоций


input_size = 1918  # Количество точек для распознавания
num_classes = 4 # Количество категорий (эмоций)

# Инициализация модели
model = EmotionNet(input_size, num_classes)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
model.eval()

# Масштабироание данных
scaler_mean = np.load(SCALER_MEAN)
scaler_scale = np.load(SCALER_SCALE)

# Работа с MediaPipe (извлечение координат основных точек лица)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# CSV файл для сохранения данных
output_csv = "emotion_data.csv"

# Эмоции (добавлена 'lying' на ближайшие планы)
emotions = ["neutral", "happy", "sad", "angry", "lying"]

# Начало работы: Включение веб-камеры
cap = cv2.VideoCapture(0)

current_emotion = None
print(f"Available emotions: {emotions}")

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

        return predicted_emotion, confidence

    except ValueError as e:
        log.error(f'Error processing the image: {e}')
        return {
            'error': str(e),
            'predicted_emotion': None,
            'confidence': 0,
            'ranked_classes': []
        }


while True:

    ret, frame = cap.read()
    if not ret:
        log.error("Failed to capture frame.")
        break

    # Конвертирование изображения в RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Извлечение координат ключевых точек лица
    landmarks = preprocess_image(rgb_frame)

    try:
        predicted_emotion, confidence = predict_emotion(landmarks)
    except:
        predicted_emotion = False

    # Вывод итоговой эмоцию и вероятность, определенные при помощи модели, на экран
    if predicted_emotion:
        text_for_detection = f"ML detected emotion: {predicted_emotion} ({confidence*100:.2f}%)"
    else:
        text_for_detection = f"No emotion detected"
    cv2.putText(frame, text_for_detection, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Название изображение
    cv2.imshow("Emotion Detection", frame)




    # Обнаружение лицевых меток
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x, y, w, h = get_face_coordinates(rgb_frame)

            # Рисование квадрата вокруг лица
            if x is not None:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Классификация эмоции
            emotion = classify_emotion(face_landmarks.landmark)
            text_for_recognition = f"Coordinates based emotion: {emotion}"

            # Вывод итоговой эмоцию и вероятность, определенные при помощи координат точек, на экран
            cv2.putText(frame, text_for_recognition, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Отображение изображения
    cv2.imshow('Emotion Recognition', frame)

    # Завершение работы программы при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Очистква ресурсов
cap.release()
cv2.destroyAllWindows()
