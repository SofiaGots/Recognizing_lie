import torch
import numpy as np
import cv2
import mediapipe as mp
import os


from loguru import logger as log

from lib.emotion_net import EmotionNet
from lib.dataset import get_data

from lib.landmark import preprocess_image
from lib import encoder

DATATEST_DIR = os.environ['DATATEST_DIR']

MODELS_DIR = os.environ['MODELS_DIR']
MODEL_FILE = os.path.join(MODELS_DIR, os.environ['MODEL_FILE'])
SCALER_MEAN = os.path.join(MODELS_DIR, os.environ['SCALER_MEAN'])
SCALER_SCALE = os.path.join(MODELS_DIR, os.environ['SCALER_SCALE'])
LABEL_CLASSES = os.path.join(MODELS_DIR, os.environ['LABEL_CLASSES'])


input_size = 1918  # Количество точек для распознавания
num_classes = 4 # Количество категорий (эмоций)

model = EmotionNet(input_size, num_classes)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
model.eval()

scaler_mean = np.load(SCALER_MEAN)
scaler_scale = np.load(SCALER_SCALE)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# CSV file for saving data
output_csv = "emotion_data.csv"

# Define emotions
emotions = ["neutral", "happy", "sad", "angry", "lying"]  # Add more emotions as needed

# Start webcam and collect data
cap = cv2.VideoCapture(0)

current_emotion = None
print(f"Available emotions: {emotions}")


def predict_emotion(landmarks):
    '''
    Evaluate emotion from an image using a pre-trained model with enhanced output
    '''
    try:
        landmarks_normalized = (landmarks - scaler_mean) / scaler_scale
        input_tensor = torch.tensor(landmarks_normalized, dtype=torch.float32).unsqueeze(0)

        # Perform emotion evaluation
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy().flatten()
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]

        # Rank emotions with more context
        # ranked_classes = sorted(
        #     zip(label_classes, probabilities),
        #     key=lambda x: x[1],
        #     reverse=True
        # )

        # primary_emotion = label_classes[predicted_class_idx]
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
    

# Функция для классификации эмоций на основе точек лица с нормализацией
def classify_emotion(landmarks):
    # Нормализация координат относительно ширины лица
    left_cheek = np.array([landmarks[234].x, landmarks[234].y])
    right_cheek = np.array([landmarks[454].x, landmarks[454].y])
    face_width = np.linalg.norm(right_cheek - left_cheek)

    # Нормализованные ключевые точки
    left_mouth = np.array([landmarks[61].x, landmarks[61].y]) / face_width
    right_mouth = np.array([landmarks[291].x, landmarks[291].y]) / face_width
    upper_lip = np.array([landmarks[13].x, landmarks[13].y]) / face_width
    lower_lip = np.array([landmarks[14].x, landmarks[14].y]) / face_width
    left_eyebrow = np.array([landmarks[55].x, landmarks[55].y]) / face_width
    right_eyebrow = np.array([landmarks[285].x, landmarks[285].y]) / face_width
    nose_bridge = np.array([landmarks[168].x, landmarks[168].y]) / face_width

    # Расчет расстояний и пропорций
    brow_distance = np.linalg.norm(left_eyebrow - right_eyebrow) / face_width
    left_brow_to_nose = np.abs(left_eyebrow[1] - nose_bridge[1])  # Вертикальное смещение левой брови от носа
    right_brow_to_nose = np.abs(right_eyebrow[1] - nose_bridge[1])  # Вертикальное смещение правой брови от носа
    brow_to_nose_threshold = 0.02  # Порог нахмуривания

    mouth_corners_up = ((left_mouth[1] < upper_lip[1] - 0.01) and (right_mouth[1] < upper_lip[1] - 0.01))  # Уголки рта значительно выше верхней губы
    mouth_corners_down = ((left_mouth[1] > lower_lip[1] + 0.01) and (right_mouth[1] > lower_lip[1] + 0.01))  # Уголки рта значительно ниже нижней губы

    # Определение эмоции на основе вычисленных параметров
    if brow_distance < 0.15 and left_brow_to_nose < brow_to_nose_threshold and right_brow_to_nose < brow_to_nose_threshold:  # Брови приближены и опущены
        return "Angry"
    elif mouth_corners_down:  # Уголки рта опущены
        return "Sad"
    elif mouth_corners_up:  # Уголки рта подняты
        return "Happy"
    else:  # Все остальные случаи
        return "Neutral"


while True:
    ret, frame = cap.read()
    if not ret:
        log.error("Failed to capture frame.")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    landmarks = preprocess_image(rgb_frame)


    # # Detect facial landmarks
    # results = face_mesh.process(rgb_frame)

    # # If landmarks are detected
    # if results.multi_face_landmarks:
    #     for face_landmarks in results.multi_face_landmarks:
    #         # Draw landmarks on the frame
    #         for landmark in face_landmarks.landmark:
    #             x = int(landmark.x * frame.shape[1])
    #             y = int(landmark.y * frame.shape[0])
    #             cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    #         # Prepare landmarks as input for the model
    #         landmarks = [coord for landmark in face_landmarks.landmark for coord in (landmark.x, landmark.y)]

            # Predict emotion
    try:
        predicted_emotion, confidence = predict_emotion(landmarks)
    except:
        predicted_emotion = False

            # Display predicted emotion on the frame
    if predicted_emotion:
        text = f"Emotion: {predicted_emotion} ({confidence*100:.2f}%)"
    else:
        text = f"No emotion detected"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Emotion Detection", frame)



    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils

    # Обнаружение лицевых меток
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Координаты ограничивающего квадрата вокруг лица
            x_coords = [landmark.x for landmark in face_landmarks.landmark]
            y_coords = [landmark.y for landmark in face_landmarks.landmark]

            x_min = int(min(x_coords) * frame.shape[1])
            x_max = int(max(x_coords) * frame.shape[1])
            y_min = int(min(y_coords) * frame.shape[0])
            y_max = int(max(y_coords) * frame.shape[0])

            # Рисование квадрата вокруг лица
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Классификация эмоции
            emotion = classify_emotion(face_landmarks.landmark)

            # Отображение эмоции над квадратом
            cv2.putText(frame, emotion, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Отображение изображения
    cv2.imshow('Emotion Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
