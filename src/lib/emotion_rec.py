import numpy as np

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
        return "angry"
    elif mouth_corners_down:  # Уголки рта опущены
        return "sad"
    elif mouth_corners_up:  # Уголки рта подняты
        return "happy"
    else:  # Все остальные случаи
        return "neutral"