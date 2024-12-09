import cv2
import mediapipe as mp
import numpy as np
import math

from itertools import combinations

face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,             # Обработка входных изображений (стоит True)
    max_num_faces=1,                    # Максимальное количество лиц для обнаружения
    refine_landmarks=True,              # Детализация точек
    min_detection_confidence=0.7        # Минимальная уверенность модели для успешного обнаружения
)

# Функция 1 вычисления признаков (нормализация всех координат относительно носа)
def feature1_norm_relative_coord(landmarks: list):
    # Базовая точка отсчета (кончик носа)
    base_x, base_y, base_z = landmarks[4]

    normalized_coords = []
    for x, y, z in landmarks:
        normalized_coords.extend([
            x - base_x,
            y - base_y,
            z - base_z
        ])
    max_value = max(map(abs, normalized_coords))
    return [coord / max_value for coord in normalized_coords]

# Функция 2 вычисления признаков (нормализация всех расстояний между всеми точками)
def feature2_parwise_distance(landmarks: list):
    pairwise_distances = []
    for i, j in combinations(range(len(landmarks)), 2):
        dist = math.sqrt(
            (landmarks[i][0] - landmarks[j][0]) ** 2 +
            (landmarks[i][1] - landmarks[j][1]) ** 2 +
            (landmarks[i][2] - landmarks[j][2]) ** 2
        )
        pairwise_distances.append(dist)
    max_dist = max(pairwise_distances)
    return [d / max_dist for d in pairwise_distances]

# Функция 3 вычисления признаков (рассчет углов между тройками точек)
def feature3_angles(landmarks: list):
    angles = []
    key_triplets = [
        (33, 4, 263),  # глаза - нос - глаза
        (61, 4, 291),  # правый уголок рта - нос - левый уголок рта
        (0, 4, 17),    # подбородок - нос - верх лоб
    ]
    for i, j, k in key_triplets:
        # Вектора для заданных точек
        v1 = np.array(landmarks[j]) - np.array(landmarks[i])
        v2 = np.array(landmarks[k]) - np.array(landmarks[i])
        # Скалярное произведение (угол между векторами)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        angle = np.arccos(dot_product / (norm_v1 * norm_v2))
        angles.append(np.degrees(angle))

    return angles

# Функция 4 вычисления признаков (нормализация расстояния между ключевыми точками (глаза, рот, нос))
def feature4_euclidean_distance(landmarks: list):
    # Расстояние между точками
    def euclidean_distance(p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2
        )
    
    eye_distance = euclidean_distance(landmarks[33], landmarks[263])
    mouth_width = euclidean_distance(landmarks[61], landmarks[291])
    nose_length = euclidean_distance(landmarks[4], landmarks[0])
    face_height = euclidean_distance(landmarks[0], landmarks[10])
    # Нормализация на высоту лица
    return [
        eye_distance / face_height,
        mouth_width / face_height,
        nose_length / face_height,
    ]

# Функция 5 вычисления признаков (нормализация расстояния каждой точки до кончика носа)
def feature5_euclidean_distance(landmarks: list):
    # Базовая точка отсчета (кончик носа)
    base_x, base_y, base_z = landmarks[4]

    normalized_coords = []
    for x, y, z in landmarks:
        normalized_coords.append(
            math.sqrt(
                (x - base_x) ** 2 +
                (y - base_y) ** 2 +
                (z - base_z) ** 2
            )
        )

    max_value = max(map(abs, normalized_coords))
    return [coord / max_value for coord in normalized_coords]

# Создается массив с особенностями точек на лице
# Данный код (а также функции, на которые он ссылается) полностью скопирован из интернета, так как я не умею делать массив с таким огромным количеством измерений
def preprocess_landmarks(landmarks: list):

    _landmarks = []

    _landmarks.extend(feature1_norm_relative_coord(landmarks))
    _landmarks.extend(feature3_angles(landmarks))
    _landmarks.extend(feature4_euclidean_distance(landmarks))
    _landmarks.extend(feature5_euclidean_distance(landmarks))

    return _landmarks

# Преобразование относительных координат в абсолютные (так как первоначально координаты имеют значение от 0 до 1)
# Код скопирован из интернета, так как я не знала, что такую функцию нужно делать
def landmarks_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_points = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = min(int(landmark.z * image_height), image_height - 1)

        landmark_points.append([landmark_x, landmark_y, landmark_z])

    return landmark_points

# Получение координат меток
def get_landmarks(image, face_mesh=face_mesh):

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = landmarks_list(image, face_landmarks)

        return landmarks

    return None

# Оставить только лицо для распознавания
def get_face_coordinates(image):
    results = face_detection.process(image)

    if not results.detections:
        print(f'No face detected in the image')
        return None, None, None, None


    # Получение первого распознавания лица
    detection = results.detections[0]

    # Извлечение ограничивающую рамку для лица
    ih, iw, _ = image.shape
    bbox = detection.location_data.relative_bounding_box

    # Преобразование относительных координат в абсолютные
    x = int(bbox.xmin * iw)
    y = int(bbox.ymin * ih)
    w = int(bbox.width * iw)
    h = int(bbox.height * ih)

    # Отступы от лица, чтобы было удобнее воспринимать
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(iw - x, w + 2 * pad)
    h = min(ih - y, h + 2 * pad)

    return x, y, w, h

# Обрезание изображения только до области лица (для более подробной работы с эмоцией человека)
def crop_face(image, x, y, w, h):

    face_crop = image[y:y+h, x:x+w]

    return np.ascontiguousarray(face_crop)

# Считывание изображения
def read_image(path: str):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Извлечение точек и вычисление признаков
def preprocess_image(image: str):

    if isinstance(image, str):
        image = read_image(image)

    image = cv2.flip(image, 1)
    # Нахождение точек
    x, y, w, h = get_face_coordinates(image)
    # Обрезание лица
    if x is not None:
        image = crop_face(image, x, y, w, h)

    if image is not None:
        # Извлечение точек
        landmarks_lst = get_landmarks(image)
        if landmarks_lst:
            # Вычисление признаков
            landmarks_norm = preprocess_landmarks(landmarks_lst)

            return landmarks_norm

        return None