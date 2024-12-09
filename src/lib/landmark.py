import cv2
import mediapipe as mp
import numpy as np
import math

from itertools import combinations

face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,             # If set to false, the solution treats the input images as a video stream.
    max_num_faces=1,                    # Maximum number of faces to detect.
    refine_landmarks=True,              # Whether to further refine the landmark coordinates around the eyes and lips, and output additional landmarks around the irises
    min_detection_confidence=0.7        # Minimum confidence value ([0.0, 1.0]) from the face detection model for the detection to be considered successful. Default to 0.5.
)


def feature1_norm_relative_coord(landmarks: list):
    base_x, base_y, base_z = landmarks[4]            # Base reference point (e.g., nose tip, landmark 4)

    normalized_coords = []
    for x, y, z in landmarks:
        normalized_coords.extend([
            x - base_x,
            y - base_y,
            z - base_z
        ])
    max_value = max(map(abs, normalized_coords))
    return [coord / max_value for coord in normalized_coords]


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


def feature3_angles(landmarks: list):
    angles = []
    key_triplets = [
        (33, 4, 263),  # Eyes-nose-eyes
        (61, 4, 291),  # Mouth corners-nose
        (0, 4, 17),    # Chin-nose-top of forehead
    ]
    for i, j, k in key_triplets:
        v1 = np.array(landmarks[j]) - np.array(landmarks[i])
        v2 = np.array(landmarks[k]) - np.array(landmarks[i])
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        angle = np.arccos(dot_product / (norm_v1 * norm_v2))
        angles.append(np.degrees(angle))

    return angles


def feature4_euclidean_distance(landmarks: list):

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
    return [
        eye_distance / face_height,
        mouth_width / face_height,
        nose_length / face_height,
    ]

def feature5_euclidean_distance(landmarks: list):

    base_x, base_y, base_z = landmarks[4]            # Base reference point (e.g., nose tip, landmark 4)

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


def preprocess_landmarks(landmarks: list):

    _landmarks = []

    _landmarks.extend(feature1_norm_relative_coord(landmarks))
    # _landmarks.extend(feature2_parwise_distance(landmarks))
    _landmarks.extend(feature3_angles(landmarks))
    _landmarks.extend(feature4_euclidean_distance(landmarks))
    _landmarks.extend(feature5_euclidean_distance(landmarks))

    return _landmarks


def landmarks_list(image, landmarks):
    # ORIGINAL
    # landmarks = [coord for landmark in face_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_points = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = min(int(landmark.z * image_height), image_height - 1)

        landmark_points.append([landmark_x, landmark_y, landmark_z])

    return landmark_points


def get_landmarks(image, face_mesh=face_mesh):
    '''
    Получить координаты меток
    '''

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = landmarks_list(image, face_landmarks)
            # landmarks = [coord for landmark in face_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]

        return landmarks

    return None


def get_face_coordinates(image):
    '''
    Оставить только лицо для распознавания
    '''
    results = face_detection.process(image)

    if not results.detections:
        print(f'No face detected in the image')
        return None, None, None, None


    # Get the first face detection
    detection = results.detections[0]

    # Extract face bounding box
    ih, iw, _ = image.shape
    bbox = detection.location_data.relative_bounding_box

    # Convert relative coordinates to absolute
    x = int(bbox.xmin * iw)
    y = int(bbox.ymin * ih)
    w = int(bbox.width * iw)
    h = int(bbox.height * ih)

    # Add some padding
    # pad = int(max(w, h) * 0.2)
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(iw - x, w + 2 * pad)
    h = min(ih - y, h + 2 * pad)

    return x, y, w, h


def crop_face(image, x, y, w, h):

    face_crop = image[y:y+h, x:x+w]

    return np.ascontiguousarray(face_crop)


def read_image(path: str):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(image: str):

    if isinstance(image, str):
        image = read_image(image)

    image = cv2.flip(image, 1)
    x, y, w, h = get_face_coordinates(image)
    if x is not None:
        image = crop_face(image, x, y, w, h)

    # resize?
    # face_resized = cv2.resize(face_crop, (96, 96))
    # show_face(face_resized)

    if image is not None:
        landmarks_lst = get_landmarks(image)
        if landmarks_lst:
            landmarks_norm = preprocess_landmarks(landmarks_lst)

            return landmarks_norm

        return None