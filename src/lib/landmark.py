import cv2
import mediapipe as mp


face_detection = mp.solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,             # If set to false, the solution treats the input images as a video stream.
    max_num_faces=1,                    # Maximum number of faces to detect.
    refine_landmarks=True,              # Whether to further refine the landmark coordinates around the eyes and lips, and output additional landmarks around the irises
    min_detection_confidence=0.7        # Minimum confidence value ([0.0, 1.0]) from the face detection model for the detection to be considered successful. Default to 0.5.
)


def preprocess_landmarks(landmarks: list):

    _landmarks = []

    base_x, base_y = landmarks[0][0], landmarks[0][1]
    for landmark_point in landmarks:
        _landmarks.extend([
            landmark_point[0] - base_x,
            landmark_point[1] - base_y
        ])

    # normalization
    max_value = max(list(map(abs, _landmarks)))
    _landmarks = list(map(lambda n: n / max_value, _landmarks))

    return _landmarks


def landmarks_list(image, landmarks):
    # ORIGINAL
    # landmarks = [coord for landmark in face_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_points = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z

        landmark_points.append([landmark_x, landmark_y])

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


def crop_face(image):
    """
    Detect and crop face from image
    """
    # Detect face
    results = face_detection.process(image)

    if not results.detections:
        raise ValueError('No face detected in the image.')

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

    # Optional: Add some padding
    pad = int(max(w, h) * 0.2)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(iw - x, w + 2 * pad)
    h = min(ih - y, h + 2 * pad)

    # Crop face
    face_crop = image[y:y+h, x:x+w]

    return face_crop


def read_image(path: str):
    image = cv2.imread(path)
    # image = cv2.flip(image, 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(path: str):
    image = read_image(path)

    # image = crop_face(image)

    # resize?
    # face_resized = cv2.resize(face_crop, (96, 96))
    # show_face(face_resized)

    landmarks_lst = get_landmarks(image)
    if landmarks_lst:
        landmarks_norm = preprocess_landmarks(landmarks_lst)

        return landmarks_norm

    return None