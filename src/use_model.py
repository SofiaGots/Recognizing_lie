import torch
import numpy as np
import cv2
import mediapipe as mp
import os
import yaml

from loguru import logger as log

from lib.emotion_net import EmotionNet
from lib.dataset import get_data

DATATEST_DIR = os.environ['DATATEST_DIR']

MODELS_DIR = os.environ['MODELS_DIR']
MODEL_FILE = os.path.join(MODELS_DIR, os.environ['MODEL_FILE'])
SCALER_MEAN = os.path.join(MODELS_DIR, os.environ['SCALER_MEAN'])
SCALER_SCALE = os.path.join(MODELS_DIR, os.environ['SCALER_SCALE'])
LABEL_CLASSES = os.path.join(MODELS_DIR, os.environ['LABEL_CLASSES'])


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)


input_size = 478 * 3  # Number of facial landmark coordinates (478 points * x, y, z)
label_classes = np.load(LABEL_CLASSES, allow_pickle=True)
num_classes = len(label_classes)  # Number of emotion classes

model = EmotionNet(input_size, num_classes)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
model.eval()

scaler_mean = np.load(SCALER_MEAN)
scaler_scale = np.load(SCALER_SCALE)


def preprocess_image(image_path):
    '''
    Preprocess image: obtain landmarks, normalize, and prepare for the model
    '''
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Could not read the image from {image_path}.')

    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        raise ValueError('No face landmarks detected in the image.')

    # Extract landmarks
    landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        landmarks = [coord for landmark in face_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]
        break  # Process only the first detected face

    # Convert landmarks to numpy array
    landmarks = np.array(landmarks)

    # Normalize using saved scaler parameters
    landmarks = (landmarks - scaler_mean) / scaler_scale

    # Reshape for model input
    return torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)

def predict_emotion(image_path):
    '''
    Evaluate emotion from an image using a pre-trained model with enhanced output
    '''
    try:
        input_tensor = preprocess_image(image_path)

        # Perform emotion evaluation
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy().flatten()
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]

        # Rank emotions with more context
        ranked_classes = sorted(
            zip(label_classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        # Enhanced emotion interpretation
        emotion_interpretation = {
            'neutral': 'Calm, composed, no strong emotional expression',
            'smile': 'Feeling joy, pleasure, or contentment',
            'sad': 'Experiencing sorrow, unhappiness, or melancholy',
            'angry': 'Feeling irritation, frustration, or rage',
            # 'Surprised': 'Experiencing unexpected or sudden reaction',
            # 'Fearful': 'Sensing anxiety, threat, or nervousness'
        }

        primary_emotion = label_classes[predicted_class_idx]

        return {
            'predicted_emotion': primary_emotion,
            'confidence': confidence,
            'emotion_description': emotion_interpretation.get(primary_emotion, 'Undefined emotional state'),
            'ranked_classes': [
                {
                    'emotion': emotion,
                    'probability': prob * 100,
                    'description': emotion_interpretation.get(emotion, 'Undefined emotional state')
                }
                for emotion, prob in ranked_classes
            ],
            'logits': outputs.numpy().flatten()
        }

    except ValueError as e:
        log.error(f'Error processing the image: {e}')
        return {
            'error': str(e),
            'predicted_emotion': None,
            'confidence': 0,
            'ranked_classes': []
        }

def visualize_emotion_results(image_path, result):
    '''
    Optional: Visualize emotion detection results on the image
    '''
    try:
        image = cv2.imread(image_path)

        # Add text overlay with emotion results
        text_primary = f"Emotion: {result['predicted_emotion']} ({result['confidence']*100:.2f}%)"
        text_description = result['emotion_description']

        cv2.putText(image, text_primary, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, text_description, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Optionally save or display the image
        cv2.imwrite('emotion_result.jpg', image)
        # Uncomment the following line if you want to display the image
        # cv2.imshow('Emotion Detection', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        log.error(f'Error visualizing results: {e}')

if __name__ == '__main__':
    # image_path = './data_check/photo_1_2024-12-03_21-16-23.jpg'

    for emotion, files in get_data(DATATEST_DIR):
        for file in files:
            try:
                result = predict_emotion(file)

                print('========================RESULTS===================================================')
                print(f'File: {file}')
                print(f'Original Emotion: {emotion}')
                print(f"Predicted Emotion: {result['predicted_emotion']}")
                print(f"Confidence: {result['confidence'] * 100:.2f}%")
                print(f"Emotion Description: {result['emotion_description']}")
                print("Ranked Emotions:")
                for res_emotion in result['ranked_classes'][:3]:
                    print(f"- {res_emotion['emotion']}: {res_emotion['probability']:.2f}% - {res_emotion['description']}")

                # Optional visualization
                # visualize_emotion_results(image_path, result)

            except Exception as e:
                log.error(f"Unexpected error in emotion detection: {e}")