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


def predict_emotion(landmarks):
    try:
        landmarks_normalized = (landmarks - scaler_mean) / scaler_scale
        input_tensor = torch.tensor(landmarks_normalized, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy().flatten()
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]

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

                landmarks = preprocess_image(image=file)

                result = predict_emotion(landmarks)

                # print('========================RESULTS===================================================')
                print(f'File: {file}')
                print(f'Original Emotion: {emotion}')
                print(f"Predicted Emotion: {result['predicted_emotion']}")
                print(f"Confidence: {result['confidence'] * 100:.2f}%")

            except Exception as e:
                log.error(f"Unexpected error in emotion detection: {e}")



if __name__ == '__main__':
    main()