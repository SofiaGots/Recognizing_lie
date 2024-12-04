import cv2
import mediapipe as mp
import csv
from loguru import logger as log
import os
import time
import yaml

from lib.dataset import get_data


DATASET_DIR = os.environ['DATASET_DIR']
MODELS_DIR = os.environ['MODELS_DIR']
DATASET_RES = os.path.join(MODELS_DIR, os.environ['DATASET_RES'])


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


def clean_data():
    '''
    Удалить старые данные
    '''
    try:
        os.remove(DATASET_RES)
    except:
        pass


def get_landmarks(file: str) -> list[int]:
    '''
    Получить координаты меток и вернуть их как list
    '''
    image = cv2.imread(file)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [coord for landmark in face_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]

        return landmarks

    return None


def main():

    clean_data()

    with open(DATASET_RES, mode="a", newline="") as file:
        csv_writer = csv.writer(file)

        for emotion, files in get_data(path=DATASET_DIR):
            for file in files:
                landmarks = get_landmarks(file)
                if landmarks:
                    csv_writer.writerow([emotion] + landmarks)

if __name__ == '__main__':
    main()
