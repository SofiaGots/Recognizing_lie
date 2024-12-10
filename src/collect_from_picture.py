import csv
import os

from lib.dataset import get_data
from lib.landmark import preprocess_image
from lib.encoder import encode

DATASET_DIR = os.environ['DATASET_DIR'] # Путь к наборам данных
MODELS_DIR = os.environ['MODELS_DIR'] # Путь к модели
DATASET_RES = os.path.join(MODELS_DIR, os.environ['DATASET_RES']) # Путь к файлу, где сохранятся данные

# Удаление файла с результатами, если он существует
def clean_data():
    try:
        os.remove(DATASET_RES)
    except:
        pass


def main():

    clean_data()
    # Данный отрывок был написан с помощью папы, так как я еще не успела разобраться с конструкцией "with open ... as"
    with open(DATASET_RES, mode="a", newline="") as file:
        # Файл открывается в режиме редактора
        csv_writer = csv.writer(file)
        # Получение данных из набора данных
        for emotion, files in get_data(path=DATASET_DIR):
            for file in files:
                # Нормализованные координаты точек лица
                landmarks = preprocess_image(image=file)

                if landmarks:
                    csv_writer.writerow([encode(emotion)] + landmarks)


if __name__ == '__main__':
    main()
