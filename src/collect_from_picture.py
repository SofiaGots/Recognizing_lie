import csv
import os

from lib.dataset import get_data
from lib.landmark import preprocess_image
from lib.encoder import encode

DATASET_DIR = os.environ['DATASET_DIR']
MODELS_DIR = os.environ['MODELS_DIR']
DATASET_RES = os.path.join(MODELS_DIR, os.environ['DATASET_RES'])


def clean_data():
    try:
        os.remove(DATASET_RES)
    except:
        pass


def main():

    clean_data()

    with open(DATASET_RES, mode="a", newline="") as file:
        csv_writer = csv.writer(file)

        for emotion, files in get_data(path=DATASET_DIR):
            for file in files:

                landmarks = preprocess_image(path=file)

                if landmarks:
                    csv_writer.writerow([encode(emotion)] + landmarks)


if __name__ == '__main__':
    main()
