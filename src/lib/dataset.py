import os


# Функция, которая возвращает эмоцию и координаты точек на лице
def get_data(path):
    for emotion in os.listdir(path):
        if emotion != '.DS_Store':
            files = [os.path.join(path, emotion, filename) for filename in os.listdir(os.path.join(path, emotion))]
            yield emotion, files
