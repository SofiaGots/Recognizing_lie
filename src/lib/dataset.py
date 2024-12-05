import os
import yaml

# def dataset1(config_file=DATASET_CONFIG):
#     with open(config_file, 'r') as config_obj:
#         dataset = yaml.safe_load(config_obj)

#     for key, value in dataset.items():
#         yield key, value


def get_data(path):
    for emotion in os.listdir(path):
        if emotion != '.DS_Store':
            files = [os.path.join(path, emotion, filename) for filename in os.listdir(os.path.join(path, emotion))]
            yield emotion, files
