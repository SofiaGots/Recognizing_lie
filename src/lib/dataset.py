import numpy as np
import os

from lib import encoder

def get_data(path):
    for emotion in os.listdir(path):
        if emotion != '.DS_Store':
            files = [os.path.join(path, emotion, filename) for filename in os.listdir(os.path.join(path, emotion))]
            yield emotion, files


def check(data, landmarks, emotion):
    """
    Comprehensive dataset analysis
    """
    print("\n--- Dataset Analysis ---")
    print(f"Total samples: {len(data)}")
    print(f"Landmarks shape: {landmarks.shape}")

    # Unique emotions and their counts
    unique_emotions, counts = np.unique(emotion, return_counts=True)
    print("\nEmotion Distribution:")
    for label, count in zip(unique_emotions, counts):
        print(f"Emotion {label} ({encoder.decode(label)}): {count} samples ({count/len(emotion)*100:.2f}%)")

    # Check landmarks statistics
    print("\nLandmarks Statistics:")
    print(f"Mean: {np.mean(landmarks)}")
    print(f"Std: {np.std(landmarks)}")
    print(f"Min: {np.min(landmarks)}")
    print(f"Max: {np.max(landmarks)}")

    # Verify unique labels
    unique_labels = np.unique(emotion)
    print(f"\nUnique labels found: {unique_labels}")
    print(f"Decoder mapping: {encoder.category}")
