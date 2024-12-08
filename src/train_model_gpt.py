import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from lib.emotion_net import EmotionNet
from lib import encoder

# Environment Variables and Paths
DATASET_DIR = os.environ['DATASET_DIR']
MODELS_DIR = os.environ['MODELS_DIR']
DATASET_RES = os.path.join(MODELS_DIR, os.environ['DATASET_RES'])

MODEL_FILE = os.path.join(MODELS_DIR, os.environ['MODEL_FILE'])
SCALER_MEAN = os.path.join(MODELS_DIR, os.environ['SCALER_MEAN'])
SCALER_SCALE = os.path.join(MODELS_DIR, os.environ['SCALER_SCALE'])
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, 'confusion_matrix.png')
TRAINING_CURVE_PATH = os.path.join(MODELS_DIR, 'training_curve.png')

RANDOM_STATE = int(os.environ['RANDOM_STATE'])

def check_dataset_balance(labels):
    """
    Print dataset class distribution
    """
    unique, counts = np.unique(labels, return_counts=True)
    print("\nDataset Distribution:")
    for label, count in zip(unique, counts):
        print(f"Emotion {encoder.decode(label)}: {count} samples ({count/len(labels)*100:.2f}%)")

    return unique, counts

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

def plot_training_curve(train_losses, val_losses):
    """
    Plot and save training and validation losses
    """
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_CURVE_PATH)
    plt.close()

def train_model():
    # Load and preprocess data
    data = pd.read_csv(DATASET_RES, header=None)
    landmarks = data.iloc[:, 1:].values  # Landmarks
    emotion = data.iloc[:, 0].values     # Emotions

    # Check dataset balance
    unique_labels, label_counts = check_dataset_balance(emotion)

    # Scale data
    scaler = StandardScaler()
    landmarks_normalized = scaler.fit_transform(landmarks)
    np.save(SCALER_MEAN, scaler.mean_)
    np.save(SCALER_SCALE, scaler.scale_)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        landmarks_normalized, emotion,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=emotion  # Ensure balanced split
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Initialize the model
    input_size = X_train.shape[1]
    num_classes = len(encoder.category)
    model = EmotionNet(input_size, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0 / count for count in label_counts],
            dtype=torch.float32
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # Training parameters
    num_epochs = 1000
    best_loss = float('inf')
    early_stop_patience = 50
    early_stop_counter = 0

    # Tracking losses
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Record losses
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}")

        # Model checkpointing and early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

    # Final model evaluation
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_classes = torch.argmax(y_pred, axis=1)
        accuracy = (y_pred_classes == y_test).sum().item() / y_test.size(0)

        print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")

        # Detailed Classification Report
        class_names = [encoder.decode(label) for label in unique_labels]
        print("\nClassification Report:")
        print(classification_report(
            y_test.numpy(),
            y_pred_classes.numpy(),
            target_names=class_names
        ))

    # Visualization
    plot_confusion_matrix(y_test.numpy(), y_pred_classes.numpy(), class_names)
    plot_training_curve(train_losses, val_losses)

    print(f"\nConfusion Matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Training Curve saved to: {TRAINING_CURVE_PATH}")


def main():
    train_model()


if __name__ == '__main__':
    main()
