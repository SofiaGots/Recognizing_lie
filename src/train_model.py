import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.emotion_net import EmotionNet
from lib import encoder


DATASET_DIR = os.environ['DATASET_DIR']
MODELS_DIR = os.environ['MODELS_DIR']
DATASET_RES = os.path.join(MODELS_DIR, os.environ['DATASET_RES'])

MODEL_FILE = os.path.join(MODELS_DIR, os.environ['MODEL_FILE'])
SCALER_MEAN = os.path.join(MODELS_DIR, os.environ['SCALER_MEAN'])
SCALER_SCALE = os.path.join(MODELS_DIR, os.environ['SCALER_SCALE'])

RANDOM_STATE = int(os.environ['RANDOM_STATE'])          # for experiment repeatability


# Load and preprocess data
data = pd.read_csv(DATASET_RES, header=None)
landmarks = data.iloc[:, 1:].values  # Landmarks
emotion = data.iloc[:, 0].values   # Emotions

# Scale data
scaler = StandardScaler()
landmarks_normalized = scaler.fit_transform(landmarks)
np.save(SCALER_MEAN, scaler.mean_)
np.save(SCALER_SCALE, scaler.scale_)


# Split data
X_train, X_test, y_train, y_test = train_test_split(landmarks_normalized, emotion, test_size=0.2, random_state=RANDOM_STATE)

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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 550
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    # Изменение параметров нейросетей
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = torch.argmax(y_pred, axis=1)
    accuracy = (y_pred_classes == y_test).sum().item() / y_test.size(0)
    print(f"Accuracy: {accuracy * 100:.2f}%")

torch.save(model.state_dict(), MODEL_FILE)
