import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from lib.emotion_net import EmotionNet, EmotionNet_1


DATASET_DIR = os.environ['DATASET_DIR']
MODELS_DIR = os.environ['MODELS_DIR']
DATASET_RES = os.path.join(MODELS_DIR, os.environ['DATASET_RES'])

MODEL_FILE = os.path.join(MODELS_DIR, os.environ['MODEL_FILE'])
SCALER_MEAN = os.path.join(MODELS_DIR, os.environ['SCALER_MEAN'])
SCALER_SCALE = os.path.join(MODELS_DIR, os.environ['SCALER_SCALE'])
LABEL_CLASSES = os.path.join(MODELS_DIR, os.environ['LABEL_CLASSES'])

RANDOM_STATE = int(os.environ['RANDOM_STATE'])


# Load and preprocess data
data = pd.read_csv(DATASET_RES, header=None)
X = data.iloc[:, 1:].values  # Landmarks
y = data.iloc[:, 0].values   # Emotions

# Encode emotions
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
np.save(LABEL_CLASSES, label_encoder.classes_)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler mean and scale values for later use in inference
np.save(SCALER_MEAN, scaler.mean_)
np.save(SCALER_SCALE, scaler.scale_)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Initialize the model
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
model = EmotionNet(input_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
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
