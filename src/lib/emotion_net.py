import torch.nn as nn


class EmotionNet_1(nn.Module):
    '''
    Опредление нейросети
    '''
    def __init__(self, input_size, num_classes):
        super(EmotionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class EmotionNet(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.3):
        super(EmotionNet, self).__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Classification layer
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output