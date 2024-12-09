import torch.nn as nn

# Попытка 1 найти встроенную модель, которая сможет правильно обучаться для моей задачи
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

# Попытка 2 найти встроенную модель, которая сможет правильно обучаться для моей задачи
class EmotionNet2(nn.Module):
    '''
    Опредление нейросети
    '''
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


# Попытка 3 (работающая) найти встроенную модель, которая сможет правильно обучаться для моей задачи
# https://claude.site/artifacts/332a763e-dd71-4fee-b49d-e1b7546dee40
# Данный класс полностью скопирован из интернета, так как я не умею создавать встроенный модели
class EmotionNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EmotionNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),    # Увеличение количества нейронов первого слоя
            nn.BatchNorm1d(128),          # Добавление Batch нормализации
            nn.ReLU(),                    # Активация изменений
            nn.Dropout(0.3),              # Снижение уровня точности (погрешность)
            nn.Linear(128, 64),           # Добавление большей глубины
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)