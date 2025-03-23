# celm.py

import torch
import torch.nn as nn


class CELM(nn.Module):
    def __init__(self, num_classes=10):
        super(CELM, self).__init__()

        # Camadas convolucionais com pesos fixos
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Fixar os pesos das camadas convolucionais
        with torch.no_grad():
            nn.init.normal_(self.conv1.weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.conv1.bias)
            nn.init.normal_(self.conv2.weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.conv2.bias)

        # Desativar o gradiente para essas camadas
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False

        # Camada de saída (os pesos serão calculados analiticamente)
        self.output_weights = None
        self.output_size = num_classes
        self.feature_size = 64 * 7 * 7

    def extract_features(self, x):
        """Extrai características usando camadas convolucionais fixas"""
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract_features(x)

        # Se estivermos no modo de inferência com pesos já calculados
        if self.output_weights is not None and self.training == False:
            return x @ self.output_weights

        return x  # Retorna apenas as características durante o treinamento