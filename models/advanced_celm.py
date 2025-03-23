# advanced_celm.py

import torch
import torch.nn as nn
import numpy as np


class AdvancedCELM(nn.Module):
    def __init__(self, num_classes=10):
        super(AdvancedCELM, self).__init__()

        # Primeira etapa de extração com diferentes tipos de filtros
        self.gabor_filters = nn.Conv2d(1, 10, kernel_size=7, stride=1, padding=3)
        self.pca_filters = nn.Conv2d(1, 10, kernel_size=7, stride=1, padding=3)
        self.patch_filters = nn.Conv2d(1, 10, kernel_size=7, stride=1, padding=3)

        self.pool1 = nn.MaxPool2d(kernel_size=7, stride=3)

        # Segunda etapa de extração (6 Gabor filters)
        self.gabor_filters2 = nn.Conv2d(30, 6, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.relu = nn.ReLU()

        # Inicializar filtros
        self._init_filters()

        # Camada de saída (a ser calculada analiticamente)
        self.output_weights = None
        self.output_size = num_classes

        # Calcular o tamanho da saída da extração de características
        self._calculate_feature_size()

    def _init_filters(self):
        # Inicializar filtros Gabor
        self._init_gabor_filters(self.gabor_filters, wavelength=6)
        self._init_gabor_filters(self.gabor_filters2, wavelength=3)

        # Inicializar filtros PCA (simplificado)
        with torch.no_grad():
            nn.init.normal_(self.pca_filters.weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.pca_filters.bias)

        # Inicializar filtros Patch (simplificado)
        with torch.no_grad():
            nn.init.uniform_(self.patch_filters.weight, -0.5, 0.5)
            nn.init.zeros_(self.patch_filters.bias)

        # Desativar o gradiente para todas as camadas convolucionais
        for param in self.parameters():
            param.requires_grad = False

    def _init_gabor_filters(self, conv_layer, wavelength):
        # Implementação simplificada de filtros Gabor
        with torch.no_grad():
            for i in range(conv_layer.weight.size(0)):
                kernel_size = conv_layer.weight.size(2)
                sigma = 0.56 * wavelength
                theta = i * np.pi / conv_layer.weight.size(0)

                # Criar filtro Gabor
                for y in range(kernel_size):
                    for x in range(kernel_size):
                        y_c = y - kernel_size // 2
                        x_c = x - kernel_size // 2

                        x_theta = x_c * np.cos(theta) + y_c * np.sin(theta)
                        y_theta = -x_c * np.sin(theta) + y_c * np.cos(theta)

                        gb = np.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2)) * np.cos(
                            2 * np.pi * x_theta / wavelength)

                        for c in range(conv_layer.weight.size(1)):
                            conv_layer.weight[i, c, y, x] = gb

                conv_layer.bias[i] = 0.0

    def _calculate_feature_size(self):
        # Calcular o tamanho das características de saída
        x = torch.zeros(1, 1, 28, 28)
        x = self.extract_features(x)
        self.feature_size = x.numel()

    def extract_features(self, x):
        # Primeira etapa - processamento paralelo com diferentes filtros
        x_gabor = self.relu(self.gabor_filters(x))
        x_pca = self.relu(self.pca_filters(x))
        x_patch = self.relu(self.patch_filters(x))

        # Concatenar os mapas de características
        x = torch.cat([x_gabor, x_pca, x_patch], dim=1)
        x = self.pool1(x)

        # Segunda etapa
        x = self.relu(self.gabor_filters2(x))
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract_features(x)

        # Se estivermos no modo de inferência com pesos já calculados
        if self.output_weights is not None and self.training == False:
            return x @ self.output_weights

        return x