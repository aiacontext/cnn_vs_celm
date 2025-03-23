# models/optimized_celm.py (Corrigido)

import torch
import torch.nn as nn
import numpy as np
import math


class OptimizedCELM(nn.Module):
    """
    CELM Otimizada com:
    1. Melhor inicialização de filtros
    2. Camadas convolucionais ortogonais
    3. Arquitetura simplificada mas eficiente
    4. Normalização consistente entre treino e inferência
    """

    def __init__(self, num_classes=10, input_channels=1, feature_dim=64):
        super(OptimizedCELM, self).__init__()

        # Parâmetros para ajuste de otimização
        self.scale_factor = 0.5  # Scale factor ótimo encontrado pelo tuning

        # Primeira camada convolucional - kernels maiores para capturar mais informação
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        # Pooling para redução de dimensionalidade
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Segunda camada convolucional (reduzida para diminuir dimensionalidade)
        self.conv2 = nn.Conv2d(32, feature_dim, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Camada de ativação não-linear e mais estável que ReLU
        self.activation = nn.SELU()  # SELU ajuda na normalização e estabilidade

        # Inicializar filtros convolucionais de maneira eficiente
        self._init_filters()

        # Calcular tamanho de saída (para MNIST: feature_dim filtros, 7x7 após pooling)
        # 28x28 -> 14x14 -> 7x7
        self.feature_size = feature_dim * 7 * 7

        # Pesos da camada de saída (a serem calculados analiticamente)
        self.output_weights = None
        self.output_size = num_classes

        # Registrar buffers para parâmetros de normalização (inicializados como None)
        self.register_buffer('norm_mean', None)
        self.register_buffer('norm_std', None)

    def _init_filters(self):
        """Inicializa os filtros convolucionais usando técnicas avançadas"""
        # 1. Inicialização ortogonal para conv1 - preserva norma do sinal
        nn.init.orthogonal_(self.conv1.weight, gain=self.scale_factor)
        nn.init.zeros_(self.conv1.bias)

        # 2. Inicialização Kaiming para conv2 - bom para ativações SELU/ReLU
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)

        # 3. Fixar os pesos (não treináveis)
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False

    def extract_features(self, x):
        """Extrai características usando camadas convolucionais fixas"""
        # Primeira camada
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)

        # Segunda camada
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)

        # Achatar para características
        x = torch.flatten(x, 1)
        return x

    def normalize_features(self, x):
        """
        Normaliza as características usando os parâmetros armazenados.
        Aplicado tanto durante o treinamento quanto durante a inferência.
        """
        if self.norm_mean is not None and self.norm_std is not None:
            # Converter para mesmo device se necessário
            if x.device != self.norm_mean.device:
                self.norm_mean = self.norm_mean.to(x.device)
                self.norm_std = self.norm_std.to(x.device)

            # Aplicar normalização
            x = (x - self.norm_mean) / self.norm_std
        return x

    def forward(self, x):
        """Forward pass com normalização consistente"""
        # Extrair características
        x = self.extract_features(x)

        # Se estivermos no modo de inferência com pesos já calculados
        if self.output_weights is not None and not self.training:
            # Normalizar características (mesmo processo usado no treinamento)
            x = self.normalize_features(x)

            # Converter para o mesmo dispositivo, se necessário
            if x.device != self.output_weights.device:
                self.output_weights = self.output_weights.to(x.device)

            # Aplicar camada de saída
            return x @ self.output_weights

        return x  # Retorna só características durante treinamento

    def set_normalization_params(self, mean, std):
        """Define os parâmetros de normalização"""
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).float()

        self.norm_mean = mean
        self.norm_std = std