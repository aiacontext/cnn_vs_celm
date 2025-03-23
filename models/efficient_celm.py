# models/efficient_celm.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA


class EfficientCELM(nn.Module):
    """
    CELM simplificada e eficiente com:
    1. Filtros predefinidos
    2. Uma única camada convolucional mais potente
    3. Compressão PCA integrada
    4. Sem complexity overkill
    """

    def __init__(self, num_classes=10, input_channels=1, num_filters=64, pca_components=128):
        super(EfficientCELM, self).__init__()

        # Única camada convolucional potente com kernels maiores para capturar mais contexto
        self.conv = nn.Conv2d(input_channels, num_filters, kernel_size=5, stride=1, padding=2)

        # Pooling em 2 estágios para redução significativa de dimensionalidade
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # Pooling adicional para redução mais agressiva
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)  # 7x7 -> 3x3 (ou 4x4 -> 2x2)

        # Ativação simples e eficiente
        self.activation = nn.ReLU()

        # Inicialização simples dos filtros com kernels pré-definidos
        self._init_predefined_filters()

        # Configuração para PCA
        self.use_pca = pca_components > 0
        self.pca_components = pca_components
        self.pca = None
        self.pca_fitted = False

        # Calcular tamanho das características
        size_after_conv = 28 // 4  # Após 2 estágios de pooling

        # Em algumas configurações o valor pode ser 3 em outras 4, dependendo do padding
        if size_after_conv == 7:
            # Divisão por 2 novamente para o pool3
            size_after_pool = 3
        else:
            size_after_pool = 2

        self.raw_feature_size = num_filters * size_after_pool * size_after_pool
        self.feature_size = pca_components if self.use_pca else self.raw_feature_size

        # Pesos da camada de saída (a serem calculados analiticamente)
        self.output_weights = None
        self.output_size = num_classes

        # Registrar buffers para parâmetros de normalização
        self.register_buffer('norm_mean', None)
        self.register_buffer('norm_std', None)

    def _init_predefined_filters(self):
        """Inicializa os filtros convolucionais com kernels pré-definidos"""
        # Obter número de filtros
        num_filters = self.conv.weight.size(0)

        with torch.no_grad():
            # 1. Filtros de detecção de bordas em várias orientações
            n_edge_filters = min(8, num_filters)

            # Filtro horizontal de detecção de bordas (Sobel)
            sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Filtro vertical de detecção de bordas (Sobel)
            sobel_y = torch.tensor([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Laplaciano para detecção de bordas
            laplacian = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Filtro gaussiano para suavização
            gaussian = torch.tensor([
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 16.0

            # Copiar esses filtros para os primeiros slots
            if n_edge_filters >= 1:
                # Repetir para todos os canais de entrada
                for c in range(self.conv.weight.size(1)):
                    self.conv.weight[0, c, :3, :3] = sobel_x

            if n_edge_filters >= 2:
                for c in range(self.conv.weight.size(1)):
                    self.conv.weight[1, c, :3, :3] = sobel_y

            if n_edge_filters >= 3:
                for c in range(self.conv.weight.size(1)):
                    self.conv.weight[2, c, :3, :3] = laplacian

            if n_edge_filters >= 4:
                for c in range(self.conv.weight.size(1)):
                    self.conv.weight[3, c, :3, :3] = gaussian

            if n_edge_filters >= 5:
                # Adicionar filtros em outras orientações (45°, 135°)
                sobel_45 = torch.tensor([
                    [-2, -1, 0],
                    [-1, 0, 1],
                    [0, 1, 2]
                ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                sobel_135 = torch.tensor([
                    [0, 1, 2],
                    [-1, 0, 1],
                    [-2, -1, 0]
                ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                for c in range(self.conv.weight.size(1)):
                    self.conv.weight[4, c, :3, :3] = sobel_45

                if n_edge_filters >= 6:
                    for c in range(self.conv.weight.size(1)):
                        self.conv.weight[5, c, :3, :3] = sobel_135

            # 2. Preencher os restantes com distribuição normal
            if num_filters > n_edge_filters:
                nn.init.normal_(self.conv.weight[n_edge_filters:], mean=0.0, std=0.01)
                nn.init.zeros_(self.conv.bias[n_edge_filters:])

            # Fixar os pesos (não treináveis)
            for param in self.conv.parameters():
                param.requires_grad = False

    def extract_features(self, x):
        """Extrai características usando a camada convolucional fixa e pooling"""
        # Aplicar convolução
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)

        # Achatar para características
        x = torch.flatten(x, 1)
        return x

    def fit_pca(self, features):
        """Ajusta PCA nos dados de características extraídas"""
        if not self.use_pca:
            return features

        # Converter para numpy para PCA
        if torch.is_tensor(features):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        # Criar e ajustar PCA
        self.pca = PCA(n_components=self.pca_components)
        transformed = self.pca.fit_transform(features_np)

        self.pca_fitted = True

        # Retornar dados transformados
        if torch.is_tensor(features):
            return torch.from_numpy(transformed).to(features.device)
        return transformed

    def apply_pca(self, features):
        """Aplica PCA pré-ajustado nas características"""
        if not self.use_pca or not self.pca_fitted:
            return features

        # Converter para numpy para PCA
        if torch.is_tensor(features):
            features_np = features.cpu().numpy()
            transformed = self.pca.transform(features_np)
            return torch.from_numpy(transformed).float().to(features.device)
        else:
            return self.pca.transform(features)

    def normalize_features(self, x):
        """Normaliza as características usando os parâmetros armazenados"""
        if self.norm_mean is not None and self.norm_std is not None:
            # Converter para mesmo device se necessário
            if x.device != self.norm_mean.device:
                self.norm_mean = self.norm_mean.to(x.device)
                self.norm_std = self.norm_std.to(x.device)

            # Aplicar normalização
            x = (x - self.norm_mean) / self.norm_std
        return x

    def forward(self, x):
        """Forward pass com normalização e PCA"""
        # Extrair características
        x = self.extract_features(x)

        # Aplicar PCA se configurado
        if self.use_pca and self.pca_fitted:
            x = self.apply_pca(x)

        # Se estivermos no modo de inferência com pesos já calculados
        if self.output_weights is not None and not self.training:
            # Normalizar características
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