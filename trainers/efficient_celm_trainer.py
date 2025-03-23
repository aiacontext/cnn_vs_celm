# trainers/efficient_celm_trainer.py

import torch
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg


def train_efficient_celm(model, train_loader, device, reg_param=0.01):
    """
    Treina uma CELM Eficiente com algoritmos state-of-art para matrizes esparsas.

    Args:
        model: Modelo CELM a ser treinado
        train_loader: DataLoader com dados de treinamento
        device: Dispositivo para computação (CPU/GPU)
        reg_param: Parâmetro de regularização Ridge

    Returns:
        Dictionary com resultados do treinamento
    """
    model.to(device)
    model.train()
    start_time = time.time()

    # Coletar características e rótulos
    print("Extraindo características de treinamento...")
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            batch_features = model.extract_features(inputs)
            features.append(batch_features.cpu())
            labels.append(targets)

    # Concatenar resultados
    X = torch.cat(features, dim=0)
    y_temp = torch.cat(labels, dim=0)

    # Converter para numpy para processamento eficiente
    X_np = X.numpy()

    # Aplicar PCA se configurado
    if model.use_pca:
        print(f"Aplicando PCA para reduzir dimensionalidade de {X_np.shape[1]} para {model.pca_components}")
        X_np = model.fit_pca(X_np)

    # Normalizar características para melhorar estabilidade numérica
    X_mean = np.mean(X_np, axis=0, keepdims=True)
    X_std = np.std(X_np, axis=0, keepdims=True) + 1e-8
    X_np = (X_np - X_mean) / X_std

    # Armazenar parâmetros de normalização no modelo (IMPORTANTE!)
    model.set_normalization_params(X_mean, X_std)

    # Converter rótulos para one-hot
    num_classes = model.output_size
    num_samples = y_temp.size(0)
    y_np = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        y_np[i, y_temp[i]] = 1

    print(f"Calculando pesos de saída analiticamente. X shape: {X_np.shape}, y shape: {y_np.shape}")

    # Calcular pesos de saída usando estado da arte para sistemas lineares esparsos
    beta = sparse_ridge_regression(X_np, y_np, reg_param)

    # Armazenar os pesos no modelo
    model.output_weights = torch.from_numpy(beta).float().to(device)

    # Capturar o tempo de treinamento
    training_time = time.time() - start_time

    return {
        'model': model,
        'training_time': training_time
    }


def sparse_ridge_regression(X, y, alpha):
    """
    Resolve regressão Ridge usando algoritmos otimizados para matrizes esparsas.
    Utiliza LSQR (Least Squares QR) que é especialmente eficiente para sistemas
    de grande escala, inclusive quando a matriz de design é esparsa.

    Args:
        X: Matriz de características (n_samples, n_features)
        y: Matriz de rótulos one-hot (n_samples, n_classes)
        alpha: Parâmetro de regularização

    Returns:
        beta: Coeficientes da regressão (n_features, n_classes)
    """
    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    beta = np.zeros((n_features, n_targets))

    # Resolver para cada classe separadamente
    for i in range(n_targets):
        # Configurar o sistema linear aumentado para incluir regularização L2
        if n_samples >= n_features:
            # Método para quando temos mais amostras que características
            # [X; sqrt(alpha)*I] * beta = [y; 0]
            X_augmented = sp.vstack([X, np.sqrt(alpha) * sp.eye(n_features)])
            y_augmented = np.vstack([y[:, i:i + 1], np.zeros((n_features, 1))])

            # Resolver usando LSQR (muito eficiente para sistemas esparsos)
            result, istop, itn, r1norm = scipy.sparse.linalg.lsqr(X_augmented, y_augmented.flatten())[:4]
            beta[:, i] = result

            # Informação de diagnóstico
            if istop != 1:
                print(f"LSQR warning for target {i}: istop={istop}, itn={itn}, r1norm={r1norm}")
                # Fallback para svd como último recurso
                if np.isnan(result).any() or np.isinf(result).any():
                    print(f"LSQR failed for target {i}, using SVD fallback")
                    beta[:, i] = ridge_regression_svd(X, y[:, i:i + 1], alpha)[:, 0]
        else:
            # Para n_features > n_samples, usar o dual é mais eficiente
            XXT = X @ X.T
            # Adicionar regularização
            for j in range(n_samples):
                XXT[j, j] += alpha

            # Resolver o sistema dual
            try:
                # Usar solver esparso LGMRES (mais rápido para sistemas grandes)
                solution, info = scipy.sparse.linalg.lgmres(XXT, y[:, i])

                if info != 0:
                    raise np.linalg.LinAlgError(f"LGMRES não convergiu: {info}")

                beta[:, i] = X.T @ solution
            except np.linalg.LinAlgError:
                print(f"LGMRES failed for target {i}, using SVD fallback")
                beta[:, i] = ridge_regression_svd(X, y[:, i:i + 1], alpha)[:, 0]

    return beta


def ridge_regression_svd(X, y, alpha):
    """
    Resolve regressão Ridge usando SVD.
    Usado como fallback quando os métodos esparsos falham.
    """
    U, s, Vh = scipy.linalg.svd(X, full_matrices=False)

    # Aplicar regularização nos valores singulares
    d = s / (s ** 2 + alpha)

    # Calcular coeficientes ridge
    beta = (Vh.T * d[:, np.newaxis]) @ (U.T @ y)

    return beta