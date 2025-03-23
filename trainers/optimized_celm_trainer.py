# trainers/optimized_celm_trainer.py (Corrigido)

import torch
import numpy as np
import time
import scipy.linalg


def train_optimized_celm(model, train_loader, device, reg_param=0.01, ridge_solver='svd'):
    """
    Treina uma CELM Otimizada com método analítico melhorado.

    Args:
        model: Modelo CELM a ser treinado
        train_loader: DataLoader com dados de treinamento
        device: Dispositivo para computação (CPU/GPU)
        reg_param: Parâmetro de regularização Ridge
        ridge_solver: Método para resolver regressão Ridge ('cholesky', 'svd', ou 'iterative')

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

    # Calcular pesos de saída usando o método escolhido
    if ridge_solver == 'cholesky':
        beta = ridge_regression_cholesky(X_np, y_np, reg_param)
    elif ridge_solver == 'svd':
        beta = ridge_regression_svd(X_np, y_np, reg_param)
    elif ridge_solver == 'iterative':
        beta = ridge_regression_iterative(X_np, y_np, reg_param)
    else:
        raise ValueError(f"Método desconhecido: {ridge_solver}")

    # Armazenar os pesos diretamente (sem normalização adicional, pois já foi aplicada aos dados)
    # IMPORTANTE: Não normalizamos os pesos aqui, pois a normalização já é aplicada nas features
    model.output_weights = torch.from_numpy(beta).float().to(device)

    # Capturar o tempo de treinamento
    training_time = time.time() - start_time

    return {
        'model': model,
        'training_time': training_time
    }


def ridge_regression_cholesky(X, y, alpha):
    """
    Resolve regressão Ridge usando decomposição de Cholesky (método mais estável).

    Ridge: beta = (X^T X + alpha*I)^(-1) X^T y
    """
    n_samples, n_features = X.shape

    # Método mais eficiente quando n_samples >= n_features
    if n_samples >= n_features:
        # Formar matriz de Gram
        XTX = X.T @ X

        # Adicionar regularização
        reg_matrix = alpha * np.eye(n_features)
        XTX_reg = XTX + reg_matrix

        # Resolver sistema usando Cholesky (muito mais estável que inversão direta)
        try:
            L = scipy.linalg.cholesky(XTX_reg, lower=True)
            temp = scipy.linalg.solve_triangular(L, X.T @ y, lower=True)
            beta = scipy.linalg.solve_triangular(L.T, temp, lower=False)
        except np.linalg.LinAlgError:
            print("Erro na decomposição de Cholesky. Usando SVD como fallback.")
            beta = ridge_regression_svd(X, y, alpha)
    else:
        # Para n_features > n_samples, usamos o dual
        # beta = X^T (X X^T + alpha*I)^(-1) y
        XXT = X @ X.T
        reg_matrix = alpha * np.eye(n_samples)
        XXT_reg = XXT + reg_matrix

        try:
            L = scipy.linalg.cholesky(XXT_reg, lower=True)
            temp = scipy.linalg.solve_triangular(L, y, lower=True)
            solution = scipy.linalg.solve_triangular(L.T, temp, lower=False)
            beta = X.T @ solution
        except np.linalg.LinAlgError:
            print("Erro na decomposição de Cholesky na forma dual. Usando SVD como fallback.")
            beta = ridge_regression_svd(X, y, alpha)

    return beta


def ridge_regression_svd(X, y, alpha):
    """
    Resolve regressão Ridge usando SVD.
    Mais lento que Cholesky, mas mais estável para matrizes mal condicionadas.
    """
    # Usar scipy.linalg.svd para melhor estabilidade numérica
    U, s, Vh = scipy.linalg.svd(X, full_matrices=False, lapack_driver='gesdd')

    # Aplicar regularização nos valores singulares
    d = s / (s ** 2 + alpha)

    # Calcular coeficientes ridge
    beta = (Vh.T * d[:, np.newaxis]) @ (U.T @ y)

    return beta


def ridge_regression_iterative(X, y, alpha, max_iter=100, tol=1e-4):
    """
    Resolve regressão Ridge usando método iterativo de gradiente conjugado.
    Bom para datasets muito grandes quando n_features >> n_samples.
    """
    from scipy.sparse.linalg import cg

    n_samples, n_features = X.shape
    n_targets = y.shape[1]
    beta = np.zeros((n_features, n_targets))

    # Definir função para multiplicação por (X^T X + alpha*I)
    def matvec(v):
        return X.T @ (X @ v) + alpha * v

    # Resolver para cada alvo separadamente
    for i in range(n_targets):
        b = X.T @ y[:, i]
        beta[:, i], info = cg(matvec, b, tol=tol, maxiter=max_iter)

        if info > 0:
            print(f"Aviso: o método CG não convergiu para o alvo {i}.")
        elif info < 0:
            print(f"Erro: problema com o método CG para o alvo {i}. Usando SVD.")
            beta[:, i] = ridge_regression_svd(X, y[:, i:i + 1], alpha)[:, 0]

    return beta