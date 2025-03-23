# celm_trainer.py

import torch
import numpy as np
import time


def train_celm(model, train_loader, device, reg_param=0.1):
    """Treina uma CELM calculando analiticamente os pesos da camada de saída"""
    model.to(device)
    model.train()
    start_time = time.time()

    # Coletar todas as características e rótulos
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            batch_features = model.extract_features(inputs) if hasattr(model, 'extract_features') else model(inputs)
            features.append(batch_features.cpu())
            labels.append(targets)

    # Concatenar todas as características e converter para one-hot
    X = torch.cat(features, dim=0).numpy()
    y_temp = torch.cat(labels, dim=0).numpy()

    # Converter rótulos para one-hot
    num_classes = model.output_size
    y = np.zeros((y_temp.shape[0], num_classes))
    for i in range(y_temp.shape[0]):
        y[i, y_temp[i]] = 1

    print(f"Calculando pesos de saída analiticamente: X shape {X.shape}, y shape {y.shape}")

    # Calcular pseudo-inversa e pesos de saída
    try:
        if X.shape[0] > X.shape[1]:  # Mais exemplos que características
            # W = (X^T X + λI)^(-1) X^T y
            XTX = X.T @ X + reg_param * np.eye(X.shape[1])
            inv_XTX = np.linalg.inv(XTX)
            beta = inv_XTX @ X.T @ y
        else:  # Mais características que exemplos
            # W = X^T (X X^T + λI)^(-1) y
            XXT = X @ X.T + reg_param * np.eye(X.shape[0])
            inv_XXT = np.linalg.inv(XXT)
            beta = X.T @ inv_XXT @ y
    except np.linalg.LinAlgError:
        print("Erro na inversão da matriz. Usando SVD para calcular a pseudo-inversa.")
        # Usar SVD como alternativa mais estável
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        s_inv = np.diag(1.0 / (s + reg_param))
        beta = Vh.T @ s_inv @ U.T @ y

    # Armazenar os pesos calculados no modelo
    model.output_weights = torch.from_numpy(beta).float().to(device)

    training_time = time.time() - start_time

    return {
        'model': model,
        'training_time': training_time
    }