# utils/evalution.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, test_loader, device, model_name="Model"):
    """Avalia um modelo treinado no conjunto de teste"""
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Obter previsões
            _, predictions = torch.max(outputs, 1)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Calcular métricas
    accuracy = accuracy_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    class_report = classification_report(all_targets, all_preds, output_dict=True)

    print(f"{model_name} - Acurácia: {accuracy:.4f}")

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets)
    }