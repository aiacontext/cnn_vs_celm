# visualization/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_comparison(results, save_path=None):
    """
    Plota uma comparação de acurácia e tempo de treinamento para os diferentes modelos.

    Args:
        results (dict): Dicionário contendo resultados de cada modelo
        save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico é exibido.
    """
    # Preparar dados
    model_names = list(results.keys())
    accuracies = [results[name]['evaluation']['accuracy'] * 100 for name in model_names]
    train_times = [results[name]['training_time'] for name in model_names]

    # Configurar subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico de acurácia
    bar_width = 0.5
    x_pos = np.arange(len(model_names))
    ax1.bar(x_pos, accuracies, bar_width, color='skyblue', edgecolor='black')
    ax1.set_ylabel('Acurácia (%)')
    ax1.set_title('Comparação de Acurácia')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    for i, acc in enumerate(accuracies):
        ax1.text(i, acc - 5, f'{acc:.2f}%', ha='center', va='bottom',
                 color='black', fontweight='bold')

    # Gráfico de tempo de treinamento
    ax2.bar(x_pos, train_times, bar_width, color='lightgreen', edgecolor='black')
    ax2.set_ylabel('Tempo de Treinamento (s)')
    ax2.set_title('Comparação de Tempo de Treinamento')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    for i, time in enumerate(train_times):
        ax2.text(i, time - (max(train_times) * 0.1), f'{time:.2f}s',
                 ha='center', va='bottom', color='black', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm, class_names, title='Matriz de Confusão', save_path=None):
    """
    Plota uma matriz de confusão.

    Args:
        cm (array): Matriz de confusão
        class_names (list): Lista com nomes das classes
        title (str): Título do gráfico
        save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico é exibido.
    """
    plt.figure(figsize=(10, 8))

    # Normalizar a matriz para porcentagens
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plotar usando seaborn
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title(title, fontsize=16)
    plt.ylabel('Classes Reais', fontsize=12)
    plt.xlabel('Classes Previstas', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_speedup(speedup_data, base_model="CNN Tradicional", save_path=None):
    """
    Plota o speedup de diferentes modelos em relação ao modelo base.

    Args:
        speedup_data (dict): Dicionário com nomes dos modelos e seus speedups
        base_model (str): Nome do modelo base de comparação
        save_path (str, optional): Caminho para salvar o gráfico. Se None, o gráfico é exibido.
    """
    model_names = list(speedup_data.keys())
    speedups = list(speedup_data.values())

    plt.figure(figsize=(10, 6))

    bar_width = 0.5
    x_pos = np.arange(len(model_names))
    bars = plt.bar(x_pos, speedups, bar_width, color='salmon', edgecolor='black')

    # Adicionar linha de referência para o modelo base (speedup = 1)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7,
                label=f'Referência ({base_model})')

    plt.ylabel(f'Speedup (vs {base_model})', fontsize=12)
    plt.title(f'Ganho de Velocidade em Relação a {base_model}', fontsize=16)
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{speedups[i]:.2f}x', ha='center', va='bottom',
                 color='black', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()