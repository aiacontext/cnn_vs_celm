#!/usr/bin/env python
# main.py - Script principal para comparar todos os modelos CELM com suporte a MPS

import argparse
import os
import torch
import time
import gc

from models.traditional_cnn import TraditionalCNN
from models.celm import CELM
from models.advanced_celm import AdvancedCELM
from models.optimized_celm import OptimizedCELM
from models.efficient_celm import EfficientCELM
from trainers.traditional_trainer import train_traditional_cnn
from trainers.celm_trainer import train_celm
from trainers.optimized_celm_trainer import train_optimized_celm
from trainers.efficient_celm_trainer import train_efficient_celm
from utils.data_loader import load_mnist, get_device
from utils.evaluation import evaluate_model
from visualization.plotting import plot_training_comparison, plot_confusion_matrix, plot_speedup


def parse_args():
    parser = argparse.ArgumentParser(description='Comparação entre todos os modelos CELM')

    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for traditional CNN')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for data')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory for output files')

    # Parâmetros para CELM Otimizada
    parser.add_argument('--opt-reg-param', type=float, default=0.01, help='Regularization parameter for optimized CELM')
    parser.add_argument('--opt-scale-factor', type=float, default=0.5, help='Scale factor for optimized CELM')
    parser.add_argument('--opt-feature-dim', type=int, default=64, help='Feature dimension for optimized CELM')

    # Parâmetros para CELM Eficiente
    parser.add_argument('--eff-reg-param', type=float, default=0.01, help='Regularization parameter for efficient CELM')
    parser.add_argument('--eff-num-filters', type=int, default=64, help='Number of filters for efficient CELM')
    parser.add_argument('--eff-pca-components', type=int, default=128, help='PCA components for efficient CELM')

    # Modelos a incluir na comparação
    parser.add_argument('--skip-cnn', action='store_true', help='Skip traditional CNN')
    parser.add_argument('--skip-celm-basic', action='store_true', help='Skip basic CELM')
    parser.add_argument('--skip-celm-advanced', action='store_true', help='Skip advanced CELM')
    parser.add_argument('--skip-celm-optimized', action='store_true', help='Skip optimized CELM')
    parser.add_argument('--skip-celm-efficient', action='store_true', help='Skip efficient CELM')

    return parser.parse_args()


def clean_memory(device):
    """Limpa memória com suporte adequado para CUDA, MPS e CPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS não tem uma função equivalente ao empty_cache(),
        # mas podemos garantir que a coleta de lixo seja eficaz
        # forçando a sincronização com o dispositivo
        if device.type == 'mps':
            torch.mps.synchronize()


def main():
    # Parse command line arguments
    args = parse_args()

    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)

    # Configurar dispositivo e carregar dados
    device = get_device()

    train_loader, test_loader = load_mnist(batch_size=args.batch_size, data_dir=args.data_dir)
    print(
        f"Dataset MNIST carregado: {len(train_loader.dataset)} exemplos de treino, {len(test_loader.dataset)} exemplos de teste")

    # Dicionário para armazenar resultados
    results = {}

    # 1. Treinar CNN Tradicional (se não for skipped)
    if not args.skip_cnn:
        print("\n--- Treinando CNN Tradicional ---")
        # Limpar memória
        clean_memory(device)

        start_time = time.time()
        cnn_model = TraditionalCNN()
        cnn_results = train_traditional_cnn(
            model=cnn_model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs
        )

        # Avaliar CNN Tradicional
        print("\n--- Avaliando CNN Tradicional ---")
        cnn_eval = evaluate_model(
            model=cnn_results['model'],
            test_loader=test_loader,
            device=device,
            model_name="CNN Tradicional"
        )

        # Armazenar resultados
        results['CNN Tradicional'] = {
            'accuracy': cnn_eval['accuracy'],
            'training_time': cnn_results['training_time'],
            'total_time': time.time() - start_time,
            'evaluation': cnn_eval
        }

        # Liberar memória
        del cnn_model
        clean_memory(device)

    # 2. Treinar CELM Básica (se não for skipped)
    if not args.skip_celm_basic:
        print("\n--- Treinando CELM Básica ---")
        # Limpar memória
        clean_memory(device)

        start_time = time.time()
        basic_celm_model = CELM()
        basic_celm_results = train_celm(
            model=basic_celm_model,
            train_loader=train_loader,
            device=device
        )

        # Avaliar CELM Básica
        print("\n--- Avaliando CELM Básica ---")
        basic_celm_eval = evaluate_model(
            model=basic_celm_results['model'],
            test_loader=test_loader,
            device=device,
            model_name="CELM Básica"
        )

        # Armazenar resultados
        results['CELM Básica'] = {
            'accuracy': basic_celm_eval['accuracy'],
            'training_time': basic_celm_results['training_time'],
            'total_time': time.time() - start_time,
            'evaluation': basic_celm_eval
        }

        # Liberar memória
        del basic_celm_model
        clean_memory(device)

    # 3. Treinar CELM Avançada (se não for skipped)
    if not args.skip_celm_advanced:
        print("\n--- Treinando CELM Avançada ---")
        # Limpar memória
        clean_memory(device)

        start_time = time.time()
        advanced_celm_model = AdvancedCELM()
        advanced_celm_results = train_celm(
            model=advanced_celm_model,
            train_loader=train_loader,
            device=device
        )

        # Avaliar CELM Avançada
        print("\n--- Avaliando CELM Avançada ---")
        advanced_celm_eval = evaluate_model(
            model=advanced_celm_results['model'],
            test_loader=test_loader,
            device=device,
            model_name="CELM Avançada"
        )

        # Armazenar resultados
        results['CELM Avançada'] = {
            'accuracy': advanced_celm_eval['accuracy'],
            'training_time': advanced_celm_results['training_time'],
            'total_time': time.time() - start_time,
            'evaluation': advanced_celm_eval
        }

        # Liberar memória
        del advanced_celm_model
        clean_memory(device)

    # 4. Treinar CELM Otimizada (se não for skipped)
    if not args.skip_celm_optimized:
        print("\n--- Treinando CELM Otimizada ---")
        # Limpar memória
        clean_memory(device)

        start_time = time.time()
        optimized_celm_model = OptimizedCELM(feature_dim=args.opt_feature_dim)
        # Aplicar scale factor (importante para a performance)
        optimized_celm_model.scale_factor = args.opt_scale_factor
        optimized_celm_model._init_filters()  # Reinicializar com o novo fator

        optimized_celm_results = train_optimized_celm(
            model=optimized_celm_model,
            train_loader=train_loader,
            device=device,
            reg_param=args.opt_reg_param,
            ridge_solver='svd'
        )

        # Avaliar CELM Otimizada
        print("\n--- Avaliando CELM Otimizada ---")
        optimized_celm_eval = evaluate_model(
            model=optimized_celm_results['model'],
            test_loader=test_loader,
            device=device,
            model_name="CELM Otimizada"
        )

        # Armazenar resultados
        results['CELM Otimizada'] = {
            'accuracy': optimized_celm_eval['accuracy'],
            'training_time': optimized_celm_results['training_time'],
            'total_time': time.time() - start_time,
            'evaluation': optimized_celm_eval
        }

        # Liberar memória
        del optimized_celm_model
        clean_memory(device)

    # 5. Treinar CELM Eficiente (se não for skipped)
    if not args.skip_celm_efficient:
        print("\n--- Treinando CELM Eficiente ---")
        # Limpar memória
        clean_memory(device)

        start_time = time.time()
        efficient_celm_model = EfficientCELM(
            num_filters=args.eff_num_filters,
            pca_components=args.eff_pca_components
        )

        efficient_celm_results = train_efficient_celm(
            model=efficient_celm_model,
            train_loader=train_loader,
            device=device,
            reg_param=args.eff_reg_param
        )

        # Avaliar CELM Eficiente
        print("\n--- Avaliando CELM Eficiente ---")
        efficient_celm_eval = evaluate_model(
            model=efficient_celm_results['model'],
            test_loader=test_loader,
            device=device,
            model_name="CELM Eficiente"
        )

        # Armazenar resultados
        results['CELM Eficiente'] = {
            'accuracy': efficient_celm_eval['accuracy'],
            'training_time': efficient_celm_results['training_time'],
            'total_time': time.time() - start_time,
            'evaluation': efficient_celm_eval
        }

        # Liberar memória
        del efficient_celm_model
        clean_memory(device)

    # Verificar se temos modelos suficientes para comparação
    if len(results) < 1:
        print("Nenhum modelo foi treinado. Por favor, desative as flags de 'skip'.")
        return

    # Visualizar resultados
    try:
        # Preparar dados para visualização
        plot_data = {name: {'evaluation': result['evaluation'], 'training_time': result['training_time']}
                     for name, result in results.items()}

        plot_training_comparison(
            results=plot_data,
            save_path=os.path.join(args.output_dir, 'training_comparison.png')
        )

        # Criar matrizes de confusão para todos os modelos
        class_names = [str(i) for i in range(10)]  # Classes 0-9 para MNIST

        for model_name, model_results in results.items():
            cm = model_results['evaluation']['confusion_matrix']
            plot_confusion_matrix(
                cm=cm,
                class_names=class_names,
                title=f'Matriz de Confusão - {model_name}',
                save_path=os.path.join(args.output_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
            )

        # Calcular e plotar speedups em relação à CNN Tradicional
        if 'CNN Tradicional' in results:
            cnn_time = results['CNN Tradicional']['training_time']
            speedup_data = {}

            for model_name, model_results in results.items():
                if model_name != 'CNN Tradicional':
                    model_time = model_results['training_time']
                    speedup_data[model_name] = cnn_time / model_time

            if speedup_data:
                plot_speedup(
                    speedup_data=speedup_data,
                    base_model="CNN Tradicional",
                    save_path=os.path.join(args.output_dir, 'speedup.png')
                )
    except Exception as e:
        print(f"Erro ao gerar visualizações: {e}")

    # Imprimir resumo detalhado
    print("\n----- Resultados Detalhados da Comparação -----")

    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print(f"  - Acurácia: {model_results['accuracy']:.4f}")
        print(f"  - Tempo de treinamento: {model_results['training_time']:.2f} segundos")

        if model_name == 'CELM Otimizada':
            print(f"  - Configurações:")
            print(f"    * Feature Dimension: {args.opt_feature_dim}")
            print(f"    * Scale Factor: {args.opt_scale_factor}")
            print(f"    * Regularização (λ): {args.opt_reg_param}")

        if model_name == 'CELM Eficiente':
            print(f"  - Configurações:")
            print(f"    * Número de filtros: {args.eff_num_filters}")
            print(f"    * Componentes PCA: {args.eff_pca_components}")
            print(f"    * Regularização (λ): {args.eff_reg_param}")

    # Calcular speedups e eficiência
    if 'CNN Tradicional' in results:
        cnn_time = results['CNN Tradicional']['training_time']
        cnn_acc = results['CNN Tradicional']['accuracy']

        print("\nSpeedups vs CNN Tradicional:")
        for model_name, model_results in results.items():
            if model_name != 'CNN Tradicional':
                speedup = cnn_time / model_results['training_time']
                print(f"  - {model_name}: {speedup:.2f}x")

        print("\nEficiência (acurácia/tempo):")
        cnn_efficiency = cnn_acc / cnn_time
        print(f"  - CNN Tradicional: {cnn_efficiency:.6f}")

        for model_name, model_results in results.items():
            if model_name != 'CNN Tradicional':
                model_efficiency = model_results['accuracy'] / model_results['training_time']
                rel_efficiency = model_efficiency / cnn_efficiency
                print(f"  - {model_name}: {model_efficiency:.6f} ({rel_efficiency:.2f}x vs CNN)")

    # Salvar resultados para referência futura
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write("----- Resultados da Comparação de Modelos -----\n\n")

        for model_name, model_results in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  - Acurácia: {model_results['accuracy']:.4f}\n")
            f.write(f"  - Tempo de treinamento: {model_results['training_time']:.2f} segundos\n")

            if model_name == 'CELM Otimizada':
                f.write(f"  - Configurações:\n")
                f.write(f"    * Feature Dimension: {args.opt_feature_dim}\n")
                f.write(f"    * Scale Factor: {args.opt_scale_factor}\n")
                f.write(f"    * Regularização (λ): {args.opt_reg_param}\n")

            if model_name == 'CELM Eficiente':
                f.write(f"  - Configurações:\n")
                f.write(f"    * Número de filtros: {args.eff_num_filters}\n")
                f.write(f"    * Componentes PCA: {args.eff_pca_components}\n")
                f.write(f"    * Regularização (λ): {args.eff_reg_param}\n")

            f.write("\n")

        if 'CNN Tradicional' in results:
            cnn_time = results['CNN Tradicional']['training_time']
            cnn_acc = results['CNN Tradicional']['accuracy']
            cnn_efficiency = cnn_acc / cnn_time

            f.write("Speedups vs CNN Tradicional:\n")
            for model_name, model_results in results.items():
                if model_name != 'CNN Tradicional':
                    speedup = cnn_time / model_results['training_time']
                    f.write(f"  - {model_name}: {speedup:.2f}x\n")

            f.write("\nEficiência (acurácia/tempo):\n")
            f.write(f"  - CNN Tradicional: {cnn_efficiency:.6f}\n")

            for model_name, model_results in results.items():
                if model_name != 'CNN Tradicional':
                    model_efficiency = model_results['accuracy'] / model_results['training_time']
                    rel_efficiency = model_efficiency / cnn_efficiency
                    f.write(f"  - {model_name}: {model_efficiency:.6f} ({rel_efficiency:.2f}x vs CNN)\n")

    print(f"\nOs resultados foram salvos em: {os.path.join(args.output_dir, 'results.txt')}")


if __name__ == "__main__":
    main()