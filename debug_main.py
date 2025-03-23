# debug_main.py
# Script simplificado para testar e verificar a estrutura do projeto

import os
import sys


def check_imports():
    """Verifica se todas as importações necessárias estão funcionando"""
    print("Verificando importações...")

    try:
        # Testar importações de modelos
        print("Importando modelos...")
        from models.traditional_cnn import TraditionalCNN
        from models.celm import CELM
        from models.advanced_celm import AdvancedCELM

        try:
            from models.hybrid_celm import HybridCELM
            print("✓ HybridCELM importado com sucesso")
        except ImportError as e:
            print(f"✗ Erro ao importar HybridCELM: {e}")
            print("  Verifique se o arquivo models/hybrid_celm.py existe e está no formato correto.")

        # Testar importações de trainers
        print("\nImportando trainers...")
        from trainers.traditional_trainer import train_traditional_cnn
        from trainers.celm_trainer import train_celm

        try:
            from trainers.hybrid_celm_trainer import train_hybrid_celm
            print("✓ train_hybrid_celm importado com sucesso")
        except ImportError as e:
            print(f"✗ Erro ao importar train_hybrid_celm: {e}")
            print("  Verifique se o arquivo trainers/hybrid_celm_trainer.py existe e está no formato correto.")

        # Testar importações de utils
        print("\nImportando utils...")
        from utils.data_loader import load_mnist, get_device
        from utils.evaluation import evaluate_model

        # Testar importações de visualization
        print("\nImportando visualization...")
        try:
            from visualization.plotting import plot_training_comparison, plot_confusion_matrix, plot_speedup
            print("✓ Módulos de visualização importados com sucesso")
        except ImportError as e:
            print(f"✗ Erro ao importar módulos de visualização: {e}")
            print("  Verifique se o arquivo visualization/plotting.py existe e está no formato correto.")

        print("\n✓ Maioria das importações funcionando corretamente!")

    except ImportError as e:
        print(f"✗ Erro de importação: {e}")
        print("  Verifique a estrutura do projeto e os caminhos dos arquivos.")


def check_directory_structure():
    """Verifica a estrutura de diretórios do projeto"""
    print("\nVerificando estrutura de diretórios...")

    # Verificar diretórios principais
    directories = ["models", "trainers", "utils", "visualization"]
    for directory in directories:
        if os.path.isdir(directory):
            print(f"✓ Diretório '{directory}' encontrado")
        else:
            print(f"✗ Diretório '{directory}' não encontrado")
            os.makedirs(directory, exist_ok=True)
            print(f"  Diretório '{directory}' criado")

    # Verificar arquivos principais
    files_to_check = [
        "models/traditional_cnn.py",
        "models/celm.py",
        "models/advanced_celm.py",
        "models/hybrid_celm.py",
        "trainers/traditional_trainer.py",
        "trainers/celm_trainer.py",
        "trainers/hybrid_celm_trainer.py",
        "utils/data_loader.py",
        "utils/evaluation.py",
        "visualization/plotting.py",
        "main.py"
    ]

    for filepath in files_to_check:
        if os.path.isfile(filepath):
            print(f"✓ Arquivo '{filepath}' encontrado")
        else:
            print(f"✗ Arquivo '{filepath}' não encontrado")


def check_python_path():
    """Verifica a configuração do PYTHONPATH"""
    print("\nVerificando PYTHONPATH...")
    print("Diretório atual:", os.getcwd())
    print("Python path:", sys.path)

    # Adicionar o diretório atual ao path se não estiver lá
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        print(f"Adicionando diretório atual '{current_dir}' ao Python path")
        sys.path.append(current_dir)

    print("\nSugestão: execute este comando antes de rodar o main.py:")
    print(f"export PYTHONPATH={current_dir}:$PYTHONPATH")


def create_init_files():
    """Cria arquivos __init__.py em todos os diretórios necessários"""
    print("\nCriando arquivos __init__.py...")

    directories = ["models", "trainers", "utils", "visualization"]
    for directory in directories:
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.isfile(init_file):
            try:
                os.makedirs(directory, exist_ok=True)
                with open(init_file, 'w') as f:
                    f.write("# Arquivo de inicialização para o pacote\n")
                print(f"✓ Criado {init_file}")
            except Exception as e:
                print(f"✗ Erro ao criar {init_file}: {e}")
        else:
            print(f"✓ {init_file} já existe")


def check_basic_model_functionality():
    """Testa a criação básica dos modelos"""
    print("\nTestando criação básica dos modelos...")

    try:
        from models.traditional_cnn import TraditionalCNN
        model = TraditionalCNN()
        print("✓ TraditionalCNN criado com sucesso")
    except Exception as e:
        print(f"✗ Erro ao criar TraditionalCNN: {e}")

    try:
        from models.celm import CELM
        model = CELM()
        print("✓ CELM criado com sucesso")
    except Exception as e:
        print(f"✗ Erro ao criar CELM: {e}")

    try:
        from models.hybrid_celm import HybridCELM
        model = HybridCELM()
        print("✓ HybridCELM criado com sucesso")
    except Exception as e:
        print(f"✗ Erro ao criar HybridCELM: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("FERRAMENTA DE DIAGNÓSTICO DO PROJETO HYBRID-CELM")
    print("=" * 50)

    check_directory_structure()
    create_init_files()
    check_imports()
    check_python_path()
    try:
        check_basic_model_functionality()
    except Exception as e:
        print(f"\nErro ao testar funcionalidade básica: {e}")

    print("\n" + "=" * 50)
    print("PRÓXIMOS PASSOS SUGERIDOS:")
    print("=" * 50)
    print("1. Verifique se todos os arquivos necessários estão presentes")
    print("2. Adicione o diretório do projeto ao Python path")
    print("3. Execute 'python main.py' com a estrutura corrigida")
    print("=" * 50)