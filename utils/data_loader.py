# utils/data_loader.py

import torch
import torchvision
import torchvision.transforms as transforms
import os
import ssl


def get_device():
    """Determina o dispositivo disponível (MPS, CUDA ou CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def load_mnist(batch_size=64, data_dir='./data'):
    """Carrega e prepara o dataset MNIST"""

    # Corrige o problema de certificado SSL no macOS
    ssl._create_default_https_context = ssl._create_unverified_context

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Cria o diretório de dados se não existir
    os.makedirs(data_dir, exist_ok=True)

    try:
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
    except Exception as e:
        print(f"Erro ao baixar MNIST usando torchvision: {e}")
        print("Tentando abordagem alternativa...")

        # Abordagem alternativa: usar uma fonte mais confiável
        import urllib.request
        import gzip
        from pathlib import Path

        # Criar diretórios necessários
        mnist_dir = Path(data_dir) / "MNIST" / "raw"
        mnist_dir.mkdir(parents=True, exist_ok=True)

        # URLs para o dataset MNIST (fonte alternativa)
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        files = {
            "train_images.gz": "train-images-idx3-ubyte.gz",
            "train_labels.gz": "train-labels-idx1-ubyte.gz",
            "test_images.gz": "t10k-images-idx3-ubyte.gz",
            "test_labels.gz": "t10k-labels-idx1-ubyte.gz"
        }

        # Baixar arquivos
        for file_key, file_name in files.items():
            file_path = mnist_dir / file_name
            if not file_path.exists():
                print(f"Baixando {file_name}...")
                url = base_url + file_name
                urllib.request.urlretrieve(url, file_path)

        # Tentar carregar o dataset novamente
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=False,  # Já baixado
            transform=transform
        )

        test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=False,  # Já baixado
            transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader