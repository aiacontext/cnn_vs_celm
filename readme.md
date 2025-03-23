# CNN vs CELM: Comparação de Modelos de Redes Neurais

Este repositório contém implementações de diferentes arquiteturas de redes neurais para comparação de desempenho, com foco especial em modelos Convolutional Extreme Learning Machine (CELM) e suas variantes.

## Modelos Implementados

- **CNN Tradicional**: Modelo baseado em backpropagation com treinamento de todas as camadas.
- **CELM Básica**: Implementação básica da CELM com inicialização aleatória.
- **CELM Avançada**: Versão avançada com filtros Gabor e arquitetura melhorada.
- **CELM Otimizada**: CELM com otimizações para normalização e estabilidade numérica.
- **CELM Eficiente**: Implementação simplificada com filtros predefinidos e seleção de características via PCA.

## Estrutura do Projeto

```
cnn_vs_celm/
├── models/             # Implementações dos modelos
│   ├── traditional_cnn.py
│   ├── celm.py
│   ├── advanced_celm.py
│   ├── optimized_celm.py
│   └── efficient_celm.py
├── trainers/           # Implementações dos treinadores
│   ├── traditional_trainer.py
│   ├── celm_trainer.py
│   ├── optimized_celm_trainer.py
│   └── efficient_celm_trainer.py
├── utils/              # Funções utilitárias
│   ├── data_loader.py
│   └── evaluation.py
├── visualization/      # Visualização e análise de resultados
│   └── plotting.py
└── main.py             # Script principal para execução
```

## Requisitos

O projeto requer Python 3.8+ e as seguintes bibliotecas:
- torch
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn

Todos os requisitos estão listados no arquivo `requirements.txt`.

## Instalação

```bash
# Clone o repositório
git clone https://github.com/aiacontext/cnn_vs_celm.git
cd cnn_vs_celm

# Instale as dependências
pip install -r requirements.txt
```

## Uso

### Executar Comparação Completa

Para comparar todos os modelos implementados:

```bash
python main.py
```

### Comparar Modelos Específicos

Para comparar apenas alguns modelos:

```bash
# Exemplo: comparar apenas CNN Tradicional e CELM Eficiente
python main.py --skip-celm-basic --skip-celm-advanced --skip-celm-optimized
```

### Ajustar Parâmetros da CELM Eficiente

```bash
python main.py --eff-num-filters 96 --eff-pca-components 192 --eff-reg-param 0.005
```

## Principais Inovações da CELM Eficiente

1. **Filtros Convolucionais Predefinidos**: Kernels como Sobel, Laplaciano e Gaussiano em vez de inicialização aleatória.
2. **Arquitetura Simplificada**: Uma única camada convolucional mais potente.
3. **Redução Agressiva de Dimensionalidade**: Triplo estágio de pooling para redução de parâmetros.
4. **Compressão PCA Integrada**: Redução de dimensionalidade e extração de características relevantes.
5. **Solvers Esparsos State-of-Art**: Algoritmos LSQR e LGMRES para solução eficiente de sistemas lineares.

## Resultados

Os resultados das comparações são salvos no diretório `results/`, incluindo:
- Gráficos de acurácia
- Gráficos de tempo de treinamento
- Matrizes de confusão
- Métricas de speedup
- Eficiência (acurácia/tempo)

## Citação

Se você utilizar este código em sua pesquisa, por favor cite:

```bibtex
@software{leitao2025cnn,
  author = {Leitao Filho, Antonio de Sousa},
  title = {CNN vs CELM: Comparação de Modelos de Redes Neurais},
  institution = {Aia Context},
  year = {2025},
  url = {https://github.com/aiacontext/cnn_vs_celm}
}
```

## Licença

Este projeto está licenciado sob os termos da licença MIT.
