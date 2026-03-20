# Tech Challenge – Fase 1

## Sistema Inteligente de Suporte ao Diagnóstico em Saúde Feminina

**Curso:** Pós-Graduação em IA para Devs – FIAP  
**Grupo:** GFA

---

## Sobre o Projeto

Este projeto desenvolve uma solução de Machine Learning para apoiar profissionais de saúde na **identificação precoce de riscos e diagnósticos** relacionados à saúde da mulher.

O sistema analisa dados médicos estruturados para detectar padrões de risco, como:

- **Câncer de mama** (dataset Wisconsin)
- **Síndrome dos Ovários Policísticos – PCOS** (dataset Kaggle)

---

## Estrutura do Projeto

```
TechChallenge1GFA/
│
├── data/
│   ├── raw/            # Datasets originais baixados do Kaggle (não versionar arquivos grandes)
│   └── processed/      # Dados após limpeza e transformações
│
├── notebooks/
│   ├── 01_eda.ipynb           # Análise Exploratória de Dados
│   ├── 02_preprocessing.ipynb # Pré-processamento e pipeline
│   ├── 03_modeling.ipynb      # Treino e avaliação dos modelos
│   └── 04_cnn_extra.ipynb     # EXTRA: CNN para imagens de mamografia
│
├── src/
│   ├── utils.py           # Funções auxiliares reutilizáveis
│   ├── preprocessing.py   # Pipeline de pré-processamento
│   └── models.py          # Definição e avaliação dos modelos
│
├── outputs/
│   ├── figures/    # Gráficos e visualizações exportados
│   └── models/     # Modelos treinados salvos (.pkl)
│
├── requirements.txt   # Dependências Python
├── Dockerfile         # Container para reprodutibilidade
└── README.md
```

---

## Datasets

| Dataset                 | Problema                                | Fonte                                                                                       |
| ----------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------- |
| Breast Cancer Wisconsin | Classificação: tumor maligno ou benigno | [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)           |
| PCOS Dataset            | Classificação: presença de SOP          | [Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos) |
| CBIS-DDSM _(extra)_     | Classificação de mamografias por CNN    | [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)     |

> **Como usar:** Baixe os datasets dos links acima e coloque os arquivos na pasta `data/raw/`.

---

## Etapas do Projeto

### 1. Análise Exploratória (EDA) — `01_eda.ipynb`

- Carregamento dos dados
- Estatísticas descritivas (média, mediana, desvio padrão)
- Visualização de distribuições e correlações
- Identificação de valores ausentes e outliers

### 2. Pré-processamento — `02_preprocessing.ipynb`

- Limpeza de dados (valores nulos, inconsistências)
- Conversão de variáveis categóricas (Label Encoding / One-Hot)
- Normalização/padronização de variáveis numéricas
- Análise de correlação e seleção de features

### 3. Modelagem — `03_modeling.ipynb`

- Três algoritmos implementados:
  - Regressão Logística
  - Árvore de Decisão
  - K-Nearest Neighbors (KNN)
- Divisão treino/teste (80/20)
- Métricas: Accuracy, Recall, F1-score, Matriz de Confusão
- Explicabilidade: Feature Importance + SHAP

### 4. CNN – Extra — `04_cnn_extra.ipynb`

- Rede Neural Convolucional para análise de imagens de mamografia
- Transfer Learning com modelo pré-treinado
- Avaliação com métricas de classificação de imagem

---

## Como Executar

### Com Docker

```bash
docker build -t tech-challenge-gfa .
docker run -p 8888:8888 tech-challenge-gfa
```

### Sem Docker

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## Tecnologias

- Python 3.11
- Jupyter Notebook
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- SHAP
- TensorFlow / Keras _(extra CNN)_
- Docker
