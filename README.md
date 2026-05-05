# Tech Challenge – Fase 1

## Sistema Inteligente de Suporte ao Diagnóstico em Saúde Feminina

**Curso:** Pós-Graduação em IA para Devs – FIAP  
**Grupo/Integrantes:** Guilherme Ferreira de Arruda  
**Proposta escolhida:** A — Saúde da Mulher

---

## Sobre o Projeto

Este projeto tem como foco, desenvolver uma solução com uso de Machine Learning para analisar dados que podem ajudar na **identificação precoce de riscos e diagnósticos** relacionados à saúde da mulher, neste caso mais específico, se trata do ** Câncer de Mama **.

> **Relatório Técnico completo:** [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md)  
> Contém análise exploratória, estratégias de pré-processamento, modelos avaliados, métricas, explicabilidade e análise crítica dos resultados.

---

## DatasetS Utilizados

| Dataset                 | Problema                                | Fonte                                                                             |
| ----------------------- | --------------------------------------- | --------------------------------------------------------------------------------- |
| Breast Cancer Wisconsin | Classificação: tumor maligno ou benigno | [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) |

> **Como usar:** Baixe o dataset do link acima e coloque o arquivo na pasta `data/raw/`.

### Configurando a Kaggle API (opcional)

Em vez de baixar manualmente, você pode usar a Kaggle API:

**1. Instale a biblioteca:**

```bash
pip install kaggle
```

**2. Obtenha seu token de autenticação:**

- Acesse [kaggle.com](https://www.kaggle.com) > sua conta > **Settings** > **API** > **Create New Token**
- Será baixado um arquivo `kaggle.json` com seu `username` e `key`

**3. Coloque o arquivo na pasta correta:**

- **Linux/Mac:** `~/.kaggle/kaggle.json`
- **Windows:** `C:\Users\<seu-usuario>\.kaggle\kaggle.json`

```bash
# Linux/Mac: ajuste as permissões
chmod 600 ~/.kaggle/kaggle.json
```

**4. Baixe o dataset:**

```bash
# Breast Cancer Wisconsin
kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p data/raw/ --unzip
```

---

## Etapas do Projeto

### 1. Análise Exploratória (EDA) — `analise_exploratoria.ipynb`

- Carregamento dos dados
- Estatísticas descritivas (média, mediana, desvio padrão)
- Visualização de distribuições e correlações
- Identificação de valores ausentes e outliers

### 2. Pré-processamento — `preprocessamento.ipynb`

- Limpeza de dados (valores nulos, inconsistências)
- Conversão de variáveis categóricas (Label Encoding / One-Hot)
- Normalização/padronização de variáveis numéricas
- Análise de correlação e seleção de features

### 3. Modelagem — `modelagem.ipynb`

- Três algoritmos implementados:
  - Regressão Logística
  - Árvore de Decisão
  - K-Nearest Neighbors / KNN
- Divisão treino/teste
- Métricas: Accuracy, Recall, F1-score, Matriz de Confusão...

### 4. Avaliação e Explicabilidade — `explicabilidade.ipynb`

- Consolidação das métricas finais da Regressão Logística
- Feature Importance por coeficientes
- SHAP global (bar e beeswarm)
- SHAP local (caso correto e falso negativo)
- Análise de erros (FN/FP)
- Ajuste controlado dentro do escopo (class_weight + threshold)

### Modelo Final Oficial

- Arquivo: `outputs/models/modelo_final_oficial.pkl`
- Scaler: `outputs/models/scaler_final_oficial.pkl`
- Melhor configuração encontrada: Regressão Logística com `class_weight='balanced'`
- Métricas no teste:
  - Accuracy: 0.9737
  - Precision maligno: 0.9756
  - Recall maligno: 0.9524
  - F1 maligno: 0.9639
  - Falsos negativos: 2
  - Falsos positivos: 1

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

### Inferência Rápida (CSV)

Use o script `src/inferencia.py` para gerar predições em novos dados sem precisar abrir o Jupyter. Ele carrega o modelo e o scaler salvos, aplica a padronização e salva as predições em CSV.

- `--input`: caminho para o CSV com as features (sem a coluna alvo)
- `--output`: caminho onde o arquivo de predições será salvo

```bash
python src/inferencia.py --input data/processed/X_test_raw.csv --output outputs/models/predicoes_teste.csv
```

---

## Tecnologias

- Python 3.11
- Jupyter Notebook
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- SHAP
- Docker
