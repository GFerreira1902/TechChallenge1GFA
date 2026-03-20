"""
utils.py
--------
Funções auxiliares reutilizáveis em todo o projeto.
"""

import os
import joblib
import matplotlib.pyplot as plt


def salvar_figura(nome_arquivo: str, pasta: str = "outputs/figures"):
    """
    Salva o gráfico atual do matplotlib como PNG.
    
    Parâmetros:
        nome_arquivo: nome do arquivo (ex: 'distribuicao_idade.png')
        pasta: pasta de destino (padrão: outputs/figures)
    """
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, nome_arquivo)
    plt.savefig(caminho, bbox_inches="tight", dpi=150)
    print(f"Figura salva em: {caminho}")


def salvar_modelo(modelo, nome_arquivo: str, pasta: str = "outputs/models"):
    """
    Salva um modelo treinado em disco usando joblib.

    Parâmetros:
        modelo: objeto do modelo scikit-learn já treinado
        nome_arquivo: nome do arquivo (ex: 'regressao_logistica.pkl')
        pasta: pasta de destino (padrão: outputs/models)
    """
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, nome_arquivo)
    joblib.dump(modelo, caminho)
    print(f"Modelo salvo em: {caminho}")


def carregar_modelo(nome_arquivo: str, pasta: str = "outputs/models"):
    """
    Carrega um modelo previamente salvo.

    Parâmetros:
        nome_arquivo: nome do arquivo (ex: 'regressao_logistica.pkl')
        pasta: pasta onde o modelo está salvo

    Retorna:
        modelo carregado
    """
    caminho = os.path.join(pasta, nome_arquivo)
    return joblib.load(caminho)
