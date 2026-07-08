from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import ltn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.ltn_predicates import (
    criar_predicados_ltn,
    TrainableBinaryPredicate,
    TrainableTernaryPredicate,
    SmoothCloseToPredicate,
)

from src.ltn_axioms import (
    criar_operadores_logicos,
    criar_base_conhecimento,
    calcular_sat_agg,
    avaliar_axiomas_individualmente,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "datasets_gerados"
RESULTS_DIR = PROJECT_ROOT / "results" / "training"

OBJETOS_CSV = DATA_DIR / "objetos_sinteticos.csv"
VETORES_NPY = DATA_DIR / "objetos_sinteticos_vetores.npy"
GT_PARES_CSV = DATA_DIR / "ground_truth_pares.csv"
GT_TRIPLAS_CSV = DATA_DIR / "ground_truth_triplas.csv"


# Relações binárias que serão realmente treinadas por redes neurais.
#
# Importante:
# closeTo foi removido desta lista.
# Motivo:
# closeTo é uma relação geométrica direta, baseada em distância euclidiana.
# Ela será usada como predicado fuzzy determinístico por meio de SmoothCloseToPredicate.
RELACOES_BINARIAS_TREINAVEIS = [
    "leftOf",
    "rightOf",
    "below",
    "above",
    "canStack",
]


# Relações binárias que serão avaliadas nas métricas finais.
#
# Aqui closeTo continua aparecendo, mas agora será avaliado como predicado
# fuzzy determinístico, não como rede neural treinável.
RELACOES_BINARIAS_AVALIADAS = [
    "leftOf",
    "rightOf",
    "below",
    "above",
    "closeTo",
    "canStack",
]


RELACOES_TERNARIAS_TREINAVEIS = [
    "inBetween",
]


def definir_seed(seed: int = 42) -> None:
    """
    Define a seed para tornar o treinamento reprodutível.

    Isso afeta:
    - inicialização dos pesos das redes;
    - separação treino/teste;
    - operações aleatórias do NumPy e PyTorch.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def carregar_objetos_e_ground_truth(
    device: torch.device,
) -> dict[str, Any]:
    """
    Carrega os dados necessários para o treinamento.

    Arquivos usados:
    - objetos_sinteticos.csv
    - objetos_sinteticos_vetores.npy
    - ground_truth_pares.csv
    - ground_truth_triplas.csv
    """

    df_objetos = pd.read_csv(OBJETOS_CSV)
    vetores = np.load(VETORES_NPY).astype(np.float32)

    tensor_objetos = torch.tensor(
        vetores,
        dtype=torch.float32,
        device=device,
    )

    df_pares = pd.read_csv(GT_PARES_CSV)
    df_triplas = pd.read_csv(GT_TRIPLAS_CSV)

    return {
        "df_objetos": df_objetos,
        "tensor_objetos": tensor_objetos,
        "df_pares": df_pares,
        "df_triplas": df_triplas,
    }


def separar_treino_teste(
    df: pd.DataFrame,
    frac_treino: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa um DataFrame em treino e teste.

    A separação é aleatória, mas reprodutível por causa da seed.
    """

    df_treino = df.sample(frac=frac_treino, random_state=seed)
    df_teste = df.drop(df_treino.index)

    return df_treino.reset_index(drop=True), df_teste.reset_index(drop=True)


def calcular_balanceamento_relacao(
    df: pd.DataFrame,
    nome_relacao: str,
) -> dict[str, float]:
    """
    Calcula o balanceamento de uma relação binária ou ternária.

    Retorna:
    - total
    - positivos
    - negativos
    - proporcao_positivos
    - proporcao_negativos
    - pos_weight

    O pos_weight é calculado como:

        negativos / positivos

    Se houver poucos positivos, esse valor fica maior.
    Ele será usado para aumentar o peso dos exemplos positivos na loss.

    Observação:
    closeTo pode aparecer no balanceamento e na avaliação, mas não será
    usado na loss supervisionada quando estiver fora da lista de relações treináveis.
    """

    if nome_relacao not in df.columns:
        raise ValueError(f"A relação {nome_relacao} não existe no DataFrame.")

    total = len(df)
    positivos = int((df[nome_relacao] == 1).sum())
    negativos = int((df[nome_relacao] == 0).sum())

    proporcao_positivos = positivos / total if total > 0 else 0.0
    proporcao_negativos = negativos / total if total > 0 else 0.0

    if positivos == 0:
        pos_weight = 1.0
    else:
        pos_weight = negativos / positivos

    pos_weight = max(float(pos_weight), 1.0)

    return {
        "total": float(total),
        "positivos": float(positivos),
        "negativos": float(negativos),
        "proporcao_positivos": float(proporcao_positivos),
        "proporcao_negativos": float(proporcao_negativos),
        "pos_weight": float(pos_weight),
    }


def calcular_pesos_positivos(
    df_pares_treino: pd.DataFrame,
    df_triplas_treino: pd.DataFrame,
) -> dict[str, float]:
    """
    Calcula pesos positivos automáticos para as relações treináveis.

    Nesta versão:
    - closeTo não recebe peso para treinamento, pois não é treinável;
    - canStack continua recebendo peso, pois é desbalanceado e treinável;
    - inBetween também pode receber peso, embora seja relativamente balanceado.
    """

    pesos: dict[str, float] = {}

    for relacao in RELACOES_BINARIAS_TREINAVEIS:
        info = calcular_balanceamento_relacao(df_pares_treino, relacao)
        pesos[relacao] = info["pos_weight"]

    for relacao in RELACOES_TERNARIAS_TREINAVEIS:
        info = calcular_balanceamento_relacao(df_triplas_treino, relacao)
        pesos[relacao] = info["pos_weight"]

    return pesos


def imprimir_balanceamento_treino(
    df_pares_treino: pd.DataFrame,
    df_triplas_treino: pd.DataFrame,
    pesos_positivos: dict[str, float],
) -> None:
    """
    Imprime no terminal o balanceamento das relações no conjunto de treino.

    Também informa quais relações são treináveis e quais são apenas avaliadas.
    """

    print("\nBalanceamento no conjunto de treino:")

    print("\nRelações binárias avaliadas:")
    for relacao in RELACOES_BINARIAS_AVALIADAS:
        info = calcular_balanceamento_relacao(df_pares_treino, relacao)

        if relacao in RELACOES_BINARIAS_TREINAVEIS:
            status = "treinável"
            pos_weight = pesos_positivos[relacao]
        else:
            status = "determinístico / não treinável"
            pos_weight = 1.0

        print(
            f"- {relacao}: "
            f"status={status}, "
            f"positivos={int(info['positivos'])}, "
            f"negativos={int(info['negativos'])}, "
            f"prop_pos={info['proporcao_positivos']:.4f}, "
            f"pos_weight={pos_weight:.4f}"
        )

    print("\nRelações ternárias treináveis:")
    for relacao in RELACOES_TERNARIAS_TREINAVEIS:
        info = calcular_balanceamento_relacao(df_triplas_treino, relacao)

        print(
            f"- {relacao}: "
            f"status=treinável, "
            f"positivos={int(info['positivos'])}, "
            f"negativos={int(info['negativos'])}, "
            f"prop_pos={info['proporcao_positivos']:.4f}, "
            f"pos_weight={pesos_positivos[relacao]:.4f}"
        )


def montar_batch_binario(
    df_relacoes: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    nome_relacao: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Monta tensores para uma relação binária.

    Entrada:
    - id_objeto_a
    - id_objeto_b
    - coluna da relação, por exemplo leftOf

    Saída:
    - x_a
    - x_b
    - y_true
    """

    ids_a = torch.tensor(
        df_relacoes["id_objeto_a"].values,
        dtype=torch.long,
        device=tensor_objetos.device,
    )

    ids_b = torch.tensor(
        df_relacoes["id_objeto_b"].values,
        dtype=torch.long,
        device=tensor_objetos.device,
    )

    y_true = torch.tensor(
        df_relacoes[nome_relacao].values,
        dtype=torch.float32,
        device=tensor_objetos.device,
    ).view(-1, 1)

    x_a = tensor_objetos[ids_a]
    x_b = tensor_objetos[ids_b]

    return x_a, x_b, y_true


def montar_batch_ternario(
    df_relacoes: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    nome_relacao: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Monta tensores para uma relação ternária.

    Entrada:
    - id_objeto_x
    - id_objeto_y
    - id_objeto_z
    - coluna da relação, por exemplo inBetween

    Saída:
    - x
    - y
    - z
    - y_true
    """

    ids_x = torch.tensor(
        df_relacoes["id_objeto_x"].values,
        dtype=torch.long,
        device=tensor_objetos.device,
    )

    ids_y = torch.tensor(
        df_relacoes["id_objeto_y"].values,
        dtype=torch.long,
        device=tensor_objetos.device,
    )

    ids_z = torch.tensor(
        df_relacoes["id_objeto_z"].values,
        dtype=torch.long,
        device=tensor_objetos.device,
    )

    y_true = torch.tensor(
        df_relacoes[nome_relacao].values,
        dtype=torch.float32,
        device=tensor_objetos.device,
    ).view(-1, 1)

    x = tensor_objetos[ids_x]
    y = tensor_objetos[ids_y]
    z = tensor_objetos[ids_z]

    return x, y, z, y_true


def criar_predicados_treinaveis() -> tuple[
    dict[str, ltn.Predicate],
    dict[str, torch.nn.Module],
    dict[str, torch.nn.Module],
]:
    """
    Cria os predicados usados no treinamento.

    Retorna:
    - predicados:
      dicionário usado pela LTN e pelos axiomas.

    - modelos_treinaveis:
      apenas os modelos com parâmetros otimizáveis.

    - modelos_avaliacao:
      modelos usados para calcular métricas.
      Inclui tanto modelos treináveis quanto predicados determinísticos.

    Estratégia adotada:
    - leftOf, rightOf, below, above e canStack são treináveis;
    - inBetween é treinável;
    - closeTo é determinístico, usando SmoothCloseToPredicate.

    Isso evita que closeTo seja substituído por TrainableBinaryPredicate.
    """

    predicados = criar_predicados_ltn()

    modelos_treinaveis: dict[str, torch.nn.Module] = {}
    modelos_avaliacao: dict[str, torch.nn.Module] = {}

    # Garante explicitamente que closeTo será o predicado fuzzy determinístico.
    # Mesmo que criar_predicados_ltn() já faça isso, deixamos explícito aqui para
    # evitar que closeTo seja acidentalmente sobrescrito como treinável.
    modelo_close_to = SmoothCloseToPredicate()
    predicados["closeTo"] = ltn.Predicate(modelo_close_to)
    modelos_avaliacao["closeTo"] = modelo_close_to

    for nome_relacao in RELACOES_BINARIAS_TREINAVEIS:
        modelo = TrainableBinaryPredicate(input_dim=22, hidden_dim=32)
        modelos_treinaveis[nome_relacao] = modelo
        modelos_avaliacao[nome_relacao] = modelo
        predicados[nome_relacao] = ltn.Predicate(modelo)

    for nome_relacao in RELACOES_TERNARIAS_TREINAVEIS:
        modelo = TrainableTernaryPredicate(input_dim=33, hidden_dim=48)
        modelos_treinaveis[nome_relacao] = modelo
        modelos_avaliacao[nome_relacao] = modelo
        predicados[nome_relacao] = ltn.Predicate(modelo)

    return predicados, modelos_treinaveis, modelos_avaliacao


def mover_modelos_para_device(
    modelos: dict[str, torch.nn.Module],
    device: torch.device,
) -> None:
    """
    Move todos os modelos para CPU ou GPU.

    Isso inclui:
    - modelos treináveis;
    - modelos determinísticos usados na avaliação, como closeTo.
    """

    for modelo in modelos.values():
        modelo.to(device)


def binary_cross_entropy_ponderada(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    """
    Calcula binary cross entropy com peso maior para exemplos positivos.

    Como os modelos já terminam com Sigmoid, usamos F.binary_cross_entropy
    com pesos por amostra.

    Para cada exemplo:
    - se y_true = 1, peso = pos_weight
    - se y_true = 0, peso = 1
    """

    pesos_amostra = torch.ones_like(y_true)

    pesos_amostra = torch.where(
        y_true == 1.0,
        torch.full_like(y_true, float(pos_weight)),
        pesos_amostra,
    )

    loss = F.binary_cross_entropy(
        y_pred,
        y_true,
        weight=pesos_amostra,
    )

    return loss


def calcular_loss_supervisionada_binaria(
    modelos_treinaveis: dict[str, torch.nn.Module],
    df_pares_treino: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    pesos_positivos: dict[str, float],
    usar_loss_ponderada: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Calcula a loss supervisionada para predicados binários treináveis.

    Importante:
    closeTo não entra aqui, pois agora é determinístico.
    """

    losses = []
    logs = {}

    for nome_relacao in RELACOES_BINARIAS_TREINAVEIS:
        x_a, x_b, y_true = montar_batch_binario(
            df_relacoes=df_pares_treino,
            tensor_objetos=tensor_objetos,
            nome_relacao=nome_relacao,
        )

        y_pred = modelos_treinaveis[nome_relacao](x_a, x_b)

        if usar_loss_ponderada:
            loss = binary_cross_entropy_ponderada(
                y_pred=y_pred,
                y_true=y_true,
                pos_weight=pesos_positivos[nome_relacao],
            )
        else:
            loss = F.binary_cross_entropy(y_pred, y_true)

        losses.append(loss)
        logs[f"loss_sup_{nome_relacao}"] = float(loss.detach().cpu().item())
        logs[f"pos_weight_{nome_relacao}"] = float(pesos_positivos[nome_relacao])

    if not losses:
        device = tensor_objetos.device
        loss_media = torch.tensor(0.0, dtype=torch.float32, device=device)
    else:
        loss_media = torch.stack(losses).mean()

    return loss_media, logs


def calcular_loss_supervisionada_ternaria(
    modelos_treinaveis: dict[str, torch.nn.Module],
    df_triplas_treino: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    pesos_positivos: dict[str, float],
    usar_loss_ponderada: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Calcula a loss supervisionada para predicados ternários.

    Atualmente, o principal predicado ternário é inBetween.
    """

    losses = []
    logs = {}

    for nome_relacao in RELACOES_TERNARIAS_TREINAVEIS:
        x, y, z, y_true = montar_batch_ternario(
            df_relacoes=df_triplas_treino,
            tensor_objetos=tensor_objetos,
            nome_relacao=nome_relacao,
        )

        y_pred = modelos_treinaveis[nome_relacao](x, y, z)

        if usar_loss_ponderada:
            loss = binary_cross_entropy_ponderada(
                y_pred=y_pred,
                y_true=y_true,
                pos_weight=pesos_positivos[nome_relacao],
            )
        else:
            loss = F.binary_cross_entropy(y_pred, y_true)

        losses.append(loss)
        logs[f"loss_sup_{nome_relacao}"] = float(loss.detach().cpu().item())
        logs[f"pos_weight_{nome_relacao}"] = float(pesos_positivos[nome_relacao])

    if not losses:
        device = tensor_objetos.device
        loss_media = torch.tensor(0.0, dtype=torch.float32, device=device)
    else:
        loss_media = torch.stack(losses).mean()

    return loss_media, logs


def calcular_loss_axiomas(
    tensor_objetos: torch.Tensor,
    predicados: dict[str, ltn.Predicate],
    operadores: dict[str, Any],
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """
    Calcula a loss lógica da base de conhecimento.

    A lógica é:

        sat = satAgg(axiomas)
        loss_axiomas = 1 - sat
    """

    axiomas = criar_base_conhecimento(
        tensor_objetos=tensor_objetos,
        predicados=predicados,
        operadores=operadores,
    )

    sat = calcular_sat_agg(
        axiomas=axiomas,
        operadores=operadores,
    )

    loss_axiomas = 1.0 - sat

    sat_individual = avaliar_axiomas_individualmente(axiomas)

    return loss_axiomas, float(sat.detach().cpu().item()), sat_individual


def calcular_metricas_binarias_classicas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calcula accuracy, precision, recall e F1.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def encontrar_melhor_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """
    Testa vários thresholds e encontra aquele que maximiza F1.

    Isso não altera o treinamento.
    Serve para diagnóstico.

    Se o F1 melhora muito com threshold menor, o problema pode estar no corte 0.5.
    """

    melhores_metricas = {
        "best_threshold": 0.5,
        "best_accuracy": 0.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "best_f1": 0.0,
    }

    thresholds = np.linspace(0.05, 0.95, 19)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        metricas = calcular_metricas_binarias_classicas(y_true, y_pred)

        if metricas["f1"] > melhores_metricas["best_f1"]:
            melhores_metricas = {
                "best_threshold": float(threshold),
                "best_accuracy": float(metricas["accuracy"]),
                "best_precision": float(metricas["precision"]),
                "best_recall": float(metricas["recall"]),
                "best_f1": float(metricas["f1"]),
            }

    return melhores_metricas


@torch.no_grad()
def avaliar_modelo_binario(
    modelos_avaliacao: dict[str, torch.nn.Module],
    df_pares_teste: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Avalia os predicados binários no conjunto de teste.

    Relações avaliadas:
    - leftOf
    - rightOf
    - below
    - above
    - closeTo
    - canStack

    Nesta versão:
    - closeTo é avaliado pelo SmoothCloseToPredicate;
    - as demais relações treináveis são avaliadas pelos modelos treinados.
    """

    resultados = {}

    for nome_relacao in RELACOES_BINARIAS_AVALIADAS:
        x_a, x_b, y_true_tensor = montar_batch_binario(
            df_relacoes=df_pares_teste,
            tensor_objetos=tensor_objetos,
            nome_relacao=nome_relacao,
        )

        modelo = modelos_avaliacao[nome_relacao]
        modelo.eval()

        y_score_tensor = modelo(x_a, x_b)
        y_pred_tensor = (y_score_tensor >= threshold).int()

        y_true = y_true_tensor.detach().cpu().numpy().astype(int).ravel()
        y_pred = y_pred_tensor.detach().cpu().numpy().astype(int).ravel()
        y_score = y_score_tensor.detach().cpu().numpy().ravel()

        metricas_padrao = calcular_metricas_binarias_classicas(y_true, y_pred)
        metricas_threshold = encontrar_melhor_threshold(y_true, y_score)

        resultados[nome_relacao] = {
            **metricas_padrao,
            **metricas_threshold,
        }

    return resultados


@torch.no_grad()
def avaliar_modelo_ternario(
    modelos_avaliacao: dict[str, torch.nn.Module],
    df_triplas_teste: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """
    Avalia os predicados ternários no conjunto de teste.
    """

    resultados = {}

    for nome_relacao in RELACOES_TERNARIAS_TREINAVEIS:
        x, y, z, y_true_tensor = montar_batch_ternario(
            df_relacoes=df_triplas_teste,
            tensor_objetos=tensor_objetos,
            nome_relacao=nome_relacao,
        )

        modelo = modelos_avaliacao[nome_relacao]
        modelo.eval()

        y_score_tensor = modelo(x, y, z)
        y_pred_tensor = (y_score_tensor >= threshold).int()

        y_true = y_true_tensor.detach().cpu().numpy().astype(int).ravel()
        y_pred = y_pred_tensor.detach().cpu().numpy().astype(int).ravel()
        y_score = y_score_tensor.detach().cpu().numpy().ravel()

        metricas_padrao = calcular_metricas_binarias_classicas(y_true, y_pred)
        metricas_threshold = encontrar_melhor_threshold(y_true, y_score)

        resultados[nome_relacao] = {
            **metricas_padrao,
            **metricas_threshold,
        }

    return resultados


def salvar_metricas_epoch(
    historico: list[dict[str, Any]],
    caminho_csv: Path,
) -> None:
    """
    Salva o histórico de treinamento em CSV.

    Colunas com dicionários são convertidas para JSON.
    """

    linhas = []

    for item in historico:
        linha = {}

        for chave, valor in item.items():
            if isinstance(valor, dict):
                linha[chave] = json.dumps(valor, ensure_ascii=False)
            else:
                linha[chave] = valor

        linhas.append(linha)

    df = pd.DataFrame(linhas)
    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")


def treinar_ltn(
    epochs: int = 250,
    lr: float = 0.001,
    peso_axiomas: float = 0.30,
    frac_treino: float = 0.8,
    threshold: float = 0.5,
    seed: int = 42,
    usar_loss_ponderada: bool = True,
) -> dict[str, Any]:
    """
    Executa o treinamento LTN.

    O treinamento combina:
    - loss supervisionada com ground_truth;
    - loss lógica com axiomas LTN.

    Nesta versão corrigida:
    - closeTo é fuzzy determinístico;
    - closeTo não é otimizado pelo Adam;
    - closeTo continua sendo avaliado nas métricas finais;
    - canStack continua treinável e usa loss ponderada;
    - os axiomas continuam usando closeTo dentro da base de conhecimento.

    Parâmetros:
    - epochs:
      número de épocas de treinamento.

    - lr:
      taxa de aprendizado.

    - peso_axiomas:
      peso da loss lógica.

    - frac_treino:
      proporção das relações usadas para treino.

    - threshold:
      corte padrão para transformar score fuzzy em classe 0/1.

    - seed:
      garante reprodutibilidade.

    - usar_loss_ponderada:
      se True, aumenta o peso dos exemplos positivos em relações desbalanceadas.

    Retorno:
    dicionário com o histórico de treinamento, os modelos de avaliação,
    o tensor de objetos e os caminhos dos arquivos gerados.
    Isso permite que scripts de experimento (por exemplo, src/run_experiments.py)
    consolidem métricas de várias execuções sem reler os arquivos.
    """

    definir_seed(seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device usado: {device}")

    dados = carregar_objetos_e_ground_truth(device=device)

    tensor_objetos = dados["tensor_objetos"]
    df_pares = dados["df_pares"]
    df_triplas = dados["df_triplas"]

    df_pares_treino, df_pares_teste = separar_treino_teste(
        df=df_pares,
        frac_treino=frac_treino,
        seed=seed,
    )

    df_triplas_treino, df_triplas_teste = separar_treino_teste(
        df=df_triplas,
        frac_treino=frac_treino,
        seed=seed,
    )

    pesos_positivos = calcular_pesos_positivos(
        df_pares_treino=df_pares_treino,
        df_triplas_treino=df_triplas_treino,
    )

    operadores = criar_operadores_logicos()

    predicados, modelos_treinaveis, modelos_avaliacao = criar_predicados_treinaveis()

    mover_modelos_para_device(
        modelos=modelos_avaliacao,
        device=device,
    )

    parametros = []

    for modelo in modelos_treinaveis.values():
        parametros.extend(list(modelo.parameters()))

    if not parametros:
        raise RuntimeError(
            "Nenhum parâmetro treinável foi encontrado. "
            "Verifique RELACOES_BINARIAS_TREINAVEIS e RELACOES_TERNARIAS_TREINAVEIS."
        )

    optimizer = torch.optim.Adam(parametros, lr=lr)

    historico: list[dict[str, Any]] = []

    print("\nIniciando treinamento LTN...")
    print(f"Épocas: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Peso dos axiomas: {peso_axiomas}")
    print(f"Threshold padrão: {threshold}")
    print(f"Loss ponderada: {usar_loss_ponderada}")
    print("closeTo determinístico: True")
    print(f"Pares treino/teste: {len(df_pares_treino)} / {len(df_pares_teste)}")
    print(f"Triplas treino/teste: {len(df_triplas_treino)} / {len(df_triplas_teste)}")

    imprimir_balanceamento_treino(
        df_pares_treino=df_pares_treino,
        df_triplas_treino=df_triplas_treino,
        pesos_positivos=pesos_positivos,
    )

    for epoch in range(1, epochs + 1):
        for modelo in modelos_treinaveis.values():
            modelo.train()

        optimizer.zero_grad()

        loss_binaria, logs_binarios = calcular_loss_supervisionada_binaria(
            modelos_treinaveis=modelos_treinaveis,
            df_pares_treino=df_pares_treino,
            tensor_objetos=tensor_objetos,
            pesos_positivos=pesos_positivos,
            usar_loss_ponderada=usar_loss_ponderada,
        )

        loss_ternaria, logs_ternarios = calcular_loss_supervisionada_ternaria(
            modelos_treinaveis=modelos_treinaveis,
            df_triplas_treino=df_triplas_treino,
            tensor_objetos=tensor_objetos,
            pesos_positivos=pesos_positivos,
            usar_loss_ponderada=usar_loss_ponderada,
        )

        loss_axiomas, sat_agg, sat_individual = calcular_loss_axiomas(
            tensor_objetos=tensor_objetos,
            predicados=predicados,
            operadores=operadores,
        )

        loss_supervisionada = loss_binaria + loss_ternaria

        loss_total = loss_supervisionada + peso_axiomas * loss_axiomas

        loss_total.backward()

        optimizer.step()

        for modelo in modelos_treinaveis.values():
            modelo.eval()

        metricas_binarias = avaliar_modelo_binario(
            modelos_avaliacao=modelos_avaliacao,
            df_pares_teste=df_pares_teste,
            tensor_objetos=tensor_objetos,
            threshold=threshold,
        )

        metricas_ternarias = avaliar_modelo_ternario(
            modelos_avaliacao=modelos_avaliacao,
            df_triplas_teste=df_triplas_teste,
            tensor_objetos=tensor_objetos,
            threshold=threshold,
        )

        registro = {
            "epoch": epoch,
            "loss_total": float(loss_total.detach().cpu().item()),
            "loss_supervisionada": float(loss_supervisionada.detach().cpu().item()),
            "loss_binaria": float(loss_binaria.detach().cpu().item()),
            "loss_ternaria": float(loss_ternaria.detach().cpu().item()),
            "loss_axiomas": float(loss_axiomas.detach().cpu().item()),
            "sat_agg": sat_agg,
            "metricas_binarias": metricas_binarias,
            "metricas_ternarias": metricas_ternarias,
            "logs_binarios": logs_binarios,
            "logs_ternarios": logs_ternarios,
            "sat_individual": sat_individual,
            "pesos_positivos": pesos_positivos,
            "usar_loss_ponderada": usar_loss_ponderada,
            "close_to_deterministico": True,
            "relacoes_binarias_treinaveis": RELACOES_BINARIAS_TREINAVEIS,
            "relacoes_binarias_avaliadas": RELACOES_BINARIAS_AVALIADAS,
            "relacoes_ternarias_treinaveis": RELACOES_TERNARIAS_TREINAVEIS,
        }

        historico.append(registro)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            close_f1 = metricas_binarias.get("closeTo", {}).get("f1", 0.0)
            close_best_f1 = metricas_binarias.get("closeTo", {}).get("best_f1", 0.0)
            canstack_f1 = metricas_binarias.get("canStack", {}).get("f1", 0.0)

            print(
                f"Epoch {epoch:03d} | "
                f"Loss Total: {registro['loss_total']:.4f} | "
                f"Loss Sup: {registro['loss_supervisionada']:.4f} | "
                f"Loss Axiomas: {registro['loss_axiomas']:.4f} | "
                f"satAgg: {registro['sat_agg']:.4f} | "
                f"F1 closeTo: {close_f1:.4f} | "
                f"Best F1 closeTo: {close_best_f1:.4f} | "
                f"F1 canStack: {canstack_f1:.4f}"
            )

    caminho_metricas = RESULTS_DIR / "historico_treinamento.csv"

    salvar_metricas_epoch(
        historico=historico,
        caminho_csv=caminho_metricas,
    )

    caminho_modelo = RESULTS_DIR / "modelo_ltn_treinavel.pt"

    torch.save(
        {
            "model_state_dicts": {
                nome: modelo.state_dict()
                for nome, modelo in modelos_treinaveis.items()
            },
            "config": {
                "epochs": epochs,
                "lr": lr,
                "peso_axiomas": peso_axiomas,
                "frac_treino": frac_treino,
                "threshold": threshold,
                "seed": seed,
                "usar_loss_ponderada": usar_loss_ponderada,
                "pesos_positivos": pesos_positivos,
                "close_to_deterministico": True,
                "relacoes_binarias_treinaveis": RELACOES_BINARIAS_TREINAVEIS,
                "relacoes_binarias_avaliadas": RELACOES_BINARIAS_AVALIADAS,
                "relacoes_ternarias_treinaveis": RELACOES_TERNARIAS_TREINAVEIS,
            },
        },
        caminho_modelo,
    )

    print("\nTreinamento concluído.")
    print(f"Histórico salvo em: {caminho_metricas}")
    print(f"Modelo salvo em: {caminho_modelo}")

    print("\nMétricas finais binárias:")
    print(json.dumps(historico[-1]["metricas_binarias"], indent=2, ensure_ascii=False))

    print("\nMétricas finais ternárias:")
    print(json.dumps(historico[-1]["metricas_ternarias"], indent=2, ensure_ascii=False))

    return {
        "historico": historico,
        "modelos_avaliacao": modelos_avaliacao,
        "tensor_objetos": tensor_objetos,
        "caminho_metricas": caminho_metricas,
        "caminho_modelo": caminho_modelo,
    }


if __name__ == "__main__":
    treinar_ltn(
        epochs=250,
        lr=0.001,
        peso_axiomas=0.30,
        frac_treino=0.8,
        threshold=0.5,
        seed=42,
        usar_loss_ponderada=True,
    )