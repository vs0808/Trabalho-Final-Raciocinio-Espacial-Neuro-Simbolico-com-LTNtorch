from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ltn

from src.ltn_predicates import (
    TrainableBinaryPredicate,
    TrainableTernaryPredicate,
    SmoothCloseToPredicate,
)

from src.ltn_axioms import (
    criar_operadores_logicos,
    criar_variaveis_ltn,
    formulas_extremos_horizontais,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "datasets_gerados"
TRAINING_DIR = PROJECT_ROOT / "results" / "training"
QUERIES_DIR = PROJECT_ROOT / "results" / "queries"

OBJETOS_CSV = DATA_DIR / "objetos_sinteticos.csv"
VETORES_NPY = DATA_DIR / "objetos_sinteticos_vetores.npy"
MODELO_TREINADO = TRAINING_DIR / "modelo_ltn_treinavel.pt"

RESPOSTAS_JSON = QUERIES_DIR / "respostas_consultas.json"
RESPOSTAS_CSV = QUERIES_DIR / "respostas_consultas.csv"
EXPLICACOES_MD = QUERIES_DIR / "explicacoes_consultas.md"


# Índices do vetor de 11 posições:
#
# [0] x
# [1] y
# [2] vermelho
# [3] verde
# [4] azul
# [5] círculo
# [6] quadrado
# [7] cilindro
# [8] cone
# [9] triângulo
# [10] tamanho: pequeno=0, grande=1

IDX_X = 0
IDX_Y = 1

IDX_RED = 2
IDX_GREEN = 3
IDX_BLUE = 4

IDX_CIRCLE = 5
IDX_SQUARE = 6
IDX_CYLINDER = 7
IDX_CONE = 8
IDX_TRIANGLE = 9

IDX_SIZE = 10


RELACOES_BINARIAS_TREINAVEIS_PADRAO = [
    "leftOf",
    "rightOf",
    "below",
    "above",
    "canStack",
]

RELACOES_TERNARIAS_TREINAVEIS_PADRAO = [
    "inBetween",
]


def garantir_pasta_saida() -> None:
    """
    Cria a pasta results/queries, caso ela ainda não exista.
    """

    QUERIES_DIR.mkdir(parents=True, exist_ok=True)


def validar_arquivos_entrada() -> None:
    """
    Verifica se os arquivos necessários para consulta existem.

    A consulta depende de:
    - objetos_sinteticos.csv
    - objetos_sinteticos_vetores.npy
    - modelo_ltn_treinavel.pt
    """

    arquivos = [
        OBJETOS_CSV,
        VETORES_NPY,
        MODELO_TREINADO,
    ]

    arquivos_faltantes = [arquivo for arquivo in arquivos if not arquivo.exists()]

    if arquivos_faltantes:
        linhas = [
            "Arquivos necessários não encontrados.",
            "",
            "Antes de executar as consultas, rode:",
            "python3 main.py --train --no-show",
            "",
            "Arquivos faltantes:",
        ]

        for arquivo in arquivos_faltantes:
            linhas.append(f"- {arquivo}")

        raise FileNotFoundError("\n".join(linhas))


def carregar_checkpoint(
    caminho_modelo: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Carrega o checkpoint do modelo treinado.

    O checkpoint contém:
    - model_state_dicts
    - config

    Em algumas versões recentes do PyTorch, torch.load pode usar weights_only=True
    por padrão. Por segurança, tentamos primeiro com weights_only=False.
    """

    try:
        checkpoint = torch.load(
            caminho_modelo,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            caminho_modelo,
            map_location=device,
        )

    return checkpoint


def carregar_dados_e_modelo(
    device: torch.device,
) -> dict[str, Any]:
    """
    Carrega objetos, vetores e checkpoint do modelo.
    """

    validar_arquivos_entrada()

    df_objetos = pd.read_csv(OBJETOS_CSV)
    vetores = np.load(VETORES_NPY).astype(np.float32)

    tensor_objetos = torch.tensor(
        vetores,
        dtype=torch.float32,
        device=device,
    )

    checkpoint = carregar_checkpoint(
        caminho_modelo=MODELO_TREINADO,
        device=device,
    )

    return {
        "df_objetos": df_objetos,
        "vetores": vetores,
        "tensor_objetos": tensor_objetos,
        "checkpoint": checkpoint,
    }


def criar_trainable_binary_predicate() -> nn.Module:
    """
    Cria um predicado binário treinável.

    Mantemos este helper para lidar com pequenas diferenças de assinatura
    da classe TrainableBinaryPredicate.

    Algumas versões do código usam:
        hidden_dim=32

    Outras podem usar:
        hidden=32
    """

    try:
        return TrainableBinaryPredicate(input_dim=22, hidden_dim=32)
    except TypeError:
        return TrainableBinaryPredicate(input_dim=22, hidden=32)


def criar_trainable_ternary_predicate() -> nn.Module:
    """
    Cria um predicado ternário treinável.

    Mantemos este helper para lidar com pequenas diferenças de assinatura
    da classe TrainableTernaryPredicate.
    """

    try:
        return TrainableTernaryPredicate(input_dim=33, hidden_dim=48)
    except TypeError:
        return TrainableTernaryPredicate(input_dim=33, hidden=48)


def reconstruir_modelos_para_consulta(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> dict[str, nn.Module]:
    """
    Reconstrói os modelos usados nas consultas.

    O checkpoint salva apenas os pesos das relações treináveis.

    Nesta versão do projeto:
    - closeTo é determinístico, então não aparece em model_state_dicts;
    - leftOf, rightOf, below, above e canStack são treináveis;
    - inBetween é treinável.

    Retorna um dicionário com todos os modelos necessários para consulta:
    - leftOf
    - rightOf
    - below
    - above
    - closeTo
    - canStack
    - inBetween
    """

    state_dicts = checkpoint.get("model_state_dicts", {})
    config = checkpoint.get("config", {})

    relacoes_binarias_treinaveis = config.get(
        "relacoes_binarias_treinaveis",
        RELACOES_BINARIAS_TREINAVEIS_PADRAO,
    )

    relacoes_ternarias_treinaveis = config.get(
        "relacoes_ternarias_treinaveis",
        RELACOES_TERNARIAS_TREINAVEIS_PADRAO,
    )

    modelos: dict[str, nn.Module] = {}

    # closeTo é sempre reconstruído como predicado fuzzy determinístico.
    modelo_close_to = SmoothCloseToPredicate()
    modelo_close_to.to(device)
    modelo_close_to.eval()
    modelos["closeTo"] = modelo_close_to

    for nome_relacao in relacoes_binarias_treinaveis:
        modelo = criar_trainable_binary_predicate()
        modelo.to(device)

        if nome_relacao in state_dicts:
            modelo.load_state_dict(state_dicts[nome_relacao])
        else:
            raise KeyError(
                f"O checkpoint não contém pesos para a relação binária treinável: {nome_relacao}"
            )

        modelo.eval()
        modelos[nome_relacao] = modelo

    for nome_relacao in relacoes_ternarias_treinaveis:
        modelo = criar_trainable_ternary_predicate()
        modelo.to(device)

        if nome_relacao in state_dicts:
            modelo.load_state_dict(state_dicts[nome_relacao])
        else:
            raise KeyError(
                f"O checkpoint não contém pesos para a relação ternária treinável: {nome_relacao}"
            )

        modelo.eval()
        modelos[nome_relacao] = modelo

    return modelos


def tensor_objeto(
    tensor_objetos: torch.Tensor,
    id_objeto: int,
) -> torch.Tensor:
    """
    Retorna o tensor de um objeto pelo ID.

    Saída com shape:
    [1, 11]
    """

    return tensor_objetos[id_objeto].view(1, -1)


def obter_vetor_numpy(
    vetores: np.ndarray,
    id_objeto: int,
) -> np.ndarray:
    """
    Retorna o vetor 11D de um objeto como NumPy array.
    """

    return vetores[id_objeto]


def verdade_feature(
    vetores: np.ndarray,
    id_objeto: int,
    indice: int,
) -> float:
    """
    Retorna diretamente o valor de uma característica one-hot ou contínua.
    """

    return float(vetores[id_objeto][indice])


def is_red(vetores: np.ndarray, id_objeto: int) -> float:
    return verdade_feature(vetores, id_objeto, IDX_RED)


def is_green(vetores: np.ndarray, id_objeto: int) -> float:
    return verdade_feature(vetores, id_objeto, IDX_GREEN)


def is_blue(vetores: np.ndarray, id_objeto: int) -> float:
    return verdade_feature(vetores, id_objeto, IDX_BLUE)


def is_circle(vetores: np.ndarray, id_objeto: int) -> float:
    return verdade_feature(vetores, id_objeto, IDX_CIRCLE)


def is_square(vetores: np.ndarray, id_objeto: int) -> float:
    return verdade_feature(vetores, id_objeto, IDX_SQUARE)


def is_cylinder(vetores: np.ndarray, id_objeto: int) -> float:
    return verdade_feature(vetores, id_objeto, IDX_CYLINDER)


def is_cone(vetores: np.ndarray, id_objeto: int) -> float:
    return verdade_feature(vetores, id_objeto, IDX_CONE)


def is_triangle(vetores: np.ndarray, id_objeto: int) -> float:
    return verdade_feature(vetores, id_objeto, IDX_TRIANGLE)


def is_big(vetores: np.ndarray, id_objeto: int) -> float:
    return verdade_feature(vetores, id_objeto, IDX_SIZE)


def is_small(vetores: np.ndarray, id_objeto: int) -> float:
    return 1.0 - verdade_feature(vetores, id_objeto, IDX_SIZE)


def same_size(vetores: np.ndarray, id_a: int, id_b: int) -> float:
    """
    Retorna 1 se os dois objetos possuem o mesmo tamanho e 0 caso contrário.
    """

    tamanho_a = round(float(vetores[id_a][IDX_SIZE]))
    tamanho_b = round(float(vetores[id_b][IDX_SIZE]))

    return 1.0 if tamanho_a == tamanho_b else 0.0


def same_color(vetores: np.ndarray, id_a: int, id_b: int) -> float:
    """
    Retorna 1 se os dois objetos possuem a mesma cor.
    """

    cor_a = np.argmax(vetores[id_a][IDX_RED : IDX_BLUE + 1])
    cor_b = np.argmax(vetores[id_b][IDX_RED : IDX_BLUE + 1])

    return 1.0 if cor_a == cor_b else 0.0


def same_shape(vetores: np.ndarray, id_a: int, id_b: int) -> float:
    """
    Retorna 1 se os dois objetos possuem a mesma forma.
    """

    forma_a = np.argmax(vetores[id_a][IDX_CIRCLE : IDX_TRIANGLE + 1])
    forma_b = np.argmax(vetores[id_b][IDX_CIRCLE : IDX_TRIANGLE + 1])

    return 1.0 if forma_a == forma_b else 0.0


@torch.no_grad()
def score_binario(
    modelos: dict[str, nn.Module],
    tensor_objetos: torch.Tensor,
    nome_relacao: str,
    id_a: int,
    id_b: int,
) -> float:
    """
    Calcula o grau de verdade de uma relação binária.

    Exemplos:
    - leftOf(a,b)
    - rightOf(a,b)
    - below(a,b)
    - above(a,b)
    - closeTo(a,b)
    - canStack(a,b)
    """

    if nome_relacao not in modelos:
        raise KeyError(f"Modelo não encontrado para relação binária: {nome_relacao}")

    x_a = tensor_objeto(tensor_objetos, id_a)
    x_b = tensor_objeto(tensor_objetos, id_b)

    valor = modelos[nome_relacao](x_a, x_b)

    return float(valor.detach().cpu().item())


@torch.no_grad()
def score_ternario(
    modelos: dict[str, nn.Module],
    tensor_objetos: torch.Tensor,
    nome_relacao: str,
    id_x: int,
    id_y: int,
    id_z: int,
) -> float:
    """
    Calcula o grau de verdade de uma relação ternária.

    Exemplo:
    - inBetween(x,y,z)
    """

    if nome_relacao not in modelos:
        raise KeyError(f"Modelo não encontrado para relação ternária: {nome_relacao}")

    x = tensor_objeto(tensor_objetos, id_x)
    y = tensor_objeto(tensor_objetos, id_y)
    z = tensor_objeto(tensor_objetos, id_z)

    valor = modelos[nome_relacao](x, y, z)

    return float(valor.detach().cpu().item())


def fuzzy_and_prod(*valores: float) -> float:
    """
    Conjunção fuzzy usando produto.

    Esse operador é coerente com o AndProd usado na LTN.
    """

    resultado = 1.0

    for valor in valores:
        resultado *= float(valor)

    return float(resultado)


def fuzzy_or_prob_sum(a: float, b: float) -> float:
    """
    Disjunção fuzzy usando soma probabilística.

    Or(a,b) = a + b - a*b
    """

    return float(a + b - a * b)


def fuzzy_implies_reichenbach(p: float, q: float) -> float:
    """
    Implicação fuzzy de Reichenbach.

    I(p,q) = 1 - p + p*q
    """

    return float(1.0 - p + p * q)


def forall_aggreg_pmean_error(
    valores: list[float],
    p: int = 2,
) -> float:
    """
    Agregador universal aproximado, inspirado no AggregPMeanError(p=2).

    Fórmula:
        1 - mean((1 - valores)^p)^(1/p)

    Quanto mais próximo de 1, mais a regra universal é satisfeita.
    """

    if not valores:
        return 1.0

    array = np.array(valores, dtype=np.float64)
    erros = np.power(1.0 - array, p)

    return float(1.0 - np.power(np.mean(erros), 1.0 / p))


def objeto_para_dict(
    df_objetos: pd.DataFrame,
    id_objeto: int,
) -> dict[str, Any]:
    """
    Retorna as informações legíveis de um objeto.
    """

    linha = df_objetos.loc[df_objetos["id_objeto"] == id_objeto].iloc[0]

    return {
        "id_objeto": int(linha["id_objeto"]),
        "x": float(linha["x"]),
        "y": float(linha["y"]),
        "cor": str(linha["cor"]),
        "forma": str(linha["forma"]),
        "tamanho": str(linha["tamanho"]),
    }


def ids_distintos(*ids: int) -> bool:
    """
    Retorna True se todos os IDs forem distintos.
    """

    return len(set(ids)) == len(ids)


def resposta_booleana(
    valor_verdade: float,
    threshold: float = 0.5,
) -> bool:
    """
    Transforma grau de verdade fuzzy em resposta booleana.
    """

    return bool(valor_verdade >= threshold)


def texto_resposta_booleana(
    valor_verdade: float,
    threshold: float = 0.5,
) -> str:
    """
    Retorna 'Sim' ou 'Não' a partir de um grau de verdade.
    """

    return "Sim" if resposta_booleana(valor_verdade, threshold) else "Não"


def consulta_1_objeto_pequeno_abaixo_cilindro_esquerda_quadrado(
    df_objetos: pd.DataFrame,
    vetores: np.ndarray,
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
    exigir_objetos_distintos: bool = True,
) -> dict[str, Any]:
    """
    Consulta 1.

    Pergunta:
    Existe algum objeto pequeno que esteja abaixo de um cilindro
    e à esquerda de um quadrado?

    Fórmula aproximada:

    ∃x (
        isSmall(x)
        ∧ ∃y (isCylinder(y) ∧ below(x,y))
        ∧ ∃z (isSquare(z) ∧ leftOf(x,z))
    )

    Implementação por busca:
    - percorremos todos os candidatos x, y, z;
    - calculamos a conjunção fuzzy;
    - escolhemos a melhor evidência.
    """

    n = len(df_objetos)

    melhor_score = -1.0
    melhor_evidencia: dict[str, Any] | None = None

    for id_x in range(n):
        for id_y in range(n):
            for id_z in range(n):
                if exigir_objetos_distintos and not ids_distintos(id_x, id_y, id_z):
                    continue

                v_small = is_small(vetores, id_x)
                v_cylinder = is_cylinder(vetores, id_y)
                v_below = score_binario(modelos, tensor_objetos, "below", id_x, id_y)
                v_square = is_square(vetores, id_z)
                v_left = score_binario(modelos, tensor_objetos, "leftOf", id_x, id_z)

                score = fuzzy_and_prod(
                    v_small,
                    v_cylinder,
                    v_below,
                    v_square,
                    v_left,
                )

                if score > melhor_score:
                    melhor_score = score
                    melhor_evidencia = {
                        "objeto_x": objeto_para_dict(df_objetos, id_x),
                        "objeto_y": objeto_para_dict(df_objetos, id_y),
                        "objeto_z": objeto_para_dict(df_objetos, id_z),
                        "componentes": {
                            f"isSmall({id_x})": v_small,
                            f"isCylinder({id_y})": v_cylinder,
                            f"below({id_x},{id_y})": v_below,
                            f"isSquare({id_z})": v_square,
                            f"leftOf({id_x},{id_z})": v_left,
                        },
                    }

    return {
        "id": "consulta_1",
        "pergunta": "Existe algum objeto pequeno abaixo de um cilindro e à esquerda de um quadrado?",
        "formula": "∃x(IsSmall(x) ∧ ∃y(IsCylinder(y) ∧ Below(x,y)) ∧ ∃z(IsSquare(z) ∧ LeftOf(x,z)))",
        "valor_verdade": float(melhor_score),
        "resposta": texto_resposta_booleana(melhor_score),
        "melhor_evidencia": melhor_evidencia,
    }


def consulta_2_cone_verde_entre_dois_objetos(
    df_objetos: pd.DataFrame,
    vetores: np.ndarray,
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
    exigir_objetos_distintos: bool = True,
) -> dict[str, Any]:
    """
    Consulta 2.

    Pergunta:
    Existe um cone verde entre dois objetos?

    Fórmula aproximada:

    ∃x,y,z (
        isCone(x)
        ∧ isGreen(x)
        ∧ inBetween(x,y,z)
    )
    """

    n = len(df_objetos)

    melhor_score = -1.0
    melhor_evidencia: dict[str, Any] | None = None

    for id_x in range(n):
        for id_y in range(n):
            for id_z in range(n):
                if exigir_objetos_distintos and not ids_distintos(id_x, id_y, id_z):
                    continue

                v_cone = is_cone(vetores, id_x)
                v_green = is_green(vetores, id_x)
                v_between = score_ternario(
                    modelos,
                    tensor_objetos,
                    "inBetween",
                    id_x,
                    id_y,
                    id_z,
                )

                score = fuzzy_and_prod(
                    v_cone,
                    v_green,
                    v_between,
                )

                if score > melhor_score:
                    melhor_score = score
                    melhor_evidencia = {
                        "objeto_x": objeto_para_dict(df_objetos, id_x),
                        "objeto_y": objeto_para_dict(df_objetos, id_y),
                        "objeto_z": objeto_para_dict(df_objetos, id_z),
                        "componentes": {
                            f"isCone({id_x})": v_cone,
                            f"isGreen({id_x})": v_green,
                            f"inBetween({id_x},{id_y},{id_z})": v_between,
                        },
                    }

    return {
        "id": "consulta_2",
        "pergunta": "Existe um cone verde entre dois objetos?",
        "formula": "∃x,y,z(IsCone(x) ∧ IsGreen(x) ∧ InBetween(x,y,z))",
        "valor_verdade": float(melhor_score),
        "resposta": texto_resposta_booleana(melhor_score),
        "melhor_evidencia": melhor_evidencia,
    }


def consulta_3_triangulos_proximos_mesmo_tamanho(
    df_objetos: pd.DataFrame,
    vetores: np.ndarray,
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
    exigir_objetos_distintos: bool = True,
) -> dict[str, Any]:
    """
    Consulta 3.

    Pergunta:
    Se dois objetos são triângulos e estão próximos,
    então eles têm o mesmo tamanho?

    Fórmula aproximada:

    ∀x,y (
        (isTriangle(x) ∧ isTriangle(y) ∧ closeTo(x,y))
        → sameSize(x,y)
    )

    Para cada par, calculamos:
    - premissa;
    - conclusão;
    - implicação fuzzy de Reichenbach.

    Depois agregamos todas as implicações com um agregador universal.
    """

    n = len(df_objetos)

    implicacoes = []
    pares_avaliados = []

    pior_score = 2.0
    pior_evidencia: dict[str, Any] | None = None

    melhor_premissa = -1.0
    melhor_caso_premissa: dict[str, Any] | None = None

    for id_x in range(n):
        for id_y in range(n):
            if exigir_objetos_distintos and id_x == id_y:
                continue

            v_triangle_x = is_triangle(vetores, id_x)
            v_triangle_y = is_triangle(vetores, id_y)
            v_close = score_binario(modelos, tensor_objetos, "closeTo", id_x, id_y)
            v_same_size = same_size(vetores, id_x, id_y)

            premissa = fuzzy_and_prod(
                v_triangle_x,
                v_triangle_y,
                v_close,
            )

            implicacao = fuzzy_implies_reichenbach(
                premissa,
                v_same_size,
            )

            implicacoes.append(implicacao)

            registro = {
                "id_x": id_x,
                "id_y": id_y,
                "premissa": premissa,
                "conclusao_sameSize": v_same_size,
                "implicacao": implicacao,
                "componentes": {
                    f"isTriangle({id_x})": v_triangle_x,
                    f"isTriangle({id_y})": v_triangle_y,
                    f"closeTo({id_x},{id_y})": v_close,
                    f"sameSize({id_x},{id_y})": v_same_size,
                },
            }

            pares_avaliados.append(registro)

            if implicacao < pior_score:
                pior_score = implicacao
                pior_evidencia = {
                    "objeto_x": objeto_para_dict(df_objetos, id_x),
                    "objeto_y": objeto_para_dict(df_objetos, id_y),
                    **registro,
                }

            if premissa > melhor_premissa:
                melhor_premissa = premissa
                melhor_caso_premissa = {
                    "objeto_x": objeto_para_dict(df_objetos, id_x),
                    "objeto_y": objeto_para_dict(df_objetos, id_y),
                    **registro,
                }

    valor_universal = forall_aggreg_pmean_error(implicacoes, p=2)

    return {
        "id": "consulta_3",
        "pergunta": "Se dois triângulos estão próximos, eles têm o mesmo tamanho?",
        "formula": "∀x,y((IsTriangle(x) ∧ IsTriangle(y) ∧ CloseTo(x,y)) → SameSize(x,y))",
        "valor_verdade": float(valor_universal),
        "resposta": texto_resposta_booleana(valor_universal),
        "pior_evidencia": pior_evidencia,
        "melhor_caso_com_premissa_ativa": melhor_caso_premissa,
    }


def consulta_4_existe_empilhamento(
    df_objetos: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
    exigir_objetos_distintos: bool = True,
) -> dict[str, Any]:
    """
    Consulta extra.

    Pergunta:
    Existe algum objeto que pode ser empilhado sobre outro?

    Fórmula aproximada:

    ∃x,y canStack(x,y)
    """

    n = len(df_objetos)

    melhor_score = -1.0
    melhor_evidencia: dict[str, Any] | None = None

    for id_x in range(n):
        for id_y in range(n):
            if exigir_objetos_distintos and id_x == id_y:
                continue

            v_can_stack = score_binario(
                modelos,
                tensor_objetos,
                "canStack",
                id_x,
                id_y,
            )

            if v_can_stack > melhor_score:
                melhor_score = v_can_stack
                melhor_evidencia = {
                    "objeto_x": objeto_para_dict(df_objetos, id_x),
                    "objeto_y": objeto_para_dict(df_objetos, id_y),
                    "componentes": {
                        f"canStack({id_x},{id_y})": v_can_stack,
                    },
                }

    return {
        "id": "consulta_4",
        "pergunta": "Existe algum objeto que pode ser empilhado sobre outro?",
        "formula": "∃x,y CanStack(x,y)",
        "valor_verdade": float(melhor_score),
        "resposta": texto_resposta_booleana(melhor_score),
        "melhor_evidencia": melhor_evidencia,
    }


@torch.no_grad()
def avaliar_formulas_extremos_horizontais(
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
) -> dict[str, float]:
    """
    Avalia as fórmulas lastOnTheLeft e lastOnTheRight com os quantificadores
    fuzzy do LTNtorch.

    Fórmulas:
    - lastOnTheLeft  = ∃x (∀y leftOf(x, y))
    - lastOnTheRight = ∃x (∀y rightOf(x, y))

    Quantificadores usados:
    - Forall com AggregPMeanError(p=2)
    - Exists com AggregPMean(p=2)
    """

    operadores = criar_operadores_logicos()
    variaveis = criar_variaveis_ltn(tensor_objetos)

    predicados = {
        "leftOf": ltn.Predicate(modelos["leftOf"]),
        "rightOf": ltn.Predicate(modelos["rightOf"]),
    }

    formulas = formulas_extremos_horizontais(
        variaveis=variaveis,
        predicados=predicados,
        operadores=operadores,
    )

    return {
        nome: float(formula.value.detach().cpu().item())
        for nome, formula in formulas.items()
    }


def consulta_extremo_horizontal(
    df_objetos: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
    nome_formula: str,
    nome_relacao: str,
    id_consulta: str,
    pergunta: str,
    formula_texto: str,
) -> dict[str, Any]:
    """
    Consulta genérica para os extremos horizontais da cena.

    O valor de verdade oficial da consulta vem dos quantificadores do LTNtorch,
    por meio de avaliar_formulas_extremos_horizontais.

    Além do valor agregado, esta função calcula uma evidência interpretável:
    para cada candidato x, replicamos o ∀y com o agregador AggregPMeanError(p=2)
    e reportamos o objeto que maximiza a quantificação universal.

    Observação:
    como leftOf e rightOf são irreflexivos e o ∀y inclui o próprio x,
    o valor de verdade nunca chega a 1, mesmo para o objeto do extremo.
    Por isso a resposta é dada pela existência de um candidato claramente
    dominante, e não apenas pelo corte padrão de 0.5.
    """

    valores_formulas = avaliar_formulas_extremos_horizontais(
        tensor_objetos=tensor_objetos,
        modelos=modelos,
    )

    valor_formula = valores_formulas[nome_formula]

    n = len(df_objetos)

    melhor_id = -1
    melhor_forall = -1.0
    forall_por_objeto: dict[int, float] = {}

    for id_x in range(n):
        scores = [
            score_binario(modelos, tensor_objetos, nome_relacao, id_x, id_y)
            for id_y in range(n)
        ]

        valor_forall = forall_aggreg_pmean_error(scores, p=2)
        forall_por_objeto[id_x] = valor_forall

        if valor_forall > melhor_forall:
            melhor_forall = valor_forall
            melhor_id = id_x

    melhor_evidencia = {
        "objeto_x": objeto_para_dict(df_objetos, melhor_id),
        "componentes": {
            f"Forall_y {nome_relacao}({melhor_id},y)": melhor_forall,
            f"Exists_x Forall_y {nome_relacao}(x,y) [LTN]": valor_formula,
        },
    }

    return {
        "id": id_consulta,
        "pergunta": pergunta,
        "formula": formula_texto,
        "valor_verdade": float(valor_formula),
        "resposta": texto_resposta_booleana(valor_formula),
        "melhor_evidencia": melhor_evidencia,
        "forall_por_objeto": {
            str(id_x): float(valor) for id_x, valor in forall_por_objeto.items()
        },
    }


def consulta_5_last_on_the_left(
    df_objetos: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
) -> dict[str, Any]:
    """
    Consulta 5.

    Pergunta:
    Existe um objeto que esteja à esquerda de todos os outros?

    Fórmula:
    lastOnTheLeft(x) = ∃x (∀y leftOf(x, y))
    """

    return consulta_extremo_horizontal(
        df_objetos=df_objetos,
        tensor_objetos=tensor_objetos,
        modelos=modelos,
        nome_formula="lastOnTheLeft",
        nome_relacao="leftOf",
        id_consulta="consulta_5",
        pergunta="Existe um objeto que esteja à esquerda de todos os outros (último da esquerda)?",
        formula_texto="lastOnTheLeft(x) = ∃x(∀y LeftOf(x,y))",
    )


def consulta_6_last_on_the_right(
    df_objetos: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
) -> dict[str, Any]:
    """
    Consulta 6.

    Pergunta:
    Existe um objeto que esteja à direita de todos os outros?

    Fórmula:
    lastOnTheRight(x) = ∃x (∀y rightOf(x, y))
    """

    return consulta_extremo_horizontal(
        df_objetos=df_objetos,
        tensor_objetos=tensor_objetos,
        modelos=modelos,
        nome_formula="lastOnTheRight",
        nome_relacao="rightOf",
        id_consulta="consulta_6",
        pergunta="Existe um objeto que esteja à direita de todos os outros (último da direita)?",
        formula_texto="lastOnTheRight(x) = ∃x(∀y RightOf(x,y))",
    )


def gerar_top_pares_binarios(
    df_objetos: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
    relacoes: list[str] | None = None,
    top_k: int = 10,
    exigir_objetos_distintos: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Gera os top pares para cada relação binária.

    Isso ajuda na explicabilidade e no relatório.

    Exemplo:
    - top 10 pares com maior leftOf
    - top 10 pares com maior below
    - top 10 pares com maior canStack
    """

    if relacoes is None:
        relacoes = [
            "leftOf",
            "rightOf",
            "below",
            "above",
            "closeTo",
            "canStack",
        ]

    n = len(df_objetos)

    resultado: dict[str, list[dict[str, Any]]] = {}

    for relacao in relacoes:
        registros = []

        for id_a in range(n):
            for id_b in range(n):
                if exigir_objetos_distintos and id_a == id_b:
                    continue

                score = score_binario(
                    modelos,
                    tensor_objetos,
                    relacao,
                    id_a,
                    id_b,
                )

                registros.append(
                    {
                        "id_a": id_a,
                        "id_b": id_b,
                        "score": float(score),
                        "objeto_a": objeto_para_dict(df_objetos, id_a),
                        "objeto_b": objeto_para_dict(df_objetos, id_b),
                    }
                )

        registros = sorted(
            registros,
            key=lambda item: item["score"],
            reverse=True,
        )

        resultado[relacao] = registros[:top_k]

    return resultado


def gerar_top_triplas_inbetween(
    df_objetos: pd.DataFrame,
    tensor_objetos: torch.Tensor,
    modelos: dict[str, nn.Module],
    top_k: int = 10,
    exigir_objetos_distintos: bool = True,
) -> list[dict[str, Any]]:
    """
    Gera as top triplas com maior score de inBetween.
    """

    n = len(df_objetos)

    registros = []

    for id_x in range(n):
        for id_y in range(n):
            for id_z in range(n):
                if exigir_objetos_distintos and not ids_distintos(id_x, id_y, id_z):
                    continue

                score = score_ternario(
                    modelos,
                    tensor_objetos,
                    "inBetween",
                    id_x,
                    id_y,
                    id_z,
                )

                registros.append(
                    {
                        "id_x": id_x,
                        "id_y": id_y,
                        "id_z": id_z,
                        "score": float(score),
                        "objeto_x": objeto_para_dict(df_objetos, id_x),
                        "objeto_y": objeto_para_dict(df_objetos, id_y),
                        "objeto_z": objeto_para_dict(df_objetos, id_z),
                    }
                )

    registros = sorted(
        registros,
        key=lambda item: item["score"],
        reverse=True,
    )

    return registros[:top_k]


def converter_consultas_para_csv(
    consultas: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Converte as consultas principais em uma tabela simples para CSV.
    """

    linhas = []

    for consulta in consultas:
        linhas.append(
            {
                "id": consulta["id"],
                "pergunta": consulta["pergunta"],
                "formula": consulta["formula"],
                "valor_verdade": consulta["valor_verdade"],
                "resposta": consulta["resposta"],
            }
        )

    return pd.DataFrame(linhas)


def formatar_objeto_md(
    objeto: dict[str, Any],
) -> str:
    """
    Formata um objeto para Markdown.
    """

    return (
        f"Objeto {objeto['id_objeto']} "
        f"({objeto['cor']}, {objeto['forma']}, {objeto['tamanho']}, "
        f"x={objeto['x']:.4f}, y={objeto['y']:.4f})"
    )


def formatar_componentes_md(
    componentes: dict[str, float],
) -> list[str]:
    """
    Formata componentes de uma evidência.
    """

    linhas = []

    for nome, valor in componentes.items():
        linhas.append(f"- `{nome}` = `{valor:.4f}`")

    return linhas


def gerar_markdown_explicacoes(
    consultas: list[dict[str, Any]],
    top_pares: dict[str, list[dict[str, Any]]],
    top_triplas: list[dict[str, Any]],
) -> str:
    """
    Gera o conteúdo Markdown com explicações das consultas.
    """

    linhas = []

    linhas.append("# Explicações das consultas LTN")
    linhas.append("")
    linhas.append(
        "Este arquivo apresenta as respostas das consultas compostas executadas sobre o modelo LTN treinado."
    )
    linhas.append(
        "Os valores de verdade estão no intervalo `[0, 1]`, em que valores próximos de `1` indicam maior satisfação da consulta."
    )
    linhas.append("")

    linhas.append("## 1. Consultas principais")
    linhas.append("")

    for consulta in consultas:
        linhas.append(f"### {consulta['id']} — {consulta['pergunta']}")
        linhas.append("")
        linhas.append(f"**Fórmula:** `{consulta['formula']}`")
        linhas.append("")
        linhas.append(f"**Valor de verdade:** `{consulta['valor_verdade']:.4f}`")
        linhas.append("")
        linhas.append(f"**Resposta:** `{consulta['resposta']}`")
        linhas.append("")

        if "melhor_evidencia" in consulta and consulta["melhor_evidencia"]:
            evidencia = consulta["melhor_evidencia"]

            linhas.append("**Melhor evidência encontrada:**")
            linhas.append("")

            for chave_objeto in ["objeto_x", "objeto_y", "objeto_z"]:
                if chave_objeto in evidencia:
                    linhas.append(f"- {chave_objeto}: {formatar_objeto_md(evidencia[chave_objeto])}")

            if "componentes" in evidencia:
                linhas.append("")
                linhas.append("**Componentes da evidência:**")
                linhas.extend(formatar_componentes_md(evidencia["componentes"]))

        if "pior_evidencia" in consulta and consulta["pior_evidencia"]:
            evidencia = consulta["pior_evidencia"]

            linhas.append("**Pior caso avaliado para a regra universal:**")
            linhas.append("")

            for chave_objeto in ["objeto_x", "objeto_y"]:
                if chave_objeto in evidencia:
                    linhas.append(f"- {chave_objeto}: {formatar_objeto_md(evidencia[chave_objeto])}")

            if "componentes" in evidencia:
                linhas.append("")
                linhas.append("**Componentes do pior caso:**")
                linhas.extend(formatar_componentes_md(evidencia["componentes"]))

            linhas.append("")
            linhas.append(f"- Premissa = `{evidencia.get('premissa', 0.0):.4f}`")
            linhas.append(f"- Conclusão sameSize = `{evidencia.get('conclusao_sameSize', 0.0):.4f}`")
            linhas.append(f"- Implicação = `{evidencia.get('implicacao', 0.0):.4f}`")

        if "melhor_caso_com_premissa_ativa" in consulta and consulta["melhor_caso_com_premissa_ativa"]:
            evidencia = consulta["melhor_caso_com_premissa_ativa"]

            linhas.append("")
            linhas.append("**Caso com premissa mais ativa:**")
            linhas.append("")

            for chave_objeto in ["objeto_x", "objeto_y"]:
                if chave_objeto in evidencia:
                    linhas.append(f"- {chave_objeto}: {formatar_objeto_md(evidencia[chave_objeto])}")

            if "componentes" in evidencia:
                linhas.append("")
                linhas.append("**Componentes do caso com premissa mais ativa:**")
                linhas.extend(formatar_componentes_md(evidencia["componentes"]))

        linhas.append("")
        linhas.append("---")
        linhas.append("")

    linhas.append("## 2. Top pares por relação binária")
    linhas.append("")

    for relacao, registros in top_pares.items():
        linhas.append(f"### Relação `{relacao}`")
        linhas.append("")

        for i, item in enumerate(registros, start=1):
            linhas.append(
                f"{i}. `{relacao}({item['id_a']},{item['id_b']})` = `{item['score']:.4f}`"
            )
            linhas.append(f"   - A: {formatar_objeto_md(item['objeto_a'])}")
            linhas.append(f"   - B: {formatar_objeto_md(item['objeto_b'])}")

        linhas.append("")

    linhas.append("## 3. Top triplas para `inBetween`")
    linhas.append("")

    for i, item in enumerate(top_triplas, start=1):
        linhas.append(
            f"{i}. `inBetween({item['id_x']},{item['id_y']},{item['id_z']})` = `{item['score']:.4f}`"
        )
        linhas.append(f"   - X: {formatar_objeto_md(item['objeto_x'])}")
        linhas.append(f"   - Y: {formatar_objeto_md(item['objeto_y'])}")
        linhas.append(f"   - Z: {formatar_objeto_md(item['objeto_z'])}")

    linhas.append("")

    return "\n".join(linhas)


def salvar_resultados(
    consultas: list[dict[str, Any]],
    top_pares: dict[str, list[dict[str, Any]]],
    top_triplas: list[dict[str, Any]],
) -> dict[str, Path]:
    """
    Salva os resultados das consultas em JSON, CSV e Markdown.
    """

    garantir_pasta_saida()

    resultado_completo = {
        "consultas": consultas,
        "top_pares_binarios": top_pares,
        "top_triplas_inbetween": top_triplas,
    }

    RESPOSTAS_JSON.write_text(
        json.dumps(resultado_completo, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    df_consultas = converter_consultas_para_csv(consultas)
    df_consultas.to_csv(
        RESPOSTAS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    markdown = gerar_markdown_explicacoes(
        consultas=consultas,
        top_pares=top_pares,
        top_triplas=top_triplas,
    )

    EXPLICACOES_MD.write_text(
        markdown,
        encoding="utf-8",
    )

    return {
        "json": RESPOSTAS_JSON,
        "csv": RESPOSTAS_CSV,
        "markdown": EXPLICACOES_MD,
    }


def imprimir_resumo_console(
    consultas: list[dict[str, Any]],
    caminhos_saida: dict[str, Path],
) -> None:
    """
    Exibe no terminal um resumo das consultas.
    """

    print("=" * 80)
    print("CONSULTAS LTN")
    print("=" * 80)

    print("\nResultados principais:")

    for consulta in consultas:
        print(
            f"- {consulta['id']}: "
            f"{consulta['resposta']} "
            f"(valor={consulta['valor_verdade']:.4f})"
        )
        print(f"  Pergunta: {consulta['pergunta']}")

    print("\nArquivos gerados:")

    for nome, caminho in caminhos_saida.items():
        print(f"- {nome}: {caminho}")

    print("\nConsultas concluídas com sucesso.")


def executar_consultas_ltn() -> None:
    """
    Executa o fluxo completo de consultas LTN.

    Etapas:
    1. Carrega dados e checkpoint.
    2. Reconstrói modelos treináveis e predicado determinístico closeTo.
    3. Executa consultas compostas.
    4. Gera top pares e top triplas.
    5. Salva JSON, CSV e Markdown.
    """

    garantir_pasta_saida()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device usado nas consultas: {device}")

    dados = carregar_dados_e_modelo(device=device)

    df_objetos = dados["df_objetos"]
    vetores = dados["vetores"]
    tensor_objetos = dados["tensor_objetos"]
    checkpoint = dados["checkpoint"]

    modelos = reconstruir_modelos_para_consulta(
        checkpoint=checkpoint,
        device=device,
    )

    consultas = [
        consulta_1_objeto_pequeno_abaixo_cilindro_esquerda_quadrado(
            df_objetos=df_objetos,
            vetores=vetores,
            tensor_objetos=tensor_objetos,
            modelos=modelos,
            exigir_objetos_distintos=True,
        ),
        consulta_2_cone_verde_entre_dois_objetos(
            df_objetos=df_objetos,
            vetores=vetores,
            tensor_objetos=tensor_objetos,
            modelos=modelos,
            exigir_objetos_distintos=True,
        ),
        consulta_3_triangulos_proximos_mesmo_tamanho(
            df_objetos=df_objetos,
            vetores=vetores,
            tensor_objetos=tensor_objetos,
            modelos=modelos,
            exigir_objetos_distintos=True,
        ),
        consulta_4_existe_empilhamento(
            df_objetos=df_objetos,
            tensor_objetos=tensor_objetos,
            modelos=modelos,
            exigir_objetos_distintos=True,
        ),
        consulta_5_last_on_the_left(
            df_objetos=df_objetos,
            tensor_objetos=tensor_objetos,
            modelos=modelos,
        ),
        consulta_6_last_on_the_right(
            df_objetos=df_objetos,
            tensor_objetos=tensor_objetos,
            modelos=modelos,
        ),
    ]

    top_pares = gerar_top_pares_binarios(
        df_objetos=df_objetos,
        tensor_objetos=tensor_objetos,
        modelos=modelos,
        top_k=10,
        exigir_objetos_distintos=True,
    )

    top_triplas = gerar_top_triplas_inbetween(
        df_objetos=df_objetos,
        tensor_objetos=tensor_objetos,
        modelos=modelos,
        top_k=10,
        exigir_objetos_distintos=True,
    )

    caminhos_saida = salvar_resultados(
        consultas=consultas,
        top_pares=top_pares,
        top_triplas=top_triplas,
    )

    imprimir_resumo_console(
        consultas=consultas,
        caminhos_saida=caminhos_saida,
    )


if __name__ == "__main__":
    executar_consultas_ltn()