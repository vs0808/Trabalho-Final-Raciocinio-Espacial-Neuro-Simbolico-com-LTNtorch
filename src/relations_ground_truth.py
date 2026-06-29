import json
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CORES = ["vermelho", "verde", "azul"]

FORMAS = ["circulo", "quadrado", "cilindro", "cone", "triangulo"]

TAMANHOS = ["pequeno", "grande"]


DEFAULT_CLOSE_THRESHOLD = 0.25
DEFAULT_STABLE_HORIZONTAL_THRESHOLD = 0.12
DEFAULT_SEGMENT_TOLERANCE = 0.05


def get_valor(obj: pd.Series | dict[str, Any], coluna: str) -> Any:
    """
    Acessa um valor de um objeto, seja ele uma linha de DataFrame
    ou um dicionário Python.
    """

    return obj[coluna]


def get_id(obj: pd.Series | dict[str, Any]) -> int:
    """
    Retorna o identificador inteiro do objeto.
    """

    return int(get_valor(obj, "id_objeto"))


def get_x(obj: pd.Series | dict[str, Any]) -> float:
    """
    Retorna a coordenada x do objeto.
    """

    return float(get_valor(obj, "x"))


def get_y(obj: pd.Series | dict[str, Any]) -> float:
    """
    Retorna a coordenada y do objeto.
    """

    return float(get_valor(obj, "y"))


def get_posicao(obj: pd.Series | dict[str, Any]) -> np.ndarray:
    """
    Retorna a posição 2D do objeto como array NumPy: [x, y].
    """

    return np.array([get_x(obj), get_y(obj)], dtype=np.float32)


def get_cor(obj: pd.Series | dict[str, Any]) -> str:
    """
    Retorna a cor textual do objeto.
    """

    return str(get_valor(obj, "cor"))


def get_forma(obj: pd.Series | dict[str, Any]) -> str:
    """
    Retorna a forma textual do objeto.
    """

    return str(get_valor(obj, "forma"))


def get_tamanho(obj: pd.Series | dict[str, Any]) -> str:
    """
    Retorna o tamanho textual do objeto.
    """

    return str(get_valor(obj, "tamanho"))


def distancia_euclidiana(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
) -> float:
    """
    Calcula a distância Euclidiana entre dois objetos no plano 2D.

    Fórmula:
    sqrt((x_a - x_b)^2 + (y_a - y_b)^2)
    """

    pos_a = get_posicao(obj_a)
    pos_b = get_posicao(obj_b)

    return float(np.linalg.norm(pos_a - pos_b))


def is_red(obj: pd.Series | dict[str, Any]) -> bool:
    return get_cor(obj) == "vermelho"


def is_green(obj: pd.Series | dict[str, Any]) -> bool:
    return get_cor(obj) == "verde"


def is_blue(obj: pd.Series | dict[str, Any]) -> bool:
    return get_cor(obj) == "azul"


def is_circle(obj: pd.Series | dict[str, Any]) -> bool:
    return get_forma(obj) == "circulo"


def is_square(obj: pd.Series | dict[str, Any]) -> bool:
    return get_forma(obj) == "quadrado"


def is_cylinder(obj: pd.Series | dict[str, Any]) -> bool:
    return get_forma(obj) == "cilindro"


def is_cone(obj: pd.Series | dict[str, Any]) -> bool:
    return get_forma(obj) == "cone"


def is_triangle(obj: pd.Series | dict[str, Any]) -> bool:
    return get_forma(obj) == "triangulo"


def is_small(obj: pd.Series | dict[str, Any]) -> bool:
    return get_tamanho(obj) == "pequeno"


def is_big(obj: pd.Series | dict[str, Any]) -> bool:
    return get_tamanho(obj) == "grande"


def same_color(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
) -> bool:
    """
    Verifica se dois objetos têm a mesma cor.
    """

    return get_cor(obj_a) == get_cor(obj_b)


def same_shape(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
) -> bool:
    """
    Verifica se dois objetos têm a mesma forma.
    """

    return get_forma(obj_a) == get_forma(obj_b)


def same_size(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
) -> bool:
    """
    Verifica se dois objetos têm o mesmo tamanho.
    """

    return get_tamanho(obj_a) == get_tamanho(obj_b)


def left_of(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
) -> bool:
    """
    Verifica se obj_a está à esquerda de obj_b.

    Como o eixo x cresce da esquerda para a direita:
    obj_a está à esquerda de obj_b se x_a < x_b.
    """

    return get_x(obj_a) < get_x(obj_b)


def right_of(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
) -> bool:
    """
    Verifica se obj_a está à direita de obj_b.

    Como o eixo x cresce da esquerda para a direita:
    obj_a está à direita de obj_b se x_a > x_b.
    """

    return get_x(obj_a) > get_x(obj_b)


def below(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
) -> bool:
    """
    Verifica se obj_a está abaixo de obj_b.

    Assumimos um plano cartesiano padrão, no qual y maior significa mais alto.
    Portanto:
    obj_a está abaixo de obj_b se y_a < y_b.
    """

    return get_y(obj_a) < get_y(obj_b)


def above(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
) -> bool:
    """
    Verifica se obj_a está acima de obj_b.

    Assumimos um plano cartesiano padrão, no qual y maior significa mais alto.
    Portanto:
    obj_a está acima de obj_b se y_a > y_b.
    """

    return get_y(obj_a) > get_y(obj_b)


def close_to(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
    threshold: float = DEFAULT_CLOSE_THRESHOLD,
) -> bool:
    """
    Verifica se dois objetos estão próximos.

    Como as coordenadas estão normalizadas entre 0 e 1,
    um threshold de 0.25 significa que objetos com distância menor
    que 25% da escala total são considerados próximos.
    """

    if get_id(obj_a) == get_id(obj_b):
        return False

    return distancia_euclidiana(obj_a, obj_b) < threshold


def horizontally_aligned(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
    tolerance: float = DEFAULT_SEGMENT_TOLERANCE,
) -> bool:
    """
    Verifica se dois objetos estão aproximadamente alinhados no eixo horizontal.

    Isso significa que a diferença entre seus valores de y é pequena.
    """

    return abs(get_y(obj_a) - get_y(obj_b)) <= tolerance


def vertically_aligned(
    obj_a: pd.Series | dict[str, Any],
    obj_b: pd.Series | dict[str, Any],
    tolerance: float = DEFAULT_SEGMENT_TOLERANCE,
) -> bool:
    """
    Verifica se dois objetos estão aproximadamente alinhados no eixo vertical.

    Isso significa que a diferença entre seus valores de x é pequena.
    """

    return abs(get_x(obj_a) - get_x(obj_b)) <= tolerance


def between_values(value: float, endpoint_a: float, endpoint_b: float) -> bool:
    """
    Verifica se um valor está entre dois extremos.
    """

    menor = min(endpoint_a, endpoint_b)
    maior = max(endpoint_a, endpoint_b)

    return menor <= value <= maior


def in_between_axis(
    obj_x: pd.Series | dict[str, Any],
    obj_y: pd.Series | dict[str, Any],
    obj_z: pd.Series | dict[str, Any],
) -> bool:
    """
    Verifica se obj_x está entre obj_y e obj_z considerando eixos.

    Aqui usamos uma interpretação mais simples:
    - x está entre y e z no eixo horizontal se sua coordenada x está entre as coordenadas x de y e z.
    - x está entre y e z no eixo vertical se sua coordenada y está entre as coordenadas y de y e z.

    Essa versão é útil para consultas do tipo:
    "o objeto está entre dois outros no eixo horizontal ou vertical?"
    """

    entre_no_x = between_values(get_x(obj_x), get_x(obj_y), get_x(obj_z))
    entre_no_y = between_values(get_y(obj_x), get_y(obj_y), get_y(obj_z))

    return entre_no_x or entre_no_y


def distance_point_to_segment(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float:
    """
    Calcula a menor distância entre um ponto e um segmento de reta.

    Essa função é usada para uma interpretação geométrica mais forte de 'inBetween':
    o objeto x está entre y e z se ele está próximo do segmento de reta que liga y a z.
    """

    segment = segment_end - segment_start
    segment_length_squared = float(np.dot(segment, segment))

    if segment_length_squared == 0.0:
        return float(np.linalg.norm(point - segment_start))

    t = float(np.dot(point - segment_start, segment) / segment_length_squared)
    t = max(0.0, min(1.0, t))

    projection = segment_start + t * segment

    return float(np.linalg.norm(point - projection))


def in_between_segment(
    obj_x: pd.Series | dict[str, Any],
    obj_y: pd.Series | dict[str, Any],
    obj_z: pd.Series | dict[str, Any],
    tolerance: float = DEFAULT_SEGMENT_TOLERANCE,
) -> bool:
    """
    Verifica se obj_x está entre obj_y e obj_z considerando o segmento de reta.

    Essa interpretação é mais geométrica:
    - traçamos uma linha entre y e z;
    - verificamos se x está próximo dessa linha;
    - também exigimos que x esteja dentro da região delimitada pelos extremos y e z.
    """

    point = get_posicao(obj_x)
    segment_start = get_posicao(obj_y)
    segment_end = get_posicao(obj_z)

    distancia = distance_point_to_segment(point, segment_start, segment_end)

    dentro_do_intervalo_x = between_values(get_x(obj_x), get_x(obj_y), get_x(obj_z))
    dentro_do_intervalo_y = between_values(get_y(obj_x), get_y(obj_y), get_y(obj_z))

    return distancia <= tolerance and dentro_do_intervalo_x and dentro_do_intervalo_y


def in_between(
    obj_x: pd.Series | dict[str, Any],
    obj_y: pd.Series | dict[str, Any],
    obj_z: pd.Series | dict[str, Any],
    tolerance: float = DEFAULT_SEGMENT_TOLERANCE,
) -> bool:
    """
    Verifica se obj_x está entre obj_y e obj_z.

    Usamos uma definição combinada:
    - entre no eixo horizontal ou vertical;
    - ou próximo do segmento de reta entre y e z.

    Também impedimos que o mesmo objeto seja usado como x, y e z ao mesmo tempo.
    """

    ids = {get_id(obj_x), get_id(obj_y), get_id(obj_z)}

    if len(ids) < 3:
        return False

    return in_between_axis(obj_x, obj_y, obj_z) or in_between_segment(
        obj_x,
        obj_y,
        obj_z,
        tolerance=tolerance,
    )


def is_leftmost(
    obj_x: pd.Series | dict[str, Any],
    objetos: list[pd.Series | dict[str, Any]],
) -> bool:
    """
    Verifica se obj_x é o objeto mais à esquerda da cena.
    """

    return all(get_x(obj_x) <= get_x(obj_y) for obj_y in objetos)


def is_rightmost(
    obj_x: pd.Series | dict[str, Any],
    objetos: list[pd.Series | dict[str, Any]],
) -> bool:
    """
    Verifica se obj_x é o objeto mais à direita da cena.
    """

    return all(get_x(obj_x) >= get_x(obj_y) for obj_y in objetos)


def is_lowest(
    obj_x: pd.Series | dict[str, Any],
    objetos: list[pd.Series | dict[str, Any]],
) -> bool:
    """
    Verifica se obj_x é o objeto mais abaixo da cena.
    """

    return all(get_y(obj_x) <= get_y(obj_y) for obj_y in objetos)


def is_highest(
    obj_x: pd.Series | dict[str, Any],
    objetos: list[pd.Series | dict[str, Any]],
) -> bool:
    """
    Verifica se obj_x é o objeto mais acima da cena.
    """

    return all(get_y(obj_x) >= get_y(obj_y) for obj_y in objetos)


def can_stack(
    obj_x: pd.Series | dict[str, Any],
    obj_y: pd.Series | dict[str, Any],
    stable_horizontal_threshold: float = DEFAULT_STABLE_HORIZONTAL_THRESHOLD,
) -> bool:
    """
    Verifica se obj_x pode ser empilhado sobre obj_y.

    Interpretação usada:
    - x deve estar acima de y;
    - y não pode ser cone;
    - y não pode ser triângulo;
    - x e y devem ter o mesmo tamanho OU seus centroides devem estar horizontalmente próximos.

    Essa regra segue a ideia do PDF:
    objeto x pode ser empilhado sobre y se y não for cone nem triângulo e se houver estabilidade.
    """

    if get_id(obj_x) == get_id(obj_y):
        return False

    base_invalida = is_cone(obj_y) or is_triangle(obj_y)

    if base_invalida:
        return False

    x_sobre_y = above(obj_x, obj_y)

    if not x_sobre_y:
        return False

    mesmos_tamanhos = same_size(obj_x, obj_y)

    distancia_horizontal = abs(get_x(obj_x) - get_x(obj_y))
    centroide_estavel = distancia_horizontal <= stable_horizontal_threshold

    return mesmos_tamanhos or centroide_estavel


def dataframe_para_objetos(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Converte o DataFrame em uma lista de dicionários.

    Isso deixa as funções mais simples para gerar combinações de objetos.
    """

    return df.to_dict(orient="records")


def gerar_ground_truth_unarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera a tabela de verdade-terreno para predicados unários.

    Predicados unários são aqueles que recebem apenas um objeto:
    isCircle(x), isSquare(x), isSmall(x), isRed(x), etc.
    """

    objetos = dataframe_para_objetos(df)

    predicados_unarios = {
        "isRed": is_red,
        "isGreen": is_green,
        "isBlue": is_blue,
        "isCircle": is_circle,
        "isSquare": is_square,
        "isCylinder": is_cylinder,
        "isCone": is_cone,
        "isTriangle": is_triangle,
        "isSmall": is_small,
        "isBig": is_big,
    }

    registros = []

    for obj in objetos:
        for nome_predicado, funcao in predicados_unarios.items():
            registros.append(
                {
                    "id_objeto": get_id(obj),
                    "predicado": nome_predicado,
                    "y_true": int(funcao(obj)),
                }
            )

    return pd.DataFrame(registros)


def gerar_ground_truth_pares(
    df: pd.DataFrame,
    incluir_auto_relacoes: bool = True,
) -> pd.DataFrame:
    """
    Gera a tabela de verdade-terreno para predicados binários.

    Predicados binários recebem dois objetos:
    leftOf(x,y), rightOf(x,y), below(x,y), above(x,y), closeTo(x,y), etc.

    incluir_auto_relacoes=True mantém pares como (x,x).
    Isso é útil para testar axiomas como:
    um objeto não pode estar à esquerda dele mesmo.
    """

    objetos = dataframe_para_objetos(df)

    registros = []

    for obj_a, obj_b in product(objetos, objetos):
        if not incluir_auto_relacoes and get_id(obj_a) == get_id(obj_b):
            continue

        registros.append(
            {
                "id_objeto_a": get_id(obj_a),
                "id_objeto_b": get_id(obj_b),
                "leftOf": int(left_of(obj_a, obj_b)),
                "rightOf": int(right_of(obj_a, obj_b)),
                "below": int(below(obj_a, obj_b)),
                "above": int(above(obj_a, obj_b)),
                "closeTo": int(close_to(obj_a, obj_b)),
                "sameColor": int(same_color(obj_a, obj_b)),
                "sameShape": int(same_shape(obj_a, obj_b)),
                "sameSize": int(same_size(obj_a, obj_b)),
                "horizontallyAligned": int(horizontally_aligned(obj_a, obj_b)),
                "verticallyAligned": int(vertically_aligned(obj_a, obj_b)),
                "canStack": int(can_stack(obj_a, obj_b)),
                "distancia_euclidiana": round(distancia_euclidiana(obj_a, obj_b), 4),
            }
        )

    return pd.DataFrame(registros)


def gerar_ground_truth_triplas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera a tabela de verdade-terreno para predicados ternários.

    Predicados ternários recebem três objetos:
    inBetween(x,y,z)

    Aqui usamos permutations, porque a ordem importa:
    inBetween(x,y,z) significa que x é o objeto candidato a estar entre y e z.
    """

    objetos = dataframe_para_objetos(df)

    registros = []

    for obj_x, obj_y, obj_z in permutations(objetos, 3):
        registros.append(
            {
                "id_objeto_x": get_id(obj_x),
                "id_objeto_y": get_id(obj_y),
                "id_objeto_z": get_id(obj_z),
                "inBetween": int(in_between(obj_x, obj_y, obj_z)),
                "inBetweenAxis": int(in_between_axis(obj_x, obj_y, obj_z)),
                "inBetweenSegment": int(in_between_segment(obj_x, obj_y, obj_z)),
            }
        )

    return pd.DataFrame(registros)


def exportar_ground_truth(
    df_unarios: pd.DataFrame,
    df_pares: pd.DataFrame,
    df_triplas: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Exporta as tabelas de verdade-terreno em CSV.
    """

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "datasets_gerados"
    else:
        output_dir = Path(output_dir)

        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    caminho_unarios = output_dir / "ground_truth_unarios.csv"
    caminho_pares = output_dir / "ground_truth_pares.csv"
    caminho_triplas = output_dir / "ground_truth_triplas.csv"

    df_unarios.to_csv(caminho_unarios, index=False, encoding="utf-8-sig")
    df_pares.to_csv(caminho_pares, index=False, encoding="utf-8-sig")
    df_triplas.to_csv(caminho_triplas, index=False, encoding="utf-8-sig")

    return {
        "unarios": caminho_unarios,
        "pares": caminho_pares,
        "triplas": caminho_triplas,
    }


def gerar_ground_truth_completo(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    exportar: bool = True,
) -> dict[str, pd.DataFrame | dict[str, Path]]:
    """
    Gera todas as tabelas de verdade-terreno:
    - unários
    - pares
    - triplas

    Opcionalmente exporta os CSVs.
    """

    df_unarios = gerar_ground_truth_unarios(df)
    df_pares = gerar_ground_truth_pares(df)
    df_triplas = gerar_ground_truth_triplas(df)

    resultado: dict[str, pd.DataFrame | dict[str, Path]] = {
        "unarios": df_unarios,
        "pares": df_pares,
        "triplas": df_triplas,
    }

    if exportar:
        caminhos = exportar_ground_truth(
            df_unarios=df_unarios,
            df_pares=df_pares,
            df_triplas=df_triplas,
            output_dir=output_dir,
        )

        resultado["caminhos"] = caminhos

    return resultado


if __name__ == "__main__":
    caminho_csv = PROJECT_ROOT / "data" / "datasets_gerados" / "objetos_sinteticos.csv"

    df_objetos = pd.read_csv(caminho_csv)

    resultado = gerar_ground_truth_completo(df_objetos, exportar=True)

    print("Ground truth unário:")
    print(resultado["unarios"].head())

    print("\nGround truth de pares:")
    print(resultado["pares"].head())

    print("\nGround truth de triplas:")
    print(resultado["triplas"].head())

    print("\nArquivos exportados:")
    for nome, caminho in resultado["caminhos"].items():
        print(f"{nome}: {caminho}")