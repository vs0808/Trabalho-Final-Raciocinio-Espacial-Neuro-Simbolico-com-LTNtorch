import json
from pathlib import Path

import numpy as np
import pandas as pd


# Caminho absoluto da raiz do projeto.
# Como este arquivo está dentro de src/, usamos parents[1]
# para subir um nível e chegar em projeto_ltn_spatial/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


CORES = ["vermelho", "verde", "azul"]

FORMAS = ["circulo", "quadrado", "cilindro", "cone", "triangulo"]

TAMANHOS = ["pequeno", "grande"]


def criar_vetor_11(x: float, y: float, cor: str, forma: str, tamanho: str) -> list[float]:
    """
    Cria o vetor de características de tamanho 11 para um objeto.

    Estrutura do vetor:
    [0]      -> posição x
    [1]      -> posição y
    [2:5]    -> cor one-hot: vermelho, verde, azul
    [5:10]   -> forma one-hot: circulo, quadrado, cilindro, cone, triangulo
    [10]     -> tamanho: pequeno = 0.0, grande = 1.0
    """

    if cor not in CORES:
        raise ValueError(f"Cor inválida: {cor}. Use uma das opções: {CORES}")

    if forma not in FORMAS:
        raise ValueError(f"Forma inválida: {forma}. Use uma das opções: {FORMAS}")

    if tamanho not in TAMANHOS:
        raise ValueError(f"Tamanho inválido: {tamanho}. Use uma das opções: {TAMANHOS}")

    cor_one_hot = [1.0 if cor == c else 0.0 for c in CORES]
    forma_one_hot = [1.0 if forma == f else 0.0 for f in FORMAS]
    tamanho_binario = 0.0 if tamanho == "pequeno" else 1.0

    vetor = [
        float(x),
        float(y),
        *cor_one_hot,
        *forma_one_hot,
        tamanho_binario,
    ]

    if len(vetor) != 11:
        raise ValueError(f"O vetor deveria ter tamanho 11, mas possui tamanho {len(vetor)}.")

    return vetor


def gerar_dados_sinteticos(
    n_objetos: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Gera uma tabela sintética com objetos aleatórios para o projeto LTN.

    Cada objeto possui:
    - id_objeto
    - posição x normalizada entre 0.0 e 1.0
    - posição y normalizada entre 0.0 e 1.0
    - cor
    - forma
    - tamanho
    - vetor_11

    O parâmetro seed garante reprodutibilidade.
    Com a mesma seed, os mesmos dados serão gerados novamente.
    """

    rng = np.random.default_rng(seed)

    registros = []

    for id_objeto in range(n_objetos):
        x = rng.uniform(0.0, 1.0)
        y = rng.uniform(0.0, 1.0)

        cor = rng.choice(CORES)
        forma = rng.choice(FORMAS)
        tamanho = rng.choice(TAMANHOS)

        vetor_11 = criar_vetor_11(
            x=x,
            y=y,
            cor=cor,
            forma=forma,
            tamanho=tamanho,
        )

        registros.append(
            {
                "id_objeto": id_objeto,
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "cor": cor,
                "forma": forma,
                "tamanho": tamanho,
                "vetor_11": json.dumps(vetor_11),
            }
        )

    df = pd.DataFrame(registros)

    return df


def exportar_dados(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    nome_base: str = "objetos_sinteticos",
) -> dict[str, Path]:
    """
    Exporta os dados sintéticos em dois formatos:

    1. CSV:
       Formato tabular, fácil de abrir no Excel e conferir.

    2. NPY:
       Matriz NumPy com shape (n_objetos, 11), útil para uso posterior
       com PyTorch e LTNtorch.

    Por padrão, salva fora da pasta src, em:
    projeto_ltn_spatial/data/datasets_gerados/
    """

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "datasets_gerados"
    else:
        output_dir = Path(output_dir)

        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    caminho_csv = output_dir / f"{nome_base}.csv"
    caminho_npy = output_dir / f"{nome_base}_vetores.npy"

    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

    matriz_vetores = np.array(
        [json.loads(vetor) for vetor in df["vetor_11"]],
        dtype=np.float32,
    )

    np.save(caminho_npy, matriz_vetores)

    return {
        "csv": caminho_csv,
        "npy": caminho_npy,
    }


if __name__ == "__main__":
    df = gerar_dados_sinteticos(n_objetos=25, seed=42)
    caminhos = exportar_dados(df)

    print(df)
    print("\nArquivos exportados:")
    for tipo, caminho in caminhos.items():
        print(f"{tipo}: {caminho}")