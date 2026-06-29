from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Caminho absoluto da raiz do projeto.
# Como este arquivo está dentro de src/, usamos parents[1]
# para chegar na pasta projeto_ltn_spatial/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


MAPA_CORES = {
    "vermelho": "red",
    "verde": "green",
    "azul": "blue",
}


MAPA_MARCADORES = {
    "circulo": "o",
    "quadrado": "s",
    "cilindro": "D",
    "cone": "^",
    "triangulo": "v",
}


MAPA_TAMANHOS = {
    "pequeno": 120,
    "grande": 260,
}


def plotar_cenario(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    nome_arquivo: str = "cenario_sintetico.png",
    mostrar: bool = True,
) -> Path:
    """
    Plota o cenário 2D com os objetos sintéticos.

    Cada objeto é representado visualmente por:
    - posição x, y no plano
    - cor visual de acordo com o atributo 'cor'
    - marcador de acordo com o atributo 'forma'
    - tamanho visual de acordo com o atributo 'tamanho'
    - rótulo com id_objeto

    Por padrão, salva fora da pasta src, em:
    projeto_ltn_spatial/results/plots/
    """

    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "plots"
    else:
        output_dir = Path(output_dir)

        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    caminho_plot = output_dir / nome_arquivo

    fig, ax = plt.subplots(figsize=(10, 8))

    for _, linha in df.iterrows():
        cor_visual = MAPA_CORES[linha["cor"]]
        marcador = MAPA_MARCADORES[linha["forma"]]
        tamanho_visual = MAPA_TAMANHOS[linha["tamanho"]]

        ax.scatter(
            linha["x"],
            linha["y"],
            c=cor_visual,
            marker=marcador,
            s=tamanho_visual,
            alpha=0.75,
            edgecolors="black",
            linewidths=1,
        )

        ax.text(
            linha["x"] + 0.01,
            linha["y"] + 0.01,
            str(linha["id_objeto"]),
            fontsize=9,
        )

    ax.set_title("Cenário sintético 2D com 25 objetos", fontsize=14)
    ax.set_xlabel("Posição X normalizada")
    ax.set_ylabel("Posição Y normalizada")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.grid(True, linestyle="--", alpha=0.4)

    legenda_formas = []

    for forma, marcador in MAPA_MARCADORES.items():
        item = ax.scatter(
            [],
            [],
            marker=marcador,
            c="gray",
            s=120,
            edgecolors="black",
            label=forma,
        )
        legenda_formas.append(item)

    ax.legend(
        handles=legenda_formas,
        title="Formas",
        loc="upper right",
        bbox_to_anchor=(1.28, 1.0),
    )

    plt.tight_layout()
    plt.savefig(caminho_plot, dpi=300)

    if mostrar:
        plt.show()

    plt.close(fig)

    return caminho_plot


if __name__ == "__main__":
    caminho_csv = PROJECT_ROOT / "data" / "datasets_gerados" / "objetos_sinteticos.csv"

    df = pd.read_csv(caminho_csv)
    caminho_plot = plotar_cenario(df)

    print(f"Plot salvo em: {caminho_plot}")