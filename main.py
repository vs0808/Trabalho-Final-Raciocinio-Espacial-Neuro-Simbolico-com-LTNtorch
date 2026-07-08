from pathlib import Path
import argparse

from src.data_generator import gerar_dados_sinteticos, exportar_dados
from src.plot_scene import plotar_cenario
from src.relations_ground_truth import gerar_ground_truth_completo


PROJECT_ROOT = Path(__file__).resolve().parent


def preparar_dados(
    n_objetos: int = 25,
    seed: int = 42,
    mostrar_plot: bool = True,
) -> None:
    """
    Executa a etapa de preparação do projeto.

    Esta função é responsável por:

    1. Gerar os dados sintéticos dos objetos.
    2. Exportar os objetos em CSV.
    3. Exportar os vetores em NPY.
    4. Gerar o gráfico da cena 2D.
    5. Gerar as tabelas de verdade-terreno:
       - predicados unários
       - predicados binários
       - predicados ternários

    Essa etapa deve ser executada antes do treinamento LTN.
    """

    data_dir = PROJECT_ROOT / "data" / "datasets_gerados"
    plots_dir = PROJECT_ROOT / "results" / "plots"

    print("=" * 80)
    print("ETAPA 1 — GERAÇÃO DOS DADOS SINTÉTICOS")
    print("=" * 80)

    print(f"\nGerando {n_objetos} objetos sintéticos com seed={seed}...")

    df_objetos = gerar_dados_sinteticos(
        n_objetos=n_objetos,
        seed=seed,
    )

    print("\nPrévia dos dados gerados:")
    print(df_objetos.head())

    print("\nExportando dados sintéticos...")

    caminhos_dados = exportar_dados(
        df=df_objetos,
        output_dir=data_dir,
        nome_base="objetos_sinteticos",
    )

    print(f"CSV salvo em: {caminhos_dados['csv']}")
    print(f"NPY salvo em: {caminhos_dados['npy']}")

    print("\n" + "=" * 80)
    print("ETAPA 2 — PLOTAGEM DO CENÁRIO 2D")
    print("=" * 80)

    caminho_plot = plotar_cenario(
        df=df_objetos,
        output_dir=plots_dir,
        nome_arquivo="cenario_sintetico.png",
        mostrar=mostrar_plot,
    )

    print(f"\nPlot salvo em: {caminho_plot}")

    print("\n" + "=" * 80)
    print("ETAPA 3 — GERAÇÃO DA VERDADE-TERRENO")
    print("=" * 80)

    resultado_ground_truth = gerar_ground_truth_completo(
        df=df_objetos,
        output_dir=data_dir,
        exportar=True,
    )

    caminhos_ground_truth = resultado_ground_truth["caminhos"]

    print("\nGround truth exportado:")
    print(f"Unários: {caminhos_ground_truth['unarios']}")
    print(f"Pares: {caminhos_ground_truth['pares']}")
    print(f"Triplas: {caminhos_ground_truth['triplas']}")

    print("\nResumo dos arquivos gerados:")

    print("\nDados dos objetos:")
    print(f"- {caminhos_dados['csv']}")
    print(f"- {caminhos_dados['npy']}")

    print("\nVisualização:")
    print(f"- {caminho_plot}")

    print("\nVerdade-terreno:")
    print(f"- {caminhos_ground_truth['unarios']}")
    print(f"- {caminhos_ground_truth['pares']}")
    print(f"- {caminhos_ground_truth['triplas']}")

    print("\nPreparação concluída com sucesso.")


def executar_treinamento(seed: int = 42) -> None:
    """
    Executa o treinamento LTN.

    O import é feito dentro da função de propósito.

    Motivo:
    - a preparação dos dados não precisa carregar torch/ltn;
    - se o ambiente ainda não tiver LTNtorch instalado, o usuário ainda consegue
      gerar os dados sintéticos e a verdade-terreno;
    - o treinamento fica opcional.

    A seed recebida é a mesma usada na geração dos dados, garantindo que
    execuções com seeds distintas sejam totalmente independentes.
    """

    print("\n" + "=" * 80)
    print("ETAPA 4 — TREINAMENTO LTN")
    print("=" * 80)

    from src.train_ltn import treinar_ltn

    treinar_ltn(
        epochs=200,
        lr=0.001,
        peso_axiomas=0.30,
        frac_treino=0.8,
        threshold=0.5,
        seed=seed,
    )


def main() -> None:
    """
    Ponto de entrada principal do projeto.

    Por padrão, este script prepara os dados.

    Uso comum:
        python main.py

    Para também executar o treinamento:
        python main.py --train

    Para não exibir o gráfico na tela:
        python main.py --no-show

    Para alterar a seed:
        python main.py --seed 43

    Para alterar a quantidade de objetos:
        python main.py --n-objetos 25
    """

    parser = argparse.ArgumentParser(
        description="Pipeline inicial do projeto LTN Spatial Reasoning."
    )

    parser.add_argument(
        "--n-objetos",
        type=int,
        default=25,
        help="Quantidade de objetos sintéticos a serem gerados. Padrão: 25.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed usada para reprodutibilidade dos dados sintéticos. Padrão: 42.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Não exibe o gráfico na tela, apenas salva o arquivo PNG.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Executa o treinamento LTN após preparar os dados.",
    )

    args = parser.parse_args()

    preparar_dados(
        n_objetos=args.n_objetos,
        seed=args.seed,
        mostrar_plot=not args.no_show,
    )

    if args.train:
        executar_treinamento(seed=args.seed)


if __name__ == "__main__":
    main()