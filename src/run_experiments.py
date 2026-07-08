from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

# Backend não interativo: o experimento roda várias vezes seguidas e não deve
# abrir janelas de gráfico.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import ltn

from main import preparar_dados
from src.train_ltn import treinar_ltn, RESULTS_DIR as TRAINING_DIR
from src.ltn_axioms import (
    criar_operadores_logicos,
    criar_variaveis_ltn,
    formulas_extremos_horizontais,
)


EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "experiments"

CSV_CONSOLIDADO = EXPERIMENTS_DIR / "experimentos_multi_seed.csv"
MD_CONSOLIDADO = EXPERIMENTS_DIR / "experimentos_multi_seed.md"


# Seeds usadas nas 5 execuções independentes.
#
# Cada seed controla:
# - a geração dos dados sintéticos (posições, cores, formas, tamanhos);
# - a separação treino/teste;
# - a inicialização dos pesos das redes neurais.
SEEDS_PADRAO = [42, 43, 44, 45, 46]


# Métricas clássicas consolidadas por predicado.
METRICAS_CLASSICAS = ["accuracy", "precision", "recall", "f1"]


@torch.no_grad()
def avaliar_formulas_extremos(
    modelos_avaliacao: dict[str, torch.nn.Module],
    tensor_objetos: torch.Tensor,
) -> dict[str, float]:
    """
    Avalia lastOnTheLeft e lastOnTheRight sobre o modelo recém-treinado.

    Fórmulas:
    - lastOnTheLeft  = ∃x (∀y leftOf(x, y))
    - lastOnTheRight = ∃x (∀y rightOf(x, y))

    A avaliação usa os quantificadores do LTNtorch:
    Forall com AggregPMeanError(p=2) e Exists com AggregPMean(p=2).
    """

    operadores = criar_operadores_logicos()
    variaveis = criar_variaveis_ltn(tensor_objetos)

    predicados = {
        "leftOf": ltn.Predicate(modelos_avaliacao["leftOf"]),
        "rightOf": ltn.Predicate(modelos_avaliacao["rightOf"]),
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


def executar_uma_execucao(
    seed: int,
    n_objetos: int = 25,
    epochs: int = 200,
) -> dict[str, Any]:
    """
    Executa uma rodada completa do pipeline para uma seed:

    1. Gera os dados sintéticos e a verdade-terreno (equivalente a
       python3 main.py --no-show --seed N).
    2. Treina a LTN com a mesma seed (equivalente à etapa --train).
    3. Coleta, da última época:
       - satAgg geral;
       - satisfatibilidade individual de cada axioma;
       - Accuracy, Precision, Recall e F1 de cada predicado avaliado.
    4. Avalia as fórmulas lastOnTheLeft e lastOnTheRight.

    Retorna um dicionário com todos os resultados da execução.
    """

    print("\n" + "#" * 80)
    print(f"# EXECUÇÃO COM SEED = {seed}")
    print("#" * 80)

    preparar_dados(
        n_objetos=n_objetos,
        seed=seed,
        mostrar_plot=False,
    )

    resultado_treino = treinar_ltn(
        epochs=epochs,
        lr=0.001,
        peso_axiomas=0.30,
        frac_treino=0.8,
        threshold=0.5,
        seed=seed,
        usar_loss_ponderada=True,
    )

    registro_final = resultado_treino["historico"][-1]

    valores_formulas = avaliar_formulas_extremos(
        modelos_avaliacao=resultado_treino["modelos_avaliacao"],
        tensor_objetos=resultado_treino["tensor_objetos"],
    )

    metricas_predicados: dict[str, dict[str, float]] = {}
    metricas_predicados.update(registro_final["metricas_binarias"])
    metricas_predicados.update(registro_final["metricas_ternarias"])

    return {
        "seed": seed,
        "sat_agg": float(registro_final["sat_agg"]),
        "sat_axiomas": dict(registro_final["sat_individual"]),
        "formulas": valores_formulas,
        "metricas_predicados": metricas_predicados,
        "caminho_historico": resultado_treino["caminho_metricas"],
    }


def preservar_artefatos_da_execucao(execucao: dict[str, Any]) -> None:
    """
    Copia os artefatos de cada execução para results/experiments/seed_<N>/.

    Como o pipeline sobrescreve results/training/historico_treinamento.csv a
    cada treinamento, guardamos uma cópia por seed junto com um resumo JSON.
    """

    seed_dir = EXPERIMENTS_DIR / f"seed_{execucao['seed']}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    caminho_historico = Path(execucao["caminho_historico"])

    if caminho_historico.exists():
        shutil.copy2(caminho_historico, seed_dir / caminho_historico.name)

    resumo = {
        "seed": execucao["seed"],
        "sat_agg": execucao["sat_agg"],
        "formulas": execucao["formulas"],
        "sat_axiomas": execucao["sat_axiomas"],
        "metricas_predicados": execucao["metricas_predicados"],
    }

    (seed_dir / "resumo_execucao.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def montar_tabela_consolidada(execucoes: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Monta a tabela consolidada em formato largo: uma linha por execução.

    Colunas:
    - seed
    - sat_agg
    - formula_lastOnTheLeft, formula_lastOnTheRight
    - sat_<axioma>, para cada axioma da base de conhecimento
    - <predicado>_accuracy, <predicado>_precision, <predicado>_recall,
      <predicado>_f1, para cada predicado avaliado
    """

    linhas = []

    for execucao in execucoes:
        linha: dict[str, Any] = {
            "seed": execucao["seed"],
            "sat_agg": execucao["sat_agg"],
        }

        for nome_formula, valor in execucao["formulas"].items():
            linha[f"formula_{nome_formula}"] = valor

        for nome_axioma, valor in execucao["sat_axiomas"].items():
            linha[f"sat_{nome_axioma}"] = valor

        for predicado, metricas in execucao["metricas_predicados"].items():
            for metrica in METRICAS_CLASSICAS:
                linha[f"{predicado}_{metrica}"] = metricas[metrica]

        linhas.append(linha)

    return pd.DataFrame(linhas)


def tabela_sat_por_axioma(execucoes: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Tabela com a satisfatibilidade de cada axioma por seed, mais média e desvio.

    Linhas: axiomas (incluindo satAgg geral e as fórmulas de consulta).
    Colunas: uma por seed, mais media e desvio_padrao.
    """

    dados: dict[str, dict[str, float]] = {}

    for execucao in execucoes:
        coluna = f"seed_{execucao['seed']}"

        dados.setdefault("satAgg (geral)", {})[coluna] = execucao["sat_agg"]

        for nome_formula, valor in execucao["formulas"].items():
            dados.setdefault(f"formula {nome_formula}", {})[coluna] = valor

        for nome_axioma, valor in execucao["sat_axiomas"].items():
            dados.setdefault(nome_axioma, {})[coluna] = valor

    df = pd.DataFrame(dados).T
    df.index.name = "axioma / formula"

    df["media"] = df.mean(axis=1)
    df["desvio_padrao"] = df.drop(columns=["media"]).std(axis=1)

    return df


def tabelas_metricas_por_predicado(
    execucoes: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """
    Uma tabela por métrica clássica (accuracy, precision, recall, f1).

    Linhas: predicados avaliados.
    Colunas: uma por seed, mais media e desvio_padrao.
    """

    tabelas: dict[str, pd.DataFrame] = {}

    for metrica in METRICAS_CLASSICAS:
        dados: dict[str, dict[str, float]] = {}

        for execucao in execucoes:
            coluna = f"seed_{execucao['seed']}"

            for predicado, metricas in execucao["metricas_predicados"].items():
                dados.setdefault(predicado, {})[coluna] = metricas[metrica]

        df = pd.DataFrame(dados).T
        df.index.name = "predicado"

        df["media"] = df.mean(axis=1)
        df["desvio_padrao"] = df.drop(columns=["media"]).std(axis=1)

        tabelas[metrica] = df

    return tabelas


def gerar_markdown_consolidado(
    execucoes: list[dict[str, Any]],
    epochs: int,
    n_objetos: int,
) -> str:
    """
    Gera o relatório Markdown com as tabelas consolidadas das 5 execuções.
    """

    seeds = [execucao["seed"] for execucao in execucoes]

    linhas: list[str] = []

    linhas.append("# Experimentos com múltiplas seeds")
    linhas.append("")
    linhas.append(
        f"Este relatório consolida {len(execucoes)} execuções independentes do pipeline "
        "(geração de dados + treinamento LTN), uma para cada seed."
    )
    linhas.append("")
    linhas.append(f"- Seeds: {seeds}")
    linhas.append(f"- Objetos por cena: {n_objetos}")
    linhas.append(f"- Épocas de treinamento: {epochs}")
    linhas.append(
        "- Valores de satisfatibilidade (sat) referem-se à última época de treinamento."
    )
    linhas.append(
        "- As métricas clássicas (Accuracy, Precision, Recall, F1) são calculadas "
        "no conjunto de teste com threshold 0.5."
    )
    linhas.append("")

    linhas.append("## 1. satAgg geral e fórmulas de consulta por execução")
    linhas.append("")
    linhas.append(
        "As fórmulas `lastOnTheLeft = ∃x(∀y leftOf(x,y))` e "
        "`lastOnTheRight = ∃x(∀y rightOf(x,y))` são avaliadas com os "
        "quantificadores do LTNtorch (Forall com AggregPMeanError(p=2) e "
        "Exists com AggregPMean(p=2)) sobre o modelo treinado de cada execução."
    )
    linhas.append("")

    df_resumo = pd.DataFrame(
        [
            {
                "seed": execucao["seed"],
                "satAgg": execucao["sat_agg"],
                "lastOnTheLeft": execucao["formulas"]["lastOnTheLeft"],
                "lastOnTheRight": execucao["formulas"]["lastOnTheRight"],
            }
            for execucao in execucoes
        ]
    ).set_index("seed")

    linhas.append(df_resumo.round(4).to_markdown())
    linhas.append("")

    linhas.append("## 2. Satisfatibilidade por axioma e por execução")
    linhas.append("")

    df_axiomas = tabela_sat_por_axioma(execucoes)
    linhas.append(df_axiomas.round(4).to_markdown())
    linhas.append("")

    linhas.append("## 3. Métricas clássicas dos predicados por execução")
    linhas.append("")

    tabelas_metricas = tabelas_metricas_por_predicado(execucoes)

    titulos = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }

    for metrica in METRICAS_CLASSICAS:
        linhas.append(f"### 3.{METRICAS_CLASSICAS.index(metrica) + 1}. {titulos[metrica]}")
        linhas.append("")
        linhas.append(tabelas_metricas[metrica].round(4).to_markdown())
        linhas.append("")

    linhas.append("## 4. Arquivos gerados")
    linhas.append("")
    linhas.append(f"- Tabela consolidada (formato largo): `{CSV_CONSOLIDADO.relative_to(PROJECT_ROOT)}`")
    linhas.append(
        "- Artefatos por execução (histórico de treinamento e resumo JSON): "
        "`results/experiments/seed_<N>/`"
    )
    linhas.append("")
    linhas.append(
        "Observação: os arquivos em `data/datasets_gerados/` e `results/training/` "
        f"correspondem à última execução (seed {seeds[-1]}), pois o pipeline os sobrescreve."
    )
    linhas.append("")

    return "\n".join(linhas)


def executar_experimentos(
    seeds: list[int],
    n_objetos: int = 25,
    epochs: int = 200,
) -> None:
    """
    Executa o experimento completo com múltiplas seeds e consolida os resultados.

    Saídas:
    - results/experiments/experimentos_multi_seed.csv
    - results/experiments/experimentos_multi_seed.md
    - results/experiments/seed_<N>/ com artefatos por execução
    """

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    execucoes: list[dict[str, Any]] = []

    for seed in seeds:
        execucao = executar_uma_execucao(
            seed=seed,
            n_objetos=n_objetos,
            epochs=epochs,
        )

        preservar_artefatos_da_execucao(execucao)
        execucoes.append(execucao)

    df_consolidado = montar_tabela_consolidada(execucoes)
    df_consolidado.to_csv(CSV_CONSOLIDADO, index=False, encoding="utf-8-sig")

    markdown = gerar_markdown_consolidado(
        execucoes=execucoes,
        epochs=epochs,
        n_objetos=n_objetos,
    )

    MD_CONSOLIDADO.write_text(markdown, encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXPERIMENTOS CONCLUÍDOS")
    print("=" * 80)

    print(f"\nSeeds executadas: {seeds}")
    print(f"Tabela CSV consolidada: {CSV_CONSOLIDADO}")
    print(f"Relatório Markdown: {MD_CONSOLIDADO}")

    print("\nResumo (satAgg e fórmulas por seed):")

    for execucao in execucoes:
        print(
            f"- seed {execucao['seed']}: "
            f"satAgg={execucao['sat_agg']:.4f}, "
            f"lastOnTheLeft={execucao['formulas']['lastOnTheLeft']:.4f}, "
            f"lastOnTheRight={execucao['formulas']['lastOnTheRight']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Executa o pipeline completo (dados + treinamento LTN) com múltiplas "
            "seeds e consolida satAgg por axioma/fórmula e métricas clássicas "
            "em results/experiments/."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS_PADRAO,
        help=f"Seeds das execuções. Padrão: {SEEDS_PADRAO}.",
    )

    parser.add_argument(
        "--n-objetos",
        type=int,
        default=25,
        help="Quantidade de objetos sintéticos por cena. Padrão: 25.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Épocas de treinamento por execução. Padrão: 200.",
    )

    args = parser.parse_args()

    executar_experimentos(
        seeds=args.seeds,
        n_objetos=args.n_objetos,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
