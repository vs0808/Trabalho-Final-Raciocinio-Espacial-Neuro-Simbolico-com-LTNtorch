from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "datasets_gerados"
TRAINING_DIR = PROJECT_ROOT / "results" / "training"

# Diretório oficial para métricas e diagnósticos.
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PLOTS_DIR = METRICS_DIR / "plots"

GROUND_TRUTH_PARES = DATA_DIR / "ground_truth_pares.csv"
GROUND_TRUTH_TRIPLAS = DATA_DIR / "ground_truth_triplas.csv"
HISTORICO_TREINAMENTO = TRAINING_DIR / "historico_treinamento.csv"

RELACOES_BINARIAS = [
    "leftOf",
    "rightOf",
    "below",
    "above",
    "closeTo",
    "canStack",
]

RELACOES_TERNARIAS = [
    "inBetween",
]


def garantir_pastas_saida() -> None:
    """
    Cria as pastas de métricas, caso elas ainda não existam.

    Estrutura gerada:

    results/
    └── metrics/
        ├── balanceamento_relacoes_binarias.csv
        ├── balanceamento_relacoes_ternarias.csv
        ├── metricas_finais_binarias.csv
        ├── metricas_finais_ternarias.csv
        ├── sat_individual_final.csv
        ├── resumo_diagnostico.md
        ├── metricas_classicas.md
        └── plots/
            ├── loss_total.png
            ├── losses_componentes.png
            ├── satagg.png
            ├── f1_final_por_predicado.png
            └── proporcao_positivos_por_relacao.png
    """

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def validar_arquivos_entrada() -> None:
    """
    Verifica se os arquivos necessários para a análise existem.

    Esta análise depende de três arquivos principais:

    1. ground_truth_pares.csv
       Contém a verdade-terreno das relações binárias.

    2. ground_truth_triplas.csv
       Contém a verdade-terreno das relações ternárias.

    3. historico_treinamento.csv
       Contém a evolução do treinamento e as métricas por época.
    """

    arquivos = [
        GROUND_TRUTH_PARES,
        GROUND_TRUTH_TRIPLAS,
        HISTORICO_TREINAMENTO,
    ]

    arquivos_faltantes = [arquivo for arquivo in arquivos if not arquivo.exists()]

    if arquivos_faltantes:
        mensagem = [
            "Arquivos necessários não encontrados.",
            "",
            "Antes de rodar esta análise, execute:",
            "python3 main.py --train --no-show",
            "",
            "Arquivos faltantes:",
        ]

        for arquivo in arquivos_faltantes:
            mensagem.append(f"- {arquivo}")

        raise FileNotFoundError("\n".join(mensagem))


def carregar_dados() -> dict[str, pd.DataFrame]:
    """
    Carrega os arquivos necessários para análise.

    Retorna:
    - df_pares
    - df_triplas
    - df_historico
    """

    validar_arquivos_entrada()

    df_pares = pd.read_csv(GROUND_TRUTH_PARES)
    df_triplas = pd.read_csv(GROUND_TRUTH_TRIPLAS)
    df_historico = pd.read_csv(HISTORICO_TREINAMENTO)

    return {
        "pares": df_pares,
        "triplas": df_triplas,
        "historico": df_historico,
    }


def parse_json_cell(valor: Any) -> dict[str, Any]:
    """
    Converte uma célula JSON do CSV em dicionário Python.

    Algumas colunas do historico_treinamento.csv são dicionários salvos como texto JSON.
    Exemplo:

    {
        "leftOf": {
            "accuracy": 0.856,
            "precision": 0.85,
            "recall": 0.83,
            "f1": 0.84
        }
    }
    """

    if isinstance(valor, dict):
        return valor

    if pd.isna(valor):
        return {}

    if not isinstance(valor, str):
        return {}

    valor = valor.strip()

    if not valor:
        return {}

    try:
        return json.loads(valor)
    except json.JSONDecodeError:
        return {}


def calcular_balanceamento_relacoes(
    df: pd.DataFrame,
    relacoes: list[str],
) -> pd.DataFrame:
    """
    Calcula a quantidade de positivos e negativos para cada relação.

    Para cada predicado, calcula:
    - total de exemplos
    - quantidade de positivos
    - quantidade de negativos
    - proporção de positivos
    - proporção de negativos

    Isso é essencial para identificar desbalanceamento de classes.
    """

    registros = []

    for relacao in relacoes:
        if relacao not in df.columns:
            continue

        total = len(df)
        positivos = int((df[relacao] == 1).sum())
        negativos = int((df[relacao] == 0).sum())

        proporcao_positivos = positivos / total if total > 0 else 0.0
        proporcao_negativos = negativos / total if total > 0 else 0.0

        registros.append(
            {
                "relacao": relacao,
                "total": total,
                "positivos": positivos,
                "negativos": negativos,
                "proporcao_positivos": proporcao_positivos,
                "proporcao_negativos": proporcao_negativos,
            }
        )

    return pd.DataFrame(registros)


def extrair_metricas_finais(
    df_historico: pd.DataFrame,
    coluna_metricas: str,
) -> pd.DataFrame:
    """
    Extrai as métricas finais de uma coluna JSON do histórico.

    Pode ser usada para:
    - metricas_binarias
    - metricas_ternarias

    Retorna uma tabela com:
    relacao | accuracy | precision | recall | f1

    Caso existam métricas adicionais, como best_threshold e best_f1,
    elas também são preservadas.
    """

    if df_historico.empty:
        return pd.DataFrame()

    ultima_linha = df_historico.iloc[-1]

    if coluna_metricas not in ultima_linha:
        return pd.DataFrame()

    metricas = parse_json_cell(ultima_linha[coluna_metricas])

    registros = []

    for relacao, valores in metricas.items():
        registro = {
            "relacao": relacao,
            "accuracy": valores.get("accuracy", 0.0),
            "precision": valores.get("precision", 0.0),
            "recall": valores.get("recall", 0.0),
            "f1": valores.get("f1", 0.0),
        }

        if "best_threshold" in valores:
            registro["best_threshold"] = valores.get("best_threshold", 0.5)

        if "best_accuracy" in valores:
            registro["best_accuracy"] = valores.get("best_accuracy", 0.0)

        if "best_precision" in valores:
            registro["best_precision"] = valores.get("best_precision", 0.0)

        if "best_recall" in valores:
            registro["best_recall"] = valores.get("best_recall", 0.0)

        if "best_f1" in valores:
            registro["best_f1"] = valores.get("best_f1", 0.0)

        registros.append(registro)

    return pd.DataFrame(registros)


def extrair_sat_individual_final(df_historico: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai a satisfatibilidade individual dos axiomas na última época.

    Retorna:
    axioma | satisfatibilidade
    """

    if df_historico.empty:
        return pd.DataFrame()

    ultima_linha = df_historico.iloc[-1]

    if "sat_individual" not in ultima_linha:
        return pd.DataFrame()

    sat_individual = parse_json_cell(ultima_linha["sat_individual"])

    registros = []

    for axioma, valor in sat_individual.items():
        registros.append(
            {
                "axioma": axioma,
                "satisfatibilidade": valor,
            }
        )

    df_sat = pd.DataFrame(registros)

    if not df_sat.empty:
        df_sat = df_sat.sort_values(
            by="satisfatibilidade",
            ascending=True,
        ).reset_index(drop=True)

    return df_sat


def salvar_tabelas(
    df_balanceamento_binario: pd.DataFrame,
    df_balanceamento_ternario: pd.DataFrame,
    df_metricas_binarias: pd.DataFrame,
    df_metricas_ternarias: pd.DataFrame,
    df_sat_individual: pd.DataFrame,
) -> dict[str, Path]:
    """
    Salva as principais tabelas da análise em CSV dentro de results/metrics/.
    """

    caminhos = {
        "balanceamento_binario": METRICS_DIR / "balanceamento_relacoes_binarias.csv",
        "balanceamento_ternario": METRICS_DIR / "balanceamento_relacoes_ternarias.csv",
        "metricas_binarias": METRICS_DIR / "metricas_finais_binarias.csv",
        "metricas_ternarias": METRICS_DIR / "metricas_finais_ternarias.csv",
        "sat_individual": METRICS_DIR / "sat_individual_final.csv",
    }

    df_balanceamento_binario.to_csv(
        caminhos["balanceamento_binario"],
        index=False,
        encoding="utf-8-sig",
    )

    df_balanceamento_ternario.to_csv(
        caminhos["balanceamento_ternario"],
        index=False,
        encoding="utf-8-sig",
    )

    df_metricas_binarias.to_csv(
        caminhos["metricas_binarias"],
        index=False,
        encoding="utf-8-sig",
    )

    df_metricas_ternarias.to_csv(
        caminhos["metricas_ternarias"],
        index=False,
        encoding="utf-8-sig",
    )

    df_sat_individual.to_csv(
        caminhos["sat_individual"],
        index=False,
        encoding="utf-8-sig",
    )

    return caminhos


def plotar_loss_total(df_historico: pd.DataFrame) -> Path:
    """
    Gera gráfico da loss total ao longo das épocas.
    """

    caminho = PLOTS_DIR / "loss_total.png"

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_historico["epoch"],
        df_historico["loss_total"],
        marker="o",
        linewidth=1,
    )
    plt.title("Evolução da Loss Total")
    plt.xlabel("Época")
    plt.ylabel("Loss Total")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()

    return caminho


def plotar_losses_componentes(df_historico: pd.DataFrame) -> Path:
    """
    Gera gráfico com as losses supervisionada e lógica.
    """

    caminho = PLOTS_DIR / "losses_componentes.png"

    plt.figure(figsize=(10, 6))

    if "loss_supervisionada" in df_historico.columns:
        plt.plot(
            df_historico["epoch"],
            df_historico["loss_supervisionada"],
            marker="o",
            linewidth=1,
            label="Loss supervisionada",
        )

    if "loss_axiomas" in df_historico.columns:
        plt.plot(
            df_historico["epoch"],
            df_historico["loss_axiomas"],
            marker="o",
            linewidth=1,
            label="Loss dos axiomas",
        )

    plt.title("Evolução das Componentes da Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()

    return caminho


def plotar_satagg(df_historico: pd.DataFrame) -> Path:
    """
    Gera gráfico do satAgg ao longo das épocas.
    """

    caminho = PLOTS_DIR / "satagg.png"

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_historico["epoch"],
        df_historico["sat_agg"],
        marker="o",
        linewidth=1,
    )
    plt.title("Evolução do satAgg")
    plt.xlabel("Época")
    plt.ylabel("satAgg")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()

    return caminho


def plotar_f1_final(
    df_metricas_binarias: pd.DataFrame,
    df_metricas_ternarias: pd.DataFrame,
) -> Path:
    """
    Gera gráfico comparando o F1 final de cada predicado.
    """

    caminho = PLOTS_DIR / "f1_final_por_predicado.png"

    df_metricas = pd.concat(
        [df_metricas_binarias, df_metricas_ternarias],
        ignore_index=True,
    )

    if df_metricas.empty:
        return caminho

    df_metricas = df_metricas.sort_values(by="f1", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df_metricas["relacao"], df_metricas["f1"])
    plt.title("F1 Final por Predicado")
    plt.xlabel("Predicado")
    plt.ylabel("F1 Score")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()

    return caminho


def plotar_balanceamento(
    df_balanceamento_binario: pd.DataFrame,
    df_balanceamento_ternario: pd.DataFrame,
) -> Path:
    """
    Gera gráfico da proporção de positivos por relação.
    """

    caminho = PLOTS_DIR / "proporcao_positivos_por_relacao.png"

    df_balanceamento = pd.concat(
        [df_balanceamento_binario, df_balanceamento_ternario],
        ignore_index=True,
    )

    if df_balanceamento.empty:
        return caminho

    df_balanceamento = df_balanceamento.sort_values(
        by="proporcao_positivos",
        ascending=False,
    )

    plt.figure(figsize=(10, 6))
    plt.bar(
        df_balanceamento["relacao"],
        df_balanceamento["proporcao_positivos"],
    )
    plt.title("Proporção de Casos Positivos por Relação")
    plt.xlabel("Relação")
    plt.ylabel("Proporção de positivos")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()

    return caminho


def gerar_graficos(
    df_historico: pd.DataFrame,
    df_balanceamento_binario: pd.DataFrame,
    df_balanceamento_ternario: pd.DataFrame,
    df_metricas_binarias: pd.DataFrame,
    df_metricas_ternarias: pd.DataFrame,
) -> dict[str, Path]:
    """
    Gera todos os gráficos da análise dentro de results/metrics/plots/.
    """

    caminhos = {
        "loss_total": plotar_loss_total(df_historico),
        "losses_componentes": plotar_losses_componentes(df_historico),
        "satagg": plotar_satagg(df_historico),
        "f1_final": plotar_f1_final(df_metricas_binarias, df_metricas_ternarias),
        "balanceamento": plotar_balanceamento(
            df_balanceamento_binario,
            df_balanceamento_ternario,
        ),
    }

    return caminhos


def gerar_arquivo_metricas_classicas() -> Path:
    """
    Gera um arquivo Markdown explicando as métricas clássicas usadas na avaliação.

    Arquivo gerado:
    - results/metrics/metricas_classicas.md

    Métricas documentadas:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    """

    caminho = METRICS_DIR / "metricas_classicas.md"

    linhas = []

    linhas.append("# Métricas clássicas de avaliação")
    linhas.append("")
    linhas.append(
        "Este arquivo descreve as métricas clássicas utilizadas para avaliar "
        "os predicados aprendidos ou definidos no projeto LTN."
    )
    linhas.append("")
    linhas.append(
        "As métricas foram calculadas comparando as predições do modelo com a verdade-terreno."
    )
    linhas.append("")
    linhas.append("## 1. Termos da matriz de confusão")
    linhas.append("")
    linhas.append("Antes de definir as métricas, usamos quatro quantidades principais:")
    linhas.append("")
    linhas.append("- **TP — True Positive:** casos positivos corretamente previstos como positivos.")
    linhas.append("- **TN — True Negative:** casos negativos corretamente previstos como negativos.")
    linhas.append("- **FP — False Positive:** casos negativos previstos incorretamente como positivos.")
    linhas.append("- **FN — False Negative:** casos positivos previstos incorretamente como negativos.")
    linhas.append("")
    linhas.append("## 2. Accuracy")
    linhas.append("")
    linhas.append(
        "A **Accuracy** representa a proporção de instâncias corretamente classificadas "
        "entre todas as instâncias avaliadas."
    )
    linhas.append("")
    linhas.append("Ela considera tanto os acertos positivos quanto os acertos negativos.")
    linhas.append("")
    linhas.append("```text")
    linhas.append("Accuracy = (TP + TN) / (TP + TN + FP + FN)")
    linhas.append("```")
    linhas.append("")
    linhas.append("Em notação matemática:")
    linhas.append("")
    linhas.append("```latex")
    linhas.append(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
    linhas.append("```")
    linhas.append("")
    linhas.append(
        "No projeto, essa métrica indica o percentual geral de acertos de cada predicado."
    )
    linhas.append("")
    linhas.append(
        "Entretanto, a Accuracy pode ser enganosa quando existe desbalanceamento de classes. "
        "Por exemplo, se uma relação possui muitos casos negativos e poucos positivos, "
        "um modelo pode obter alta Accuracy apenas prevendo quase tudo como negativo."
    )
    linhas.append("")
    linhas.append("## 3. Precision")
    linhas.append("")
    linhas.append(
        "A **Precision** mede, entre todos os casos que o modelo classificou como positivos, "
        "quantos realmente eram positivos."
    )
    linhas.append("")
    linhas.append("```text")
    linhas.append("Precision = TP / (TP + FP)")
    linhas.append("```")
    linhas.append("")
    linhas.append("Em notação matemática:")
    linhas.append("")
    linhas.append("```latex")
    linhas.append(r"\text{Precision} = \frac{TP}{TP + FP}")
    linhas.append("```")
    linhas.append("")
    linhas.append(
        "No projeto, uma Precision alta significa que, quando o modelo afirma que uma relação é verdadeira, "
        "ele costuma estar correto."
    )
    linhas.append("")
    linhas.append(
        "Por exemplo: se `canStack(x,y)` tem Precision alta, então os pares que o modelo identifica "
        "como empilháveis tendem a ser realmente empilháveis."
    )
    linhas.append("")
    linhas.append("## 4. Recall")
    linhas.append("")
    linhas.append(
        "O **Recall**, também chamado de sensibilidade, mede a proporção de casos positivos reais "
        "que foram corretamente encontrados pelo modelo."
    )
    linhas.append("")
    linhas.append("```text")
    linhas.append("Recall = TP / (TP + FN)")
    linhas.append("```")
    linhas.append("")
    linhas.append("Em notação matemática:")
    linhas.append("")
    linhas.append("```latex")
    linhas.append(r"\text{Recall} = \frac{TP}{TP + FN}")
    linhas.append("```")
    linhas.append("")
    linhas.append(
        "No projeto, um Recall alto significa que o modelo consegue encontrar a maioria dos exemplos "
        "verdadeiros de uma relação."
    )
    linhas.append("")
    linhas.append(
        "Por exemplo: se `inBetween(x,y,z)` tem Recall alto, então o modelo encontra boa parte das triplas "
        "em que um objeto realmente está entre dois outros."
    )
    linhas.append("")
    linhas.append("## 5. F1 Score")
    linhas.append("")
    linhas.append("O **F1 Score** é a média harmônica entre Precision e Recall.")
    linhas.append("")
    linhas.append(
        "Ele busca equilibrar a capacidade do modelo de evitar falsos positivos "
        "e, ao mesmo tempo, encontrar os positivos reais."
    )
    linhas.append("")
    linhas.append("```text")
    linhas.append("F1 Score = 2 * (Precision * Recall) / (Precision + Recall)")
    linhas.append("```")
    linhas.append("")
    linhas.append("Em notação matemática:")
    linhas.append("")
    linhas.append("```latex")
    linhas.append(
        r"\text{F1 Score} = 2 \cdot "
        r"\frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}"
    )
    linhas.append("```")
    linhas.append("")
    linhas.append(
        "No projeto, o F1 Score é especialmente importante para relações desbalanceadas, "
        "como `closeTo` e `canStack`."
    )
    linhas.append("")
    linhas.append(
        "Quando uma relação tem poucos exemplos positivos, a Accuracy pode parecer boa mesmo que o modelo "
        "não esteja aprendendo os positivos. Por isso, o F1 Score ajuda a avaliar melhor o desempenho real."
    )
    linhas.append("")
    linhas.append("## 6. Interpretação no projeto LTN")
    linhas.append("")
    linhas.append("As métricas foram usadas para avaliar os seguintes predicados binários:")
    linhas.append("")
    linhas.append("- `leftOf`")
    linhas.append("- `rightOf`")
    linhas.append("- `below`")
    linhas.append("- `above`")
    linhas.append("- `closeTo`")
    linhas.append("- `canStack`")
    linhas.append("")
    linhas.append("E o seguinte predicado ternário:")
    linhas.append("")
    linhas.append("- `inBetween`")
    linhas.append("")
    linhas.append("A leitura geral é:")
    linhas.append("")
    linhas.append("- **Accuracy** mostra o acerto geral.")
    linhas.append("- **Precision** mostra a confiabilidade das previsões positivas.")
    linhas.append("- **Recall** mostra a capacidade de encontrar positivos reais.")
    linhas.append("- **F1 Score** mostra o equilíbrio entre Precision e Recall.")
    linhas.append("")
    linhas.append(
        "No contexto deste projeto, o F1 Score é a métrica mais importante para analisar relações "
        "com desbalanceamento de classes."
    )
    linhas.append("")

    caminho.write_text("\n".join(linhas), encoding="utf-8")

    return caminho


def obter_metrica(
    df_metricas: pd.DataFrame,
    relacao: str,
    metrica: str,
) -> float | None:
    """
    Busca o valor de uma métrica específica para uma relação.
    """

    if df_metricas.empty:
        return None

    if metrica not in df_metricas.columns:
        return None

    filtro = df_metricas["relacao"] == relacao

    if not filtro.any():
        return None

    return float(df_metricas.loc[filtro, metrica].iloc[0])


def obter_balanceamento(
    df_balanceamento: pd.DataFrame,
    relacao: str,
) -> dict[str, Any]:
    """
    Busca o balanceamento de uma relação específica.
    """

    if df_balanceamento.empty:
        return {}

    filtro = df_balanceamento["relacao"] == relacao

    if not filtro.any():
        return {}

    linha = df_balanceamento.loc[filtro].iloc[0].to_dict()

    return linha


def dataframe_to_markdown_seguro(df: pd.DataFrame) -> str:
    """
    Converte DataFrame para Markdown.

    Caso o pacote tabulate não esteja instalado, usa uma representação textual simples.
    """

    if df.empty:
        return "Tabela vazia."

    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def gerar_diagnostico_textual(
    df_historico: pd.DataFrame,
    df_balanceamento_binario: pd.DataFrame,
    df_balanceamento_ternario: pd.DataFrame,
    df_metricas_binarias: pd.DataFrame,
    df_metricas_ternarias: pd.DataFrame,
    df_sat_individual: pd.DataFrame,
    caminhos_tabelas: dict[str, Path],
    caminhos_graficos: dict[str, Path],
    caminho_metricas_classicas: Path,
) -> Path:
    """
    Gera um arquivo Markdown com o resumo diagnóstico da análise.

    Esse arquivo é útil para:
    - relatório final;
    - interpretação dos resultados;
    - decisão do próximo passo.
    """

    caminho = METRICS_DIR / "resumo_diagnostico.md"

    primeira_epoca = df_historico.iloc[0]
    ultima_epoca = df_historico.iloc[-1]

    loss_inicial = float(primeira_epoca["loss_total"])
    loss_final = float(ultima_epoca["loss_total"])

    sat_inicial = float(primeira_epoca["sat_agg"])
    sat_final = float(ultima_epoca["sat_agg"])

    close_f1 = obter_metrica(df_metricas_binarias, "closeTo", "f1")
    canstack_f1 = obter_metrica(df_metricas_binarias, "canStack", "f1")
    inbetween_f1 = obter_metrica(df_metricas_ternarias, "inBetween", "f1")

    close_balanceamento = obter_balanceamento(df_balanceamento_binario, "closeTo")
    canstack_balanceamento = obter_balanceamento(df_balanceamento_binario, "canStack")

    linhas = []

    linhas.append("# Resumo diagnóstico do treinamento LTN")
    linhas.append("")
    linhas.append("## 1. Visão geral")
    linhas.append("")
    linhas.append(
        "Este arquivo resume a análise do treinamento realizado com os predicados LTN treináveis."
    )
    linhas.append(
        "O objetivo é verificar se o modelo está aprendendo as relações espaciais e se há problemas de desbalanceamento."
    )
    linhas.append("")

    linhas.append("## 2. Evolução geral do treinamento")
    linhas.append("")
    linhas.append(f"- Loss total inicial: `{loss_inicial:.4f}`")
    linhas.append(f"- Loss total final: `{loss_final:.4f}`")
    linhas.append(f"- satAgg inicial: `{sat_inicial:.4f}`")
    linhas.append(f"- satAgg final: `{sat_final:.4f}`")
    linhas.append("")

    if loss_final < loss_inicial:
        linhas.append(
            "A loss total diminuiu ao longo do treinamento, indicando que o modelo reduziu o erro global."
        )
    else:
        linhas.append(
            "A loss total não diminuiu. Isso indica que o treinamento pode precisar de ajuste."
        )

    if sat_final > sat_inicial:
        linhas.append(
            "O satAgg aumentou, indicando que a base de conhecimento ficou mais satisfeita ao final do treinamento."
        )
    else:
        linhas.append(
            "O satAgg não aumentou. Isso indica que os axiomas podem não estar sendo bem incorporados."
        )

    linhas.append("")
    linhas.append("## 3. Balanceamento das relações")
    linhas.append("")
    linhas.append("### Relações binárias")
    linhas.append("")
    linhas.append(dataframe_to_markdown_seguro(df_balanceamento_binario))
    linhas.append("")
    linhas.append("### Relações ternárias")
    linhas.append("")
    linhas.append(dataframe_to_markdown_seguro(df_balanceamento_ternario))
    linhas.append("")

    linhas.append("## 4. Métricas finais")
    linhas.append("")
    linhas.append(
        f"As fórmulas das métricas clássicas utilizadas nesta avaliação estão documentadas em: `{caminho_metricas_classicas}`."
    )
    linhas.append("")
    linhas.append("### Predicados binários")
    linhas.append("")
    linhas.append(dataframe_to_markdown_seguro(df_metricas_binarias))
    linhas.append("")
    linhas.append("### Predicados ternários")
    linhas.append("")
    linhas.append(dataframe_to_markdown_seguro(df_metricas_ternarias))
    linhas.append("")

    linhas.append("## 5. Diagnóstico específico")
    linhas.append("")

    if close_f1 is not None:
        linhas.append(f"- F1 de `closeTo`: `{close_f1:.4f}`")

        if close_balanceamento:
            linhas.append(
                f"  - Positivos em `closeTo`: `{int(close_balanceamento['positivos'])}`"
            )
            linhas.append(
                f"  - Negativos em `closeTo`: `{int(close_balanceamento['negativos'])}`"
            )
            linhas.append(
                f"  - Proporção de positivos em `closeTo`: `{float(close_balanceamento['proporcao_positivos']):.4f}`"
            )

        if close_f1 == 0:
            linhas.append(
                "  - Diagnóstico: o modelo não está detectando corretamente os casos positivos de proximidade."
            )
            linhas.append(
                "  - Hipótese provável: desbalanceamento de classes ou threshold inadequado."
            )
        elif close_f1 >= 0.95:
            linhas.append(
                "  - Diagnóstico: o predicado `closeTo` está com desempenho excelente."
            )
            linhas.append(
                "  - Interpretação: isso é esperado quando `closeTo` é tratado como predicado fuzzy determinístico baseado em distância."
            )

    linhas.append("")

    if canstack_f1 is not None:
        linhas.append(f"- F1 de `canStack`: `{canstack_f1:.4f}`")

        if canstack_balanceamento:
            linhas.append(
                f"  - Positivos em `canStack`: `{int(canstack_balanceamento['positivos'])}`"
            )
            linhas.append(
                f"  - Negativos em `canStack`: `{int(canstack_balanceamento['negativos'])}`"
            )
            linhas.append(
                f"  - Proporção de positivos em `canStack`: `{float(canstack_balanceamento['proporcao_positivos']):.4f}`"
            )

        if canstack_f1 == 0:
            linhas.append(
                "  - Diagnóstico: o modelo não está detectando corretamente os casos positivos de empilhamento."
            )
            linhas.append(
                "  - Hipótese provável: relação rara e composta por muitas condições simultâneas."
            )
        elif canstack_f1 >= 0.5:
            linhas.append(
                "  - Diagnóstico: `canStack` apresentou desempenho intermediário ou bom para uma relação composta."
            )
            linhas.append(
                "  - Interpretação: como `canStack` depende de múltiplas condições espaciais e semânticas, é esperado que seja mais difícil que relações simples."
            )

    linhas.append("")

    if inbetween_f1 is not None:
        linhas.append(f"- F1 de `inBetween`: `{inbetween_f1:.4f}`")
        linhas.append(
            "  - Diagnóstico: por ser uma relação ternária, é esperado que seja mais difícil que leftOf/rightOf."
        )

    linhas.append("")
    linhas.append("## 6. Axiomas com menor satisfatibilidade")
    linhas.append("")

    if not df_sat_individual.empty:
        linhas.append(dataframe_to_markdown_seguro(df_sat_individual.head(10)))
    else:
        linhas.append("Nenhuma informação de satisfatibilidade individual foi encontrada.")

    linhas.append("")
    linhas.append("## 7. Arquivos gerados")
    linhas.append("")
    linhas.append("### Tabelas")
    linhas.append("")

    for nome, caminho_tabela in caminhos_tabelas.items():
        linhas.append(f"- `{nome}`: `{caminho_tabela}`")

    linhas.append("")
    linhas.append("### Gráficos")
    linhas.append("")

    for nome, caminho_grafico in caminhos_graficos.items():
        linhas.append(f"- `{nome}`: `{caminho_grafico}`")

    linhas.append("")
    linhas.append("### Fórmulas das métricas")
    linhas.append("")
    linhas.append(f"- `metricas_classicas`: `{caminho_metricas_classicas}`")

    linhas.append("")
    linhas.append("## 8. Próximo passo recomendado")
    linhas.append("")
    linhas.append(
        "Com as métricas consolidadas em `results/metrics/`, o próximo módulo natural é `src/query_ltn.py`."
    )
    linhas.append("")
    linhas.append(
        "Esse módulo deve carregar o modelo treinado, executar consultas compostas e gerar explicações textuais para cada resposta."
    )
    linhas.append("")
    linhas.append("As consultas principais devem incluir:")
    linhas.append("")
    linhas.append("1. objeto pequeno abaixo de cilindro e à esquerda de quadrado;")
    linhas.append("2. cone verde entre dois objetos;")
    linhas.append("3. regra universal sobre triângulos próximos terem o mesmo tamanho;")
    linhas.append("4. existência de objeto que possa ser empilhado sobre outro.")
    linhas.append("")
    linhas.append(
        "Os resultados dessas consultas devem ser salvos em `results/queries/`."
    )

    caminho.write_text("\n".join(linhas), encoding="utf-8")

    return caminho


def imprimir_resumo_console(
    df_historico: pd.DataFrame,
    df_balanceamento_binario: pd.DataFrame,
    df_balanceamento_ternario: pd.DataFrame,
    df_metricas_binarias: pd.DataFrame,
    df_metricas_ternarias: pd.DataFrame,
    df_sat_individual: pd.DataFrame,
    caminhos_tabelas: dict[str, Path],
    caminhos_graficos: dict[str, Path],
    caminho_diagnostico: Path,
    caminho_metricas_classicas: Path,
) -> None:
    """
    Exibe no terminal um resumo amigável da análise.
    """

    primeira_epoca = df_historico.iloc[0]
    ultima_epoca = df_historico.iloc[-1]

    print("=" * 80)
    print("ANÁLISE DAS MÉTRICAS DO TREINAMENTO LTN")
    print("=" * 80)

    print("\nEvolução geral:")
    print(f"- Loss total inicial: {float(primeira_epoca['loss_total']):.4f}")
    print(f"- Loss total final:   {float(ultima_epoca['loss_total']):.4f}")
    print(f"- satAgg inicial:     {float(primeira_epoca['sat_agg']):.4f}")
    print(f"- satAgg final:       {float(ultima_epoca['sat_agg']):.4f}")

    print("\nBalanceamento das relações binárias:")
    print(df_balanceamento_binario)

    print("\nBalanceamento das relações ternárias:")
    print(df_balanceamento_ternario)

    print("\nMétricas finais binárias:")
    print(df_metricas_binarias)

    print("\nMétricas finais ternárias:")
    print(df_metricas_ternarias)

    print("\nAxiomas com menor satisfatibilidade:")
    if not df_sat_individual.empty:
        print(df_sat_individual.head(10))
    else:
        print("Nenhum dado encontrado.")

    print("\nArquivos de métricas gerados:")

    print("\nTabelas:")
    for nome, caminho in caminhos_tabelas.items():
        print(f"- {nome}: {caminho}")

    print("\nGráficos:")
    for nome, caminho in caminhos_graficos.items():
        print(f"- {nome}: {caminho}")

    print(f"\nResumo diagnóstico: {caminho_diagnostico}")
    print(f"Arquivo com fórmulas das métricas clássicas: {caminho_metricas_classicas}")
    print("\nAnálise concluída com sucesso.")


def analisar_treinamento() -> None:
    """
    Executa a análise completa dos resultados de treinamento.

    Fluxo:
    1. Carrega ground truth e histórico.
    2. Calcula balanceamento das classes.
    3. Extrai métricas finais.
    4. Extrai satisfatibilidade individual dos axiomas.
    5. Salva tabelas em results/metrics/.
    6. Gera gráficos em results/metrics/plots/.
    7. Gera arquivo com fórmulas das métricas clássicas.
    8. Gera resumo diagnóstico em results/metrics/.
    9. Exibe resumo no terminal.
    """

    garantir_pastas_saida()

    dados = carregar_dados()

    df_pares = dados["pares"]
    df_triplas = dados["triplas"]
    df_historico = dados["historico"]

    df_balanceamento_binario = calcular_balanceamento_relacoes(
        df=df_pares,
        relacoes=RELACOES_BINARIAS,
    )

    df_balanceamento_ternario = calcular_balanceamento_relacoes(
        df=df_triplas,
        relacoes=RELACOES_TERNARIAS,
    )

    df_metricas_binarias = extrair_metricas_finais(
        df_historico=df_historico,
        coluna_metricas="metricas_binarias",
    )

    df_metricas_ternarias = extrair_metricas_finais(
        df_historico=df_historico,
        coluna_metricas="metricas_ternarias",
    )

    df_sat_individual = extrair_sat_individual_final(
        df_historico=df_historico,
    )

    caminhos_tabelas = salvar_tabelas(
        df_balanceamento_binario=df_balanceamento_binario,
        df_balanceamento_ternario=df_balanceamento_ternario,
        df_metricas_binarias=df_metricas_binarias,
        df_metricas_ternarias=df_metricas_ternarias,
        df_sat_individual=df_sat_individual,
    )

    caminhos_graficos = gerar_graficos(
        df_historico=df_historico,
        df_balanceamento_binario=df_balanceamento_binario,
        df_balanceamento_ternario=df_balanceamento_ternario,
        df_metricas_binarias=df_metricas_binarias,
        df_metricas_ternarias=df_metricas_ternarias,
    )

    caminho_metricas_classicas = gerar_arquivo_metricas_classicas()

    caminho_diagnostico = gerar_diagnostico_textual(
        df_historico=df_historico,
        df_balanceamento_binario=df_balanceamento_binario,
        df_balanceamento_ternario=df_balanceamento_ternario,
        df_metricas_binarias=df_metricas_binarias,
        df_metricas_ternarias=df_metricas_ternarias,
        df_sat_individual=df_sat_individual,
        caminhos_tabelas=caminhos_tabelas,
        caminhos_graficos=caminhos_graficos,
        caminho_metricas_classicas=caminho_metricas_classicas,
    )

    imprimir_resumo_console(
        df_historico=df_historico,
        df_balanceamento_binario=df_balanceamento_binario,
        df_balanceamento_ternario=df_balanceamento_ternario,
        df_metricas_binarias=df_metricas_binarias,
        df_metricas_ternarias=df_metricas_ternarias,
        df_sat_individual=df_sat_individual,
        caminhos_tabelas=caminhos_tabelas,
        caminhos_graficos=caminhos_graficos,
        caminho_diagnostico=caminho_diagnostico,
        caminho_metricas_classicas=caminho_metricas_classicas,
    )


if __name__ == "__main__":
    analisar_treinamento()