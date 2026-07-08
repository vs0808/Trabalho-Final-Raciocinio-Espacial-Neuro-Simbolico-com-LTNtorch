# Resumo diagnóstico do treinamento LTN

## 1. Visão geral

Este arquivo resume a análise do treinamento realizado com os predicados LTN treináveis.
O objetivo é verificar se o modelo está aprendendo as relações espaciais e se há problemas de desbalanceamento.

## 2. Evolução geral do treinamento

- Loss total inicial: `1.5604`
- Loss total final: `1.0372`
- satAgg inicial: `0.8007`
- satAgg final: `0.8378`

A loss total diminuiu ao longo do treinamento, indicando que o modelo reduziu o erro global.
O satAgg aumentou, indicando que a base de conhecimento ficou mais satisfeita ao final do treinamento.

## 3. Balanceamento das relações

### Relações binárias

| relacao   |   total |   positivos |   negativos |   proporcao_positivos |   proporcao_negativos |
|:----------|--------:|------------:|------------:|----------------------:|----------------------:|
| leftOf    |     625 |         300 |         325 |                0.48   |                0.52   |
| rightOf   |     625 |         300 |         325 |                0.48   |                0.52   |
| below     |     625 |         300 |         325 |                0.48   |                0.52   |
| above     |     625 |         300 |         325 |                0.48   |                0.52   |
| closeTo   |     625 |         102 |         523 |                0.1632 |                0.8368 |
| canStack  |     625 |         116 |         509 |                0.1856 |                0.8144 |

### Relações ternárias

| relacao   |   total |   positivos |   negativos |   proporcao_positivos |   proporcao_negativos |
|:----------|--------:|------------:|------------:|----------------------:|----------------------:|
| inBetween |   13800 |        7556 |        6244 |              0.547536 |              0.452464 |

## 4. Métricas finais

As fórmulas das métricas clássicas utilizadas nesta avaliação estão documentadas em: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/metricas_classicas.md`.

### Predicados binários

| relacao   |   accuracy |   precision |   recall |       f1 |   best_threshold |   best_accuracy |   best_precision |   best_recall |   best_f1 |
|:----------|-----------:|------------:|---------:|---------:|-----------------:|----------------:|-----------------:|--------------:|----------:|
| leftOf    |      0.848 |    0.833333 | 0.847458 | 0.840336 |             0.4  |           0.84  |         0.774648 |      0.932203 |  0.846154 |
| rightOf   |      0.856 |    0.90566  | 0.786885 | 0.842105 |             0.45 |           0.856 |         0.890909 |      0.803279 |  0.844828 |
| below     |      0.816 |    0.847826 | 0.709091 | 0.772277 |             0.4  |           0.792 |         0.716418 |      0.872727 |  0.786885 |
| above     |      0.832 |    0.87931  | 0.784615 | 0.829268 |             0.45 |           0.848 |         0.838235 |      0.876923 |  0.857143 |
| closeTo   |      1     |    1        | 1        | 1        |             0.45 |           1     |         1        |      1        |  1        |
| canStack  |      0.784 |    0.510638 | 0.857143 | 0.64     |             0.5  |           0.784 |         0.510638 |      0.857143 |  0.64     |

### Predicados ternários

| relacao   |   accuracy |   precision |   recall |       f1 |   best_threshold |   best_accuracy |   best_precision |   best_recall |   best_f1 |
|:----------|-----------:|------------:|---------:|---------:|-----------------:|----------------:|-----------------:|--------------:|----------:|
| inBetween |   0.721739 |    0.692308 | 0.855092 | 0.765138 |              0.5 |        0.721739 |         0.692308 |      0.855092 |  0.765138 |

## 5. Diagnóstico específico

- F1 de `closeTo`: `1.0000`
  - Positivos em `closeTo`: `102`
  - Negativos em `closeTo`: `523`
  - Proporção de positivos em `closeTo`: `0.1632`
  - Diagnóstico: o predicado `closeTo` está com desempenho excelente.
  - Interpretação: isso é esperado quando `closeTo` é tratado como predicado fuzzy determinístico baseado em distância.

- F1 de `canStack`: `0.6400`
  - Positivos em `canStack`: `116`
  - Negativos em `canStack`: `509`
  - Proporção de positivos em `canStack`: `0.1856`
  - Diagnóstico: `canStack` apresentou desempenho intermediário ou bom para uma relação composta.
  - Interpretação: como `canStack` depende de múltiplas condições espaciais e semânticas, é esperado que seja mais difícil que relações simples.

- F1 de `inBetween`: `0.7651`
  - Diagnóstico: por ser uma relação ternária, é esperado que seja mais difícil que leftOf/rightOf.

## 6. Axiomas com menor satisfatibilidade

| axioma                                  |   satisfatibilidade |
|:----------------------------------------|--------------------:|
| below_irreflexividade                   |            0.573922 |
| above_irreflexividade                   |            0.574821 |
| left_irreflexividade                    |            0.58603  |
| right_irreflexividade                   |            0.587456 |
| inbetween_implica_configuracao_espacial |            0.761649 |
| above_implica_below_inverso             |            0.797148 |
| below_implica_above_inverso             |            0.808886 |
| can_stack_implica_above                 |            0.814449 |
| left_implica_right_inverso              |            0.82237  |
| right_implica_left_inverso              |            0.826675 |

## 7. Arquivos gerados

### Tabelas

- `balanceamento_binario`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/balanceamento_relacoes_binarias.csv`
- `balanceamento_ternario`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/balanceamento_relacoes_ternarias.csv`
- `metricas_binarias`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/metricas_finais_binarias.csv`
- `metricas_ternarias`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/metricas_finais_ternarias.csv`
- `sat_individual`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/sat_individual_final.csv`

### Gráficos

- `loss_total`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/plots/loss_total.png`
- `losses_componentes`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/plots/losses_componentes.png`
- `satagg`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/plots/satagg.png`
- `f1_final`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/plots/f1_final_por_predicado.png`
- `balanceamento`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/plots/proporcao_positivos_por_relacao.png`

### Fórmulas das métricas

- `metricas_classicas`: `/Users/thalesaraujods/Developer/Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/.claude/worktrees/bold-neumann-10a099/results/metrics/metricas_classicas.md`

## 8. Próximo passo recomendado

Com as métricas consolidadas em `results/metrics/`, o próximo módulo natural é `src/query_ltn.py`.

Esse módulo deve carregar o modelo treinado, executar consultas compostas e gerar explicações textuais para cada resposta.

As consultas principais devem incluir:

1. objeto pequeno abaixo de cilindro e à esquerda de quadrado;
2. cone verde entre dois objetos;
3. regra universal sobre triângulos próximos terem o mesmo tamanho;
4. existência de objeto que possa ser empilhado sobre outro.

Os resultados dessas consultas devem ser salvos em `results/queries/`.