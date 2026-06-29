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

### Predicados binários

| relacao   |   accuracy |   precision |   recall |       f1 |
|:----------|-----------:|------------:|---------:|---------:|
| leftOf    |      0.848 |    0.833333 | 0.847458 | 0.840336 |
| rightOf   |      0.856 |    0.90566  | 0.786885 | 0.842105 |
| below     |      0.816 |    0.847826 | 0.709091 | 0.772277 |
| above     |      0.832 |    0.87931  | 0.784615 | 0.829268 |
| closeTo   |      1     |    1        | 1        | 1        |
| canStack  |      0.784 |    0.510638 | 0.857143 | 0.64     |

### Predicados ternários

| relacao   |   accuracy |   precision |   recall |       f1 |
|:----------|-----------:|------------:|---------:|---------:|
| inBetween |   0.722101 |    0.692478 | 0.855776 | 0.765515 |

## 5. Diagnóstico específico

- F1 de `closeTo`: `1.0000`
  - Positivos em `closeTo`: `102`
  - Negativos em `closeTo`: `523`
  - Proporção de positivos em `closeTo`: `0.1632`

- F1 de `canStack`: `0.6400`
  - Positivos em `canStack`: `116`
  - Negativos em `canStack`: `509`
  - Proporção de positivos em `canStack`: `0.1856`

- F1 de `inBetween`: `0.7655`
  - Diagnóstico: por ser uma relação ternária, é esperado que seja mais difícil que leftOf/rightOf.

## 6. Axiomas com menor satisfatibilidade

| axioma                                  |   satisfatibilidade |
|:----------------------------------------|--------------------:|
| below_irreflexividade                   |            0.573941 |
| above_irreflexividade                   |            0.574883 |
| left_irreflexividade                    |            0.58603  |
| right_irreflexividade                   |            0.587452 |
| inbetween_implica_configuracao_espacial |            0.76166  |
| above_implica_below_inverso             |            0.797164 |
| below_implica_above_inverso             |            0.808869 |
| can_stack_implica_above                 |            0.814434 |
| left_implica_right_inverso              |            0.822371 |
| right_implica_left_inverso              |            0.826674 |

## 7. Arquivos gerados

### Tabelas

- `balanceamento_binario`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/balanceamento_relacoes_binarias.csv`
- `balanceamento_ternario`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/balanceamento_relacoes_ternarias.csv`
- `metricas_binarias`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/metricas_finais_binarias.csv`
- `metricas_ternarias`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/metricas_finais_ternarias.csv`
- `sat_individual`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/sat_individual_final.csv`

### Gráficos

- `loss_total`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/plots/loss_total.png`
- `losses_componentes`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/plots/losses_componentes.png`
- `satagg`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/plots/satagg.png`
- `f1_final`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/plots/f1_final_por_predicado.png`
- `balanceamento`: `/home/viniciussousa/Documentos/IA-EC034/TrabalhoFinal/Trabalho-Final-de-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/results/analysis/plots/proporcao_positivos_por_relacao.png`
