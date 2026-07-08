# Relatório Final — Raciocínio Espacial Neuro-Simbólico com LTNtorch

## 1. Objetivo

Este relatório consolida os resultados do sistema neuro-simbólico desenvolvido para raciocínio espacial sobre uma cena sintética 2D, combinando aprendizado supervisionado (redes neurais) com restrições lógicas diferenciáveis (Logic Tensor Networks, via LTNtorch).

O objetivo é responder perguntas espaciais sobre objetos sintéticos — posição, cor, forma, tamanho e relações entre eles (`leftOf`, `rightOf`, `below`, `above`, `closeTo`, `canStack`, `inBetween`) — aprendendo predicados a partir de exemplos e, simultaneamente, respeitando axiomas lógicos que descrevem a coerência esperada dessas relações.

## 2. Metodologia

### 2.1. Dados sintéticos

Foram gerados 25 objetos aleatórios (seed=42) em um plano 2D normalizado `[0, 1] x [0, 1]`, cada um com:

- posição `(x, y)`;
- cor: vermelho, verde ou azul;
- forma: círculo, quadrado, cilindro, cone ou triângulo;
- tamanho: pequeno ou grande.

Cada objeto é representado por um vetor de 11 posições (coordenadas + one-hot de cor + one-hot de forma + tamanho binário), conforme `src/data_generator.py`.

### 2.2. Verdade-terreno

A partir das posições e atributos, `src/relations_ground_truth.py` calcula deterministicamente:

- **10 predicados unários** (isRed, isGreen, isBlue, isCircle, isSquare, isCylinder, isCone, isTriangle, isSmall, isBig);
- **9 predicados binários** avaliados sobre os 625 pares (25×25) de objetos, incluindo autopares: `leftOf`, `rightOf`, `below`, `above`, `closeTo`, `sameColor`, `sameShape`, `sameSize`, `canStack`;
- **1 predicado ternário**, `inBetween`, avaliado sobre as 13.800 triplas (25×24×23) de objetos distintos.

Os limiares geométricos usados são: `closeTo` (distância < 0.25), `canStack` (alinhamento horizontal < 0.12) e `inBetween` (tolerância de segmento de reta de 0.05).

### 2.3. Predicados LTN

Os predicados são de três tipos (`src/ltn_predicates.py`):

1. **Predicados de atributo**, calculados diretamente do vetor do objeto (isRed, isTriangle, isSmall, sameSize, etc.);
2. **Predicado fuzzy determinístico**: `closeTo`, calculado por uma função geométrica diferenciável da distância euclidiana — decisão de design que se mostrou acertada (ver seção 3.2);
3. **Predicados treináveis**: `leftOf`, `rightOf`, `below`, `above`, `canStack` (redes binárias) e `inBetween` (rede ternária), aprendidos a partir da verdade-terreno e regularizados por axiomas lógicos.

### 2.4. Axiomas e treinamento

A base de conhecimento lógica (`src/ltn_axioms.py`) inclui axiomas de taxonomia (cobertura/exclusividade de cor, forma e tamanho), de relações horizontais e verticais (irreflexividade, assimetria, inversão, transitividade), de proximidade (`closeTo` irreflexivo e simétrico) e de composição (`canStack(x,y) → above(x,y)`, restrições sobre a base de empilhamento, e uma regra ligando `inBetween` à configuração espacial).

O treinamento (`src/train_ltn.py`) combina:

```text
loss_total = loss_supervisionada + peso_axiomas * (1 - satAgg)
```

com `peso_axiomas = 0.30`, 200 épocas, `lr = 0.001`, 80% dos pares/triplas para treino e 20% para teste, e loss ponderada para lidar com o desbalanceamento de classes de `canStack`.

## 3. Resultados

### 3.1. Evolução do treinamento

| Métrica | Inicial (época 1) | Final (época 200) |
|---|---|---|
| Loss total | 1.5604 | 1.0372 |
| Loss supervisionada | 1.5006 | 0.9885 |
| Loss dos axiomas | 0.1993 | 0.1622 |
| satAgg | 0.8007 | 0.8378 |

A loss total caiu de forma consistente ao longo das 200 épocas, e o `satAgg` aumentou de 0.8007 para 0.8378, indicando que o modelo aprendeu as relações ao mesmo tempo em que satisfez progressivamente mais os axiomas lógicos (gráficos em `results/metrics/plots/loss_total.png`, `losses_componentes.png` e `satagg.png`).

### 3.2. Métricas finais por predicado

| Predicado | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| `leftOf` | 0.848 | 0.833 | 0.847 | **0.8403** |
| `rightOf` | 0.856 | 0.906 | 0.787 | **0.8421** |
| `below` | 0.816 | 0.848 | 0.709 | **0.7723** |
| `above` | 0.832 | 0.879 | 0.785 | **0.8293** |
| `closeTo` | 1.000 | 1.000 | 1.000 | **1.0000** |
| `canStack` | 0.784 | 0.511 | 0.857 | **0.6400** |
| `inBetween` (ternário) | 0.722 | 0.692 | 0.856 | **0.7655** |

Interpretação:

- As relações espaciais binárias simples (`leftOf`, `rightOf`, `below`, `above`) atingiram F1 entre 0.77 e 0.84, um desempenho consistente e equilibrado entre precisão e recall.
- `closeTo`, tratado como predicado fuzzy determinístico baseado em distância euclidiana em vez de rede treinável, atingiu F1 = 1.0 — confirmando que relações geométricas diretas são melhor representadas por funções diferenciáveis conhecidas do que aprendidas do zero.
- `canStack` teve o desempenho mais fraco (F1 = 0.64), esperado por ser uma relação composta (depende de forma, tamanho e alinhamento) e fortemente desbalanceada (apenas 18.6% de positivos no treino).
- `inBetween`, por ser ternário e envolver uma combinatória muito maior (13.800 triplas), teve F1 = 0.7655, um resultado aceitável dada a complexidade adicional.

### 3.3. Balanceamento das classes

`leftOf`, `rightOf`, `below` e `above` estão bem balanceados (~48% de positivos). Já `closeTo` (16.3%) e `canStack` (18.6%) são fortemente desbalanceados, o que motivou o uso de loss ponderada (`pos_weight`) durante o treinamento — ver `results/metrics/balanceamento_relacoes_binarias.csv`.

### 3.4. Satisfatibilidade individual dos axiomas

Os axiomas com menor satisfatibilidade ao final do treinamento foram os de irreflexividade (`below_irreflexividade` = 0.574, `above_irreflexividade` = 0.575, `left_irreflexividade` = 0.586, `right_irreflexividade` = 0.587), seguidos pela regra que liga `inBetween` à configuração espacial (0.762). Isso sugere que, embora o modelo aprenda bem os casos positivos, ainda atribui algum grau de verdade residual a relações que deveriam ser estritamente falsas para um objeto em relação a si mesmo — um ponto de possível refinamento futuro (ver `results/metrics/sat_individual_final.csv`).

## 4. Consultas compostas

O módulo `src/query_ltn.py` avaliou quatro consultas lógicas sobre o modelo treinado:

1. **"Existe algum objeto pequeno abaixo de um cilindro e à esquerda de um quadrado?"** → **Sim** (valor = 0.7274). Evidência: objeto 21 (triângulo pequeno) está abaixo do objeto 3 (cilindro grande, grau 0.837) e à esquerda do objeto 15 (quadrado grande, grau 0.869).
2. **"Existe um cone verde entre dois objetos?"** → **Sim** (valor = 0.8392). Evidência: objeto 22 (cone verde) está entre os objetos 24 e 6.
3. **"Se dois triângulos estão próximos, eles têm o mesmo tamanho?"** (regra universal) → **Sim** (valor = 0.9266), mas com um contraexemplo relevante: os objetos 0 (triângulo pequeno) e 17 (triângulo grande) estão próximos (`closeTo` = 0.942) e têm tamanhos diferentes (`sameSize` = 0.0), o que reduz o grau de satisfação da implicação para 0.058 nesse caso específico — mostrando que a regra é majoritariamente, mas não universalmente, satisfeita.
4. **"Existe algum objeto que pode ser empilhado sobre outro?"** → **Sim** (valor = 0.9499). Evidência: objeto 6 (triângulo grande) pode ser empilhado sobre o objeto 14 (círculo grande).

Os resultados completos, incluindo os top-10 pares/triplas por relação, estão em `results/queries/explicacoes_consultas.md`, `respostas_consultas.json` e `respostas_consultas.csv`.

## 5. Conclusão

O experimento confirma a proposta central do projeto: a combinação de aprendizado neural com axiomas lógicos diferenciáveis (LTNtorch) permite que o modelo aprenda relações espaciais a partir de exemplos ao mesmo tempo em que respeita, de forma crescente, uma base de conhecimento lógica explícita. Os resultados numéricos obtidos nesta execução reproduzem os valores documentados no `README.md` do projeto, confirmando a reprodutibilidade do pipeline (seed=42).

A decisão de tratar `closeTo` como predicado fuzzy determinístico, em vez de treinável, foi a que mais impactou positivamente o desempenho geral (F1 = 1.0), reforçando a lição de que relações geométricas diretas não precisam ser aprendidas quando podem ser expressas analiticamente. Por outro lado, `canStack` e `inBetween` — relações compostas ou de maior aridade — seguem como os pontos de maior dificuldade, sugerindo caminhos futuros: mais dados de treino, engenharia de features específica para `canStack`, ou uma regra lógica adicional que penalize explicitamente a autorreferência nas relações irreflexivas.

## 6. Como reproduzir

```bash
python main.py --train --no-show
python src/analyze_training_results.py
python src/query_ltn.py
```

Todos os artefatos citados neste relatório (tabelas, gráficos, modelo treinado, respostas às consultas) estão em `results/` e `data/datasets_gerados/`.
