# Métricas clássicas de avaliação

Este arquivo descreve as métricas clássicas utilizadas para avaliar os predicados aprendidos ou definidos no projeto LTN.

As métricas foram calculadas comparando as predições do modelo com a verdade-terreno.

## 1. Termos da matriz de confusão

Antes de definir as métricas, usamos quatro quantidades principais:

- **TP — True Positive:** casos positivos corretamente previstos como positivos.
- **TN — True Negative:** casos negativos corretamente previstos como negativos.
- **FP — False Positive:** casos negativos previstos incorretamente como positivos.
- **FN — False Negative:** casos positivos previstos incorretamente como negativos.

## 2. Accuracy

A **Accuracy** representa a proporção de instâncias corretamente classificadas entre todas as instâncias avaliadas.

Ela considera tanto os acertos positivos quanto os acertos negativos.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Em notação matemática:

```latex
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
```

No projeto, essa métrica indica o percentual geral de acertos de cada predicado.

Entretanto, a Accuracy pode ser enganosa quando existe desbalanceamento de classes. Por exemplo, se uma relação possui muitos casos negativos e poucos positivos, um modelo pode obter alta Accuracy apenas prevendo quase tudo como negativo.

## 3. Precision

A **Precision** mede, entre todos os casos que o modelo classificou como positivos, quantos realmente eram positivos.

```text
Precision = TP / (TP + FP)
```

Em notação matemática:

```latex
\text{Precision} = \frac{TP}{TP + FP}
```

No projeto, uma Precision alta significa que, quando o modelo afirma que uma relação é verdadeira, ele costuma estar correto.

Por exemplo: se `canStack(x,y)` tem Precision alta, então os pares que o modelo identifica como empilháveis tendem a ser realmente empilháveis.

## 4. Recall

O **Recall**, também chamado de sensibilidade, mede a proporção de casos positivos reais que foram corretamente encontrados pelo modelo.

```text
Recall = TP / (TP + FN)
```

Em notação matemática:

```latex
\text{Recall} = \frac{TP}{TP + FN}
```

No projeto, um Recall alto significa que o modelo consegue encontrar a maioria dos exemplos verdadeiros de uma relação.

Por exemplo: se `inBetween(x,y,z)` tem Recall alto, então o modelo encontra boa parte das triplas em que um objeto realmente está entre dois outros.

## 5. F1 Score

O **F1 Score** é a média harmônica entre Precision e Recall.

Ele busca equilibrar a capacidade do modelo de evitar falsos positivos e, ao mesmo tempo, encontrar os positivos reais.

```text
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```

Em notação matemática:

```latex
\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

No projeto, o F1 Score é especialmente importante para relações desbalanceadas, como `closeTo` e `canStack`.

Quando uma relação tem poucos exemplos positivos, a Accuracy pode parecer boa mesmo que o modelo não esteja aprendendo os positivos. Por isso, o F1 Score ajuda a avaliar melhor o desempenho real.

## 6. Interpretação no projeto LTN

As métricas foram usadas para avaliar os seguintes predicados binários:

- `leftOf`
- `rightOf`
- `below`
- `above`
- `closeTo`
- `canStack`

E o seguinte predicado ternário:

- `inBetween`

A leitura geral é:

- **Accuracy** mostra o acerto geral.
- **Precision** mostra a confiabilidade das previsões positivas.
- **Recall** mostra a capacidade de encontrar positivos reais.
- **F1 Score** mostra o equilíbrio entre Precision e Recall.

No contexto deste projeto, o F1 Score é a métrica mais importante para analisar relações com desbalanceamento de classes.
