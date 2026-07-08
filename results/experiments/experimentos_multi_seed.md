# Experimentos com múltiplas seeds

Este relatório consolida 5 execuções independentes do pipeline (geração de dados + treinamento LTN), uma para cada seed.

- Seeds: [42, 43, 44, 45, 46]
- Objetos por cena: 25
- Épocas de treinamento: 200
- Valores de satisfatibilidade (sat) referem-se à última época de treinamento.
- As métricas clássicas (Accuracy, Precision, Recall, F1) são calculadas no conjunto de teste com threshold 0.5.

## 1. satAgg geral e fórmulas de consulta por execução

As fórmulas `lastOnTheLeft = ∃x(∀y leftOf(x,y))` e `lastOnTheRight = ∃x(∀y rightOf(x,y))` são avaliadas com os quantificadores do LTNtorch (Forall com AggregPMeanError(p=2) e Exists com AggregPMean(p=2)) sobre o modelo treinado de cada execução.

|   seed |   satAgg |   lastOnTheLeft |   lastOnTheRight |
|-------:|---------:|----------------:|-----------------:|
|     42 |   0.8378 |          0.4611 |           0.4567 |
|     43 |   0.8359 |          0.4603 |           0.4655 |
|     44 |   0.8406 |          0.4478 |           0.4557 |
|     45 |   0.837  |          0.4639 |           0.4532 |
|     46 |   0.8394 |          0.4588 |           0.4675 |

## 2. Satisfatibilidade por axioma e por execução

| axioma / formula                        |   seed_42 |   seed_43 |   seed_44 |   seed_45 |   seed_46 |   media |   desvio_padrao |
|:----------------------------------------|----------:|----------:|----------:|----------:|----------:|--------:|----------------:|
| satAgg (geral)                          |    0.8378 |    0.8359 |    0.8406 |    0.837  |    0.8394 |  0.8381 |          0.0019 |
| formula lastOnTheLeft                   |    0.4611 |    0.4603 |    0.4478 |    0.4639 |    0.4588 |  0.4584 |          0.0062 |
| formula lastOnTheRight                  |    0.4567 |    0.4655 |    0.4557 |    0.4532 |    0.4675 |  0.4597 |          0.0064 |
| cor_cobertura                           |    0.9997 |    0.9998 |    0.9997 |    0.9997 |    0.9997 |  0.9997 |          0      |
| cor_exclusividade_0_1                   |    0.9998 |    0.9998 |    0.9998 |    0.9998 |    0.9998 |  0.9998 |          0      |
| cor_exclusividade_0_2                   |    0.9998 |    0.9998 |    0.9998 |    0.9998 |    0.9998 |  0.9998 |          0      |
| cor_exclusividade_1_2                   |    0.9998 |    0.9998 |    0.9998 |    0.9998 |    0.9998 |  0.9998 |          0      |
| forma_cobertura                         |    0.9996 |    0.9996 |    0.9996 |    0.9996 |    0.9996 |  0.9996 |          0      |
| forma_exclusividade_0_1                 |    0.9999 |    0.9999 |    0.9999 |    0.9998 |    0.9998 |  0.9999 |          0      |
| forma_exclusividade_0_2                 |    0.9998 |    0.9998 |    0.9999 |    0.9998 |    0.9999 |  0.9998 |          0      |
| forma_exclusividade_0_3                 |    0.9999 |    0.9999 |    0.9999 |    0.9998 |    0.9998 |  0.9999 |          0      |
| forma_exclusividade_0_4                 |    0.9998 |    0.9999 |    0.9999 |    0.9999 |    0.9999 |  0.9999 |          0      |
| forma_exclusividade_1_2                 |    0.9999 |    0.9998 |    0.9999 |    0.9998 |    0.9999 |  0.9999 |          0      |
| forma_exclusividade_1_3                 |    0.9999 |    0.9999 |    0.9998 |    0.9999 |    0.9998 |  0.9999 |          0      |
| forma_exclusividade_1_4                 |    0.9999 |    0.9999 |    0.9998 |    0.9999 |    0.9999 |  0.9999 |          0      |
| forma_exclusividade_2_3                 |    0.9999 |    0.9998 |    0.9999 |    0.9998 |    0.9999 |  0.9999 |          0      |
| forma_exclusividade_2_4                 |    0.9998 |    0.9998 |    0.9998 |    0.9999 |    0.9999 |  0.9998 |          0      |
| forma_exclusividade_3_4                 |    0.9999 |    0.9999 |    0.9998 |    0.9999 |    0.9999 |  0.9999 |          0      |
| tamanho_cobertura                       |    0.9998 |    0.9998 |    0.9998 |    0.9998 |    0.9998 |  0.9998 |          0      |
| tamanho_exclusividade                   |    0.9998 |    0.9998 |    0.9998 |    0.9998 |    0.9998 |  0.9998 |          0      |
| left_irreflexividade                    |    0.586  |    0.592  |    0.5891 |    0.5789 |    0.6135 |  0.5919 |          0.013  |
| right_irreflexividade                   |    0.5875 |    0.5744 |    0.5679 |    0.5723 |    0.5592 |  0.5722 |          0.0103 |
| left_assimetria                         |    0.8561 |    0.84   |    0.8733 |    0.8432 |    0.835  |  0.8495 |          0.0154 |
| right_assimetria                        |    0.8585 |    0.841  |    0.8571 |    0.8507 |    0.8357 |  0.8486 |          0.01   |
| left_implica_right_inverso              |    0.8224 |    0.8091 |    0.8376 |    0.8074 |    0.8034 |  0.816  |          0.014  |
| right_implica_left_inverso              |    0.8267 |    0.8045 |    0.8207 |    0.8169 |    0.804  |  0.8146 |          0.0101 |
| left_transitividade                     |    0.9378 |    0.9252 |    0.9445 |    0.9292 |    0.9239 |  0.9321 |          0.0088 |
| right_transitividade                    |    0.9381 |    0.9274 |    0.941  |    0.9298 |    0.9235 |  0.932  |          0.0073 |
| below_irreflexividade                   |    0.5739 |    0.5711 |    0.5824 |    0.5544 |    0.5759 |  0.5715 |          0.0105 |
| above_irreflexividade                   |    0.5748 |    0.5698 |    0.5733 |    0.5811 |    0.5839 |  0.5766 |          0.0058 |
| below_assimetria                        |    0.839  |    0.8456 |    0.8509 |    0.845  |    0.8619 |  0.8485 |          0.0086 |
| above_assimetria                        |    0.835  |    0.8337 |    0.8646 |    0.8596 |    0.8601 |  0.8506 |          0.015  |
| below_implica_above_inverso             |    0.8089 |    0.8134 |    0.8207 |    0.8135 |    0.8283 |  0.8169 |          0.0076 |
| above_implica_below_inverso             |    0.7971 |    0.8017 |    0.824  |    0.8224 |    0.8243 |  0.8139 |          0.0133 |
| below_transitividade                    |    0.9237 |    0.9289 |    0.9349 |    0.9331 |    0.9395 |  0.932  |          0.006  |
| above_transitividade                    |    0.9254 |    0.9248 |    0.9417 |    0.939  |    0.9396 |  0.9341 |          0.0083 |
| close_irreflexividade                   |    0.9999 |    0.9999 |    0.9999 |    0.9999 |    0.9999 |  0.9999 |          0      |
| close_implica_close_inverso_1           |    0.8943 |    0.9097 |    0.9085 |    0.9055 |    0.9023 |  0.904  |          0.0062 |
| close_implica_close_inverso_2           |    0.8943 |    0.9097 |    0.9085 |    0.9055 |    0.9023 |  0.904  |          0.0062 |
| triangulos_proximos_mesmo_tamanho       |    0.9281 |    0.9914 |    0.9101 |    0.9998 |    0.9601 |  0.9579 |          0.0389 |
| can_stack_implica_above                 |    0.8144 |    0.7986 |    0.8344 |    0.8268 |    0.8325 |  0.8213 |          0.0149 |
| can_stack_base_nao_cone                 |    0.9565 |    0.9718 |    0.9478 |    0.9636 |    0.956  |  0.9592 |          0.009  |
| can_stack_base_nao_triangulo            |    0.9648 |    0.9676 |    0.9509 |    0.9747 |    0.9606 |  0.9637 |          0.0088 |
| inbetween_implica_configuracao_espacial |    0.7616 |    0.7635 |    0.7561 |    0.7583 |    0.7584 |  0.7596 |          0.003  |

## 3. Métricas clássicas dos predicados por execução

### 3.1. Accuracy

| predicado   |   seed_42 |   seed_43 |   seed_44 |   seed_45 |   seed_46 |   media |   desvio_padrao |
|:------------|----------:|----------:|----------:|----------:|----------:|--------:|----------------:|
| leftOf      |    0.848  |    0.872  |    0.84   |    0.848  |    0.88   |  0.8576 |          0.0173 |
| rightOf     |    0.856  |    0.896  |    0.856  |    0.832  |    0.856  |  0.8592 |          0.023  |
| below       |    0.816  |    0.8    |    0.832  |    0.808  |    0.872  |  0.8256 |          0.0285 |
| above       |    0.832  |    0.768  |    0.872  |    0.784  |    0.848  |  0.8208 |          0.0437 |
| closeTo     |    1      |    1      |    1      |    1      |    1      |  1      |          0      |
| canStack    |    0.784  |    0.776  |    0.832  |    0.736  |    0.736  |  0.7728 |          0.0398 |
| inBetween   |    0.7217 |    0.7134 |    0.7185 |    0.6964 |    0.7025 |  0.7105 |          0.0107 |

### 3.2. Precision

| predicado   |   seed_42 |   seed_43 |   seed_44 |   seed_45 |   seed_46 |   media |   desvio_padrao |
|:------------|----------:|----------:|----------:|----------:|----------:|--------:|----------------:|
| leftOf      |    0.8333 |    0.9318 |    0.8919 |    0.8833 |    0.88   |  0.8841 |          0.0351 |
| rightOf     |    0.9057 |    0.9492 |    0.8657 |    0.8571 |    0.8806 |  0.8916 |          0.037  |
| below       |    0.8478 |    0.8958 |    0.8475 |    0.7407 |    0.9608 |  0.8585 |          0.0806 |
| above       |    0.8793 |    0.7258 |    0.913  |    0.9074 |    0.9167 |  0.8684 |          0.0811 |
| closeTo     |    1      |    1      |    1      |    1      |    1      |  1      |          0      |
| canStack    |    0.5106 |    0.4615 |    0.4118 |    0.5283 |    0.3182 |  0.4461 |          0.0847 |
| inBetween   |    0.6923 |    0.7012 |    0.7197 |    0.6967 |    0.7057 |  0.7031 |          0.0105 |

### 3.3. Recall

| predicado   |   seed_42 |   seed_43 |   seed_44 |   seed_45 |   seed_46 |   media |   desvio_padrao |
|:------------|----------:|----------:|----------:|----------:|----------:|--------:|----------------:|
| leftOf      |    0.8475 |    0.7593 |    0.6735 |    0.8154 |    0.8302 |  0.7852 |          0.0707 |
| rightOf     |    0.7869 |    0.8485 |    0.8657 |    0.75   |    0.8551 |  0.8212 |          0.0503 |
| below       |    0.7091 |    0.6825 |    0.8065 |    0.8    |    0.7778 |  0.7552 |          0.056  |
| above       |    0.7846 |    0.7895 |    0.7778 |    0.6901 |    0.7458 |  0.7576 |          0.0414 |
| closeTo     |    1      |    1      |    1      |    1      |    1      |  1      |          0      |
| canStack    |    0.8571 |    1      |    0.9333 |    0.7778 |    0.8235 |  0.8784 |          0.0886 |
| inBetween   |    0.8551 |    0.8528 |    0.8159 |    0.7804 |    0.7802 |  0.8169 |          0.0368 |

### 3.4. F1

| predicado   |   seed_42 |   seed_43 |   seed_44 |   seed_45 |   seed_46 |   media |   desvio_padrao |
|:------------|----------:|----------:|----------:|----------:|----------:|--------:|----------------:|
| leftOf      |    0.8403 |    0.8367 |    0.7674 |    0.848  |    0.8544 |  0.8294 |          0.0353 |
| rightOf     |    0.8421 |    0.896  |    0.8657 |    0.8    |    0.8676 |  0.8543 |          0.0359 |
| below       |    0.7723 |    0.7748 |    0.8264 |    0.7692 |    0.8596 |  0.8005 |          0.0406 |
| above       |    0.8293 |    0.7563 |    0.84   |    0.784  |    0.8224 |  0.8064 |          0.0351 |
| closeTo     |    1      |    1      |    1      |    1      |    1      |  1      |          0      |
| canStack    |    0.64   |    0.6316 |    0.5714 |    0.6292 |    0.459  |  0.5862 |          0.0762 |
| inBetween   |    0.7651 |    0.7696 |    0.7648 |    0.7361 |    0.7411 |  0.7553 |          0.0155 |

## 4. Arquivos gerados

- Tabela consolidada (formato largo): `results/experiments/experimentos_multi_seed.csv`
- Artefatos por execução (histórico de treinamento e resumo JSON): `results/experiments/seed_<N>/`

Observação: os arquivos em `data/datasets_gerados/` e `results/training/` correspondem à última execução (seed 46), pois o pipeline os sobrescreve.
