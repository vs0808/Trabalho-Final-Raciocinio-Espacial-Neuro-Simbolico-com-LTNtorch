# Trabalho Final de IA — Raciocínio Espacial Neuro-Simbólico com LTNtorch

Este repositório contém o desenvolvimento de um agente neuro-simbólico para raciocínio espacial usando **LTNtorch**. O projeto trabalha com objetos sintéticos em um plano 2D, representados por vetores de atributos, e combina:

- geração de dados sintéticos;
- visualização de cena 2D;
- construção de verdade-terreno;
- definição de predicados lógicos e fuzzy;
- treinamento com redes neurais e axiomas lógicos;
- análise de métricas;
- consultas compostas;
- geração de explicações para as respostas.

A proposta central é demonstrar como um sistema pode aprender e raciocinar sobre relações espaciais, combinando aprendizado supervisionado com restrições lógicas diferenciáveis.

---

## 1. Visão geral do projeto

O objetivo do projeto é construir um sistema capaz de responder perguntas espaciais sobre uma cena sintética formada por objetos geométricos.

Cada objeto possui:

- posição no plano 2D;
- cor;
- forma;
- tamanho.

A partir desses objetos, o sistema aprende ou calcula relações como:

- `leftOf(x, y)`: o objeto `x` está à esquerda de `y`;
- `rightOf(x, y)`: o objeto `x` está à direita de `y`;
- `below(x, y)`: o objeto `x` está abaixo de `y`;
- `above(x, y)`: o objeto `x` está acima de `y`;
- `closeTo(x, y)`: o objeto `x` está próximo de `y`;
- `inBetween(x, y, z)`: o objeto `x` está entre `y` e `z`;
- `canStack(x, y)`: o objeto `x` pode ser empilhado sobre `y`.

Além das relações espaciais, o sistema também usa predicados de atributo:

- `isRed(x)`;
- `isGreen(x)`;
- `isBlue(x)`;
- `isCircle(x)`;
- `isSquare(x)`;
- `isCylinder(x)`;
- `isCone(x)`;
- `isTriangle(x)`;
- `isSmall(x)`;
- `isBig(x)`;
- `sameColor(x, y)`;
- `sameShape(x, y)`;
- `sameSize(x, y)`.

O projeto é inspirado em uma versão simplificada do tipo de raciocínio presente no CLEVR, mas, em vez de usar imagens reais, trabalha diretamente com vetores estruturados.

---

## 2. Ideia neuro-simbólica

A abordagem neuro-simbólica combina dois mundos:

1. **Aprendizado neural**
   - redes neurais aprendem predicados a partir de exemplos;
   - cada predicado retorna um valor de verdade fuzzy entre `0` e `1`.

2. **Raciocínio simbólico**
   - regras lógicas são usadas como axiomas;
   - o modelo é penalizado quando viola essas regras;
   - a satisfatibilidade lógica é medida por `satAgg`.

Com LTNtorch, as fórmulas lógicas são diferenciáveis. Isso permite treinar o modelo considerando simultaneamente:

- erro supervisionado em relação à verdade-terreno;
- grau de satisfação dos axiomas lógicos.

---

## 3. Representação dos objetos

Cada objeto é representado por um vetor de 11 posições:

```text
[0]  x normalizado
[1]  y normalizado
[2]  cor vermelho
[3]  cor verde
[4]  cor azul
[5]  forma círculo
[6]  forma quadrado
[7]  forma cilindro
[8]  forma cone
[9]  forma triângulo
[10] tamanho
```

O tamanho é representado assim:

```text
pequeno = 0.0
grande  = 1.0
```

Exemplo conceitual de vetor:

```text
[0.7740, 0.4389, 0, 1, 0, 0, 0, 0, 0, 1, 0]
```

Esse vetor representa um objeto em uma posição `(x, y)`, com cor verde, forma triângulo e tamanho pequeno.

---

## 4. Estrutura esperada do projeto

A estrutura do projeto fica organizada da seguinte forma:

```text
Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch/
├── main.py
├── README.md
├── requirements.txt
│
├── data/
│   └── datasets_gerados/
│       ├── objetos_sinteticos.csv
│       ├── objetos_sinteticos_vetores.npy
│       ├── ground_truth_unarios.csv
│       ├── ground_truth_pares.csv
│       └── ground_truth_triplas.csv
│
├── notebooks/
│   └── 01_exploracao_dados.ipynb
│
├── results/
│   ├── plots/
│   │   └── cenario_sintetico.png
│   │
│   ├── training/
│   │   ├── historico_treinamento.csv
│   │   └── modelo_ltn_treinavel.pt
│   │
│   ├── metrics/
│   │   ├── balanceamento_relacoes_binarias.csv
│   │   ├── balanceamento_relacoes_ternarias.csv
│   │   ├── metricas_finais_binarias.csv
│   │   ├── metricas_finais_ternarias.csv
│   │   ├── sat_individual_final.csv
│   │   ├── resumo_diagnostico.md
│   │   ├── metricas_classicas.md
│   │   └── plots/
│   │       ├── loss_total.png
│   │       ├── losses_componentes.png
│   │       ├── satagg.png
│   │       ├── f1_final_por_predicado.png
│   │       └── proporcao_positivos_por_relacao.png
│   │
│   └── queries/
│       ├── respostas_consultas.json
│       ├── respostas_consultas.csv
│       └── explicacoes_consultas.md
│
└── src/
    ├── __init__.py
    ├── data_generator.py
    ├── plot_scene.py
    ├── relations_ground_truth.py
    ├── ltn_predicates.py
    ├── ltn_axioms.py
    ├── train_ltn.py
    ├── analyze_training_results.py
    └── query_ltn.py
```

---

## 5. Descrição dos principais arquivos

### 5.1. `main.py`

Arquivo principal do projeto.

Ele executa a preparação inicial dos dados e pode chamar o treinamento.

Responsabilidades:

- gerar objetos sintéticos;
- salvar CSV e NPY;
- gerar visualização da cena;
- criar verdade-terreno;
- opcionalmente executar o treinamento.

Comando básico:

```bash
python3 main.py --no-show
```

Com treinamento:

```bash
python3 main.py --train --no-show
```

---

### 5.2. `src/data_generator.py`

Responsável por gerar a cena sintética.

Ele cria 25 objetos, cada um contendo:

- `id_objeto`;
- coordenada `x`;
- coordenada `y`;
- cor;
- forma;
- tamanho;
- vetor de 11 posições.

Saídas principais:

```text
data/datasets_gerados/objetos_sinteticos.csv
data/datasets_gerados/objetos_sinteticos_vetores.npy
```

---

### 5.3. `src/plot_scene.py`

Responsável por gerar uma imagem da cena 2D.

A cena mostra os objetos distribuídos no plano, usando:

- cores visuais;
- marcadores diferentes por forma;
- tamanho visual diferente para objetos pequenos e grandes.

Saída:

```text
results/plots/cenario_sintetico.png
```

---

### 5.4. `src/relations_ground_truth.py`

Responsável por gerar a verdade-terreno das relações.

Esse arquivo calcula deterministicamente:

#### Predicados unários

```text
isRed
isGreen
isBlue
isCircle
isSquare
isCylinder
isCone
isTriangle
isSmall
isBig
```

#### Predicados binários

```text
sameColor
sameShape
sameSize
leftOf
rightOf
below
above
closeTo
canStack
```

#### Predicado ternário

```text
inBetween
```

Saídas:

```text
data/datasets_gerados/ground_truth_unarios.csv
data/datasets_gerados/ground_truth_pares.csv
data/datasets_gerados/ground_truth_triplas.csv
```

---

### 5.5. `src/ltn_predicates.py`

Define os predicados usados pela LTN.

Há três grupos principais:

#### 1. Predicados de atributo

São calculados diretamente a partir do vetor do objeto.

Exemplos:

```text
isRed(x)
isTriangle(x)
isSmall(x)
sameSize(x,y)
```

#### 2. Predicados fuzzy determinísticos

São calculados por fórmulas geométricas diferenciáveis.

O caso mais importante é:

```text
closeTo(x,y)
```

No projeto, `closeTo` foi tratado como predicado fuzzy determinístico, pois ele depende diretamente da distância euclidiana entre objetos.

A decisão final foi:

```text
closeTo(x,y) = alto se x e y são objetos diferentes e estão próximos no plano.
```

Isso corrigiu o desempenho desse predicado e melhorou a consistência lógica.

#### 3. Predicados treináveis

São redes neurais pequenas que aprendem relações a partir da verdade-terreno e dos axiomas.

Exemplos:

```text
leftOf
rightOf
below
above
canStack
inBetween
```

---

### 5.6. `src/ltn_axioms.py`

Define a base de conhecimento lógica.

Os axiomas principais envolvem:

#### Taxonomia

- todo objeto tem uma cor;
- um objeto não deve ter duas cores ao mesmo tempo;
- todo objeto tem uma forma;
- um objeto não deve ter duas formas ao mesmo tempo;
- todo objeto é pequeno ou grande;
- um objeto não deve ser pequeno e grande ao mesmo tempo.

#### Relações horizontais

- irreflexividade de `leftOf` e `rightOf`;
- assimetria;
- relação inversa entre `leftOf` e `rightOf`;
- transitividade.

#### Relações verticais

- irreflexividade de `below` e `above`;
- assimetria;
- relação inversa entre `below` e `above`;
- transitividade.

#### Proximidade e composição

- `closeTo` é irreflexivo;
- `closeTo` é simétrico;
- triângulos próximos devem ter o mesmo tamanho;
- `canStack(x,y)` implica `above(x,y)`;
- a base de empilhamento não deve ser cone;
- a base de empilhamento não deve ser triângulo.

#### `inBetween`

A relação `inBetween(x,y,z)` é conectada a uma configuração espacial coerente envolvendo relações horizontais e verticais.

---

### 5.7. `src/train_ltn.py`

Responsável pelo treinamento do modelo neuro-simbólico.

Ele combina:

```text
loss_total = loss_supervisionada + peso_axiomas * loss_axiomas
```

Onde:

- `loss_supervisionada` mede erro em relação à verdade-terreno;
- `loss_axiomas = 1 - satAgg`;
- `satAgg` mede o grau de satisfação dos axiomas;
- `peso_axiomas` controla a influência da lógica no treinamento.

Na versão final:

- `closeTo` é determinístico;
- `canStack` continua treinável;
- `leftOf`, `rightOf`, `below` e `above` continuam treináveis;
- `inBetween` continua treinável;
- relações desbalanceadas usam loss ponderada.

Saídas:

```text
results/training/historico_treinamento.csv
results/training/modelo_ltn_treinavel.pt
```

---

### 5.8. `src/analyze_training_results.py`

Responsável por analisar os resultados do treinamento.

Ele gera:

```text
results/metrics/balanceamento_relacoes_binarias.csv
results/metrics/balanceamento_relacoes_ternarias.csv
results/metrics/metricas_finais_binarias.csv
results/metrics/metricas_finais_ternarias.csv
results/metrics/sat_individual_final.csv
results/metrics/resumo_diagnostico.md
results/metrics/metricas_classicas.md
results/metrics/plots/
```

Esse módulo analisa:

- balanceamento das classes;
- métricas finais;
- evolução da loss;
- evolução do `satAgg`;
- satisfatibilidade individual dos axiomas;
- fórmulas das métricas clássicas.

---

### 5.9. `src/query_ltn.py`

Responsável por carregar o modelo treinado e executar consultas compostas.

Ele gera respostas para perguntas como:

```text
Existe algum objeto pequeno abaixo de um cilindro e à esquerda de um quadrado?
Existe um cone verde entre dois objetos?
Se dois triângulos estão próximos, eles têm o mesmo tamanho?
Existe algum objeto que pode ser empilhado sobre outro?
```

Saídas:

```text
results/queries/respostas_consultas.json
results/queries/respostas_consultas.csv
results/queries/explicacoes_consultas.md
```

---

## 6. Como preparar o ambiente

### 6.1. Clonar ou acessar o projeto

Entre na raiz do projeto:

```bash
cd Trabalho-Final-Raciocinio-Espacial-Neuro-Simbolico-com-LTNtorch
```

---

### 6.2. Criar ambiente virtual

No Linux:

```bash
python3 -m venv venv
```

Ativar:

```bash
source venv/bin/activate
```

No Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 6.3. Instalar dependências

Com o ambiente virtual ativo:

```bash
pip install -r requirements.txt
```

O `requirements.txt` do projeto contém:

```text
numpy
pandas
matplotlib
torch
ltntorch
scikit-learn
```

O pacote `ltntorch` (LTNtorch) é o que fornece o módulo `ltn` usado nos imports do código.

---

## 7. Como executar o projeto

O padrão adotado para execução é:

```bash
python3 main.py --train --no-show
python3 src/analyze_training_results.py
python3 src/query_ltn.py
```

Esse padrão mantém a execução simples, direta e compatível com a organização em `src/`.

---

### 7.1. Gerar dados, imagem e verdade-terreno

```bash
python3 main.py --no-show
```

Esse comando gera:

```text
data/datasets_gerados/objetos_sinteticos.csv
data/datasets_gerados/objetos_sinteticos_vetores.npy
data/datasets_gerados/ground_truth_unarios.csv
data/datasets_gerados/ground_truth_pares.csv
data/datasets_gerados/ground_truth_triplas.csv
results/plots/cenario_sintetico.png
```

---

### 7.2. Gerar dados e treinar o modelo

```bash
python3 main.py --train --no-show
```

Esse comando executa:

1. geração dos dados sintéticos;
2. visualização da cena;
3. geração da verdade-terreno;
4. treinamento LTN.

Saídas principais:

```text
results/training/historico_treinamento.csv
results/training/modelo_ltn_treinavel.pt
```

---

### 7.3. Analisar métricas

```bash
python3 src/analyze_training_results.py
```

Esse comando gera:

```text
results/metrics/
```

Com arquivos como:

```text
balanceamento_relacoes_binarias.csv
balanceamento_relacoes_ternarias.csv
metricas_finais_binarias.csv
metricas_finais_ternarias.csv
sat_individual_final.csv
resumo_diagnostico.md
metricas_classicas.md
```

E gráficos em:

```text
results/metrics/plots/
```

---

### 7.4. Executar consultas compostas

```bash
python3 src/query_ltn.py
```

Esse comando gera:

```text
results/queries/respostas_consultas.json
results/queries/respostas_consultas.csv
results/queries/explicacoes_consultas.md
```

---

## 8. Saídas geradas

### 8.1. `objetos_sinteticos.csv`

Contém a tabela dos objetos gerados.

Colunas principais:

```text
id_objeto
x
y
cor
forma
tamanho
vetor_11
```

Serve para inspeção humana e para conferir os objetos da cena.

---

### 8.2. `objetos_sinteticos_vetores.npy`

Contém os vetores numéricos dos objetos em formato NumPy.

Esse arquivo é usado pelo treinamento.

---

### 8.3. `cenario_sintetico.png`

Imagem da cena 2D com todos os objetos.

Serve para visualizar a distribuição espacial.

---

### 8.4. `ground_truth_unarios.csv`

Contém os rótulos verdadeiros dos predicados unários.

Exemplo de predicados:

```text
isRed
isGreen
isBlue
isCircle
isSquare
isCylinder
isCone
isTriangle
isSmall
isBig
```

---

### 8.5. `ground_truth_pares.csv`

Contém relações binárias entre pares de objetos.

Exemplos:

```text
leftOf
rightOf
below
above
closeTo
canStack
sameColor
sameShape
sameSize
```

Como há 25 objetos, são avaliados:

```text
25 × 25 = 625 pares
```

---

### 8.6. `ground_truth_triplas.csv`

Contém relações ternárias entre trios de objetos.

Principal relação:

```text
inBetween
```

Como usamos objetos distintos e a ordem importa, são avaliadas:

```text
25 × 24 × 23 = 13.800 triplas
```

---

### 8.7. `historico_treinamento.csv`

Arquivo de histórico do treinamento.

Contém, por época:

- `loss_total`;
- `loss_supervisionada`;
- `loss_binaria`;
- `loss_ternaria`;
- `loss_axiomas`;
- `sat_agg`;
- métricas binárias;
- métricas ternárias;
- satisfatibilidade individual dos axiomas;
- pesos positivos usados na loss ponderada.

Esse arquivo é usado pelo módulo de análise.

---

### 8.8. `modelo_ltn_treinavel.pt`

Arquivo do modelo treinado.

Ele contém:

- pesos dos predicados treináveis;
- configurações de treinamento;
- lista das relações treinadas;
- indicação de que `closeTo` é determinístico.

Esse arquivo é usado por `src/query_ltn.py`.

---

### 8.9. `resumo_diagnostico.md`

Resumo textual dos resultados do treinamento.

Inclui:

- evolução da loss;
- evolução do `satAgg`;
- balanceamento das relações;
- métricas finais;
- diagnóstico por predicado;
- axiomas com menor satisfatibilidade;
- próximo passo recomendado.

---

### 8.10. `metricas_classicas.md`

Explica as métricas clássicas usadas no projeto:

- Accuracy;
- Precision;
- Recall;
- F1 Score.

Também define os termos:

- TP;
- TN;
- FP;
- FN.

Esse arquivo serve como apoio teórico para o relatório final.

---

### 8.11. `respostas_consultas.json`

Arquivo estruturado com todas as respostas das consultas.

Contém:

- pergunta;
- fórmula;
- valor de verdade;
- resposta booleana;
- melhor evidência;
- componentes da evidência.

---

### 8.12. `respostas_consultas.csv`

Tabela simplificada com as consultas principais.

Serve para inserir facilmente no relatório.

---

### 8.13. `explicacoes_consultas.md`

Arquivo textual explicando as respostas das consultas.

Inclui:

- fórmula da consulta;
- valor de verdade;
- resposta;
- objetos envolvidos;
- componentes que sustentaram a resposta;
- top pares por relação;
- top triplas de `inBetween`.

---

## 9. Métricas usadas

O projeto usa as seguintes métricas clássicas:

### 9.1. Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Mede o percentual geral de acertos.

Pode ser enganosa quando há desbalanceamento.

---

### 9.2. Precision

```text
Precision = TP / (TP + FP)
```

Mede, entre os casos previstos como positivos, quantos realmente eram positivos.

---

### 9.3. Recall

```text
Recall = TP / (TP + FN)
```

Mede, entre os positivos reais, quantos foram encontrados pelo modelo.

---

### 9.4. F1 Score

```text
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
```

Mede o equilíbrio entre Precision e Recall.

No projeto, o F1 Score é especialmente importante para relações desbalanceadas.

---

## 10. Resultados atuais do treinamento

A versão final analisada apresentou os seguintes resultados principais:

### Predicados binários

```text
leftOf    F1 ≈ 0.8403
rightOf   F1 ≈ 0.8421
below     F1 ≈ 0.7723
above     F1 ≈ 0.8293
closeTo   F1 = 1.0000
canStack  F1 ≈ 0.6400
```

### Predicado ternário

```text
inBetween F1 ≈ 0.7655
```

### Satisfatibilidade lógica

```text
satAgg inicial ≈ 0.8007
satAgg final   ≈ 0.8378
```

A interpretação geral é:

- relações espaciais simples tiveram bom desempenho;
- `closeTo` ficou excelente ao ser tratado como predicado fuzzy determinístico;
- `canStack` teve desempenho intermediário, esperado por ser relação composta;
- `inBetween` teve desempenho aceitável para uma relação ternária;
- os axiomas foram progressivamente mais satisfeitos durante o treinamento.

---

## 11. Consultas compostas

O módulo `query_ltn.py` executa consultas compostas, como:

### Consulta 1

```text
Existe algum objeto pequeno abaixo de um cilindro e à esquerda de um quadrado?
```

Fórmula aproximada:

```text
∃x(IsSmall(x) ∧ ∃y(IsCylinder(y) ∧ Below(x,y)) ∧ ∃z(IsSquare(z) ∧ LeftOf(x,z)))
```

---

### Consulta 2

```text
Existe um cone verde entre dois objetos?
```

Fórmula aproximada:

```text
∃x,y,z(IsCone(x) ∧ IsGreen(x) ∧ InBetween(x,y,z))
```

---

### Consulta 3

```text
Se dois triângulos estão próximos, eles têm o mesmo tamanho?
```

Fórmula aproximada:

```text
∀x,y((IsTriangle(x) ∧ IsTriangle(y) ∧ CloseTo(x,y)) → SameSize(x,y))
```

---

### Consulta 4

```text
Existe algum objeto que pode ser empilhado sobre outro?
```

Fórmula aproximada:

```text
∃x,y CanStack(x,y)
```

---

### Consulta 5

```text
Existe um objeto que esteja à esquerda de todos os outros (último da esquerda)?
```

Fórmula:

```text
lastOnTheLeft(x) = ∃x(∀y LeftOf(x,y))
```

---

### Consulta 6

```text
Existe um objeto que esteja à direita de todos os outros (último da direita)?
```

Fórmula:

```text
lastOnTheRight(x) = ∃x(∀y RightOf(x,y))
```

As fórmulas das consultas 5 e 6 são construídas em `src/ltn_axioms.py`
(função `formulas_extremos_horizontais`) e avaliadas em `src/query_ltn.py`
com os quantificadores fuzzy do LTNtorch:

- `Forall` com `AggregPMeanError(p=2)`;
- `Exists` com `AggregPMean(p=2)`.

Observação importante: como o `∀y` percorre todos os objetos, incluindo o caso
`y = x`, e `leftOf`/`rightOf` são irreflexivos, o valor de verdade dessas
fórmulas nunca chega a 1, mesmo para o objeto do extremo. Por isso, além do
valor agregado, `query_ltn.py` reporta o objeto que maximiza a quantificação
universal (a melhor evidência), permitindo identificar qual objeto é o extremo
da cena.

---

## 12. Organização do diretório `notebooks/`

O diretório `notebooks/` fica na raiz do projeto e contém atualmente:

```text
notebooks/
└── 01_exploracao_dados.ipynb
```

- `01_exploracao_dados.ipynb`: exploração dos dados sintéticos gerados (`objetos_sinteticos.csv`).

Sugestão de notebooks futuros:

- `02_visualizacao_cenario.ipynb`: explicar a cena 2D;
- `03_ground_truth_relacoes.ipynb`: mostrar como as relações são calculadas;
- `04_metricas_classicas.ipynb`: explicar Accuracy, Precision, Recall e F1;
- `05_analise_treinamento.ipynb`: analisar loss, satAgg e métricas finais;
- `06_consultas_ltn.ipynb`: explorar respostas de `query_ltn.py`;
- `07_relatorio_experimentos.ipynb`: consolidar resultados para o relatório final.

---

## 13. Pipeline consolidado

A sequência principal do projeto é:

```bash
python3 main.py --train --no-show
python3 src/analyze_training_results.py
python3 src/query_ltn.py
```

Essa sequência gera:

1. dados sintéticos;
2. gráfico da cena;
3. verdade-terreno;
4. modelo treinado;
5. histórico de treinamento;
6. métricas;
7. gráficos;
8. diagnóstico;
9. consultas compostas;
10. explicações.

### 13.1. Experimento com múltiplas seeds

O script `src/run_experiments.py` repete o pipeline completo (geração de dados
+ treinamento LTN) com 5 seeds distintas e consolida os resultados:

```bash
python3 src/run_experiments.py
```

Por padrão, as seeds são `42, 43, 44, 45, 46`, com 25 objetos e 200 épocas por
execução. Esses parâmetros podem ser alterados:

```bash
python3 src/run_experiments.py --seeds 10 20 30 40 50 --epochs 200 --n-objetos 25
```

Para cada execução, o script coleta da última época de treinamento:

- o `satAgg` geral e a satisfatibilidade individual de cada axioma;
- os valores das fórmulas `lastOnTheLeft` e `lastOnTheRight`;
- Accuracy, Precision, Recall e F1 de cada predicado avaliado
  (`leftOf`, `rightOf`, `below`, `above`, `closeTo`, `canStack`, `inBetween`).

Saídas geradas:

```text
results/experiments/
├── experimentos_multi_seed.csv   # tabela consolidada, uma linha por execução
├── experimentos_multi_seed.md    # tabelas pivotadas por axioma e por métrica
└── seed_<N>/                     # histórico de treinamento e resumo JSON por seed
```

---

## 14. Conformidade com o enunciado e pendências

Esta seção compara o estado atual do projeto com as tarefas definidas no enunciado do trabalho (ICC260 — Raciocínio Espacial Neuro-Simbólico com LTNtorch).

### 14.1. Tarefa 1 — Taxonomia e Formas (2 pontos) — ✅ Completa

- [x] Geração de 25 objetos aleatórios com coordenadas, cores, formas e tamanho (`src/data_generator.py`);
- [x] Plot do cenário gerado (`results/plots/cenario_sintetico.png`);
- [x] Predicados de forma: `isCircle`, `isSquare`, `isCylinder`, `isCone`, `isTriangle`;
- [x] Predicados de tamanho: `isSmall`, `isBig`;
- [x] Axioma de forma única (exclusividade) e de cobertura (completude), incluindo também cores e tamanho (`src/ltn_axioms.py`).

### 14.2. Tarefa 2 — Raciocínio Espacial Horizontal (5 pontos) — ⚠️ Quase completa

- [x] Predicados `leftOf(x,y)`, `rightOf(x,y)`, `closeTo(x,y)`, `inBetween(x,y,z)`;
- [x] Axiomas de irreflexividade, assimetria, inverso e transitividade;
- [x] `closeTo` implementado como predicado fuzzy determinístico baseado na distância euclidiana (sigmoide sobre o threshold, variação do kernel gaussiano sugerido no enunciado);
- [x] `inBetween` com a fórmula lógica `(leftOf(y,x) ∧ rightOf(z,x)) ∨ (leftOf(z,x) ∧ rightOf(y,x))` (estendida também para o eixo vertical);
- [ ] **Pendente:** `lastOnTheLeft(x)` — fórmula `∃x (∀y leftOf(x,y))`;
- [ ] **Pendente:** `lastOnTheRight(x)` — fórmula `∃x (∀y rightOf(x,y))`;
- [ ] Opcional (não implementado): consulta existencial "existe um objeto à esquerda de todos os quadrados";
- [ ] Opcional (não implementado): restrição "todo quadrado está à direita de todo círculo".

### 14.3. Tarefa 3 — Raciocínio Vertical — ✅ Completa

- [x] Predicados `below(x,y)` e `above(x,y)`;
- [x] Axiomas de inverso e transitividade (além de irreflexividade e assimetria);
- [x] `canStack(x,y)` com axiomas: base não pode ser cone nem triângulo, e `canStack(x,y) → above(x,y)`.

### 14.4. Tarefa 4 — Raciocínio Composto (2 pontos) — ✅ Completa

- [x] Consulta 1: filtragem composta (objeto pequeno abaixo de cilindro e à esquerda de quadrado);
- [x] Consulta 2: cone verde entre dois objetos (`inBetween`);
- [x] Consulta 3: restrição de proximidade (triângulos próximos → mesmo tamanho);
- [x] Consulta extra: existência de empilhamento (`canStack`).

### 14.5. Entregas (1 ponto) — ⚠️ Parcial

- [x] Código e texto em Markdown no GitHub;
- [x] Descrição de NeSy e LTN (Seção 2);
- [x] Descrição do dataset CLEVR simplificado (Seções 1 e 3);
- [x] Valor de satisfação das fórmulas (`results/metrics/sat_individual_final.csv`) e métricas no conjunto de teste;
- [ ] **Pendente:** resultados para **5 execuções com 5 datasets aleatórios distintos** (repetir a geração de dados 5 vezes com seeds diferentes e reportar, para cada execução: satAgg de cada fórmula, Accuracy, Precision, Recall e F1). Hoje o pipeline roda com um único dataset (seed 42). É possível executar manualmente com `python3 main.py --train --no-show --seed <N>`, mas falta um script que automatize as 5 execuções e consolide os resultados em tabela.

### 14.6. Ponto Extra — ✅ Completo

- [x] Explicações de cada pergunta/raciocínio geradas em `results/queries/explicacoes_consultas.md` (melhor evidência, componentes da conjunção, pior caso das regras universais, top pares e top triplas).

### 14.7. Outras pendências técnicas

- `results/relatorio.md` está vazio — preencher com o relatório final ou remover;
- `results/analysis/` duplica o conteúdo de `results/metrics/` — manter apenas um dos dois;
- Os axiomas de irreflexividade apresentam satisfatibilidade baixa (~0.57–0.59 em `sat_individual_final.csv`) — vale investigar/ajustar antes da entrega final.

---

## 15. Observações finais

Este projeto mostra uma aplicação prática de raciocínio espacial neuro-simbólico com LTNtorch.

A principal contribuição está na combinação de:

- atributos estruturados;
- predicados treináveis;
- predicados fuzzy determinísticos;
- axiomas lógicos;
- métricas de avaliação;
- consultas compostas explicáveis.

A decisão de transformar `closeTo` em predicado fuzzy determinístico foi importante porque mostrou que nem toda relação precisa ser aprendida por rede neural. Relações geométricas diretas podem ser melhor representadas por funções diferenciáveis conhecidas, enquanto relações compostas, como `canStack`, podem ser aprendidas e regularizadas por axiomas.

### 14.1. Pendências obrigatórias do enunciado

- [x] **Fórmulas `lastOnTheLeft` e `lastOnTheRight`** — concluído.
  As fórmulas `lastOnTheLeft(x) = ∃x(∀y leftOf(x,y))` e
  `lastOnTheRight(x) = ∃x(∀y rightOf(x,y))` foram implementadas em
  `src/ltn_axioms.py` (função `formulas_extremos_horizontais`) e são avaliadas
  como as consultas 5 e 6 de `src/query_ltn.py`, usando os quantificadores do
  LTNtorch já adotados no projeto (`Forall` com `AggregPMeanError(p=2)` e
  `Exists` com `AggregPMean(p=2)`). Ver seção 11.

- [x] **Experimento com 5 execuções (múltiplas seeds)** — concluído.
  O script `src/run_experiments.py` repete a geração de dados e o treinamento
  com 5 seeds distintas (42 a 46) e consolida, por execução, o `satAgg` de cada
  fórmula/axioma e as métricas Accuracy, Precision, Recall e F1 dos predicados,
  em `results/experiments/experimentos_multi_seed.csv` e
  `results/experiments/experimentos_multi_seed.md`. Ver seção 13.1.

O projeto continua aberto para expansões, como notebooks explicativos e análise comparativa entre predicados treináveis e determinísticos.