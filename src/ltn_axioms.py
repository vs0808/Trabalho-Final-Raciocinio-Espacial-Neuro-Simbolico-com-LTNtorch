from __future__ import annotations

from typing import Any

import torch
import ltn


def criar_operadores_logicos() -> dict[str, Any]:
    """
    Cria os operadores logicos fuzzy usados pela LTN.

    A LTN nao usa logica booleana classica pura.
    Ela usa logica fuzzy diferenciavel, em que os valores de verdade ficam no intervalo [0, 1].

    Operadores usados:
    - Not: negacao padrao, 1 - p
    - And: produto, p * q
    - Or: soma probabilistica
    - Implies: implicacao de Reichenbach
    - Forall: agregador para quantificacao universal
    - Exists: agregador para quantificacao existencial
    - SatAgg: agregador geral de satisfatibilidade da base de conhecimento
    """

    operadores = {
        "Not": ltn.Connective(ltn.fuzzy_ops.NotStandard()),
        "And": ltn.Connective(ltn.fuzzy_ops.AndProd()),
        "Or": ltn.Connective(ltn.fuzzy_ops.OrProbSum()),
        "Implies": ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach()),
        "Forall": ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f"),
        "Exists": ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e"),
        "SatAgg": ltn.fuzzy_ops.SatAgg(),
    }

    return operadores


def criar_variaveis_ltn(
    tensor_objetos: torch.Tensor,
) -> dict[str, ltn.Variable]:
    """
    Cria variaveis LTN para os objetos.

    Usamos tres variaveis diferentes, x, y e z, todas apontando para o mesmo conjunto
    de objetos. Isso permite criar formulas com um, dois ou tres objetos.

    Exemplos:
    - isCircle(x)
    - leftOf(x, y)
    - inBetween(x, y, z)

    O LTNtorch faz broadcasting entre variaveis diferentes.
    Entao, em uma formula com [x, y], ele avalia todos os pares possiveis.
    Em uma formula com [x, y, z], avalia todas as triplas possiveis.
    """

    return {
        "x": ltn.Variable("x", tensor_objetos),
        "y": ltn.Variable("y", tensor_objetos),
        "z": ltn.Variable("z", tensor_objetos),
    }


def axiomas_taxonomia(
    variaveis: dict[str, ltn.Variable],
    predicados: dict[str, ltn.Predicate],
    operadores: dict[str, Any],
) -> dict[str, Any]:
    """
    Cria axiomas de taxonomia dos objetos.

    Esses axiomas garantem coerencia nas propriedades dos objetos.

    Axiomas principais:
    1. Cobertura de cor:
       todo objeto deve ter alguma cor.

    2. Exclusividade de cor:
       um objeto nao deve ter duas cores ao mesmo tempo.

    3. Cobertura de forma:
       todo objeto deve ter alguma forma.

    4. Exclusividade de forma:
       um objeto nao deve ter duas formas ao mesmo tempo.

    5. Cobertura de tamanho:
       todo objeto deve ser pequeno ou grande.

    6. Exclusividade de tamanho:
       um objeto nao deve ser pequeno e grande ao mesmo tempo.
    """

    x = variaveis["x"]

    Not = operadores["Not"]
    And = operadores["And"]
    Or = operadores["Or"]
    Forall = operadores["Forall"]

    cores = [
        predicados["isRed"],
        predicados["isGreen"],
        predicados["isBlue"],
    ]

    formas = [
        predicados["isCircle"],
        predicados["isSquare"],
        predicados["isCylinder"],
        predicados["isCone"],
        predicados["isTriangle"],
    ]

    isSmall = predicados["isSmall"]
    isBig = predicados["isBig"]

    axiomas: dict[str, Any] = {}

    axiomas["cor_cobertura"] = Forall(
        x,
        Or(
            Or(cores[0](x), cores[1](x)),
            cores[2](x),
        ),
    )

    for i in range(len(cores)):
        for j in range(i + 1, len(cores)):
            axiomas[f"cor_exclusividade_{i}_{j}"] = Forall(
                x,
                Not(
                    And(
                        cores[i](x),
                        cores[j](x),
                    )
                ),
            )

    axiomas["forma_cobertura"] = Forall(
        x,
        Or(
            Or(
                Or(formas[0](x), formas[1](x)),
                Or(formas[2](x), formas[3](x)),
            ),
            formas[4](x),
        ),
    )

    for i in range(len(formas)):
        for j in range(i + 1, len(formas)):
            axiomas[f"forma_exclusividade_{i}_{j}"] = Forall(
                x,
                Not(
                    And(
                        formas[i](x),
                        formas[j](x),
                    )
                ),
            )

    axiomas["tamanho_cobertura"] = Forall(
        x,
        Or(
            isSmall(x),
            isBig(x),
        ),
    )

    axiomas["tamanho_exclusividade"] = Forall(
        x,
        Not(
            And(
                isSmall(x),
                isBig(x),
            )
        ),
    )

    return axiomas


def axiomas_espaciais_horizontais(
    variaveis: dict[str, ltn.Variable],
    predicados: dict[str, ltn.Predicate],
    operadores: dict[str, Any],
) -> dict[str, Any]:
    """
    Cria axiomas para raciocinio espacial horizontal.

    Relacoes envolvidas:
    - leftOf(x, y)
    - rightOf(x, y)

    Axiomas:
    1. Irreflexividade:
       um objeto nao pode estar a esquerda ou a direita de si mesmo.

    2. Assimetria:
       se x esta a esquerda de y, entao y nao esta a esquerda de x.
       se x esta a direita de y, entao y nao esta a direita de x.

    3. Inverso:
       se x esta a esquerda de y, entao y esta a direita de x.
       se x esta a direita de y, entao y esta a esquerda de x.

    4. Transitividade:
       se x esta a esquerda de y e y esta a esquerda de z,
       entao x esta a esquerda de z.
    """

    x = variaveis["x"]
    y = variaveis["y"]
    z = variaveis["z"]

    Not = operadores["Not"]
    And = operadores["And"]
    Implies = operadores["Implies"]
    Forall = operadores["Forall"]

    leftOf = predicados["leftOf"]
    rightOf = predicados["rightOf"]

    axiomas: dict[str, Any] = {}

    axiomas["left_irreflexividade"] = Forall(
        x,
        Not(leftOf(x, x)),
    )

    axiomas["right_irreflexividade"] = Forall(
        x,
        Not(rightOf(x, x)),
    )

    axiomas["left_assimetria"] = Forall(
        [x, y],
        Implies(
            leftOf(x, y),
            Not(leftOf(y, x)),
        ),
    )

    axiomas["right_assimetria"] = Forall(
        [x, y],
        Implies(
            rightOf(x, y),
            Not(rightOf(y, x)),
        ),
    )

    # Equivalencia original:
    # leftOf(x, y) <=> rightOf(y, x)
    #
    # Correção:
    # Em vez de usar uma equivalencia composta com equivale(...),
    # quebramos em duas implicacoes independentes.
    # Isso evita conflito interno de shape/broadcasting no LTNtorch.

    axiomas["left_implica_right_inverso"] = Forall(
        [x, y],
        Implies(
            leftOf(x, y),
            rightOf(y, x),
        ),
    )

    axiomas["right_implica_left_inverso"] = Forall(
        [x, y],
        Implies(
            rightOf(x, y),
            leftOf(y, x),
        ),
    )

    axiomas["left_transitividade"] = Forall(
        [x, y, z],
        Implies(
            And(
                leftOf(x, y),
                leftOf(y, z),
            ),
            leftOf(x, z),
        ),
    )

    axiomas["right_transitividade"] = Forall(
        [x, y, z],
        Implies(
            And(
                rightOf(x, y),
                rightOf(y, z),
            ),
            rightOf(x, z),
        ),
    )

    return axiomas


def axiomas_espaciais_verticais(
    variaveis: dict[str, ltn.Variable],
    predicados: dict[str, ltn.Predicate],
    operadores: dict[str, Any],
) -> dict[str, Any]:
    """
    Cria axiomas para raciocinio espacial vertical.

    Relacoes envolvidas:
    - below(x, y)
    - above(x, y)

    Axiomas:
    1. Irreflexividade:
       um objeto nao pode estar abaixo ou acima de si mesmo.

    2. Assimetria:
       se x esta abaixo de y, entao y nao esta abaixo de x.
       se x esta acima de y, entao y nao esta acima de x.

    3. Inverso:
       se x esta abaixo de y, entao y esta acima de x.
       se x esta acima de y, entao y esta abaixo de x.

    4. Transitividade:
       se x esta abaixo de y e y esta abaixo de z,
       entao x esta abaixo de z.
    """

    x = variaveis["x"]
    y = variaveis["y"]
    z = variaveis["z"]

    Not = operadores["Not"]
    And = operadores["And"]
    Implies = operadores["Implies"]
    Forall = operadores["Forall"]

    below = predicados["below"]
    above = predicados["above"]

    axiomas: dict[str, Any] = {}

    axiomas["below_irreflexividade"] = Forall(
        x,
        Not(below(x, x)),
    )

    axiomas["above_irreflexividade"] = Forall(
        x,
        Not(above(x, x)),
    )

    axiomas["below_assimetria"] = Forall(
        [x, y],
        Implies(
            below(x, y),
            Not(below(y, x)),
        ),
    )

    axiomas["above_assimetria"] = Forall(
        [x, y],
        Implies(
            above(x, y),
            Not(above(y, x)),
        ),
    )

    # Equivalencia original:
    # below(x, y) <=> above(y, x)
    #
    # Correção:
    # Quebramos em duas implicacoes independentes.

    axiomas["below_implica_above_inverso"] = Forall(
        [x, y],
        Implies(
            below(x, y),
            above(y, x),
        ),
    )

    axiomas["above_implica_below_inverso"] = Forall(
        [x, y],
        Implies(
            above(x, y),
            below(y, x),
        ),
    )

    axiomas["below_transitividade"] = Forall(
        [x, y, z],
        Implies(
            And(
                below(x, y),
                below(y, z),
            ),
            below(x, z),
        ),
    )

    axiomas["above_transitividade"] = Forall(
        [x, y, z],
        Implies(
            And(
                above(x, y),
                above(y, z),
            ),
            above(x, z),
        ),
    )

    return axiomas


def axiomas_proximidade_e_composicao(
    variaveis: dict[str, ltn.Variable],
    predicados: dict[str, ltn.Predicate],
    operadores: dict[str, Any],
) -> dict[str, Any]:
    """
    Cria axiomas adicionais para proximidade e raciocinio composto.

    Axiomas:
    1. closeTo e irreflexivo:
       por decisao de modelagem, nao consideramos que um objeto esteja proximo de si mesmo.

    2. closeTo e simetrico:
       se x esta proximo de y, entao y esta proximo de x.
       A simetria e representada por duas implicacoes independentes.

    3. Regra dos triangulos proximos:
       se dois objetos sao triangulos e estao proximos,
       entao devem ter o mesmo tamanho.

    4. canStack implica above:
       se x pode ser empilhado sobre y,
       entao x deve estar acima de y.

    5. canStack nao deve ser verdadeiro quando a base e cone ou triangulo.
    """

    x = variaveis["x"]
    y = variaveis["y"]

    Not = operadores["Not"]
    And = operadores["And"]
    Implies = operadores["Implies"]
    Forall = operadores["Forall"]

    closeTo = predicados["closeTo"]
    sameSize = predicados["sameSize"]
    isTriangle = predicados["isTriangle"]
    isCone = predicados["isCone"]
    above = predicados["above"]
    canStack = predicados["canStack"]

    axiomas: dict[str, Any] = {}

    axiomas["close_irreflexividade"] = Forall(
        x,
        Not(closeTo(x, x)),
    )

    # Equivalencia original:
    # closeTo(x, y) <=> closeTo(y, x)
    #
    # Correção:
    # Quebramos em duas implicacoes independentes.

    axiomas["close_implica_close_inverso_1"] = Forall(
        [x, y],
        Implies(
            closeTo(x, y),
            closeTo(y, x),
        ),
    )

    axiomas["close_implica_close_inverso_2"] = Forall(
        [x, y],
        Implies(
            closeTo(y, x),
            closeTo(x, y),
        ),
    )

    axiomas["triangulos_proximos_mesmo_tamanho"] = Forall(
        [x, y],
        Implies(
            And(
                And(
                    isTriangle(x),
                    isTriangle(y),
                ),
                closeTo(x, y),
            ),
            sameSize(x, y),
        ),
    )

    axiomas["can_stack_implica_above"] = Forall(
        [x, y],
        Implies(
            canStack(x, y),
            above(x, y),
        ),
    )

    axiomas["can_stack_base_nao_cone"] = Forall(
        [x, y],
        Implies(
            canStack(x, y),
            Not(isCone(y)),
        ),
    )

    axiomas["can_stack_base_nao_triangulo"] = Forall(
        [x, y],
        Implies(
            canStack(x, y),
            Not(isTriangle(y)),
        ),
    )

    return axiomas


def axiomas_inbetween(
    variaveis: dict[str, ltn.Variable],
    predicados: dict[str, ltn.Predicate],
    operadores: dict[str, Any],
) -> dict[str, Any]:
    """
    Cria axiomas para a relacao inBetween(x, y, z).

    A ideia e que, se x esta entre y e z, entao deve existir uma configuracao espacial coerente.

    Para evitar restringir demais o modelo, consideramos duas possibilidades:

    1. Configuracao horizontal:
       x esta entre y e z no eixo horizontal.

    2. Configuracao vertical:
       x esta entre y e z no eixo vertical.

    Isso fica mais alinhado com o ground truth gerado anteriormente, pois o inBetween
    foi pensado como uma relacao que pode acontecer por eixo horizontal, eixo vertical
    ou por interpretacao geometrica de segmento.
    """

    x = variaveis["x"]
    y = variaveis["y"]
    z = variaveis["z"]

    And = operadores["And"]
    Or = operadores["Or"]
    Implies = operadores["Implies"]
    Forall = operadores["Forall"]

    leftOf = predicados["leftOf"]
    rightOf = predicados["rightOf"]
    below = predicados["below"]
    above = predicados["above"]
    inBetween = predicados["inBetween"]

    axiomas: dict[str, Any] = {}

    configuracao_horizontal = Or(
        And(
            leftOf(y, x),
            rightOf(z, x),
        ),
        And(
            leftOf(z, x),
            rightOf(y, x),
        ),
    )

    configuracao_vertical = Or(
        And(
            below(y, x),
            above(z, x),
        ),
        And(
            below(z, x),
            above(y, x),
        ),
    )

    configuracao_espacial = Or(
        configuracao_horizontal,
        configuracao_vertical,
    )

    axiomas["inbetween_implica_configuracao_espacial"] = Forall(
        [x, y, z],
        Implies(
            inBetween(x, y, z),
            configuracao_espacial,
        ),
    )

    return axiomas


def criar_base_conhecimento(
    tensor_objetos: torch.Tensor,
    predicados: dict[str, ltn.Predicate],
    operadores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Cria a base de conhecimento completa.

    Retorna um dicionario com todos os axiomas nomeados.
    Isso permite avaliar:
    - satAgg geral;
    - satisfatibilidade individual por axioma;
    - quais regras estao sendo violadas.
    """

    if operadores is None:
        operadores = criar_operadores_logicos()

    variaveis = criar_variaveis_ltn(tensor_objetos)

    axiomas: dict[str, Any] = {}

    axiomas.update(
        axiomas_taxonomia(
            variaveis=variaveis,
            predicados=predicados,
            operadores=operadores,
        )
    )

    axiomas.update(
        axiomas_espaciais_horizontais(
            variaveis=variaveis,
            predicados=predicados,
            operadores=operadores,
        )
    )

    axiomas.update(
        axiomas_espaciais_verticais(
            variaveis=variaveis,
            predicados=predicados,
            operadores=operadores,
        )
    )

    axiomas.update(
        axiomas_proximidade_e_composicao(
            variaveis=variaveis,
            predicados=predicados,
            operadores=operadores,
        )
    )

    axiomas.update(
        axiomas_inbetween(
            variaveis=variaveis,
            predicados=predicados,
            operadores=operadores,
        )
    )

    return axiomas


def calcular_sat_agg(
    axiomas: dict[str, Any],
    operadores: dict[str, Any],
) -> Any:
    """
    Calcula o satAgg geral da base de conhecimento.

    satAgg representa o grau agregado de satisfacao da base de conhecimento.
    Quanto mais proximo de 1, mais as regras estao sendo satisfeitas.
    """

    SatAgg = operadores["SatAgg"]

    return SatAgg(*list(axiomas.values()))


def avaliar_axiomas_individualmente(
    axiomas: dict[str, Any],
) -> dict[str, float]:
    """
    Extrai a satisfatibilidade individual de cada axioma.

    Essa funcao e util para diagnostico.
    Se o satAgg geral estiver baixo, conseguimos descobrir qual regra esta causando problema.
    """

    resultados = {}

    for nome, valor in axiomas.items():
        resultados[nome] = float(valor.value.detach().cpu().item())

    return resultados