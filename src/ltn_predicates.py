from pathlib import Path

import torch
import torch.nn as nn
import ltn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


IDX_X = 0
IDX_Y = 1

IDX_RED = 2
IDX_GREEN = 3
IDX_BLUE = 4

IDX_CIRCLE = 5
IDX_SQUARE = 6
IDX_CYLINDER = 7
IDX_CONE = 8
IDX_TRIANGLE = 9

IDX_SIZE = 10


DEFAULT_TEMPERATURE = 0.05
DEFAULT_CLOSE_THRESHOLD = 0.25
DEFAULT_STABLE_HORIZONTAL_THRESHOLD = 0.12


class UnaryFeaturePredicate(nn.Module):
    """
    Predicado unário baseado diretamente em uma posição do vetor_11.

    Exemplo:
    - isCircle(x) lê a posição IDX_CIRCLE do vetor.
    - isBig(x) lê a posição IDX_SIZE do vetor.

    Como o vetor já está codificado em one-hot, essa leitura já retorna
    um valor compatível com verdade fuzzy:
    0.0 para falso e 1.0 para verdadeiro.
    """

    def __init__(self, feature_index: int):
        super().__init__()
        self.feature_index = feature_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x[..., self.feature_index : self.feature_index + 1]
        return torch.clamp(value, 0.0, 1.0)


class SmallPredicate(nn.Module):
    """
    Predicado isSmall(x).

    No vetor_11:
    - pequeno = 0.0
    - grande = 1.0

    Portanto:
    isSmall(x) = 1 - tamanho
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x[..., IDX_SIZE : IDX_SIZE + 1]
        return torch.clamp(1.0 - size, 0.0, 1.0)


class SameFeatureGroupPredicate(nn.Module):
    """
    Predicado genérico para verificar se dois objetos compartilham
    o mesmo grupo de atributos one-hot.

    Pode ser usado para:
    - sameColor(x,y)
    - sameShape(x,y)

    Como as categorias são one-hot, o produto escalar dos blocos retorna:
    - 1 se forem iguais
    - 0 se forem diferentes
    """

    def __init__(self, start_index: int, end_index: int):
        super().__init__()
        self.start_index = start_index
        self.end_index = end_index

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_features = x[..., self.start_index : self.end_index]
        y_features = y[..., self.start_index : self.end_index]

        value = torch.sum(x_features * y_features, dim=-1, keepdim=True)

        return torch.clamp(value, 0.0, 1.0)


class SameSizePredicate(nn.Module):
    """
    Predicado sameSize(x,y).

    Como tamanho é binário:
    - pequeno = 0.0
    - grande = 1.0

    A diferença absoluta será:
    - 0 se forem iguais
    - 1 se forem diferentes

    Então:
    sameSize = 1 - abs(size_x - size_y)
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        size_x = x[..., IDX_SIZE : IDX_SIZE + 1]
        size_y = y[..., IDX_SIZE : IDX_SIZE + 1]

        value = 1.0 - torch.abs(size_x - size_y)

        return torch.clamp(value, 0.0, 1.0)


class SmoothLeftOfPredicate(nn.Module):
    """
    Predicado fuzzy leftOf(x,y).

    Versão determinística booleana:
    x.pos_x < y.pos_x

    Versão fuzzy diferenciável:
    sigmoid((y_x - x_x) / temperature)

    Se y_x for muito maior que x_x, o valor tende a 1.
    Se y_x for menor ou igual a x_x, o valor tende a 0.
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_coord = x[..., IDX_X : IDX_X + 1]
        y_coord = y[..., IDX_X : IDX_X + 1]

        return torch.sigmoid((y_coord - x_coord) / self.temperature)


class SmoothRightOfPredicate(nn.Module):
    """
    Predicado fuzzy rightOf(x,y).

    Versão determinística booleana:
    x.pos_x > y.pos_x

    Versão fuzzy diferenciável:
    sigmoid((x_x - y_x) / temperature)
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_coord = x[..., IDX_X : IDX_X + 1]
        y_coord = y[..., IDX_X : IDX_X + 1]

        return torch.sigmoid((x_coord - y_coord) / self.temperature)


class SmoothBelowPredicate(nn.Module):
    """
    Predicado fuzzy below(x,y).

    Versão determinística booleana:
    x.pos_y < y.pos_y

    Versão fuzzy diferenciável:
    sigmoid((y_y - x_y) / temperature)
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_coord = x[..., IDX_Y : IDX_Y + 1]
        y_coord = y[..., IDX_Y : IDX_Y + 1]

        return torch.sigmoid((y_coord - x_coord) / self.temperature)


class SmoothAbovePredicate(nn.Module):
    """
    Predicado fuzzy above(x,y).

    Versão determinística booleana:
    x.pos_y > y.pos_y

    Versão fuzzy diferenciável:
    sigmoid((x_y - y_y) / temperature)
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_coord = x[..., IDX_Y : IDX_Y + 1]
        y_coord = y[..., IDX_Y : IDX_Y + 1]

        return torch.sigmoid((x_coord - y_coord) / self.temperature)


class SmoothCloseToPredicate(nn.Module):
    """
    Predicado fuzzy closeTo(x,y).

    closeTo(x,y) deve ser alto quando:
    - x e y são objetos diferentes;
    - a distância euclidiana entre eles é menor que o threshold.

    Para x == y, retornamos 0 para respeitar a irreflexividade.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_CLOSE_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
        same_object_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.same_object_epsilon = same_object_epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pos_x = x[..., IDX_X : IDX_Y + 1]
        pos_y = y[..., IDX_X : IDX_Y + 1]

        distance = torch.linalg.norm(pos_x - pos_y, dim=-1, keepdim=True)

        close_truth = torch.sigmoid((self.threshold - distance) / self.temperature)

        different_objects = (distance > self.same_object_epsilon).float()

        return torch.clamp(close_truth * different_objects, 0.0, 1.0)


class SmoothAlignedHorizontalPredicate(nn.Module):
    """
    Predicado fuzzy horizontallyAligned(x,y).

    Mede se dois objetos têm coordenadas y parecidas.
    """

    def __init__(
        self,
        tolerance: float = 0.05,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__()
        self.tolerance = tolerance
        self.temperature = temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff_y = torch.abs(x[..., IDX_Y : IDX_Y + 1] - y[..., IDX_Y : IDX_Y + 1])
        return torch.sigmoid((self.tolerance - diff_y) / self.temperature)


class SmoothAlignedVerticalPredicate(nn.Module):
    """
    Predicado fuzzy verticallyAligned(x,y).

    Mede se dois objetos têm coordenadas x parecidas.
    """

    def __init__(
        self,
        tolerance: float = 0.05,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__()
        self.tolerance = tolerance
        self.temperature = temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff_x = torch.abs(x[..., IDX_X : IDX_X + 1] - y[..., IDX_X : IDX_X + 1])
        return torch.sigmoid((self.tolerance - diff_x) / self.temperature)


class SmoothInBetweenPredicate(nn.Module):
    """
    Predicado fuzzy inBetween(x,y,z).

    Interpretação:
    x está entre y e z se:
    - sua coordenada x está entre as coordenadas x de y e z;
    - ou sua coordenada y está entre as coordenadas y de y e z.

    Também adicionamos uma aproximação de segmento:
    x deve estar dentro da caixa delimitada por y e z e próximo da linha entre y e z.

    Essa versão é diferenciável e pode ser usada dentro da LTN.
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def smooth_between(
        self,
        value: torch.Tensor,
        endpoint_a: torch.Tensor,
        endpoint_b: torch.Tensor,
    ) -> torch.Tensor:
        lower = torch.minimum(endpoint_a, endpoint_b)
        upper = torch.maximum(endpoint_a, endpoint_b)

        greater_than_lower = torch.sigmoid((value - lower) / self.temperature)
        lower_than_upper = torch.sigmoid((upper - value) / self.temperature)

        return greater_than_lower * lower_than_upper

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        x_x = x[..., IDX_X : IDX_X + 1]
        x_y = x[..., IDX_Y : IDX_Y + 1]

        y_x = y[..., IDX_X : IDX_X + 1]
        y_y = y[..., IDX_Y : IDX_Y + 1]

        z_x = z[..., IDX_X : IDX_X + 1]
        z_y = z[..., IDX_Y : IDX_Y + 1]

        between_x = self.smooth_between(x_x, y_x, z_x)
        between_y = self.smooth_between(x_y, y_y, z_y)

        axis_between = torch.maximum(between_x, between_y)

        return torch.clamp(axis_between, 0.0, 1.0)


class SmoothCanStackPredicate(nn.Module):
    """
    Predicado fuzzy canStack(x,y).

    Interpretação:
    x pode ser empilhado sobre y se:
    - x está acima de y;
    - y não é cone;
    - y não é triângulo;
    - x e y têm mesmo tamanho OU seus centroides estão horizontalmente próximos.
    """

    def __init__(
        self,
        stable_horizontal_threshold: float = DEFAULT_STABLE_HORIZONTAL_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__()
        self.stable_horizontal_threshold = stable_horizontal_threshold
        self.temperature = temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_y = x[..., IDX_Y : IDX_Y + 1]
        y_y = y[..., IDX_Y : IDX_Y + 1]

        above_truth = torch.sigmoid((x_y - y_y) / self.temperature)

        y_is_cone = y[..., IDX_CONE : IDX_CONE + 1]
        y_is_triangle = y[..., IDX_TRIANGLE : IDX_TRIANGLE + 1]

        base_is_valid = 1.0 - torch.clamp(torch.maximum(y_is_cone, y_is_triangle), 0.0, 1.0)

        size_x = x[..., IDX_SIZE : IDX_SIZE + 1]
        size_y = y[..., IDX_SIZE : IDX_SIZE + 1]

        same_size_truth = 1.0 - torch.abs(size_x - size_y)

        horizontal_distance = torch.abs(x[..., IDX_X : IDX_X + 1] - y[..., IDX_X : IDX_X + 1])

        centroid_stable_truth = torch.sigmoid(
            (self.stable_horizontal_threshold - horizontal_distance) / self.temperature
        )

        stable_truth = torch.maximum(same_size_truth, centroid_stable_truth)

        value = above_truth * base_is_valid * stable_truth

        return torch.clamp(value, 0.0, 1.0)


def criar_operadores_logicos_ltn() -> dict[str, object]:
    """
    Cria os operadores lógicos fuzzy usados pela LTN.

    Esses operadores seguem a mesma lógica do material de referência:
    - NotStandard
    - AndProd
    - OrProbSum
    - ImpliesReichenbach
    - Forall com AggregPMeanError
    - Exists com AggregPMean
    - SatAgg
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


def criar_predicados_ltn() -> dict[str, ltn.Predicate]:
    """
    Cria os predicados LTN principais do projeto.

    Os predicados retornam valores fuzzy entre 0 e 1.
    """

    predicados = {
        "isRed": ltn.Predicate(UnaryFeaturePredicate(IDX_RED)),
        "isGreen": ltn.Predicate(UnaryFeaturePredicate(IDX_GREEN)),
        "isBlue": ltn.Predicate(UnaryFeaturePredicate(IDX_BLUE)),
        "isCircle": ltn.Predicate(UnaryFeaturePredicate(IDX_CIRCLE)),
        "isSquare": ltn.Predicate(UnaryFeaturePredicate(IDX_SQUARE)),
        "isCylinder": ltn.Predicate(UnaryFeaturePredicate(IDX_CYLINDER)),
        "isCone": ltn.Predicate(UnaryFeaturePredicate(IDX_CONE)),
        "isTriangle": ltn.Predicate(UnaryFeaturePredicate(IDX_TRIANGLE)),
        "isSmall": ltn.Predicate(SmallPredicate()),
        "isBig": ltn.Predicate(UnaryFeaturePredicate(IDX_SIZE)),
        "sameColor": ltn.Predicate(SameFeatureGroupPredicate(IDX_RED, IDX_BLUE + 1)),
        "sameShape": ltn.Predicate(SameFeatureGroupPredicate(IDX_CIRCLE, IDX_TRIANGLE + 1)),
        "sameSize": ltn.Predicate(SameSizePredicate()),
        "leftOf": ltn.Predicate(SmoothLeftOfPredicate()),
        "rightOf": ltn.Predicate(SmoothRightOfPredicate()),
        "below": ltn.Predicate(SmoothBelowPredicate()),
        "above": ltn.Predicate(SmoothAbovePredicate()),
        "closeTo": ltn.Predicate(SmoothCloseToPredicate()),
        "horizontallyAligned": ltn.Predicate(SmoothAlignedHorizontalPredicate()),
        "verticallyAligned": ltn.Predicate(SmoothAlignedVerticalPredicate()),
        "inBetween": ltn.Predicate(SmoothInBetweenPredicate()),
        "canStack": ltn.Predicate(SmoothCanStackPredicate()),
    }

    return predicados


class TrainableUnaryPredicate(nn.Module):
    """
    Predicado unário treinável.

    Este modelo será útil posteriormente caso a gente queira que a rede aprenda
    propriedades a partir do vetor completo, em vez de simplesmente ler a posição
    one-hot correta.

    Entrada:
    vetor_11

    Saída:
    grau de verdade entre 0 e 1
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 16):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TrainableBinaryPredicate(nn.Module):
    """
    Predicado binário treinável.

    Recebe dois objetos, concatena os vetores e aprende uma relação.

    Entrada:
    vetor_11 de x + vetor_11 de y = 22 dimensões

    Saída:
    grau de verdade entre 0 e 1
    """

    def __init__(self, input_dim: int = 22, hidden_dim: int = 32):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, y], dim=-1)
        return self.model(inputs)


class TrainableTernaryPredicate(nn.Module):
    """
    Predicado ternário treinável.

    Recebe três objetos, concatena os vetores e aprende uma relação.

    Entrada:
    vetor_11 de x + vetor_11 de y + vetor_11 de z = 33 dimensões

    Saída:
    grau de verdade entre 0 e 1
    """

    def __init__(self, input_dim: int = 33, hidden_dim: int = 48):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, y, z], dim=-1)
        return self.model(inputs)


if __name__ == "__main__":
    predicados = criar_predicados_ltn()
    operadores = criar_operadores_logicos_ltn()

    print("Predicados LTN criados:")
    for nome in predicados:
        print(f"- {nome}")

    print("\nOperadores lógicos criados:")
    for nome in operadores:
        print(f"- {nome}")