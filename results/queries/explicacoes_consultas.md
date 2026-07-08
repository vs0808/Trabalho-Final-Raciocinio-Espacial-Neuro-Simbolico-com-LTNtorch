# Explicações das consultas LTN

Este arquivo apresenta as respostas das consultas compostas executadas sobre o modelo LTN treinado.
Os valores de verdade estão no intervalo `[0, 1]`, em que valores próximos de `1` indicam maior satisfação da consulta.

## 1. Consultas principais

### consulta_1 — Existe algum objeto pequeno abaixo de um cilindro e à esquerda de um quadrado?

**Fórmula:** `∃x(IsSmall(x) ∧ ∃y(IsCylinder(y) ∧ Below(x,y)) ∧ ∃z(IsSquare(z) ∧ LeftOf(x,z)))`

**Valor de verdade:** `0.7274`

**Resposta:** `Sim`

**Melhor evidência encontrada:**

- objeto_x: Objeto 21 (vermelho, triangulo, pequeno, x=0.0583, y=0.2814)
- objeto_y: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
- objeto_z: Objeto 15 (azul, quadrado, grande, x=0.6649, y=0.7052)

**Componentes da evidência:**
- `isSmall(21)` = `1.0000`
- `isCylinder(3)` = `1.0000`
- `below(21,3)` = `0.8372`
- `isSquare(15)` = `1.0000`
- `leftOf(21,15)` = `0.8689`

---

### consulta_2 — Existe um cone verde entre dois objetos?

**Fórmula:** `∃x,y,z(IsCone(x) ∧ IsGreen(x) ∧ InBetween(x,y,z))`

**Valor de verdade:** `0.8392`

**Resposta:** `Sim`

**Melhor evidência encontrada:**

- objeto_x: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
- objeto_y: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
- objeto_z: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)

**Componentes da evidência:**
- `isCone(22)` = `1.0000`
- `isGreen(22)` = `1.0000`
- `inBetween(22,24,6)` = `0.8392`

---

### consulta_3 — Se dois triângulos estão próximos, eles têm o mesmo tamanho?

**Fórmula:** `∀x,y((IsTriangle(x) ∧ IsTriangle(y) ∧ CloseTo(x,y)) → SameSize(x,y))`

**Valor de verdade:** `0.9266`

**Resposta:** `Sim`

**Pior caso avaliado para a regra universal:**

- objeto_x: Objeto 0 (verde, triangulo, pequeno, x=0.7740, y=0.4389)
- objeto_y: Objeto 17 (vermelho, triangulo, grande, x=0.6684, y=0.4711)

**Componentes do pior caso:**
- `isTriangle(0)` = `1.0000`
- `isTriangle(17)` = `1.0000`
- `closeTo(0,17)` = `0.9423`
- `sameSize(0,17)` = `0.0000`

- Premissa = `0.9423`
- Conclusão sameSize = `0.0000`
- Implicação = `0.0577`

**Caso com premissa mais ativa:**

- objeto_x: Objeto 0 (verde, triangulo, pequeno, x=0.7740, y=0.4389)
- objeto_y: Objeto 17 (vermelho, triangulo, grande, x=0.6684, y=0.4711)

**Componentes do caso com premissa mais ativa:**
- `isTriangle(0)` = `1.0000`
- `isTriangle(17)` = `1.0000`
- `closeTo(0,17)` = `0.9423`
- `sameSize(0,17)` = `0.0000`

---

### consulta_4 — Existe algum objeto que pode ser empilhado sobre outro?

**Fórmula:** `∃x,y CanStack(x,y)`

**Valor de verdade:** `0.9499`

**Resposta:** `Sim`

**Melhor evidência encontrada:**

- objeto_x: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
- objeto_y: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)

**Componentes da evidência:**
- `canStack(6,14)` = `0.9499`

---

## 2. Top pares por relação binária

### Relação `leftOf`

1. `leftOf(8,3)` = `0.9953`
   - A: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)
   - B: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
2. `leftOf(24,3)` = `0.9936`
   - A: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - B: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
3. `leftOf(1,3)` = `0.9919`
   - A: Objeto 1 (azul, cone, grande, x=0.0942, y=0.9756)
   - B: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
4. `leftOf(8,5)` = `0.9882`
   - A: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)
   - B: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
5. `leftOf(8,18)` = `0.9858`
   - A: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)
   - B: Objeto 18 (verde, cilindro, pequeno, x=0.7650, y=0.6347)
6. `leftOf(8,6)` = `0.9839`
   - A: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)
   - B: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
7. `leftOf(8,11)` = `0.9826`
   - A: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)
   - B: Objeto 11 (vermelho, circulo, grande, x=0.6698, y=0.4372)
8. `leftOf(24,5)` = `0.9817`
   - A: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - B: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
9. `leftOf(21,3)` = `0.9813`
   - A: Objeto 21 (vermelho, triangulo, pequeno, x=0.0583, y=0.2814)
   - B: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
10. `leftOf(24,18)` = `0.9808`
   - A: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - B: Objeto 18 (verde, cilindro, pequeno, x=0.7650, y=0.6347)

### Relação `rightOf`

1. `rightOf(3,8)` = `0.9951`
   - A: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
   - B: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)
2. `rightOf(3,24)` = `0.9946`
   - A: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
   - B: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
3. `rightOf(3,1)` = `0.9875`
   - A: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
   - B: Objeto 1 (azul, cone, grande, x=0.0942, y=0.9756)
4. `rightOf(5,8)` = `0.9863`
   - A: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
   - B: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)
5. `rightOf(3,21)` = `0.9848`
   - A: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
   - B: Objeto 21 (vermelho, triangulo, pequeno, x=0.0583, y=0.2814)
6. `rightOf(18,8)` = `0.9839`
   - A: Objeto 18 (verde, cilindro, pequeno, x=0.7650, y=0.6347)
   - B: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)
7. `rightOf(5,24)` = `0.9834`
   - A: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
   - B: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
8. `rightOf(18,24)` = `0.9817`
   - A: Objeto 18 (verde, cilindro, pequeno, x=0.7650, y=0.6347)
   - B: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
9. `rightOf(11,8)` = `0.9794`
   - A: Objeto 11 (vermelho, circulo, grande, x=0.6698, y=0.4372)
   - B: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)
10. `rightOf(6,8)` = `0.9773`
   - A: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
   - B: Objeto 8 (azul, cone, pequeno, x=0.1543, y=0.6830)

### Relação `below`

1. `below(10,6)` = `0.9142`
   - A: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
   - B: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
2. `below(14,6)` = `0.9044`
   - A: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
   - B: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
3. `below(14,22)` = `0.9001`
   - A: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
   - B: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
4. `below(10,5)` = `0.8992`
   - A: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
   - B: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
5. `below(10,22)` = `0.8956`
   - A: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
   - B: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
6. `below(2,6)` = `0.8946`
   - A: Objeto 2 (azul, cilindro, grande, x=0.7861, y=0.1281)
   - B: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
7. `below(10,3)` = `0.8945`
   - A: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
   - B: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
8. `below(2,22)` = `0.8944`
   - A: Objeto 2 (azul, cilindro, grande, x=0.7861, y=0.1281)
   - B: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
9. `below(4,22)` = `0.8915`
   - A: Objeto 4 (vermelho, cilindro, grande, x=0.4434, y=0.2272)
   - B: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
10. `below(10,23)` = `0.8892`
   - A: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
   - B: Objeto 23 (verde, quadrado, pequeno, x=0.4064, y=0.8140)

### Relação `above`

1. `above(6,14)` = `0.9538`
   - A: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
   - B: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
2. `above(22,14)` = `0.9441`
   - A: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
   - B: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
3. `above(22,10)` = `0.9384`
   - A: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
   - B: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
4. `above(23,14)` = `0.9370`
   - A: Objeto 23 (verde, quadrado, pequeno, x=0.4064, y=0.8140)
   - B: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
5. `above(6,10)` = `0.9324`
   - A: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
   - B: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
6. `above(1,10)` = `0.9294`
   - A: Objeto 1 (azul, cone, grande, x=0.0942, y=0.9756)
   - B: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
7. `above(23,10)` = `0.9283`
   - A: Objeto 23 (verde, quadrado, pequeno, x=0.4064, y=0.8140)
   - B: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
8. `above(1,14)` = `0.9251`
   - A: Objeto 1 (azul, cone, grande, x=0.0942, y=0.9756)
   - B: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
9. `above(3,14)` = `0.9191`
   - A: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
   - B: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
10. `above(5,10)` = `0.9176`
   - A: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
   - B: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)

### Relação `closeTo`

1. `closeTo(11,17)` = `0.9869`
   - A: Objeto 11 (vermelho, circulo, grande, x=0.6698, y=0.4372)
   - B: Objeto 17 (vermelho, triangulo, grande, x=0.6684, y=0.4711)
2. `closeTo(17,11)` = `0.9869`
   - A: Objeto 17 (vermelho, triangulo, grande, x=0.6684, y=0.4711)
   - B: Objeto 11 (vermelho, circulo, grande, x=0.6698, y=0.4372)
3. `closeTo(7,20)` = `0.9775`
   - A: Objeto 7 (azul, cilindro, pequeno, x=0.1946, y=0.4667)
   - B: Objeto 20 (azul, triangulo, pequeno, x=0.2146, y=0.4085)
4. `closeTo(20,7)` = `0.9775`
   - A: Objeto 20 (azul, triangulo, pequeno, x=0.2146, y=0.4085)
   - B: Objeto 7 (azul, cilindro, pequeno, x=0.1946, y=0.4667)
5. `closeTo(5,18)` = `0.9769`
   - A: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
   - B: Objeto 18 (verde, cilindro, pequeno, x=0.7650, y=0.6347)
6. `closeTo(18,5)` = `0.9769`
   - A: Objeto 18 (verde, cilindro, pequeno, x=0.7650, y=0.6347)
   - B: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
7. `closeTo(4,13)` = `0.9659`
   - A: Objeto 4 (vermelho, cilindro, grande, x=0.4434, y=0.2272)
   - B: Objeto 13 (azul, quadrado, grande, x=0.3875, y=0.2883)
8. `closeTo(13,4)` = `0.9659`
   - A: Objeto 13 (azul, quadrado, grande, x=0.3875, y=0.2883)
   - B: Objeto 4 (vermelho, cilindro, grande, x=0.4434, y=0.2272)
9. `closeTo(10,14)` = `0.9638`
   - A: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
   - B: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
10. `closeTo(14,10)` = `0.9638`
   - A: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
   - B: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)

### Relação `canStack`

1. `canStack(6,14)` = `0.9499`
   - A: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
   - B: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
2. `canStack(6,2)` = `0.9463`
   - A: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
   - B: Objeto 2 (azul, cilindro, grande, x=0.7861, y=0.1281)
3. `canStack(6,4)` = `0.9424`
   - A: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
   - B: Objeto 4 (vermelho, cilindro, grande, x=0.4434, y=0.2272)
4. `canStack(6,10)` = `0.9093`
   - A: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
   - B: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)
5. `canStack(22,14)` = `0.9045`
   - A: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
   - B: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
6. `canStack(6,13)` = `0.9036`
   - A: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
   - B: Objeto 13 (azul, quadrado, grande, x=0.3875, y=0.2883)
7. `canStack(1,14)` = `0.8943`
   - A: Objeto 1 (azul, cone, grande, x=0.0942, y=0.9756)
   - B: Objeto 14 (azul, circulo, grande, x=0.1398, y=0.1999)
8. `canStack(22,4)` = `0.8926`
   - A: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
   - B: Objeto 4 (vermelho, cilindro, grande, x=0.4434, y=0.2272)
9. `canStack(22,2)` = `0.8917`
   - A: Objeto 22 (verde, cone, grande, x=0.6619, y=0.5570)
   - B: Objeto 2 (azul, cilindro, grande, x=0.7861, y=0.1281)
10. `canStack(23,10)` = `0.8914`
   - A: Objeto 23 (verde, quadrado, pequeno, x=0.4064, y=0.8140)
   - B: Objeto 10 (azul, cilindro, pequeno, x=0.1895, y=0.1299)

## 3. Top triplas para `inBetween`

1. `inBetween(16,24,3)` = `0.9375`
   - X: Objeto 16 (vermelho, circulo, pequeno, x=0.4589, y=0.5687)
   - Y: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - Z: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
2. `inBetween(9,24,3)` = `0.9357`
   - X: Objeto 9 (azul, circulo, pequeno, x=0.3258, y=0.3705)
   - Y: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - Z: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
3. `inBetween(12,24,3)` = `0.9308`
   - X: Objeto 12 (azul, triangulo, pequeno, x=0.7003, y=0.3124)
   - Y: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - Z: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
4. `inBetween(16,3,24)` = `0.9284`
   - X: Objeto 16 (vermelho, circulo, pequeno, x=0.4589, y=0.5687)
   - Y: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
   - Z: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
5. `inBetween(9,3,24)` = `0.9274`
   - X: Objeto 9 (azul, circulo, pequeno, x=0.3258, y=0.3705)
   - Y: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
   - Z: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
6. `inBetween(5,24,3)` = `0.9160`
   - X: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
   - Y: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - Z: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
7. `inBetween(12,3,24)` = `0.9147`
   - X: Objeto 12 (azul, triangulo, pequeno, x=0.7003, y=0.3124)
   - Y: Objeto 3 (verde, cilindro, grande, x=0.9268, y=0.6439)
   - Z: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
8. `inBetween(9,24,18)` = `0.9119`
   - X: Objeto 9 (azul, circulo, pequeno, x=0.3258, y=0.3705)
   - Y: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - Z: Objeto 18 (verde, cilindro, pequeno, x=0.7650, y=0.6347)
9. `inBetween(9,24,6)` = `0.9114`
   - X: Objeto 9 (azul, circulo, pequeno, x=0.3258, y=0.3705)
   - Y: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - Z: Objeto 6 (verde, triangulo, grande, x=0.3545, y=0.9707)
10. `inBetween(9,24,5)` = `0.9102`
   - X: Objeto 9 (azul, circulo, pequeno, x=0.3258, y=0.3705)
   - Y: Objeto 24 (azul, cone, grande, x=0.0227, y=0.0900)
   - Z: Objeto 5 (vermelho, circulo, grande, x=0.8276, y=0.6317)
