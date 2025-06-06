import numpy as np
from otimizacao import HillClimbing
import matplotlib.pyplot as plt

# Função 1 - queremos minimizar, então invertemos o sinal para maximizar
def f1(x1, x2):
    return - (x1**2 + x2**2)

# Restrições das variáveis
restricoes = np.array([
    [-100, 100],  # x1
    [-100, 100]   # x2
])

# Armazenar os melhores resultados
resultados = []

# Rodar 100 vezes
for _ in range(100):
    hc = HillClimbing(
        max_it=1000,
        max_viz=100,
        epsilon=0.1,
        func=f1,
        restricoes=restricoes
    )
    x_final, f_final = hc.search()  # f_final já vem com sinal corrigido
    resultados.append((round(x_final[0], 2), round(x_final[1], 2), round(f_final, 2)))

# Encontrar as soluções mais frequentes "na mão", sem Counter
# Criar lista única com strings para comparar
labels = [f"{x[0]},{x[1]},{x[2]}" for x in resultados]
valores, contagens = np.unique(labels, return_counts=True)

# Mostrar as 5 soluções mais comuns
print("Top 5 soluções mais comuns:")
top_5_indices = np.argsort(-contagens)[:5]
for i in top_5_indices:
    print(f"Solução: ({valores[i]}) | Frequência: {contagens[i]}")
