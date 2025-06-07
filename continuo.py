import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from algoritmos import BuscaGlobal, BuscaLocal, HillClimbing


def calcular_moda(solucoes, decimais=2):
    """
    Calcula a moda (solução mais frequente) em um conjunto de soluções.
    Base teórica:
      - Usada para identificar a solução mais estável em execuções repetidas
      - Importante para algoritmos estocásticos com múltiplos mínimos locais
    """
    solucoes_arredondadas = [tuple(np.round(sol, decimais)) for sol in solucoes]
    frequencia = defaultdict(int)
    for sol in solucoes_arredondadas:
        frequencia[sol] += 1
    if not frequencia:
        return None
    moda = max(frequencia, key=frequencia.get)
    return np.array(moda)


# =====================================================
# Implementação das funções do trabalho (8 problemas)
# =====================================================

def funcao_1(x1, x2):
    return x1**2 + x2**2



def funcao_2(x1, x2):
    termo1 = np.exp(-(x1 ** 2 + x2 ** 2))
    termo2 = 2 * np.exp(-(x1 - 1.7) ** 2 + (x2 - 1.7) ** 2)
    return termo1 + termo2


def funcao_3(x1, x2):
    termo1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    termo2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return termo1 + termo2 + 20 + np.exp(1)



def funcao_4(x1, x2):
    termo1 = x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + 10
    termo2 = x2 ** 2 - 10 * np.cos(2 * np.pi * x2) + 10
    return termo1 + termo2


def funcao_5(x1, x2):
    termo1 = (x1 * np.cos(x1)) / 20
    termo2 = 2 * np.exp(-(x1)**2 - (x2 - 1) ** 2)
    termo3 = 0.01 * x1 * x2
    return termo1 + termo2 + termo3


def funcao_6(x1, x2):
    termo1 = x1 * np.sin(4 * np.pi * x1)
    termo2 = -x2 * np.sin(4 * np.pi * x2 + np.pi)
    return termo1 + termo2 + 1


def funcao_7(x1, x2):
    termo1 = -np.sin(x1) * (np.sin(x1 ** 2 / np.pi)) ** (2*10) # não sei pq tá escrito 2*10 em vez de 20 no trabalho, mas vamo deixar como tá escrito kkk
    termo2 = -np.sin(x2) * (np.sin(2 * x2 ** 2 / np.pi)) ** (2*10)
    return termo1 + termo2


def funcao_8(x1, x2):
    termo1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47))))
    termo2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    return termo1 + termo2


# Configurações para cada função
funcoes = {
    '1': (funcao_1, [[-100, 100], [-100, 100]], False),
    '2': (funcao_2, [[-2, 4], [-2, 5]], True),
    '3': (funcao_3, [[-8, 8], [-8, 8]], False),
    '4': (funcao_4, [[-5.12, 5.12], [-5.12, 5.12]], False),
    '5': (funcao_5, [[-10, 10], [-10, 10]], True),
    '6': (funcao_6, [[-1, 3], [-1, 3]], True),
    '7': (funcao_7, [[0, np.pi], [0, np.pi]], False),
    '8': (funcao_8, [[-200, 200], [-200, 200]], False)
}

# =====================================================
# Execução dos experimentos (100 rodadas por função)
# =====================================================
resultados = {}
epsilon = 0.1  # Valor inicial para HillClimbing
sigma = 0.5  # Valor inicial para BuscaLocal

for numero, (f, limites, maximizar) in funcoes.items():
    print(f"\nProcessando função: {numero}")
    resultados_algoritmo = {'HillClimbing': [], 'BuscaLocal': [], 'BuscaGlobal': []}

    for _ in range(100):
        algoritmo = HillClimbing(f, limites, maximizar, epsilon)
        solucao, _, _ = algoritmo.executar()
        resultados_algoritmo['HillClimbing'].append(solucao)

    for _ in range(100):
        algoritmo = BuscaLocal(f, limites, maximizar, sigma)
        solucao, _, _ = algoritmo.executar()
        resultados_algoritmo['BuscaLocal'].append(solucao)

    for _ in range(100):
        algoritmo = BuscaGlobal(f, limites, maximizar)
        solucao, _, _ = algoritmo.executar()
        resultados_algoritmo['BuscaGlobal'].append(solucao)

    resultados[numero] = resultados_algoritmo

# =====================================================
# Cálculo das modas e exibição dos resultados
# =====================================================
print("\nRESULTADOS - SOLUÇÕES MAIS FREQUENTES (MODA)")
print("=============================================")
for funcao, dados in resultados.items():
    print(f"\nFunção: {funcao}")
    print(f"  HillClimbing: {calcular_moda(dados['HillClimbing'])}")
    print(f"  BuscaLocal: {calcular_moda(dados['BuscaLocal'])}")
    print(f"  BuscaGlobal: {calcular_moda(dados['BuscaGlobal'])}")


# =====================================================
# Visualização gráfica (exemplo para função 'esfera')
# =====================================================
def visualizar_solucoes_3d(nome_funcao, resultados):
    """Gera visualização 3D das soluções encontradas"""
    f, limites, _ = funcoes[nome_funcao]

    # Prepara grid para superfície
    x1 = np.linspace(limites[0][0], limites[0][1], 100)
    x2 = np.linspace(limites[1][0], limites[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)

    fig = plt.figure(figsize=(15, 5))

    for i, algoritmo in enumerate(['HillClimbing', 'BuscaLocal', 'BuscaGlobal']):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        # Plota superfície da função
        ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

        # Plota soluções encontradas
        solucoes = np.array(resultados[nome_funcao][algoritmo])
        z_solucoes = f(solucoes[:, 0], solucoes[:, 1])
        ax.scatter(solucoes[:, 0], solucoes[:, 1], z_solucoes,
                   c='red', s=30, alpha=0.7, label='Soluções')

        # Configurações do gráfico
        ax.set_title(f'{algoritmo} - {nome_funcao}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1,x2)')
        ax.legend()

    plt.tight_layout()
    plt.show()


# Visualiza resultados para a primeira função
visualizar_solucoes_3d('1', resultados)