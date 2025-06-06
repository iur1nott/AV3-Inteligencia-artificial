import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class HillClimbing:
    """
    Implementação do algoritmo de Subida de Encosta (Hill Climbing).
    Base teórica (páginas 24-25 do material):
      - Algoritmo de busca local que parte de um ponto inicial (limite inferior do domínio)
      - Explora vizinhança definida por |x - y| ≤ ε
      - Ideal para problemas unimodais/convexos
    """

    def __init__(self, funcao, limites, maximizar, epsilon=0.1, max_iter=1000, paciencia=100, max_vizinhos=100):
        self.funcao = funcao
        self.limites = limites  # [[min_x1, max_x1], [min_x2, max_x2]]
        self.maximizar = maximizar  # True para maximização, False para minimização
        self.epsilon = epsilon  # Raio da vizinhança
        self.max_iter = max_iter
        self.paciencia = paciencia  # Iterações sem melhoria antes de parar
        self.max_vizinhos = max_vizinhos  # Vizinhos testados por iteração

    def executar(self):
        dim = len(self.limites)
        # Inicializa no limite inferior do domínio (conforme especificação)
        x_melhor = np.array([lim[0] for lim in self.limites])
        f_melhor = self.funcao(*x_melhor)
        historico = [x_melhor.copy()]

        sem_melhoria = 0
        iteracao = 0

        while iteracao < self.max_iter and sem_melhoria < self.paciencia:
            melhorou = False

            # Testa até max_vizinhos candidatos na vizinhança
            for _ in range(self.max_vizinhos):
                # Gera candidato na vizinhança de x_melhor
                candidato = np.zeros(dim)
                for i in range(dim):
                    inf = max(self.limites[i][0], x_melhor[i] - self.epsilon)
                    sup = min(self.limites[i][1], x_melhor[i] + self.epsilon)
                    candidato[i] = np.random.uniform(inf, sup)

                f_candidato = self.funcao(*candidato)

                # Verifica se há melhoria
                if self.maximizar:
                    if f_candidato > f_melhor:
                        x_melhor = candidato.copy()
                        f_melhor = f_candidato
                        melhorou = True
                        break
                else:
                    if f_candidato < f_melhor:
                        x_melhor = candidato.copy()
                        f_melhor = f_candidato
                        melhorou = True
                        break

            # Atualiza critério de parada
            if melhorou:
                sem_melhoria = 0
            else:
                sem_melhoria += 1

            historico.append(x_melhor.copy())
            iteracao += 1

        return x_melhor, f_melhor, historico


class BuscaLocal:
    """
    Implementação da Busca Aleatória Local (LRS).
    Base teórica (páginas 37-39 do material):
      - Algoritmo estocástico que perturba a solução atual com ruído Gaussiano
      - Não utiliza gradientes, adequado para funções não-diferenciáveis
      - Mantém busca local ao explorar vizinhança próxima
    """

    def __init__(self, funcao, limites, maximizar, sigma=0.5, max_iter=1000, paciencia=100):
        self.funcao = funcao
        self.limites = limites
        self.maximizar = maximizar
        self.sigma = sigma  # Desvio padrão da perturbação
        self.max_iter = max_iter
        self.paciencia = paciencia

    def executar(self):
        dim = len(self.limites)
        # Solução inicial aleatória dentro dos limites
        x_melhor = np.array([np.random.uniform(lim[0], lim[1]) for lim in self.limites])
        f_melhor = self.funcao(*x_melhor)
        historico = [x_melhor.copy()]

        sem_melhoria = 0
        iteracao = 0

        while iteracao < self.max_iter and sem_melhoria < self.paciencia:
            # Gera candidato com perturbação Gaussiana
            candidato = x_melhor + np.random.normal(0, self.sigma, dim)

            # Aplica restrições de caixa (limites)
            for i in range(dim):
                if candidato[i] < self.limites[i][0]:
                    candidato[i] = self.limites[i][0]
                elif candidato[i] > self.limites[i][1]:
                    candidato[i] = self.limites[i][1]

            f_candidato = self.funcao(*candidato)

            # Atualiza solução se houver melhoria
            if self.maximizar:
                if f_candidato > f_melhor:
                    x_melhor = candidato.copy()
                    f_melhor = f_candidato
                    sem_melhoria = 0
                else:
                    sem_melhoria += 1
            else:
                if f_candidato < f_melhor:
                    x_melhor = candidato.copy()
                    f_melhor = f_candidato
                    sem_melhoria = 0
                else:
                    sem_melhoria += 1

            historico.append(x_melhor.copy())
            iteracao += 1

        return x_melhor, f_melhor, historico


class BuscaGlobal:
    """
    Implementação da Busca Aleatória Global (GRS).
    Base teórica (páginas 43-44 do material):
      - Algoritmo que explora uniformemente todo o espaço de busca
      - Não possui viés local, ideal para problemas multimodais
      - Mais eficiente em espaços de alta dimensão
    """

    def __init__(self, funcao, limites, maximizar, max_iter=1000, paciencia=100):
        self.funcao = funcao
        self.limites = limites
        self.maximizar = maximizar
        self.max_iter = max_iter
        self.paciencia = paciencia

    def executar(self):
        dim = len(self.limites)
        # Solução inicial aleatória
        x_melhor = np.array([np.random.uniform(lim[0], lim[1]) for lim in self.limites])
        f_melhor = self.funcao(*x_melhor)
        historico = [x_melhor.copy()]

        sem_melhoria = 0
        iteracao = 0

        while iteracao < self.max_iter and sem_melhoria < self.paciencia:
            # Gera candidato uniforme em todo o domínio
            candidato = np.array([np.random.uniform(lim[0], lim[1]) for lim in self.limites])
            f_candidato = self.funcao(*candidato)

            # Atualiza solução se houver melhoria
            if self.maximizar:
                if f_candidato > f_melhor:
                    x_melhor = candidato.copy()
                    f_melhor = f_candidato
                    sem_melhoria = 0
                else:
                    sem_melhoria += 1
            else:
                if f_candidato < f_melhor:
                    x_melhor = candidato.copy()
                    f_melhor = f_candidato
                    sem_melhoria = 0
                else:
                    sem_melhoria += 1

            historico.append(x_melhor.copy())
            iteracao += 1

        return x_melhor, f_melhor, historico


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

def funcao_esfera(x1, x2):
    return x1**2 + x2**2



def funcao_multimodal_1(x1, x2):
    """Problema 2: Maximização (página 3)"""
    termo1 = np.exp(-(x1 * 2 + x2 * 2))
    termo2 = 2 * np.exp(-((x1 - 1.7) * 2 + (x2 - 1.7) * 2))
    return termo1 + termo2


def funcao_ackley(x1, x2):
    termo1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    termo2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return termo1 + termo2 + 20 + np.exp(1)



def funcao_rastrigin(x1, x2):
    """Problema 4: Minimização (página 4)"""
    termo1 = x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + 10
    termo2 = x2 ** 2 - 10 * np.cos(2 * np.pi * x2) + 10
    return termo1 + termo2


def funcao_multimodal_2(x1, x2):
    """Problema 5: Maximização (página 4)"""
    termo1 = (x1 * np.cos(x1)) / 20
    termo2 = 2 * np.exp(-(x1 * 2) - (x2 - 1) * 2)
    termo3 = 0.01 * x1 * x2
    return termo1 + termo2 + termo3


def funcao_multimodal_3(x1, x2):
    """Problema 6: Maximização (página 5)"""
    termo1 = x1 * np.sin(4 * np.pi * x1)
    termo2 = -x2 * np.sin(4 * np.pi * x2 + np.pi)
    return termo1 + termo2 + 1


def funcao_multimodal_4(x1, x2):
    """Problema 7: Minimização (página 5)"""
    termo1 = -np.sin(x1) * (np.sin(x1 * 2 / np.pi)) * (20)
    termo2 = -np.sin(x2) * (np.sin(2 * x2 * 2 / np.pi)) * (20)
    return termo1 + termo2


def funcao_eggholder(x1, x2):
    """Problema 8: Minimização (página 6)"""
    termo1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47))))
    termo2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    return termo1 + termo2


# Configurações para cada função
funcoes = {
    'esfera': (funcao_esfera, [[-100, 100], [-100, 100]], False),
    'multimodal_1': (funcao_multimodal_1, [[-2, 4], [-2, 5]], True),
    'ackley': (funcao_ackley, [[-8, 8], [-8, 8]], False),
    'rastrigin': (funcao_rastrigin, [[-5.12, 5.12], [-5.12, 5.12]], False),
    'multimodal_2': (funcao_multimodal_2, [[-10, 10], [-10, 10]], True),
    'multimodal_3': (funcao_multimodal_3, [[-1, 3], [-1, 3]], True),
    'multimodal_4': (funcao_multimodal_4, [[0, np.pi], [0, np.pi]], False),
    'eggholder': (funcao_eggholder, [[-200, 200], [-200, 200]], False)
}

# =====================================================
# Execução dos experimentos (100 rodadas por função)
# =====================================================
resultados = {}
epsilon = 0.1  # Valor inicial para HillClimbing
sigma = 0.5  # Valor inicial para BuscaLocal

for nome, (f, limites, maximizar) in funcoes.items():
    print(f"\nProcessando função: {nome}")
    resultados_algoritmo = {'HillClimbing': [], 'BuscaLocal': [], 'BuscaGlobal': []}

    # Executa HillClimbing (100 repetições)
    for _ in range(100):
        algoritmo = HillClimbing(f, limites, maximizar, epsilon)
        solucao, _, _ = algoritmo.executar()
        resultados_algoritmo['HillClimbing'].append(solucao)

    # Executa BuscaLocal (100 repetições)
    for _ in range(100):
        algoritmo = BuscaLocal(f, limites, maximizar, sigma)
        solucao, _, _ = algoritmo.executar()
        resultados_algoritmo['BuscaLocal'].append(solucao)

    # Executa BuscaGlobal (100 repetições)
    for _ in range(100):
        algoritmo = BuscaGlobal(f, limites, maximizar)
        solucao, _, _ = algoritmo.executar()
        resultados_algoritmo['BuscaGlobal'].append(solucao)

    resultados[nome] = resultados_algoritmo

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
visualizar_solucoes_3d('esfera', resultados)