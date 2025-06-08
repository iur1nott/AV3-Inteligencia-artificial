import random
import math

def conta_ataques(solucao):
    """
    Calcula o número de pares de rainhas que se atacam mutuamente.
    solucao: lista onde o índice representa a coluna e o valor a linha
    """
    num_ataques = 0
    n = len(solucao)
    for col1 in range(n):
        for col2 in range(col1 + 1, n):
            mesma_linha = solucao[col1] == solucao[col2]
            mesma_diagonal = abs(solucao[col1] - solucao[col2]) == abs(col1 - col2)
            if mesma_linha or mesma_diagonal:
                num_ataques += 1
    return num_ataques

def funcao_objetivo(solucao):
    """
    Calcula o valor da função objetivo a ser maximizada (28 - ataques)
    Retorna 28 quando não há ataques (solução ótima)
    """
    return 28 - conta_ataques(solucao)

def gera_solucao_inicial(n=8):
    """Gera uma solução inicial aleatória com uma rainha por coluna"""
    return random.sample(range(1, n+1), n)

def gera_vizinho(solucao):
    """
    Gera uma solução vizinha através da troca de duas posições aleatórias,
    mantendo uma rainha por coluna
    """
    viz = solucao.copy()
    i, j = random.sample(range(len(solucao)), 2)
    viz[i], viz[j] = viz[j], viz[i]
    return viz

def tempera_simulada(n=8, temp_inicial=10.0, fator_resfriamento=0.99, max_iter=100000):
    """
    Implementa o algoritmo de Têmpera Simulada para o problema das 8 rainhas
    
    Parâmetros:
    n: Número de rainhas (padrão: 8)
    temp_inicial: Temperatura inicial
    fator_resfriamento: Fator de decaimento da temperatura (0 < α < 1)
    max_iter: Número máximo de iterações
    
    Retorna:
    melhor_solucao: Melhor solução encontrada
    num_ataques: Quantidade de ataques na melhor solução
    """
    solucao_corrente = gera_solucao_inicial(n)
    melhor_solucao = solucao_corrente
    temperatura = temp_inicial
    
    for iteracao in range(1, max_iter+1):
        # Critério de parada por solução ótima
        if conta_ataques(melhor_solucao) == 0:
            break
        
        vizinho = gera_vizinho(solucao_corrente)
        delta = funcao_objetivo(vizinho) - funcao_objetivo(solucao_corrente)
        
        # Aceita soluções melhores ou soluções piores com probabilidade e^(Δ/T)
        if delta > 0 or random.random() < math.exp(delta / temperatura):
            solucao_corrente = vizinho
            if funcao_objetivo(solucao_corrente) > funcao_objetivo(melhor_solucao):
                melhor_solucao = solucao_corrente
        
        # Resfriamento exponencial
        temperatura *= fator_resfriamento
        
        # Parada por temperatura mínima
        if temperatura < 1e-6:
            break
    
    return melhor_solucao, conta_ataques(melhor_solucao)

def imprime_tabuleiro(solucao):
    """Exibe visualmente o tabuleiro com as rainhas posicionadas"""
    n = len(solucao)
    print("\nTabuleiro:")
    for linha in range(n, 0, -1):
        row = ""
        for coluna in range(n):
            row += " ♛ " if solucao[coluna] == linha else " . "
        print(row)
    print(f"\nConfiguração (colunas 1-8 → linhas): {solucao}")
    print(f"Total de ataques: {conta_ataques(solucao)}")

def encontrar_todas_solucoes():
    """
    Executa múltiplas vezes o algoritmo até encontrar as 92 soluções únicas
    Retorna estatísticas sobre o processo
    """
    solucoes_unicas = set()
    total_execucoes = 0
    total_iteracoes = 0
    
    while len(solucoes_unicas) < 92:
        solucao, ataques = tempera_simulada(max_iter=5000)
        total_execucoes += 1
        
        # Considera apenas soluções ótimas
        if ataques == 0:
            # Usa tupla para ser hasheável
            solucao_tupla = tuple(solucao)
            if solucao_tupla not in solucoes_unicas:
                solucoes_unicas.add(solucao_tupla)
                print(f"Soluções encontradas: {len(solucoes_unicas)}/92")
        
        # Contabiliza iterações (aproximação)
        total_iteracoes += 5000  # Valor máximo por execução
    
    return {
        "solucoes_encontradas": len(solucoes_unicas),
        "total_execucoes": total_execucoes,
        "total_iteracoes": total_iteracoes,
        "custo_medio": total_iteracoes / total_execucoes
    }

if __name__ == "__main__":
    # Parte 1: Encontrar uma solução ótima
    solucao_otima, num_ataques = tempera_simulada()
    
    if num_ataques == 0:
        print("\nSolução ótima encontrada!")
        imprime_tabuleiro(solucao_otima)
    else:
        print("\nMelhor solução encontrada (não ótima):")
        imprime_tabuleiro(solucao_otima)
    
    # Parte 2: Encontrar todas as 92 soluções
    print("\nIniciando busca pelas 92 soluções únicas...")
    estatisticas = encontrar_todas_solucoes()
    
    print("\nEstatísticas da busca completa:")
    print(f"Execuções do algoritmo: {estatisticas['total_execucoes']}")
    print(f"Iterações totais: {estatisticas['total_iteracoes']}")
    print(f"Custo médio por execução: {estatisticas['custo_medio']:.0f} iterações")