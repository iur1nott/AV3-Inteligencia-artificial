import numpy as np

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