import numpy as np

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