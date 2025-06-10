import numpy as np

class HillClimbing:
    """
    implementação do algoritmo de Subida de Encosta (Hill Climbing)
    """

    def __init__(self, funcao, limites, maximizar, epsilon=0.1, max_iter=1000, paciencia=100, max_vizinhos=100):
        self.funcao = funcao
        self.limites = limites  # [[min_x1, max_x1], [min_x2, max_x2]]
        self.maximizar = maximizar  # True para maximização, False para minimização
        self.epsilon = epsilon  # raio da vizinhança
        self.max_iter = max_iter
        self.paciencia = paciencia  # n de iterações sem melhoria antes de parar
        self.max_vizinhos = max_vizinhos  # vizinhos testados por iteração

    def executar(self):
        dim = len(self.limites)
        # inicializa no limite inferior do domínio (conforme especificação)
        x_melhor = np.array([lim[0] for lim in self.limites])
        f_melhor = self.funcao(*x_melhor)
        historico = [x_melhor.copy()]

        sem_melhoria = 0
        iteracao = 0

        while iteracao < self.max_iter and sem_melhoria < self.paciencia:
            melhorou = False

            # testa até max_vizinhos candidatos na vizinhança
            for _ in range(self.max_vizinhos):
                # gera candidato na vizinhança de x_melhor
                candidato = np.zeros(dim)
                for i in range(dim):
                    inf = max(self.limites[i][0], x_melhor[i] - self.epsilon)
                    sup = min(self.limites[i][1], x_melhor[i] + self.epsilon)
                    candidato[i] = np.random.uniform(inf, sup)

                f_candidato = self.funcao(*candidato)

                # verifica se há melhoria
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

            # atualiza critério de parada
            if melhorou:
                sem_melhoria = 0
            else:
                sem_melhoria += 1

            historico.append(x_melhor.copy())
            iteracao += 1

        return x_melhor, f_melhor, historico