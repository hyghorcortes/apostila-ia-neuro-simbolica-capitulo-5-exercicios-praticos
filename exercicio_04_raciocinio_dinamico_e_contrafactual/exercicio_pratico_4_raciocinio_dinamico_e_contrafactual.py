# -*- coding: utf-8 -*-
"""
EXERCÍCIO PRÁTICO 4 — RACIOCÍNIO DINÂMICO: O DYNAMICS PREDICTOR E A INFERÊNCIA
                      CONTRAFACTUAL (NSDR)
=============================================================================

Capítulo 5 — "Introdução à IA Neuro-Simbólica — o Próximo Nível da IA".
Referência visual: Figura 5.5 (exemplo de programa do executor simbólico do
NSDR, para a pergunta "sem o objeto cinza, qual evento aconteceria?").

O QUE ESTE EXERCÍCIO DEMONSTRA
------------------------------
O NSCL do exercício anterior olhava para uma IMAGEM PARADA e respondia sobre
relações estáticas ("há um cubo à esquerda da esfera?"). O NSDR dá o passo
seguinte: troca imagens por VÍDEOS (o conjunto CLEVRER — clipes de 5 segundos,
25 quadros, objetos que entram, saem e colidem) e, com isso, troca relações
espaciais por CAUSALIDADE. Passam a existir quatro famílias de perguntas:

    DESCRITIVA   : "quantas colisões ocorreram?"       -> sobre o passado observado
    PREDITIVA    : "qual evento acontecerá em seguida?" -> sobre o futuro
    EXPLICATIVA  : "por que a colisão entre X e Y não aconteceria?"
    CONTRAFACTUAL: "sem o objeto cinza, qual evento aconteceria?"

As duas últimas são impossíveis para qualquer classificador puramente
perceptivo: elas perguntam sobre um mundo que NUNCA FOI FILMADO. Para responder
é preciso um motor de física — o sub-modelo Dynamics Predictor, inspirado no
PropNet — capaz de RE-SIMULAR o clipe a partir de um estado inicial alterado.

Este exercício constrói a cadeia inteira, do zero:

  (a) MUNDO FÍSICO 2D  — o "CLEVRER" do exercício: discos coloridos em uma
      arena retangular, colisões elásticas disco-disco e disco-parede
      conservando momento e energia, objetos entrando e saindo do campo de
      visão da câmera. É a única parte que conhece a verdade.
  (b) VIDEO PARSER     — o extrator quadro a quadro (o análogo do Mask R-CNN
      aplicado 25 vezes por clipe), com ruído de medição e OCLUSÃO: o que sai
      do campo de visão simplesmente não é observado.
  (c) DYNAMICS PREDICTOR — uma rede de relações treinada em numpy puro, com
      retropropagação escrita à mão: para cada par de objetos ela calcula um
      "efeito" (posições e velocidades relativas, raios), soma os efeitos sobre
      os vizinhos, junta um termo próprio e prevê o incremento de velocidade do
      objeto no próximo passo.
  (d) EXECUTOR SIMBÓLICO — ao contrário do executor QUASE-simbólico do NSCL
      (que propagava probabilidades), aqui o executor é TOTALMENTE SIMBÓLICO:
      opera sobre listas discretas de eventos com filtros e lógica. É ele que
      roda o programa da Figura 5.5.

E, por ser tudo simulado, o exercício faz algo que nenhum artigo consegue fazer
com dados reais: VALIDA o raciocínio contrafactual comparando o mundo
alternativo imaginado pela rede com o mundo alternativo obtido re-simulando a
FÍSICA VERDADEIRA sem o objeto removido.

RESTRIÇÕES DE IMPLEMENTAÇÃO
---------------------------
Somente a biblioteca padrão do Python 3 + numpy. Nenhum torch, tensorflow,
scipy, pandas ou matplotlib: a física, a rede de relações, o otimizador Adam, o
detector de eventos, a DSL e o executor simbólico são todos escritos do zero.
É justamente aí que mora o valor didático — você vê a máquina por dentro.

EXECUÇÃO
--------
    python3 exercicio_pratico_4_raciocinio_dinamico_e_contrafactual.py

Sem argumentos, sem arquivos de entrada, sem interação, sem rede. A saída é um
relatório de texto no terminal. A semente é fixa, então o relatório é sempre o
mesmo — você pode conferir seus números com os do colega ao lado.
"""

import math
import os
import textwrap

# As matrizes deste exercicio sao minusculas: o paralelismo do BLAS so
# acrescentaria sincronizacao. Uma thread e mais rapido e mais reprodutivel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTES GLOBAIS
# ---------------------------------------------------------------------------

SEMENTE = 42                 # semente fixa => saída 100% reprodutível
LARGURA = 78                 # largura das linhas do relatório

# --- O clipe de vídeo (as dimensões vêm do CLEVRER: 5 s, 25 quadros) --------
N_QUADROS = 25               # quadros por clipe
DT = 0.2                     # 0,2 s por quadro => 25 quadros = 5,0 s
QUADRO_ATUAL = 12            # "agora": tudo até aqui é observado, o resto é previsto

# --- A arena física e o campo de visão da câmera ----------------------------
# A arena tem paredes (colisão elástica). O campo de visão é um retângulo MENOR
# dentro dela: um objeto pode continuar existindo e ricocheteando FORA do campo,
# e nesses quadros ele simplesmente não é observado (oclusão).
ARENA_X = (0.0, 7.0)
ARENA_Y = (0.0, 4.5)
CAMPO_X = (0.9, 6.1)
CAMPO_Y = (0.6, 3.9)
MARGEM_INICIAL = 0.35        # os objetos nascem dentro do campo, com folga

# --- Os objetos -------------------------------------------------------------
CORES = ["vermelho", "azul", "cinza", "amarelo", "verde"]
N_MIN_OBJETOS, N_MAX_OBJETOS = 3, 5
MAX_VIZINHOS = N_MAX_OBJETOS - 1
RAIO_MIN, RAIO_MAX = 0.34, 0.46
RAPIDEZ_MIN, RAPIDEZ_MAX = 1.0, 2.0
# Todos os discos têm a MESMA massa: por isso a colisão elástica se resolve
# simplesmente TROCANDO as componentes normais das velocidades.

# --- O video parser ---------------------------------------------------------
RUIDO_POSICAO = 0.008        # desvio-padrão do erro de localização, por eixo

# --- O Dynamics Predictor ---------------------------------------------------
N_ATRIB_OBJETO = 12          # atributos próprios (velocidade, raio, folgas p/ paredes)
N_ATRIB_PAR = 9              # atributos relacionais de um par de objetos
UNIDADE_OCULTA = 24          # largura das camadas ocultas
EPOCAS = 60
TAMANHO_LOTE = 256
TAXA_APRENDIZADO = 0.006
FATOR_REFORCO_COLISAO = 3    # quantas cópias EXTRA de cada transição com impulso
LIMIAR_IMPULSO = 0.30        # |Δv| acima disso = "algo aconteceu" (colisão/parede)

# --- Os clipes --------------------------------------------------------------
N_CLIPES_TREINO = 40        # clipes com condições iniciais livres
N_CLIPES_COLISAO = 50        # clipes dirigidos: dois objetos em rota de colisão
N_CLIPES_TESTE = 30
MAX_CANDIDATOS_DEMO = 400    # orçamento de busca do clipe de demonstração

# --- O detector de eventos --------------------------------------------------
LIMIAR_CONTATO = 0.30        # folga máxima (em unidades) para considerar contato
TOLERANCIA_QUADROS = 2       # dois eventos "casam" se distam <= 2 quadros
CLIPES_NA_TABELA = 10        # linhas exibidas na tabela da seção [6]

# Contador didático: quantas vezes o extrator de quadros foi aplicado.
ESTATISTICAS_DO_PARSER = {"quadros_processados": 0, "deteccoes": 0, "oclusoes": 0}


# ===========================================================================
# PARTE 1 — O MUNDO FÍSICO 2D (o "CLEVRER" deste exercício)
# ===========================================================================

def sortear_cena(rng, n_objetos=None, em_rota_de_colisao=False):
    """Sorteia as condições iniciais de um clipe.

    Uma cena é um dicionário com quatro campos alinhados por índice de objeto:
    `cores` (nomes), `raios`, `pos0` (posições iniciais) e `vel0` (velocidades
    iniciais). As posições nascem dentro do campo de visão, com folga, para que
    todos os objetos estejam visíveis nos primeiros quadros.

    Com `em_rota_de_colisao=True` os dois primeiros objetos recebem velocidades
    apontadas um para o outro. Esses clipes formam o "currículo de colisões" do
    treino: sem eles, colisões seriam raras demais para a rede aprender.
    """
    if n_objetos is None:
        n_objetos = int(rng.integers(N_MIN_OBJETOS, N_MAX_OBJETOS + 1))

    cores = list(rng.permutation(CORES)[:n_objetos])
    raios = rng.uniform(RAIO_MIN, RAIO_MAX, size=n_objetos)

    limites_x = (CAMPO_X[0] + MARGEM_INICIAL + RAIO_MAX,
                 CAMPO_X[1] - MARGEM_INICIAL - RAIO_MAX)
    limites_y = (CAMPO_Y[0] + MARGEM_INICIAL + RAIO_MAX,
                 CAMPO_Y[1] - MARGEM_INICIAL - RAIO_MAX)

    # Amostragem por rejeição: nenhum par pode nascer sobreposto.
    posicoes = np.zeros((n_objetos, 2))
    for i in range(n_objetos):
        for _ in range(500):
            candidato = np.array([rng.uniform(*limites_x), rng.uniform(*limites_y)])
            if all(np.linalg.norm(candidato - posicoes[j]) > raios[i] + raios[j] + 0.12
                   for j in range(i)):
                posicoes[i] = candidato
                break
        else:
            posicoes[i] = candidato   # orçamento esgotado: aceita o último

    angulos = rng.uniform(0.0, 2.0 * math.pi, size=n_objetos)
    rapidez = rng.uniform(RAPIDEZ_MIN, RAPIDEZ_MAX, size=n_objetos)
    velocidades = np.column_stack([rapidez * np.cos(angulos),
                                   rapidez * np.sin(angulos)])

    if em_rota_de_colisao and n_objetos >= 2:
        direcao = posicoes[1] - posicoes[0]
        direcao = direcao / (np.linalg.norm(direcao) + 1e-12)
        perpendicular = np.array([-direcao[1], direcao[0]])
        desvio = rng.uniform(-0.25, 0.25)
        velocidades[0] = rapidez[0] * (direcao + desvio * perpendicular)
        velocidades[1] = -rapidez[1] * (direcao - desvio * perpendicular)

    return {"cores": cores, "raios": raios,
            "pos0": posicoes, "vel0": velocidades}


def passo_fisico(pos, vel, raios):
    """Avança a física verdadeira em um quadro (DT segundos).

    A ordem das operações é escolhida de propósito para que a transição seja
    uma função EXATA do estado do quadro — e, portanto, algo que uma rede possa
    aprender sem ambiguidade:

      1. detectam-se as colisões que ocorreriam DENTRO do passo (calculando a
         distância mínima entre os dois discos ao longo do intervalo, o que
         elimina o efeito de "atravessar" o outro disco em um passo grande);
      2. resolvem-se essas colisões trocando as componentes normais das
         velocidades (solução elástica para massas iguais: conserva momento e
         energia cinética exatamente);
      3. resolvem-se as colisões com as paredes da arena, invertendo a
         componente normal;
      4. só então os objetos se movem, com a velocidade JÁ corrigida.

    Assim vale a identidade `pos[t+1] = pos[t] + vel[t+1] * DT`, e prever o
    incremento de velocidade Δv basta para reproduzir toda a dinâmica.

    Retorna (nova_pos, nova_vel, lista_de_pares_que_colidiram).
    """
    v = vel.copy()
    n = len(raios)
    colisoes = []

    # (1) e (2) — pares de discos
    for i in range(n):
        for j in range(i + 1, n):
            p = pos[i] - pos[j]           # posição relativa
            u = v[i] - v[j]               # velocidade relativa
            aproximando = float(p @ u)
            if aproximando >= 0.0:
                continue                  # já estão se afastando
            uu = float(u @ u)
            if uu < 1e-12:
                continue
            instante = min(max(-aproximando / uu, 0.0), DT)
            distancia_minima = float(np.linalg.norm(p + u * instante))
            if distancia_minima > raios[i] + raios[j]:
                continue
            normal = p / (np.linalg.norm(p) + 1e-12)
            componente_i = float(v[i] @ normal)
            componente_j = float(v[j] @ normal)
            v[i] = v[i] + (componente_j - componente_i) * normal
            v[j] = v[j] + (componente_i - componente_j) * normal
            colisoes.append((i, j))

    # (3) — paredes da arena (a parede tem massa infinita: só inverte o sinal)
    for i in range(n):
        proxima = pos[i] + v[i] * DT
        if proxima[0] - raios[i] < ARENA_X[0] and v[i, 0] < 0.0:
            v[i, 0] = -v[i, 0]
        if proxima[0] + raios[i] > ARENA_X[1] and v[i, 0] > 0.0:
            v[i, 0] = -v[i, 0]
        if proxima[1] - raios[i] < ARENA_Y[0] and v[i, 1] < 0.0:
            v[i, 1] = -v[i, 1]
        if proxima[1] + raios[i] > ARENA_Y[1] and v[i, 1] > 0.0:
            v[i, 1] = -v[i, 1]

    # (4) — movimento
    return pos + v * DT, v, colisoes


def esta_no_campo(ponto):
    """Um objeto é observável quando seu centro está no campo de visão."""
    return (CAMPO_X[0] <= ponto[0] <= CAMPO_X[1] and
            CAMPO_Y[0] <= ponto[1] <= CAMPO_Y[1])


def mascara_de_visibilidade(posicoes):
    """Matriz booleana (quadros x objetos): quem está dentro do campo."""
    n_quadros, n_objetos = posicoes.shape[0], posicoes.shape[1]
    visivel = np.zeros((n_quadros, n_objetos), dtype=bool)
    for t in range(n_quadros):
        for i in range(n_objetos):
            visivel[t, i] = esta_no_campo(posicoes[t, i])
    return visivel


def criar_evento(quadro, tipo, cores):
    """Um evento é (quadro, tipo, objetos), identificados pela COR.

    Usar cores como identificadores — e não índices — é o que permite comparar
    o mundo factual com o mundo contrafactual, onde um objeto foi removido e os
    índices mudaram de lugar.
    """
    return {"quadro": int(quadro), "tipo": tipo, "cores": tuple(sorted(cores))}


def ordenar_eventos(eventos):
    """Ordem canônica: por quadro, depois por tipo, depois por cores."""
    return sorted(eventos, key=lambda e: (e["quadro"], e["tipo"], e["cores"]))


def eventos_de_campo(posicoes, cores):
    """Extrai os eventos ENTRA e SAI a partir das travessias do campo de visão."""
    visivel = mascara_de_visibilidade(posicoes)
    eventos = []
    for i, cor in enumerate(cores):
        for t in range(1, len(posicoes)):
            if visivel[t, i] and not visivel[t - 1, i]:
                eventos.append(criar_evento(t, "ENTRA", [cor]))
            elif visivel[t - 1, i] and not visivel[t, i]:
                eventos.append(criar_evento(t, "SAI", [cor]))
    return eventos


def simular_clipe(cena, n_quadros=N_QUADROS):
    """Roda a física verdadeira e devolve (posicoes, velocidades, eventos).

    As colisões são registradas pelo próprio simulador — é a verdade absoluta
    contra a qual todo o resto do exercício será comparado.
    """
    n = len(cena["raios"])
    posicoes = np.zeros((n_quadros, n, 2))
    velocidades = np.zeros((n_quadros, n, 2))
    posicoes[0] = cena["pos0"]
    velocidades[0] = cena["vel0"]

    eventos = []
    for t in range(n_quadros - 1):
        nova_pos, nova_vel, colisoes = passo_fisico(
            posicoes[t], velocidades[t], cena["raios"])
        posicoes[t + 1] = nova_pos
        velocidades[t + 1] = nova_vel
        for i, j in colisoes:
            eventos.append(criar_evento(
                t + 1, "COLIDE", [cena["cores"][i], cena["cores"][j]]))

    eventos.extend(eventos_de_campo(posicoes, cena["cores"]))
    return posicoes, velocidades, ordenar_eventos(eventos)


def remover_objeto(cena, indice):
    """Devolve uma cópia da cena SEM o objeto `indice` — o mundo contrafactual."""
    manter = [k for k in range(len(cena["raios"])) if k != indice]
    return {"cores": [cena["cores"][k] for k in manter],
            "raios": cena["raios"][manter].copy(),
            "pos0": cena["pos0"][manter].copy(),
            "vel0": cena["vel0"][manter].copy()}


# ===========================================================================
# PARTE 2 — O VIDEO PARSER (o Mask R-CNN aplicado quadro a quadro)
# ===========================================================================

def analisar_video(posicoes, cores):
    """Simula o video parser do NSDR.

    No artigo, um Mask R-CNN é aplicado a CADA UM dos 25 quadros do clipe e
    devolve, por quadro, a máscara e os atributos de cada objeto visível. Aqui
    o extrator é o mesmo laço, com duas imperfeições realistas:

      * RUÍDO   : a posição sai com erro de localização gaussiano;
      * OCLUSÃO : objeto fora do campo de visão não é detectado — a linha do
                  quadro fica com NaN, e o raciocínio terá de conviver com o
                  buraco.

    Retorna (posicoes_observadas, visivel). O contador global registra quantas
    vezes o extrator foi aplicado.
    """
    rng = np.random.default_rng(SEMENTE + 991)
    n_quadros, n_objetos = posicoes.shape[0], posicoes.shape[1]
    observadas = np.full((n_quadros, n_objetos, 2), np.nan)
    visivel = np.zeros((n_quadros, n_objetos), dtype=bool)

    for t in range(n_quadros):
        ESTATISTICAS_DO_PARSER["quadros_processados"] += 1
        for i in range(n_objetos):
            if esta_no_campo(posicoes[t, i]):
                observadas[t, i] = posicoes[t, i] + rng.normal(0.0, RUIDO_POSICAO, 2)
                visivel[t, i] = True
                ESTATISTICAS_DO_PARSER["deteccoes"] += 1
            else:
                ESTATISTICAS_DO_PARSER["oclusoes"] += 1
    del cores    # a cor é lida sem erro; fica aqui só para documentar a interface
    return observadas, visivel


def estimar_velocidades(observadas, visivel):
    """Velocidade por diferenças finitas entre quadros consecutivos.

    O parser enxerga POSIÇÕES, nunca velocidades — exatamente como o Mask R-CNN.
    Como a física deste mundo satisfaz `pos[t] = pos[t-1] + vel[t] * DT`, a
    diferença PARA TRÁS recupera `vel[t]` exatamente (a menos do ruído). No
    quadro 0 não existe quadro anterior: usa-se a diferença para a frente, que
    devolve `vel[1]` — idêntica a `vel[0]` quando nada colide no primeiro passo.
    """
    n_quadros, n_objetos = observadas.shape[0], observadas.shape[1]
    velocidades = np.full((n_quadros, n_objetos, 2), np.nan)
    for i in range(n_objetos):
        for t in range(n_quadros):
            if t >= 1 and visivel[t, i] and visivel[t - 1, i]:
                velocidades[t, i] = (observadas[t, i] - observadas[t - 1, i]) / DT
            elif t + 1 < n_quadros and visivel[t, i] and visivel[t + 1, i]:
                velocidades[t, i] = (observadas[t + 1, i] - observadas[t, i]) / DT
    return velocidades


# ===========================================================================
# PARTE 3 — O DYNAMICS PREDICTOR (rede de relações, inspirada no PropNet)
# ===========================================================================

_TABELA_VIZINHOS = {}


def indices_de_vizinhos(n):
    """Tabela (n, n-1) com, para cada objeto, os índices de todos os outros."""
    if n not in _TABELA_VIZINHOS:
        _TABELA_VIZINHOS[n] = np.array(
            [[j for j in range(n) if j != i] for i in range(n)],
            dtype=np.int64).reshape(n, max(n - 1, 0))
    return _TABELA_VIZINHOS[n]


def atributos_do_estado(pos, vel, raios):
    """Converte um quadro em (OBJ, VIZ, MASC), a entrada da rede de relacoes.

    OBJ  (n, 12)  : atributos PROPRIOS — velocidade, rapidez, raio, folga ate
                    cada uma das quatro paredes agora e folga depois de um passo
                    em linha reta. Sao as quatro ultimas que tornam o ricochete
                    aprendivel: folga futura negativa significa "vou bater".
    VIZ  (n, 4, 9): atributos RELACIONAIS de cada vizinho — posicao e velocidade
                    relativas, distancia, folga entre superficies, velocidade
                    radial, folga MINIMA ao longo do proximo passo, soma dos raios.
    MASC (n, 4)   : 1 nos vizinhos que existem, 0 no preenchimento (as cenas tem
                    de 3 a 5 objetos e o tensor tem largura fixa).
    """
    n = len(raios)
    proxima = pos + vel * DT
    obj = np.column_stack([
        vel[:, 0], vel[:, 1], np.hypot(vel[:, 0], vel[:, 1]), raios,
        pos[:, 0] - raios - ARENA_X[0], ARENA_X[1] - pos[:, 0] - raios,
        pos[:, 1] - raios - ARENA_Y[0], ARENA_Y[1] - pos[:, 1] - raios,
        proxima[:, 0] - raios - ARENA_X[0], ARENA_X[1] - proxima[:, 0] - raios,
        proxima[:, 1] - raios - ARENA_Y[0], ARENA_Y[1] - proxima[:, 1] - raios])

    # Todos os pares de uma vez (n e no maximo 5: a matriz n x n e minuscula).
    p = pos[:, None, :] - pos[None, :, :]
    u = vel[:, None, :] - vel[None, :, :]
    distancia = np.hypot(p[:, :, 0], p[:, :, 1])
    soma_raios = raios[:, None] + raios[None, :]
    projecao = (p * u).sum(axis=2)
    instante = np.clip(-projecao / np.maximum((u * u).sum(axis=2), 1e-12), 0.0, DT)
    q = p + u * instante[:, :, None]
    folga_minima = np.hypot(q[:, :, 0], q[:, :, 1]) - soma_raios
    pares = np.stack([p[:, :, 0], p[:, :, 1], u[:, :, 0], u[:, :, 1], distancia,
                      distancia - soma_raios, projecao / (distancia + 1e-9),
                      folga_minima, soma_raios], axis=2)

    viz = np.zeros((n, MAX_VIZINHOS, N_ATRIB_PAR))
    masc = np.zeros((n, MAX_VIZINHOS))
    if n > 1:
        viz[:, :n - 1] = pares[np.arange(n)[:, None], indices_de_vizinhos(n)]
        masc[:, :n - 1] = 1.0
    return obj, viz, masc


class MLP:
    """Perceptron multicamadas com ReLU, retropropagação e Adam à mão.

    Nada de framework: `avancar` guarda as ativações, `retropropagar` devolve o
    gradiente em relação à ENTRADA (é isso que permite encadear três redes) e
    `passo_adam` aplica a atualização.
    """

    def __init__(self, tamanhos, rng, saida_linear=False):
        self.pesos, self.vieses = [], []
        for entrada, saida in zip(tamanhos[:-1], tamanhos[1:]):
            escala = math.sqrt(2.0 / entrada)          # inicialização de He
            self.pesos.append(rng.normal(0.0, escala, (entrada, saida)))
            self.vieses.append(np.zeros(saida))
        self.saida_linear = saida_linear
        self.momento_pesos = [np.zeros_like(w) for w in self.pesos]
        self.momento_vieses = [np.zeros_like(b) for b in self.vieses]
        self.escala_pesos = [np.zeros_like(w) for w in self.pesos]
        self.escala_vieses = [np.zeros_like(b) for b in self.vieses]

    def avancar(self, entrada):
        """Propagação direta; memoriza o necessário para a retropropagação."""
        self.ativacoes = [entrada]
        self.pre_ativacoes = []
        atual = entrada
        ultima = len(self.pesos) - 1
        for k, (w, b) in enumerate(zip(self.pesos, self.vieses)):
            z = atual @ w + b
            self.pre_ativacoes.append(z)
            atual = z if (k == ultima and self.saida_linear) else np.maximum(z, 0.0)
            self.ativacoes.append(atual)
        return atual

    def retropropagar(self, gradiente_saida):
        """Regra da cadeia camada a camada; devolve o gradiente da entrada."""
        self.grad_pesos = [None] * len(self.pesos)
        self.grad_vieses = [None] * len(self.vieses)
        grad = gradiente_saida
        ultima = len(self.pesos) - 1
        for k in reversed(range(len(self.pesos))):
            if not (k == ultima and self.saida_linear):
                grad = grad * (self.pre_ativacoes[k] > 0.0)
            self.grad_pesos[k] = self.ativacoes[k].T @ grad
            self.grad_vieses[k] = grad.sum(axis=0)
            grad = grad @ self.pesos[k].T
        return grad

    def passo_adam(self, taxa, iteracao, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Atualização de Adam com correção de viés, escrita explicitamente."""
        correcao1 = 1.0 - beta1 ** iteracao
        correcao2 = 1.0 - beta2 ** iteracao
        for k in range(len(self.pesos)):
            for parametro, gradiente, momento, escala in (
                    (self.pesos[k], self.grad_pesos[k],
                     self.momento_pesos[k], self.escala_pesos[k]),
                    (self.vieses[k], self.grad_vieses[k],
                     self.momento_vieses[k], self.escala_vieses[k])):
                momento *= beta1
                momento += (1.0 - beta1) * gradiente
                escala *= beta2
                escala += (1.0 - beta2) * gradiente * gradiente
                parametro -= taxa * (momento / correcao1) / (
                    np.sqrt(escala / correcao2) + epsilon)


class PreditorDeDinamica:
    """O Dynamics Predictor: três redes que juntas formam uma rede de relações.

    Para cada objeto i:

        efeito(i, j) = f_relacao( atributos do par (i, j) )        [por vizinho]
        agregado(i)  = SOMA sobre os vizinhos j de efeito(i, j)    [invariante]
        proprio(i)   = f_objeto( atributos próprios de i )
        Δv(i)        = f_saida( [proprio(i), agregado(i)] )

    A soma sobre os vizinhos é o detalhe importante: ela torna a rede
    INDIFERENTE à ordem e ao NÚMERO de objetos na cena. É graças a isso que a
    mesma rede treinada em cenas de 3 a 5 objetos consegue simular o mundo
    CONTRAFACTUAL, que tem um objeto a menos e nunca foi visto no treino.
    """

    def __init__(self, rng):
        self.rede_relacao = MLP([N_ATRIB_PAR, UNIDADE_OCULTA, UNIDADE_OCULTA], rng)
        self.rede_objeto = MLP([N_ATRIB_OBJETO, UNIDADE_OCULTA], rng)
        self.rede_saida = MLP([2 * UNIDADE_OCULTA, UNIDADE_OCULTA, 2], rng,
                              saida_linear=True)
        self.media_obj = np.zeros(N_ATRIB_OBJETO)
        self.desvio_obj = np.ones(N_ATRIB_OBJETO)
        self.media_viz = np.zeros(N_ATRIB_PAR)
        self.desvio_viz = np.ones(N_ATRIB_PAR)
        self.escala_alvo = 1.0

    # -- normalização --------------------------------------------------------
    def ajustar_normalizacao(self, obj, viz, masc, alvo):
        """Padroniza as entradas e mede a escala típica de Δv."""
        self.media_obj = obj.mean(axis=0)
        self.desvio_obj = obj.std(axis=0) + 1e-6
        reais = viz.reshape(-1, N_ATRIB_PAR)[masc.reshape(-1) > 0.5]
        self.media_viz = reais.mean(axis=0)
        self.desvio_viz = reais.std(axis=0) + 1e-6
        self.escala_alvo = float(alvo.std()) + 1e-6

    def normalizar(self, obj, viz):
        return ((obj - self.media_obj) / self.desvio_obj,
                (viz - self.media_viz) / self.desvio_viz)

    # -- propagação ----------------------------------------------------------
    def avancar(self, obj, viz, masc):
        """Δv normalizado para um lote de objetos."""
        n_lote, n_viz, _ = viz.shape
        self._forma = (n_lote, n_viz)
        self._masc = masc
        proprio = self.rede_objeto.avancar(obj)
        efeitos = self.rede_relacao.avancar(viz.reshape(n_lote * n_viz, N_ATRIB_PAR))
        efeitos = efeitos.reshape(n_lote, n_viz, UNIDADE_OCULTA) * masc[:, :, None]
        agregado = efeitos.sum(axis=1)
        return self.rede_saida.avancar(np.concatenate([proprio, agregado], axis=1))

    def retropropagar(self, gradiente_saida):
        """Distribui o gradiente pelas três redes (a soma apenas o replica)."""
        n_lote, n_viz = self._forma
        grad_concat = self.rede_saida.retropropagar(gradiente_saida)
        grad_proprio = grad_concat[:, :UNIDADE_OCULTA]
        grad_agregado = grad_concat[:, UNIDADE_OCULTA:]
        grad_efeitos = (np.repeat(grad_agregado[:, None, :], n_viz, axis=1)
                        * self._masc[:, :, None])
        self.rede_relacao.retropropagar(
            grad_efeitos.reshape(n_lote * n_viz, UNIDADE_OCULTA))
        self.rede_objeto.retropropagar(grad_proprio)

    def atualizar(self, taxa, iteracao):
        for rede in (self.rede_objeto, self.rede_relacao, self.rede_saida):
            rede.passo_adam(taxa, iteracao)

    def n_parametros(self):
        total = 0
        for rede in (self.rede_objeto, self.rede_relacao, self.rede_saida):
            total += sum(w.size for w in rede.pesos)
            total += sum(b.size for b in rede.vieses)
        return total

    # -- treino --------------------------------------------------------------
    def treinar(self, obj, viz, masc, alvo, rng):
        """Descida de gradiente por minilotes; devolve o histórico da perda."""
        obj_n, viz_n = self.normalizar(obj, viz)
        alvo_n = alvo / self.escala_alvo
        historico = []
        iteracao = 0
        for epoca in range(1, EPOCAS + 1):
            ordem = rng.permutation(len(obj_n))
            soma, lotes = 0.0, 0
            for inicio in range(0, len(ordem), TAMANHO_LOTE):
                indices = ordem[inicio:inicio + TAMANHO_LOTE]
                saida = self.avancar(obj_n[indices], viz_n[indices], masc[indices])
                erro = saida - alvo_n[indices]
                soma += float(np.mean(erro ** 2))
                lotes += 1
                iteracao += 1
                self.retropropagar(2.0 * erro / erro.size)
                self.atualizar(TAXA_APRENDIZADO, iteracao)
            historico.append((epoca, soma / lotes))
        return historico

    # -- uso -----------------------------------------------------------------
    def prever_delta_v(self, pos, vel, raios):
        """Incremento de velocidade previsto para cada objeto de um quadro."""
        obj, viz, masc = atributos_do_estado(pos, vel, raios)
        obj_n, viz_n = self.normalizar(obj, viz)
        return self.avancar(obj_n, viz_n, masc) * self.escala_alvo

    def passo(self, pos, vel, raios):
        """Um quadro de simulação APRENDIDA, no lugar de `passo_fisico`."""
        nova_vel = vel + self.prever_delta_v(pos, vel, raios)
        nova_pos = pos + nova_vel * DT
        # Trava de segurança: um simulador aprendido pode, em um erro grande,
        # cuspir um objeto para fora da arena e nunca mais voltar. Limitamos a
        # posição à arena com meia unidade de folga; isso não corrige a física,
        # apenas impede que um erro isolado contamine o clipe inteiro.
        nova_pos[:, 0] = np.clip(nova_pos[:, 0], ARENA_X[0] - 0.5, ARENA_X[1] + 0.5)
        nova_pos[:, 1] = np.clip(nova_pos[:, 1], ARENA_Y[0] - 0.5, ARENA_Y[1] + 0.5)
        return nova_pos, nova_vel

    def rolar(self, pos0, vel0, raios, n_passos):
        """Rolagem livre: `n_passos` quadros aplicando a rede sobre si mesma."""
        posicoes = np.zeros((n_passos + 1, len(raios), 2))
        velocidades = np.zeros((n_passos + 1, len(raios), 2))
        posicoes[0], velocidades[0] = pos0, vel0
        for t in range(n_passos):
            posicoes[t + 1], velocidades[t + 1] = self.passo(
                posicoes[t], velocidades[t], raios)
        return posicoes, velocidades


def montar_conjunto_de_treino(clipes):
    """Empilha todas as transições de todos os clipes em tensores de treino.

    O alvo é Δv = vel[t+1] - vel[t]. Como a esmagadora maioria das transições é
    movimento livre (Δv exatamente zero), as transições COM impulso são
    replicadas `FATOR_REFORCO_COLISAO` vezes: sem esse reforço o erro quadrático
    médio seria minimizado por uma rede que simplesmente responde "nada
    acontece", e o exercício inteiro perderia a graça.
    """
    lista_obj, lista_viz, lista_masc, lista_alvo, com_impulso = [], [], [], [], []
    for posicoes, velocidades, _eventos, cena in clipes:
        for t in range(len(posicoes) - 1):
            obj, viz, masc = atributos_do_estado(
                posicoes[t], velocidades[t], cena["raios"])
            alvo = velocidades[t + 1] - velocidades[t]
            impulso = np.linalg.norm(alvo, axis=1) > LIMIAR_IMPULSO
            lista_obj.append(obj)
            lista_viz.append(viz)
            lista_masc.append(masc)
            lista_alvo.append(alvo)
            com_impulso.append(impulso)

    obj = np.concatenate(lista_obj)
    viz = np.concatenate(lista_viz)
    masc = np.concatenate(lista_masc)
    alvo = np.concatenate(lista_alvo)
    impulso = np.concatenate(com_impulso)

    extras = np.repeat(np.flatnonzero(impulso), FATOR_REFORCO_COLISAO)
    indices = np.concatenate([np.arange(len(obj)), extras])
    return (obj[indices], viz[indices], masc[indices], alvo[indices],
            int(impulso.sum()), len(obj))


# ===========================================================================
# PARTE 4 — DETECÇÃO DE EVENTOS EM UMA TRAJETÓRIA QUALQUER
# ===========================================================================

def detectar_eventos(posicoes, velocidades, cores, raios):
    """Lê eventos de uma trajetória — verdadeira ou PREVISTA pela rede.

    Uma colisão deixa duas assinaturas simultâneas, e o detector exige as duas:

      * geométrica : as superfícies dos dois discos ficam a menos de
                     LIMIAR_CONTATO uma da outra;
      * dinâmica   : pelo menos um dos dois muda de velocidade acima de
                     LIMIAR_IMPULSO naquele quadro.

    Só a primeira condição confundiria uma passagem raspante com uma colisão;
    só a segunda confundiria um ricochete na parede com uma colisão entre
    discos. Quadros consecutivos do mesmo par contam como UM evento.
    """
    n_quadros, n_objetos = posicoes.shape[0], posicoes.shape[1]
    eventos = []
    for i in range(n_objetos):
        for j in range(i + 1, n_objetos):
            ultimo_registrado = -99
            for t in range(1, n_quadros):
                folga = float(np.linalg.norm(posicoes[t, i] - posicoes[t, j])) \
                    - (raios[i] + raios[j])
                if folga > LIMIAR_CONTATO:
                    continue
                impulso = max(
                    float(np.linalg.norm(velocidades[t, i] - velocidades[t - 1, i])),
                    float(np.linalg.norm(velocidades[t, j] - velocidades[t - 1, j])))
                if impulso < LIMIAR_IMPULSO:
                    continue
                if t - ultimo_registrado <= 2:
                    continue
                ultimo_registrado = t
                eventos.append(criar_evento(t, "COLIDE", [cores[i], cores[j]]))
    eventos.extend(eventos_de_campo(posicoes, cores))
    return ordenar_eventos(eventos)


def casar_eventos(previstos, verdadeiros, tipos=None):
    """Casa duas listas de eventos e devolve (acertos, falsos, faltantes).

    Dois eventos casam quando têm o mesmo tipo, exatamente os mesmos objetos
    (pelas cores) e quadros a no máximo TOLERANCIA_QUADROS de distância — o
    detector não é obrigado a cravar o quadro exato da batida.
    """
    if tipos is not None:
        previstos = [e for e in previstos if e["tipo"] in tipos]
        verdadeiros = [e for e in verdadeiros if e["tipo"] in tipos]
    disponiveis = list(range(len(verdadeiros)))
    acertos, falsos = [], []
    for evento in previstos:
        casado = None
        for k in disponiveis:
            alvo = verdadeiros[k]
            if (alvo["tipo"] == evento["tipo"] and alvo["cores"] == evento["cores"]
                    and abs(alvo["quadro"] - evento["quadro"]) <= TOLERANCIA_QUADROS):
                casado = k
                break
        if casado is None:
            falsos.append(evento)
        else:
            disponiveis.remove(casado)
            acertos.append((evento, verdadeiros[casado]))
    faltantes = [verdadeiros[k] for k in disponiveis]
    return acertos, falsos, faltantes


def formatar_evento(evento):
    """'COLIDE(cinza, azul) @ quadro 07' — a forma legível de um evento."""
    return f"{evento['tipo']}({', '.join(evento['cores'])}) @ quadro {evento['quadro']:02d}"


# ===========================================================================
# PARTE 5 — AVALIAÇÕES DO PREDITOR
# ===========================================================================

def avaliar_um_passo(preditor, clipes):
    """Erro de um passo: rede contra a inércia (velocidade constante)."""
    erros_rede, erros_inercia = [], []
    for posicoes, velocidades, _ev, cena in clipes:
        for t in range(len(posicoes) - 1):
            previsto, _ = preditor.passo(posicoes[t], velocidades[t], cena["raios"])
            inercia = posicoes[t] + velocidades[t] * DT
            erros_rede.extend(np.linalg.norm(previsto - posicoes[t + 1], axis=1))
            erros_inercia.extend(np.linalg.norm(inercia - posicoes[t + 1], axis=1))
    return float(np.mean(erros_rede)), float(np.mean(erros_inercia))


def avaliar_rolagem(preditor, clipes, horizontes):
    """Erro médio de posição após h passos de rolagem, rede contra inércia."""
    resultado = {}
    for h in horizontes:
        erros_rede, erros_inercia = [], []
        for posicoes, velocidades, _ev, cena in clipes:
            previstas, _ = preditor.rolar(
                posicoes[0], velocidades[0], cena["raios"], h)
            inercia = posicoes[0] + velocidades[0] * (DT * h)
            erros_rede.extend(np.linalg.norm(previstas[h] - posicoes[h], axis=1))
            erros_inercia.extend(np.linalg.norm(inercia - posicoes[h], axis=1))
        resultado[h] = (float(np.mean(erros_rede)), float(np.mean(erros_inercia)))
    return resultado


def rolagem_ancorada(preditor, observadas, visivel, raios, velocidades_obs):
    """Rolagem com ANCORAGEM: usa o que o parser vê, prevê o que ele não vê.

    A cada quadro a rede propaga o estado de todos os objetos; em seguida, os
    objetos DETECTADOS têm posição e velocidade substituídas pela medição, e os
    objetos OCLUSOS seguem apenas com a previsão. É assim que o NSDR aproxima o
    movimento com informação faltante: a trajetória imaginada atravessa o
    buraco e reencontra a verdadeira quando o objeto reaparece.

    Devolve as posições propagadas ANTES da ancoragem de cada quadro — que é
    exatamente a previsão a ser comparada com a realidade.
    """
    n_quadros, n_objetos = observadas.shape[0], observadas.shape[1]
    posicoes = np.zeros((n_quadros, n_objetos, 2))
    velocidades = np.zeros((n_quadros, n_objetos, 2))
    previstas = np.zeros((n_quadros, n_objetos, 2))

    posicoes[0] = np.where(np.isnan(observadas[0]), 0.0, observadas[0])
    velocidades[0] = np.where(np.isnan(velocidades_obs[0]), 0.0, velocidades_obs[0])
    previstas[0] = posicoes[0]

    for t in range(n_quadros - 1):
        proxima_pos, proxima_vel = preditor.passo(posicoes[t], velocidades[t], raios)
        previstas[t + 1] = proxima_pos
        for i in range(n_objetos):
            if visivel[t + 1, i] and not np.isnan(observadas[t + 1, i]).any():
                proxima_pos[i] = observadas[t + 1, i]
                if not np.isnan(velocidades_obs[t + 1, i]).any():
                    proxima_vel[i] = velocidades_obs[t + 1, i]
        posicoes[t + 1], velocidades[t + 1] = proxima_pos, proxima_vel
    return previstas, posicoes


def avaliar_colisoes_futuras(preditor, clipes_parseados):
    """Precisão e revocação das colisões PREVISTAS depois do quadro atual.

    Do quadro QUADRO_ATUAL em diante o parser não vê mais nada: a rede rola
    sozinha até o fim do clipe e o detector de eventos lê as colisões dessa
    trajetória imaginada. O gabarito são as colisões que a física verdadeira
    registrou no mesmo intervalo.
    """
    acertos = falsos = faltantes = 0
    for posicoes, velocidades, eventos, cena, observadas, visivel, vel_obs in clipes_parseados:
        estado_pos = np.where(np.isnan(observadas[QUADRO_ATUAL]),
                              posicoes[QUADRO_ATUAL], observadas[QUADRO_ATUAL])
        estado_vel = np.where(np.isnan(vel_obs[QUADRO_ATUAL]),
                              velocidades[QUADRO_ATUAL], vel_obs[QUADRO_ATUAL])
        n_passos = len(posicoes) - 1 - QUADRO_ATUAL
        pos_prev, vel_prev = preditor.rolar(
            estado_pos, estado_vel, cena["raios"], n_passos)
        previstos = [dict(e, quadro=e["quadro"] + QUADRO_ATUAL)
                     for e in detectar_eventos(pos_prev, vel_prev,
                                               cena["cores"], cena["raios"])]
        verdadeiros = [e for e in eventos if e["quadro"] > QUADRO_ATUAL]
        a, f, m = casar_eventos(previstos, verdadeiros, tipos={"COLIDE"})
        acertos += len(a)
        falsos += len(f)
        faltantes += len(m)
    precisao = acertos / max(1, acertos + falsos)
    revocacao = acertos / max(1, acertos + faltantes)
    return acertos, falsos, faltantes, precisao, revocacao


# ===========================================================================
# PARTE 6 — O EXECUTOR SIMBÓLICO (totalmente simbólico, não probabilístico)
# ===========================================================================

class MundoDoClipe:
    """Tudo o que o executor simbólico pode consultar sobre um clipe.

    Repare no que NÃO está aqui: pixels, ativações, probabilidades. O executor
    do NSDR trabalha sobre uma representação já discreta — objetos com cores e
    uma lista de eventos — e é por isso que ele é TOTALMENTE simbólico, e não
    quase-simbólico como o do NSCL.
    """

    def __init__(self, cena, eventos_observados, eventos_previstos,
                 preditor, estado_inicial):
        self.cena = cena
        self.eventos_observados = eventos_observados
        self.eventos_previstos = eventos_previstos
        self.preditor = preditor
        self.estado_inicial = estado_inicial     # (posições, velocidades) do quadro 0
        self.cache_contrafactual = {}

    def indice_da_cor(self, cor):
        return self.cena["cores"].index(cor)

    def simular_sem(self, cor):
        """Re-simula o clipe inteiro SEM o objeto de cor `cor`, com a rede.

        Este é o coração da operação OBTER_CONTRAFACTUAIS: apaga-se o objeto do
        ESTADO INICIAL e o Dynamics Predictor gera um vídeo que nunca existiu.
        """
        if cor in self.cache_contrafactual:
            return self.cache_contrafactual[cor]
        indice = self.indice_da_cor(cor)
        manter = [k for k in range(len(self.cena["cores"])) if k != indice]
        pos0, vel0 = self.estado_inicial
        raios = self.cena["raios"][manter]
        cores = [self.cena["cores"][k] for k in manter]
        posicoes, velocidades = self.preditor.rolar(
            pos0[manter], vel0[manter], raios, N_QUADROS - 1)
        eventos = detectar_eventos(posicoes, velocidades, cores, raios)
        self.cache_contrafactual[cor] = (posicoes, velocidades, cores, eventos)
        return self.cache_contrafactual[cor]


# Formatação do rastro do executor: o relatório inteiro cabe em LARGURA
# colunas, inclusive as linhas de rastro, que são as mais compridas.
COLUNA_DA_OPERACAO = 30      # espaço reservado ao nome da operação no rastro
RECUO_DOS_ITENS = " " * 10   # indentação dos itens listados abaixo da operação
EVENTOS_NO_RASTRO = 4        # eventos exibidos antes de resumir com "(+N)"
EVENTOS_POR_LINHA = 2        # no máximo dois eventos por linha indentada


def descrever_valor(valor):
    """Resumo em uma linha do valor corrente do programa, para o rastro."""
    if isinstance(valor, bool):
        return "SIM" if valor else "NÃO"
    if isinstance(valor, int):
        return str(valor)
    if isinstance(valor, str):
        return valor
    if isinstance(valor, dict) and "tipo" in valor and "quadro" in valor:
        return formatar_evento(valor)
    if isinstance(valor, dict) and "tipo" in valor:
        return f"{valor['tipo']}({', '.join(valor['cores'])})"
    if isinstance(valor, list):
        if not valor:
            return "conjunto vazio"
        if isinstance(valor[0], dict):
            resumo = "; ".join(formatar_evento(e) for e in valor[:3])
            extra = "" if len(valor) <= 3 else f"; ... (+{len(valor) - 3})"
            return f"{len(valor)} evento(s): {resumo}{extra}"
        return f"{len(valor)} objeto(s): {', '.join(str(v) for v in valor)}"
    return str(valor)


def partir_valor(valor, largura_util):
    """Separa o valor corrente em (cabeça, linhas de itens) para o rastro.

    A cabeça fica na mesma linha da operação ("10 evento(s):"); os itens vão
    em linhas indentadas abaixo, no máximo EVENTOS_POR_LINHA por linha, sem
    nunca passar de `largura_util` colunas. Quando a lista de eventos é longa
    demais, as sobras viram um "(+N)" na última linha.
    """
    if isinstance(valor, list) and valor and isinstance(valor[0], dict):
        itens = [formatar_evento(e) for e in valor[:EVENTOS_NO_RASTRO]]
        if len(valor) > EVENTOS_NO_RASTRO:
            itens.append(f"... (+{len(valor) - EVENTOS_NO_RASTRO})")
        linhas, atual = [], []
        for item in itens:
            junto = "; ".join(atual + [item])
            if atual and (len(atual) >= EVENTOS_POR_LINHA
                          or len(junto) > largura_util):
                linhas.append("; ".join(atual))
                atual = [item]
            else:
                atual.append(item)
        if atual:
            linhas.append("; ".join(atual))
        return f"{len(valor)} evento(s):", linhas
    if isinstance(valor, list) and valor and not isinstance(valor[0], dict):
        texto = ", ".join(str(v) for v in valor)
        return (f"{len(valor)} objeto(s):",
                textwrap.wrap(texto, width=largura_util) or [""])
    return descrever_valor(valor), []


class ExecutorSimbolico:
    """Executa programas funcionais da DSL do NSDR e imprime o rastro.

    Um programa é uma LISTA de operações aplicadas em cadeia: a saída de uma é
    a entrada da seguinte, exatamente como nos diagramas do capítulo. Cada
    operação é determinística e inspecionável — não há um único número em ponto
    flutuante escondido no caminho.
    """

    def __init__(self, mundo):
        self.mundo = mundo

    # -- operações da DSL ----------------------------------------------------
    def op_objetos(self, _valor):
        return list(self.mundo.cena["cores"])

    def op_filtrar_cor(self, valor, cor):
        return [c for c in valor if c == cor]

    def op_unico(self, valor):
        return valor[0] if valor else None

    def op_eventos(self, _valor, escopo):
        if escopo == "observado":
            return list(self.mundo.eventos_observados)
        if escopo == "previsto":
            return list(self.mundo.eventos_previstos)
        raise ValueError(escopo)

    def op_filtrar_tipo(self, valor, tipo):
        return [e for e in valor if e["tipo"] == tipo]

    def op_filtrar_objeto(self, valor, cor):
        return [e for e in valor if cor in e["cores"]]

    def op_contar(self, valor):
        return len(valor)

    def op_primeiro(self, valor):
        return valor[0] if valor else None

    def op_parceiros(self, valor, cor):
        return [c for e in valor for c in e["cores"] if c != cor]

    def op_obter_contrafactuais(self, valor):
        cor = valor if isinstance(valor, str) else valor[0]
        return self.mundo.simular_sem(cor)[3]

    def op_programa_de_escolha(self, _valor, alternativa):
        return alternativa

    def op_pertence_a(self, valor, conjunto):
        return any(e["tipo"] == valor["tipo"] and e["cores"] == valor["cores"]
                   for e in conjunto)

    OPERACOES = {
        "OBJETOS": op_objetos,
        "FILTRAR_COR": op_filtrar_cor,
        "UNICO": op_unico,
        "EVENTOS": op_eventos,
        "FILTRAR_TIPO": op_filtrar_tipo,
        "FILTRAR_OBJETO": op_filtrar_objeto,
        "CONTAR": op_contar,
        "PRIMEIRO": op_primeiro,
        "PARCEIROS": op_parceiros,
        "OBTER_CONTRAFACTUAIS": op_obter_contrafactuais,
        "PROGRAMA_DE_ESCOLHA": op_programa_de_escolha,
        "PERTENCE_A": op_pertence_a,
    }

    def executar(self, programa, mostrar=True):
        """Roda a cadeia de operações, imprimindo o rastro de cada passo."""
        valor = None
        for k, (nome, argumentos) in enumerate(programa, start=1):
            valor = self.OPERACOES[nome](self, valor, **argumentos)
            if mostrar:
                if argumentos:
                    partes = []
                    for chave, item in argumentos.items():
                        partes.append(f"{chave}={descrever_valor(item)}"
                                      if not isinstance(item, str) else item)
                    chamada = f"{nome}({', '.join(partes)})"
                else:
                    chamada = f"{nome}()"
                if len(chamada) > COLUNA_DA_OPERACAO:
                    chamada = chamada[:COLUNA_DA_OPERACAO - 3] + "..."
                cabeca, itens = partir_valor(
                    valor, LARGURA - len(RECUO_DOS_ITENS))
                prefixo = f"    [{k}] {chamada:<{COLUNA_DA_OPERACAO}s} -> "
                if len(itens) == 1 and len(prefixo + cabeca) + 1 + \
                        len(itens[0]) <= LARGURA:
                    print(prefixo + cabeca + " " + itens[0])
                else:
                    print(prefixo + cabeca)
                    for linha in itens:
                        print(RECUO_DOS_ITENS + linha)
        return valor


# ===========================================================================
# PARTE 7 — BUSCA DO CLIPE DE DEMONSTRAÇÃO
# ===========================================================================

def procurar_clipe_de_demonstracao(rng):
    """Sorteia clipes até achar um que sustente as QUATRO famílias de pergunta.

    O clipe precisa, ao mesmo tempo:
      1. conter um objeto CINZA (é o objeto removido na Figura 5.5);
      2. ter o cinza colidindo com alguém ATÉ o quadro atual (pergunta descritiva);
      3. ter pelo menos três colisões e algum evento DEPOIS do quadro atual
         (pergunta preditiva);
      4. ter uma colisão entre dois objetos NÃO-cinza que DESAPARECE quando o
         cinza é removido da física verdadeira (perguntas contrafactual e
         explicativa);
      5. não ter colisão no primeiro passo e ter todos os objetos visíveis nos
         quadros 0 e 1, para que a velocidade inicial medida pelo parser seja
         confiável.
    """
    for _ in range(MAX_CANDIDATOS_DEMO):
        cena = sortear_cena(rng)
        if "cinza" not in cena["cores"] or len(cena["cores"]) < 4:
            continue
        posicoes, velocidades, eventos = simular_clipe(cena)

        colisoes = [e for e in eventos if e["tipo"] == "COLIDE"]
        if len(colisoes) < 3:
            continue
        if any(e["quadro"] <= 1 for e in colisoes):
            continue
        if not any("cinza" in e["cores"] and e["quadro"] <= QUADRO_ATUAL
                   for e in colisoes):
            continue
        if not any(e["quadro"] > QUADRO_ATUAL for e in eventos):
            continue

        visivel = mascara_de_visibilidade(posicoes)
        if not visivel[0].all() or not visivel[1].all():
            continue

        indice_cinza = cena["cores"].index("cinza")
        cena_cf = remover_objeto(cena, indice_cinza)
        pos_cf, vel_cf, eventos_cf = simular_clipe(cena_cf)
        conjunto_cf = {(e["tipo"], e["cores"]) for e in eventos_cf}

        impedidas = [e for e in colisoes
                     if "cinza" not in e["cores"]
                     and (e["tipo"], e["cores"]) not in conjunto_cf]
        if not impedidas:
            continue

        return {"cena": cena, "posicoes": posicoes, "velocidades": velocidades,
                "eventos": eventos, "cena_cf": cena_cf, "pos_cf": pos_cf,
                "vel_cf": vel_cf, "eventos_cf": eventos_cf,
                "colisao_impedida": impedidas[0]}
    raise RuntimeError("nenhum clipe de demonstração encontrado no orçamento de busca")


# ===========================================================================
# PARTE 8 — RELATÓRIO
# ===========================================================================

def paragrafo(texto, recuo=2):
    """Quebra um texto corrido dentro da largura do relatório."""
    espaco = " " * recuo
    return textwrap.fill(" ".join(texto.split()), width=LARGURA,
                         initial_indent=espaco, subsequent_indent=espaco,
                         break_on_hyphens=False)


def cabecalho(titulo, caractere="="):
    """Imprime um cabeçalho de seção."""
    print()
    print(caractere * LARGURA)
    print(f" {titulo}")
    print(caractere * LARGURA)


def subtitulo(titulo):
    """Imprime um separador de subseção."""
    print()
    print(f"-- {titulo} " + "-" * max(0, LARGURA - len(titulo) - 4))


def imprimir_lista_de_eventos(eventos, recuo=4, limite=None):
    """Lista de eventos, um por linha, na ordem cronológica."""
    espaco = " " * recuo
    if not eventos:
        print(espaco + "(nenhum)")
        return
    mostrados = eventos if limite is None else eventos[:limite]
    for evento in mostrados:
        print(espaco + formatar_evento(evento))
    if limite is not None and len(eventos) > limite:
        print(espaco + f"... e mais {len(eventos) - limite} evento(s)")


def imprimir_secao_mundo(demo):
    """Seção [1]: o mundo físico e o clipe escolhido."""
    cena = demo["cena"]
    cabecalho("[1] O MUNDO — UM CLEVRER SINTÉTICO DE 25 QUADROS", "-")
    print(f"  Duração do clipe .............. {N_QUADROS} quadros x {DT:.1f} s "
          f"= {N_QUADROS * DT:.1f} s")
    print(f"  Arena (com paredes) ........... "
          f"[{ARENA_X[0]:.1f}, {ARENA_X[1]:.1f}] x [{ARENA_Y[0]:.1f}, {ARENA_Y[1]:.1f}]")
    print(f"  Campo de visão da câmera ...... "
          f"[{CAMPO_X[0]:.1f}, {CAMPO_X[1]:.1f}] x [{CAMPO_Y[0]:.1f}, {CAMPO_Y[1]:.1f}]")
    print(f"  Objetos por clipe ............. {N_MIN_OBJETOS} a "
          f"{N_MAX_OBJETOS} discos")
    print(f"  Cores disponíveis ............. {', '.join(CORES)}")
    print("  Colisões ...................... elásticas, massas iguais:")
    print("                                  troca das componentes normais")
    print("                                  (momento e energia conservados)")
    print()
    print(paragrafo(
        "Objetos podem sair do campo de visão e voltar: a arena é maior que o "
        "quadro filmado. É daí que saem os eventos ENTRA e SAI — e a oclusão "
        "que o raciocínio terá de contornar."))

    subtitulo("O clipe de demonstração")
    print()
    print(f"    {'#':>2s}  {'cor':<10s}  {'raio':>5s}  {'posição inicial':>17s}  "
          f"{'velocidade inicial':>19s}")
    print("    " + "-" * 60)
    for i, cor in enumerate(cena["cores"]):
        p, v = cena["pos0"][i], cena["vel0"][i]
        print(f"    {i:2d}  {cor:<10s}  {cena['raios'][i]:5.2f}  "
              f"({p[0]:6.2f}, {p[1]:6.2f})     ({v[0]:6.2f}, {v[1]:6.2f})")
    print()
    print("  Lista VERDADEIRA de eventos do clipe (registrada pelo simulador):")
    imprimir_lista_de_eventos(demo["eventos"])


def imprimir_secao_parser(demo, observadas, visivel):
    """Seção [2]: o video parser quadro a quadro."""
    cena = demo["cena"]
    cabecalho("[2] VIDEO PARSER — O EXTRATOR APLICADO A CADA QUADRO", "-")
    print(paragrafo(
        "No NSDR, um Mask R-CNN é aplicado a CADA UM dos 25 quadros do clipe: "
        "não existe 'uma passada pelo vídeo', existem 25 passadas por imagem, "
        "e o resultado é uma tabela de objetos por quadro. Aqui o extrator é o "
        "mesmo laço, com ruído de medição e oclusão."))
    print()
    print(f"  Aplicações do extrator até aqui  {ESTATISTICAS_DO_PARSER['quadros_processados']} "
          f"quadros processados")
    print(f"  Detecções bem-sucedidas ......... {ESTATISTICAS_DO_PARSER['deteccoes']}")
    print(f"  Objetos ocultos (fora do campo) . {ESTATISTICAS_DO_PARSER['oclusoes']}")
    print(f"  Ruído de localização ............ sigma = {RUIDO_POSICAO:.3f} unidade por eixo")

    subtitulo("Mapa de visibilidade do clipe de demonstração")
    print()
    print("    quadro:  " + "".join(f"{t % 10}" for t in range(N_QUADROS)))
    for i, cor in enumerate(cena["cores"]):
        linha = "".join("#" if visivel[t, i] else "." for t in range(N_QUADROS))
        print(f"    {cor:<8s} {linha}")
    print()
    print("    ('#' = detectado pelo extrator, '.' = fora do campo de visão)")

    erro = np.linalg.norm(observadas - demo["posicoes"], axis=2)
    erro_medio = float(np.nanmean(np.where(visivel, erro, np.nan)))
    print()
    print(f"  Erro médio de localização do parser: {erro_medio:.4f} unidade")


def imprimir_secao_preditor(preditor, historico, n_com_impulso, n_amostras):
    """Seção [3]: arquitetura e treino do Dynamics Predictor."""
    cabecalho("[3] DYNAMICS PREDICTOR — UMA REDE DE RELAÇÕES (PropNet)", "-")
    print(paragrafo(
        "O sub-modelo que separa o NSDR do NSCL. Para cada PAR de objetos a "
        "rede calcula um efeito a partir do estado relacional; os efeitos dos "
        "vizinhos são SOMADOS; a soma se junta a um termo próprio e produz o "
        "incremento de velocidade do objeto no próximo quadro. Como a "
        "agregação é uma soma, a rede não depende da ordem nem do NÚMERO de "
        "objetos — e é por isso que ela consegue simular um mundo com um "
        "objeto a menos."))
    print()
    print(f"  f_relacao ..................... {N_ATRIB_PAR} -> {UNIDADE_OCULTA} "
          f"-> {UNIDADE_OCULTA}  (ReLU)")
    print(f"  f_objeto ...................... {N_ATRIB_OBJETO} -> {UNIDADE_OCULTA}  (ReLU)")
    print(f"  f_saida ....................... {2 * UNIDADE_OCULTA} -> {UNIDADE_OCULTA} "
          f"-> 2   (saída linear = Δv)")
    print(f"  Parâmetros treináveis ......... {preditor.n_parametros()}")
    print(f"  Otimizador .................... Adam, taxa {TAXA_APRENDIZADO}, "
          f"lote {TAMANHO_LOTE}, {EPOCAS} épocas")
    print()
    print(f"  Transições de treino .......... {n_amostras} "
          f"(de {N_CLIPES_TREINO + N_CLIPES_COLISAO} clipes)")
    print(f"  Delas, com impulso .............{n_com_impulso:6d} "
          f"({n_com_impulso / n_amostras:.1%}) — colisão ou parede")
    print(f"  Reforço das transições raras .. {FATOR_REFORCO_COLISAO} cópias extras "
          f"de cada uma")
    print()
    print("  Curva de treino (erro quadrático médio, unidades normalizadas):")
    print(f"      {'época':>6s}  {'perda':>10s}")
    print("      " + "-" * 18)
    marcos = [0, EPOCAS // 8, EPOCAS // 4, EPOCAS // 2,
              3 * EPOCAS // 4, EPOCAS - 1]
    for indice in marcos:
        epoca, perda = historico[indice]
        print(f"      {epoca:6d}  {perda:10.5f}")


def imprimir_secao_avaliacao(preditor, clipes_teste, clipes_parseados, faltante):
    """Seção [4]: precisão do motor de física aprendido."""
    cabecalho("[4] AVALIAÇÃO DO MOTOR DE FÍSICA APRENDIDO", "-")

    erro_rede, erro_inercia = avaliar_um_passo(preditor, clipes_teste)
    horizontes = [5, 10, N_QUADROS - 1]
    rolagens = avaliar_rolagem(preditor, clipes_teste, horizontes)

    subtitulo("Erro de previsão (posição, em unidades da arena)")
    print()
    print(f"    {'horizonte':<28s}{'Dynamics Predictor':>20s}{'inércia':>14s}")
    print("    " + "-" * 62)
    print(f"    {'1 passo (0,2 s)':<28s}{erro_rede:>20.4f}{erro_inercia:>14.4f}")
    for h in horizontes:
        rede, inercia = rolagens[h]
        rotulo = f"{h} passos ({h * DT:.1f} s)"
        if h == N_QUADROS - 1:
            rotulo += " = clipe"
        print(f"    {rotulo:<28s}{rede:>20.4f}{inercia:>14.4f}")
    print("    " + "-" * 62)
    diagonal = math.hypot(ARENA_X[1] - ARENA_X[0], ARENA_Y[1] - ARENA_Y[0])
    rede_final = rolagens[N_QUADROS - 1][0]
    print(paragrafo(
        f"Para calibrar: a diagonal da arena mede {diagonal:.2f} unidades e o "
        f"raio típico de um disco é {(RAIO_MIN + RAIO_MAX) / 2:.2f}. Depois de "
        f"{N_QUADROS - 1} passos SEM NENHUMA observação, o erro da rede é de "
        f"{rede_final:.3f} unidade ({rede_final / diagonal:.1%} da diagonal), "
        f"contra {rolagens[N_QUADROS - 1][1]:.3f} da inércia. A diferença é o "
        f"que a rede aprendeu sobre colisões.", recuo=4))

    subtitulo("Movimento com INFORMAÇÃO FALTANTE (oclusão)")
    print()
    print(paragrafo(
        f"O objeto {faltante['cor']} do clipe de teste #{faltante['clipe']} sai "
        f"do campo de visão no quadro {faltante['inicio']} e só reaparece no "
        f"quadro {faltante['fim']}. Durante esses {faltante['fim'] - faltante['inicio']} "
        f"quadros o parser não entrega nada: a rede propaga o objeto sozinha, "
        f"guiada apenas pelo próprio modelo de dinâmica.", recuo=4))
    print()
    print(f"      {'quadro':>6s}  {'parser':>8s}  {'posição verdadeira':>20s}"
          f"  {'posição prevista':>19s}  {'erro':>6s}")
    print("      " + "-" * 68)
    for linha in faltante["linhas"]:
        t, visto, verdadeira, prevista, erro = linha
        print(f"      {t:6d}  {('detecta' if visto else 'OCLUSO'):>8s}  "
              f"({verdadeira[0]:6.2f}, {verdadeira[1]:6.2f})       "
              f"({prevista[0]:6.2f}, {prevista[1]:6.2f})   {erro:6.3f}")
    print("      " + "-" * 68)
    print(paragrafo(
        f"No quadro do reaparecimento a trajetória imaginada está a "
        f"{faltante['erro_no_retorno']:.3f} unidade da real — menos de um raio "
        f"de disco. A rede não 'perdeu' o objeto: ela o reencontrou. É "
        f"exatamente essa propriedade que o capítulo destaca como crucial para "
        f"o CLEVRER, onde objetos entram e saem do quadro o tempo todo.",
        recuo=4))

    subtitulo("Previsão de COLISÕES FUTURAS (a partir do quadro 12)")
    acertos, falsos, faltantes, precisao, revocacao = avaliar_colisoes_futuras(
        preditor, clipes_parseados)
    print()
    print(f"    Colisões corretamente previstas (acertos) ....... {acertos}")
    print(f"    Colisões previstas que não aconteceram (falsos) . {falsos}")
    print(f"    Colisões que aconteceram e não foram previstas .. {faltantes}")
    print(f"    Precisão ........................................ {precisao:.1%}")
    print(f"    Revocação ....................................... {revocacao:.1%}")
    print()
    print(paragrafo(
        f"Um evento conta como acertado quando envolve os mesmos dois objetos "
        f"e cai a no máximo {TOLERANCIA_QUADROS} quadros do quadro verdadeiro. "
        f"Prever colisões a até {(N_QUADROS - 1 - QUADRO_ATUAL) * DT:.1f} "
        f"segundos de distância é raciocínio PREDITIVO — e nenhuma rede de "
        f"classificação de vídeo, por maior que seja, faz isso sem um modelo "
        f"de dinâmica.", recuo=4))
    print()
    print(paragrafo(
        f"Mas leia o placar com honestidade: revocação de {revocacao:.0%} "
        f"quer dizer que {1 - revocacao:.0%} das colisões futuras passam "
        f"batido, e precisão de {precisao:.0%} quer dizer que "
        f"{1 - precisao:.0%} das colisões anunciadas nunca acontecem. "
        f"O horizonte de {N_QUADROS - 1 - QUADRO_ATUAL} quadros é longo "
        f"demais para uma rede deste tamanho; a seção [6] mostra o mesmo "
        f"efeito, pior ainda, no mundo contrafactual.", recuo=4))
    return rolagens, (acertos, falsos, faltantes, precisao, revocacao)


def preparar_faltante(preditor, clipes_parseados):
    """Encontra e mede o melhor exemplo de raciocínio sob oclusão."""
    melhor = None
    for indice, (posicoes, velocidades, _ev, cena, observadas, visivel,
                 vel_obs) in enumerate(clipes_parseados):
        previstas, _ancoradas = rolagem_ancorada(
            preditor, observadas, visivel, cena["raios"], vel_obs)
        for i, cor in enumerate(cena["cores"]):
            t = 1
            while t < N_QUADROS:
                if visivel[t, i] or not visivel[t - 1, i]:
                    t += 1
                    continue
                fim = t
                while fim < N_QUADROS and not visivel[fim, i]:
                    fim += 1
                duracao = fim - t
                if fim < N_QUADROS and duracao >= 3:
                    erro = float(np.linalg.norm(previstas[fim, i] - posicoes[fim, i]))
                    candidato = {"clipe": indice, "cor": cor, "objeto": i,
                                 "inicio": t, "fim": fim, "duracao": duracao,
                                 "erro_no_retorno": erro,
                                 "previstas": previstas, "posicoes": posicoes,
                                 "visivel": visivel}
                    if melhor is None or duracao > melhor["duracao"]:
                        melhor = candidato
                t = fim + 1
    linhas = []
    for t in range(max(0, melhor["inicio"] - 2), min(N_QUADROS, melhor["fim"] + 2)):
        i = melhor["objeto"]
        erro = float(np.linalg.norm(
            melhor["previstas"][t, i] - melhor["posicoes"][t, i]))
        linhas.append((t, bool(melhor["visivel"][t, i]),
                       melhor["posicoes"][t, i], melhor["previstas"][t, i], erro))
    melhor["linhas"] = linhas
    return melhor


def imprimir_secao_executor(mundo, demo, preditor):
    """Seção [5]: as quatro famílias de perguntas do CLEVRER."""
    cena = demo["cena"]
    executor = ExecutorSimbolico(mundo)
    cabecalho("[5] EXECUTOR SIMBÓLICO — AS QUATRO FAMÍLIAS DE PERGUNTA", "-")
    print(paragrafo(
        "O question parser (um LSTM, no artigo) traduziria cada pergunta em "
        "linguagem natural para um destes programas funcionais. Aqui os "
        "programas já vêm escritos, para que a atenção fique onde importa: na "
        "EXECUÇÃO. Repare que nenhuma operação abaixo manipula probabilidade "
        "alguma — o executor do NSDR é totalmente simbólico."))

    # ---------------------------------------------------------------- 5.1
    subtitulo("5.1 DESCRITIVA — 'quantas colisões ocorreram?'")
    print()
    resposta = executor.executar([
        ("EVENTOS", {"escopo": "observado"}),
        ("FILTRAR_TIPO", {"tipo": "COLIDE"}),
        ("CONTAR", {}),
    ])
    print(f"    RESPOSTA: {resposta} colisões nos quadros 0 a {QUADRO_ATUAL}.")

    print()
    print("    'qual objeto colidiu com o objeto cinza?'")
    print()
    parceiros = executor.executar([
        ("EVENTOS", {"escopo": "observado"}),
        ("FILTRAR_TIPO", {"tipo": "COLIDE"}),
        ("FILTRAR_OBJETO", {"cor": "cinza"}),
        ("PARCEIROS", {"cor": "cinza"}),
    ])
    print(f"    RESPOSTA: {', '.join(parceiros) if parceiros else 'nenhum'}.")

    # ---------------------------------------------------------------- 5.2
    subtitulo("5.2 PREDITIVA — 'qual evento acontecerá em seguida?'")
    print()
    print(paragrafo(
        f"O passado acabou no quadro {QUADRO_ATUAL}. A lista consultada agora "
        f"não foi observada por ninguém: ela saiu da rolagem do Dynamics "
        f"Predictor.", recuo=4))
    print()
    proximo = executor.executar([
        ("EVENTOS", {"escopo": "previsto"}),
        ("PRIMEIRO", {}),
    ])
    verdadeiros_futuros = [e for e in demo["eventos"] if e["quadro"] > QUADRO_ATUAL]
    print(f"    RESPOSTA (prevista): "
          f"{formatar_evento(proximo) if proximo else 'nenhum evento'}")
    print(f"    Verdade (física):    "
          f"{formatar_evento(verdadeiros_futuros[0]) if verdadeiros_futuros else 'nenhum evento'}")

    # ---------------------------------------------------------------- 5.3
    subtitulo("5.3 CONTRAFACTUAL — 'sem o objeto cinza, qual evento aconteceria?'")
    print()
    print("    Programa da Figura 5.5:")
    print()
    print("      OBJETOS -> FILTRAR_COR(cinza) -> OBTER_CONTRAFACTUAIS -> PERTENCE_A")
    print("                                                                   ^")
    print("                                              PROGRAMA_DE_ESCOLHA -+")
    print()
    eventos_cf_previstos = executor.executar([
        ("OBJETOS", {}),
        ("FILTRAR_COR", {"cor": "cinza"}),
        ("UNICO", {}),
        ("OBTER_CONTRAFACTUAIS", {}),
    ])
    print()
    print(paragrafo(
        "OBTER_CONTRAFACTUAIS apagou o disco cinza do estado inicial e pediu "
        "ao Dynamics Predictor um vídeo inteiro que nunca foi filmado. O "
        "resultado é uma nova lista de eventos, sobre a qual as mesmas "
        "operações simbólicas de sempre voltam a funcionar.", recuo=4))

    alternativas = montar_alternativas(demo, eventos_cf_previstos)
    conjunto_cf_verdadeiro = {(e["tipo"], e["cores"]) for e in demo["eventos_cf"]}
    conjunto_factual = {(e["tipo"], e["cores"]) for e in demo["eventos"]}

    print()
    print("    PROGRAMA_DE_ESCOLHA — as alternativas da pergunta de múltipla escolha:")
    print()
    print(f"      {'alternativa':<34s}{'no clipe?':>11s}{'sem o cinza?':>14s}{'verdade':>10s}")
    print("      " + "-" * 69)
    acertos_alt, negativas_certas = 0, 0
    for letra, alternativa in alternativas:
        chave = (alternativa["tipo"], alternativa["cores"])
        no_clipe = "sim" if chave in conjunto_factual else "não"
        previsto = ExecutorSimbolico(mundo).op_pertence_a(
            alternativa, eventos_cf_previstos)
        verdade = chave in conjunto_cf_verdadeiro
        acertos_alt += int(previsto == verdade)
        negativas_certas += int(previsto == verdade and not verdade)
        rotulo = f"({letra}) {alternativa['tipo']}({', '.join(alternativa['cores'])})"
        print(f"      {rotulo:<34s}{no_clipe:>11s}"
              f"{('SIM' if previsto else 'NÃO'):>14s}"
              f"{('sim' if verdade else 'não'):>10s}")
    print("      " + "-" * 69)
    print()
    print(paragrafo(
        "A coluna 'sem o cinza?' é a resposta do modelo — PERTENCE_A aplicado "
        "ao mundo imaginado. A coluna 'verdade' vem de re-simular a FÍSICA "
        "VERDADEIRA sem o disco cinza; ela existe só porque este mundo é "
        "sintético, e é o que permite auditar o raciocínio contrafactual em "
        "vez de acreditar nele.", recuo=4))
    print()
    print(paragrafo(
        f"CUIDADO COM ESTA TABELA: aqui o modelo acertou "
        f"{acertos_alt} das {len(alternativas)} alternativas, mas "
        f"{negativas_certas} desses acertos são respostas NEGATIVAS ('esse "
        f"evento não aconteceria'), que são baratas — basta a colisão não "
        f"aparecer na lista imaginada. Um clipe não mede nada. A medição de "
        f"verdade, sobre os {N_CLIPES_TESTE} clipes de teste, está na seção "
        f"[6] — e lá o "
        f"resultado é ruim.", recuo=4))

    # ---------------------------------------------------------------- 5.4
    subtitulo("5.4 EXPLICATIVA — por que a remoção do cinza impediu uma colisão")
    impedida = demo["colisao_impedida"]
    cor_a, cor_b = impedida["cores"]
    print()
    print(f"    Pergunta: por que a remoção do objeto cinza impediu a colisão")
    print(f"              entre o objeto {cor_a} e o objeto {cor_b}?")
    print()

    diagnostico = explicar_colisao_impedida(demo, impedida)
    print(f"    Colisão em questão ............ {formatar_evento(impedida)}")
    print(f"    Primeira colisão com o cinza .. "
          f"{formatar_evento(diagnostico['gatilho'])}")
    print(f"    Quadro em que as trajetórias divergem ... {diagnostico['quadro_divergencia']}")
    print(f"    Objeto desviado ............... {diagnostico['cor_desviada']}")
    print()
    print(f"      {'quadro':>6s}  {'com cinza (real)':>18s}  "
          f"{'sem cinza (contraf.)':>21s}  {'desvio':>7s}")
    print("      " + "-" * 60)
    for t, real, contraf, desvio in diagnostico["linhas"]:
        print(f"      {t:6d}  ({real[0]:6.2f}, {real[1]:6.2f})       "
              f"({contraf[0]:6.2f}, {contraf[1]:6.2f})        {desvio:7.3f}")
    print("      " + "-" * 60)
    print()
    print(paragrafo(
        f"Distância mínima entre {cor_a} e {cor_b}, medida NOS QUADROS do "
        f"clipe: {diagnostico['minima_real']:.3f} no mundo real contra "
        f"{diagnostico['minima_contraf']:.3f} no mundo sem o cinza. A soma "
        f"dos raios é {diagnostico['soma_raios']:.3f}: no mundo real o "
        f"encontro mais próximo deixa uma folga de "
        f"{diagnostico['minima_real'] - diagnostico['soma_raios']:.3f}, "
        f"dentro do LIMIAR_CONTATO de {LIMIAR_CONTATO:.2f} — o toque "
        f"acontece ENTRE dois quadros, e o clipe só é amostrado a cada "
        f"{DT:.1f} s. Sem o cinza a folga sobe para "
        f"{diagnostico['minima_contraf'] - diagnostico['soma_raios']:.3f} e "
        f"os dois nunca chegam perto.", recuo=4))
    print()
    print(paragrafo(
        f"RESPOSTA: no quadro {diagnostico['gatilho']['quadro']} o disco cinza "
        f"bate em {diagnostico['cor_desviada']} e desvia sua trajetória; é a "
        f"partir do quadro {diagnostico['quadro_divergencia']} que os dois "
        f"mundos se separam. Sem esse desvio, {diagnostico['cor_desviada']} "
        f"segue em outra direção e a rota que cruzaria com {cor_b} nunca "
        f"chega a existir.", recuo=4))
    print()
    print(paragrafo(
        "Note DE ONDE veio a explicação: da comparação de duas trajetórias "
        "inspecionáveis, quadro a quadro. O mesmo sub-modelo que dá ao sistema "
        "o poder de imaginar mundos alternativos é o que fornece o material "
        "para justificar a resposta.", recuo=4))
    return alternativas


def montar_alternativas(demo, eventos_cf_previstos):
    """Constrói as alternativas de múltipla escolha da pergunta contrafactual.

    Uma alternativa é um evento SEM quadro: só o tipo e os objetos, como nas
    opções escritas das perguntas do CLEVRER. Escolhemos, quando existirem:
    (A) a colisão que o cinza causava e que some sem ele; (B) uma colisão que
    sobrevive nos dois mundos; (C) uma colisão que só existe no mundo sem o
    cinza; (D) uma colisão que não acontece em mundo nenhum.
    """
    cores = demo["cena"]["cores"]
    factuais = {(e["tipo"], e["cores"]) for e in demo["eventos"]
                if e["tipo"] == "COLIDE"}
    previstas = {(e["tipo"], e["cores"]) for e in eventos_cf_previstos
                 if e["tipo"] == "COLIDE"}
    verdadeiras_cf = {(e["tipo"], e["cores"]) for e in demo["eventos_cf"]
                      if e["tipo"] == "COLIDE"}

    escolhidas, vistas = [], set()

    def acrescentar(chave):
        if chave is not None and chave not in vistas:
            vistas.add(chave)
            escolhidas.append({"tipo": chave[0], "cores": chave[1]})

    impedida = demo["colisao_impedida"]
    acrescentar((impedida["tipo"], impedida["cores"]))
    acrescentar(next((c for c in sorted(factuais)
                      if c in verdadeiras_cf), None))
    acrescentar(next((c for c in sorted(verdadeiras_cf | previstas)
                      if c not in factuais), None))
    for a in cores:
        for b in cores:
            if a < b and "cinza" not in (a, b):
                chave = ("COLIDE", tuple(sorted((a, b))))
                if chave not in factuais and chave not in verdadeiras_cf:
                    acrescentar(chave)
                    break
        if len(escolhidas) >= 4:
            break
    return list(zip("ABCDEF", escolhidas[:4]))


def explicar_colisao_impedida(demo, impedida):
    """Compara as trajetórias factual e contrafactual e localiza a divergência."""
    cena, cena_cf = demo["cena"], demo["cena_cf"]
    posicoes, pos_cf = demo["posicoes"], demo["pos_cf"]

    gatilho = next(e for e in demo["eventos"]
                   if e["tipo"] == "COLIDE" and "cinza" in e["cores"])

    quadro_divergencia, cor_desviada = N_QUADROS, cena["cores"][0]
    for cor in cena_cf["cores"]:
        i, j = cena["cores"].index(cor), cena_cf["cores"].index(cor)
        for t in range(N_QUADROS):
            if float(np.linalg.norm(posicoes[t, i] - pos_cf[t, j])) > 0.02:
                if t < quadro_divergencia:
                    quadro_divergencia, cor_desviada = t, cor
                break

    i_a, i_b = (cena["cores"].index(c) for c in impedida["cores"])
    j_a, j_b = (cena_cf["cores"].index(c) for c in impedida["cores"])
    minima_real = float(np.min(np.linalg.norm(
        posicoes[:, i_a] - posicoes[:, i_b], axis=1)))
    minima_contraf = float(np.min(np.linalg.norm(
        pos_cf[:, j_a] - pos_cf[:, j_b], axis=1)))

    i, j = (cena["cores"].index(cor_desviada), cena_cf["cores"].index(cor_desviada))
    linhas = []
    for t in range(max(0, quadro_divergencia - 2),
                   min(N_QUADROS, quadro_divergencia + 4)):
        desvio = float(np.linalg.norm(posicoes[t, i] - pos_cf[t, j]))
        linhas.append((t, posicoes[t, i], pos_cf[t, j], desvio))

    return {"gatilho": gatilho, "quadro_divergencia": quadro_divergencia,
            "cor_desviada": cor_desviada, "linhas": linhas,
            "minima_real": minima_real, "minima_contraf": minima_contraf,
            "soma_raios": float(cena["raios"][i_a] + cena["raios"][i_b])}


def imprimir_secao_validacao(preditor, clipes_teste, demo, mundo,
                             rolagens):
    """Seção [6]: o contrafactual da rede contra o contrafactual da física."""
    cabecalho("[6] VALIDAÇÃO DO RACIOCÍNIO CONTRAFACTUAL", "-")
    print(paragrafo(
        "Para cada clipe de teste que contém um disco cinza, removemos esse "
        "disco do estado inicial e produzimos DOIS mundos alternativos: um "
        "imaginado pelo Dynamics Predictor e outro obtido re-simulando a "
        "física verdadeira. Depois comparamos as duas listas de eventos. "
        "Nenhum artigo consegue fazer isso com vídeo real — o mundo "
        "contrafactual não pode ser filmado."))

    linhas = []
    totais = {"COLIDE": [0, 0, 0], "CAMPO": [0, 0, 0]}
    for indice, (posicoes, velocidades, _ev, cena) in enumerate(clipes_teste):
        if "cinza" not in cena["cores"]:
            continue
        alvo = cena["cores"].index("cinza")
        cena_cf = remover_objeto(cena, alvo)
        pos_verd, vel_verd, eventos_verd = simular_clipe(cena_cf)
        pos_prev, vel_prev = preditor.rolar(
            cena_cf["pos0"], cena_cf["vel0"], cena_cf["raios"], N_QUADROS - 1)
        eventos_prev = detectar_eventos(pos_prev, vel_prev,
                                        cena_cf["cores"], cena_cf["raios"])
        acertos_c, falsos_c, faltantes_c = casar_eventos(
            eventos_prev, eventos_verd, tipos={"COLIDE"})
        acertos_e, falsos_e, faltantes_e = casar_eventos(
            eventos_prev, eventos_verd, tipos={"ENTRA", "SAI"})
        for chave, (a, f, m) in (("COLIDE", (acertos_c, falsos_c, faltantes_c)),
                                 ("CAMPO", (acertos_e, falsos_e, faltantes_e))):
            totais[chave][0] += len(a)
            totais[chave][1] += len(f)
            totais[chave][2] += len(m)
        erro_final = float(np.mean(np.linalg.norm(
            pos_prev[-1] - pos_verd[-1], axis=1)))
        linhas.append((indice, len(cena_cf["cores"]), len(acertos_c),
                       len(falsos_c), len(faltantes_c), erro_final))

    print()
    print(f"    {'clipe':>5s}  {'obj.':>4s}  {'colisões ok':>11s}  {'falsas':>6s}"
          f"  {'perdidas':>8s}  {'erro final':>10s}")
    print("    " + "-" * 56)
    for indice, n_obj, ok, falsas, perdidas, erro in linhas[:CLIPES_NA_TABELA]:
        print(f"    {indice:5d}  {n_obj:4d}  {ok:11d}  {falsas:6d}"
              f"  {perdidas:8d}  {erro:10.3f}")
    if len(linhas) > CLIPES_NA_TABELA:
        print(f"    {'...':>5s}   ... e mais "
              f"{len(linhas) - CLIPES_NA_TABELA} clipes com disco cinza")
    print("    " + "-" * 56)

    print()
    print(f"    {'tipo de evento':<22s}{'acertos':>9s}{'falsos':>8s}"
          f"{'perdidos':>10s}{'precisão':>10s}{'revocação':>11s}")
    print("    " + "-" * 70)
    resumo = {}
    for chave, rotulo in (("COLIDE", "COLIDE"), ("CAMPO", "ENTRA / SAI")):
        a, f, m = totais[chave]
        precisao = a / max(1, a + f)
        revocacao = a / max(1, a + m)
        resumo[chave] = (a, f, m, precisao, revocacao)
        print(f"    {rotulo:<22s}{a:>9d}{f:>8d}{m:>10d}"
              f"{precisao:>9.1%}{revocacao:>10.1%}")
    print("    " + "-" * 70)
    print()
    erro_medio_cf = float(np.mean([linha[5] for linha in linhas]))
    diagonal = math.hypot(ARENA_X[1] - ARENA_X[0], ARENA_Y[1] - ARENA_Y[0])
    print(paragrafo(
        f"{len(linhas)} clipes de teste tinham disco cinza. O erro final médio "
        f"de posição no mundo contrafactual, depois de {N_QUADROS - 1} passos "
        f"sem observação nenhuma, é de {erro_medio_cf:.3f} unidade.", recuo=4))

    # ---- leitura honesta: este é o experimento que FALHA -------------------
    ac, fc, mc, prec_c, rev_c = resumo["COLIDE"]
    ae, fe, me, prec_e, rev_e = resumo["CAMPO"]
    pior = max(linhas[:CLIPES_NA_TABELA], key=lambda linha: linha[4])
    print()
    print(paragrafo(
        f"LEITURA HONESTA DA TABELA: o raciocínio contrafactual NÃO funciona "
        f"bem aqui, e as duas linhas acima dizem isso sem rodeios. Das "
        f"{ac + mc} colisões que a física verdadeira produz nos mundos sem o "
        f"disco cinza, o Dynamics Predictor reencontra {ac} ({rev_c:.1%}), "
        f"deixa passar {mc} e ainda inventa {fc} que nunca acontecem. Nos "
        f"eventos de campo o placar é melhor, mas continua ruim: {ae} acertos "
        f"contra {me} perdidos e {fe} falsos ({rev_e:.1%} de revocação). O "
        f"clipe {pior[0]}, na tabela acima, é o retrato do problema: "
        f"acertos={pior[2]}, falsas={pior[3]}, perdidas={pior[4]}. "
        f"Quem lesse só a seção 5.3, onde as quatro alternativas do clipe de "
        f"demonstração saem certas, concluiria o contrário — e concluiria "
        f"errado.", recuo=4))
    print()
    print(paragrafo(
        f"POR QUE O ERRO SE ACUMULA: a rolagem contrafactual é feita em MALHA "
        f"ABERTA. A saída de um passo é a entrada do passo seguinte, e nenhuma "
        f"observação corrige o percurso ao longo dos {N_QUADROS - 1} passos — "
        f"o clipe inteiro é imaginado. O pequeno erro de Δv de cada passo é "
        f"integrado na posição, de modo que o desvio cresce mais ou menos com "
        f"o quadrado do tempo. Foi exatamente essa curva que a seção [4] "
        f"mediu: {rolagens[5][0]:.3f} unidade em 5 passos, "
        f"{rolagens[10][0]:.3f} em 10 e {rolagens[N_QUADROS - 1][0]:.3f} em "
        f"{N_QUADROS - 1}, contra uma diagonal de arena de {diagonal:.2f}. "
        f"Aqui, no mundo contrafactual, o erro final médio é da mesma ordem "
        f"({erro_medio_cf:.3f}).",
        recuo=4))
    print()
    print(paragrafo(
        f"E há um agravante próprio dos EVENTOS, que são discretos: a folga "
        f"entre dois discos cruza zero ou não cruza. Um erro de posição menor "
        f"que um raio ({RAIO_MIN:.2f} a {RAIO_MAX:.2f}) já decide entre "
        f"'colidiu' e 'não colidiu'. Pior: cada colisão que a rede erra de "
        f"lugar troca as velocidades de dois discos na hora errada e reescreve "
        f"TODO o resto do vídeo imaginado — as colisões seguintes passam a "
        f"acontecer entre outros pares, em outros quadros. É um regime "
        f"caótico, e é por isso que um erro de dinâmica que parece modesto "
        f"produz listas de eventos tão diferentes.", recuo=4))
    print()
    print(paragrafo(
        f"O que sobrevive ao teste, portanto, é o MECANISMO — apagar um "
        f"objeto do estado inicial, re-simular, aplicar o mesmo executor "
        f"simbólico sobre a lista imaginada — e não a exatidão do resultado. "
        f"Este é um resultado negativo, e ele está aqui de propósito: a "
        f"causa provável é o HORIZONTE de simulação, não a arquitetura: "
        f"aqui o vídeo alternativo inteiro, os {N_QUADROS - 1} passos, sai "
        f"da rede. O teste direto está ao alcance do leitor — reduzir "
        f"N_QUADROS encurta a rolagem contrafactual, e o placar desta tabela "
        f"tem de melhorar; se não melhorar, a causa é outra.", recuo=4))

    subtitulo("O mundo contrafactual do clipe de demonstração, lado a lado")
    _pos, _vel, _cores, eventos_previstos_cf = mundo.simular_sem("cinza")
    print()
    print("    Imaginado pelo Dynamics Predictor:")
    imprimir_lista_de_eventos(eventos_previstos_cf, recuo=6)
    print()
    print("    Obtido re-simulando a física verdadeira:")
    imprimir_lista_de_eventos(demo["eventos_cf"], recuo=6)
    return resumo


def item_numerado(numero, texto):
    """Formata um item numerado da interpretação dentro de LARGURA colunas."""
    return textwrap.fill(" ".join(texto.split()), width=LARGURA,
                         initial_indent=f"  {numero}. ",
                         subsequent_indent="     ",
                         break_on_hyphens=False)


def imprimir_interpretacao(resumo_cf, colisoes_futuras, rolagens, preditor):
    """Seção [7]: o fecho que amarra tudo aos conceitos do capítulo."""
    cabecalho("[7] INTERPRETAÇÃO — o que você acabou de ver")
    (acertos_cf, falsos_cf, perdidos_cf,
     precisao_cf, revocacao_cf) = resumo_cf["COLIDE"]
    acertos_fut, falsos_fut, perdidos_fut, prec_fut, rev_fut = colisoes_futuras
    erro_final = rolagens[N_QUADROS - 1][0]

    itens = [
        """NSDR = NSCL + UM MOTOR DE FÍSICA APRENDIDO. A arquitetura é a mesma
        de antes — extrator perceptivo, tradutor de perguntas, executor
        simbólico — com uma peça a mais: o Dynamics Predictor. Essa peça é o
        que transforma um sistema que DESCREVE em um sistema que tenta PREVER,
        EXPLICAR e IMAGINAR.""",

        """O EXECUTOR AQUI É TOTALMENTE SIMBÓLICO. No NSCL, o executor era
        quase-simbólico: propagava distribuições de probabilidade sobre
        objetos. No NSDR ele opera sobre listas discretas de eventos com
        filtros e lógica, como as das seções 5.1 a 5.4. Toda a incerteza foi
        empurrada para as bordas (o parser e o preditor); o raciocínio, no
        miolo, é exato — e é exatamente por isso que ele herda, sem filtro
        nenhum, os erros do preditor.""",

        f"""PREVER É SIMULAR, NÃO CLASSIFICAR. A resposta preditiva não saiu
        de um classificador de "próximo evento": saiu de rolar a física
        aprendida por {N_QUADROS - 1 - QUADRO_ATUAL} quadros e LER os eventos
        da trajetória imaginada. O placar dessa leitura, sobre os clipes de
        teste, é modesto: {acertos_fut} colisões futuras acertadas,
        {falsos_fut} imaginadas e {perdidos_fut} perdidas — precisão
        {prec_fut:.0%}, revocação {rev_fut:.0%}. Um classificador teria de ter
        visto, no treino, exatamente aquela configuração; o simulador não
        precisa, mas paga o preço em exatidão.""",

        f"""O CONTRAFACTUAL É O TESTE MAIS DURO — e AQUI ELE FALHA. "Sem o
        objeto cinza, qual evento aconteceria?" é uma pergunta sobre um vídeo
        que não existe. O MECANISMO da Figura 5.5 funciona: apagar o objeto do
        estado inicial, re-simular, aplicar o mesmo executor simbólico sobre a
        lista imaginada. O RESULTADO, não: precisão de {precisao_cf:.0%} e
        revocação de {revocacao_cf:.0%} nas colisões do mundo alternativo
        ({acertos_cf} acertadas, {falsos_cf} inventadas, {perdidos_cf}
        perdidas), medidas contra a física verdadeira. Este é um resultado
        NEGATIVO, e está no relatório de propósito: o mundo contrafactual é
        imaginado por {N_QUADROS - 1} passos em malha aberta, sem uma única
        observação para corrigir o rumo. A causa provável é essa, e não a
        arquitetura — a seção [6] mostra o erro se acumulando passo a passo.""",

        """A EXPLICAÇÃO É UM SUBPRODUTO DA SIMULAÇÃO. A seção 5.4 não
        consultou nenhum módulo de explicabilidade: bastou comparar duas
        trajetórias quadro a quadro e apontar onde elas se separam. O mesmo
        sub-modelo que habilita o raciocínio profundo é o que fornece a
        explicação, porque as trajetórias são INSPECIONÁVEIS. Compare com o
        Capítulo 4, em que a explicação tinha de ser reconstruída por fora,
        depois do treino, e de forma aproximada. Note, porém, que uma
        explicação inspecionável não é o mesmo que uma explicação correta:
        ela é tão boa quanto a trajetória que a sustenta.""",

        f"""E OS LIMITES APARECEM NOS NÚMEROS — não no texto. Depois de
        {N_QUADROS - 1} passos sem observação, o erro de posição da rede é de
        {erro_final:.3f} unidade; as colisões perdidas e as colisões
        imaginadas estão nas tabelas das seções [4] e [6], e são muitas. Um
        erro pequeno de dinâmica vira um erro GRANDE de evento, porque eventos
        são discretos: um roçar de poucos centímetros decide entre "colidiu" e
        "não colidiu", e uma colisão fora de lugar reescreve todo o resto do
        vídeo imaginado. O que este exercício demonstra, portanto, é o
        MECANISMO do raciocínio dinâmico e contrafactual, movido por um motor
        de física de brinquedo cujo alcance útil é de poucos quadros. Essa
        fragilidade é o preço de fazer o raciocínio simbólico depender de um
        substrato aprendido — e é o problema de fronteira da IA
        Neuro-Simbólica.""",
    ]
    print()
    for numero, texto in enumerate(itens, start=1):
        print(item_numerado(numero, texto))
        print()
    print("=" * LARGURA)
    print(" Fim do Exercício 4.")
    print("=" * LARGURA)


# ===========================================================================
# PARTE 9 — MAIN
# ===========================================================================

def main():
    """Executa o exercício completo e imprime o relatório didático."""
    cabecalho("EXERCÍCIO 4 — RACIOCÍNIO DINÂMICO E INFERÊNCIA CONTRAFACTUAL (NSDR)")
    print(" Capítulo 5 — Introdução à IA Neuro-Simbólica (Figura 5.5)")
    print(" De imagens paradas para vídeos; de relações espaciais para CAUSALIDADE.")

    rng = np.random.default_rng(SEMENTE)

    # -----------------------------------------------------------------------
    # 1. Os clipes
    # -----------------------------------------------------------------------
    clipes_treino = []
    for k in range(N_CLIPES_TREINO + N_CLIPES_COLISAO):
        cena = sortear_cena(rng, em_rota_de_colisao=(k >= N_CLIPES_TREINO))
        posicoes, velocidades, eventos = simular_clipe(cena)
        clipes_treino.append((posicoes, velocidades, eventos, cena))

    clipes_teste = []
    for _ in range(N_CLIPES_TESTE):
        cena = sortear_cena(rng)
        posicoes, velocidades, eventos = simular_clipe(cena)
        clipes_teste.append((posicoes, velocidades, eventos, cena))

    demo = procurar_clipe_de_demonstracao(np.random.default_rng(SEMENTE + 7))

    # -----------------------------------------------------------------------
    # 2. O video parser
    # -----------------------------------------------------------------------
    clipes_parseados = []
    for posicoes, velocidades, eventos, cena in clipes_teste:
        observadas, visivel = analisar_video(posicoes, cena["cores"])
        vel_obs = estimar_velocidades(observadas, visivel)
        clipes_parseados.append((posicoes, velocidades, eventos, cena,
                                 observadas, visivel, vel_obs))

    obs_demo, vis_demo = analisar_video(demo["posicoes"], demo["cena"]["cores"])
    vel_demo = estimar_velocidades(obs_demo, vis_demo)

    imprimir_secao_mundo(demo)
    imprimir_secao_parser(demo, obs_demo, vis_demo)

    # -----------------------------------------------------------------------
    # 3. O Dynamics Predictor
    # -----------------------------------------------------------------------
    obj, viz, masc, alvo, n_impulso, n_amostras = montar_conjunto_de_treino(
        clipes_treino)
    preditor = PreditorDeDinamica(np.random.default_rng(SEMENTE + 1))
    preditor.ajustar_normalizacao(obj, viz, masc, alvo)
    historico = preditor.treinar(obj, viz, masc, alvo,
                                 np.random.default_rng(SEMENTE + 2))
    imprimir_secao_preditor(preditor, historico, n_impulso, n_amostras)

    # -----------------------------------------------------------------------
    # 4. Avaliação do motor de física
    # -----------------------------------------------------------------------
    faltante = preparar_faltante(preditor, clipes_parseados)
    rolagens, colisoes_futuras = imprimir_secao_avaliacao(
        preditor, clipes_teste, clipes_parseados, faltante)

    # -----------------------------------------------------------------------
    # 5. Executor simbólico sobre o clipe de demonstração
    # -----------------------------------------------------------------------
    eventos_observados = [e for e in demo["eventos"] if e["quadro"] <= QUADRO_ATUAL]
    estado_pos = np.where(np.isnan(obs_demo[0]), demo["posicoes"][0], obs_demo[0])
    estado_vel = np.where(np.isnan(vel_demo[0]), demo["velocidades"][0], vel_demo[0])
    n_passos_futuros = N_QUADROS - 1 - QUADRO_ATUAL
    pos_futuras, vel_futuras = preditor.rolar(
        np.where(np.isnan(obs_demo[QUADRO_ATUAL]),
                 demo["posicoes"][QUADRO_ATUAL], obs_demo[QUADRO_ATUAL]),
        np.where(np.isnan(vel_demo[QUADRO_ATUAL]),
                 demo["velocidades"][QUADRO_ATUAL], vel_demo[QUADRO_ATUAL]),
        demo["cena"]["raios"], n_passos_futuros)
    eventos_previstos = [dict(e, quadro=e["quadro"] + QUADRO_ATUAL)
                         for e in detectar_eventos(pos_futuras, vel_futuras,
                                                   demo["cena"]["cores"],
                                                   demo["cena"]["raios"])]
    mundo = MundoDoClipe(demo["cena"], eventos_observados, eventos_previstos,
                         preditor, (estado_pos, estado_vel))
    imprimir_secao_executor(mundo, demo, preditor)

    # -----------------------------------------------------------------------
    # 6. Validação do contrafactual e fecho
    # -----------------------------------------------------------------------
    resumo_cf = imprimir_secao_validacao(preditor, clipes_teste, demo,
                                         mundo, rolagens)
    imprimir_interpretacao(resumo_cf, colisoes_futuras, rolagens,
                           preditor)


if __name__ == "__main__":
    main()
