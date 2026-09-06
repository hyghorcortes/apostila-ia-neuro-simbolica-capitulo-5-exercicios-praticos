# -*- coding: utf-8 -*-
"""
EXERCÍCIO PRÁTICO 3 — A MISTURA NEURO-SIMBÓLICA: UM NSCL SIMPLIFICADO
=====================================================================

Capítulo 5 — "Introdução à IA Neuro-Simbólica — o Próximo Nível da IA".
Referência visual: Figuras 5.3 (arquitetura híbrida) e 5.4 (visão geral do NSCL).

O QUE ESTE EXERCÍCIO DEMONSTRA
------------------------------
O Exercício 1 mostrou o ingrediente SIMBÓLICO: uma base de conhecimento escrita à
mão, que raciocina com elegância e quebra diante da primeira forma nova. O
Exercício 2 mostrou o ingrediente NEURAL: uma rede que aprende dos pixels, não
precisa de ninguém escrevendo regras — e não sabe raciocinar, além de ser faminta
por dados.

Aqui os dois ingredientes são MISTURADOS, na escala de uma sala de aula, seguindo
os três sub-modelos do NSCL (Figura 5.4):

  (a) IMAGE PARSER      : a cena é segmentada em objetos por conectividade de
                          pixels (rotulagem de componentes conexas escrita à mão)
                          e cada máscara de objeto passa por um MLP treinado do
                          zero, que devolve uma DISTRIBUIÇÃO DE PROBABILIDADE
                          sobre forma, cor, tamanho e material.

  (b) BASE DE CONHECIMENTO : as saídas do perceptor viram fatos probabilísticos —
                          forma(obj1, cubo)=0.98, cor(obj1, azul)=0.99,
                          a_esquerda_de(obj1, obj2)=1.0. Ninguém a escreveu: ela
                          saiu da rede. É o passo que faltava no Exercício 1.

  (c) QUESTION PARSER   : um parser semântico APRENDIDO (modelos lineares softmax
                          treinados por descida de gradiente sobre sacos de
                          n-gramas e sobre janelas de contexto) traduz a pergunta
                          em português para um PROGRAMA da DSL do capítulo:
                          SCENE, FILTER, RELATE, QUERY, AEQUERY, COUNT, EXIST.

  (d) EXECUTOR QUASE-SIMBÓLICO : o programa roda sobre a base probabilística com
                          operações diferenciáveis — máscaras suaves em [0,1],
                          conjunção por produto, COUNT por soma, EXIST por máximo,
                          AEQUERY por produto interno de distribuições. Cada
                          objeto recebe uma probabilidade de pertencer à saída.

E então as TRÊS COMPARAÇÕES contra uma rede ponta a ponta (versão enxuta da linha
de base do Exercício 2, reimplementada aqui para o experimento ser autocontido):
eficiência de dados, generalização composicional e explicabilidade.

RESTRIÇÕES DE IMPLEMENTAÇÃO
---------------------------
Somente a biblioteca padrão do Python 3 + numpy. Nada de scikit-learn, torch,
scipy ou PIL: a rotulagem de componentes conexas, o MLP, a retropropagação, o
parser semântico, a DSL e o executor probabilístico são todos escritos do zero.
É justamente aí que mora o valor didático — você vê a máquina por dentro.

EXECUÇÃO
--------
    python3 exercicio_pratico_3_nscl_simplificado.py

Sem argumentos, sem arquivos de entrada, sem interação, sem rede. A saída é um
relatório de texto no terminal. A semente aleatória é fixa, então o relatório é
sempre o mesmo — você pode conferir seus números com os do colega ao lado.
"""

import math
import os
import textwrap
from collections import Counter

# Uma única thread de álgebra linear. Além de tornar o tempo de execução
# previsível, isso fixa a ORDEM das somas em ponto flutuante — e é o que garante
# que duas execuções produzam um relatório idêntico byte a byte.
for _variavel in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variavel, "1")

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTES GLOBAIS
# ---------------------------------------------------------------------------

SEMENTE = 42                 # semente fixa => saída 100% reprodutível
LARGURA = 78                 # largura das linhas do relatório

# --- o mundo de cenas (um CLEVR de bolso, em duas dimensões) ---------------
ALTURA_CENA = 19             # linhas da grade de pixels
LARGURA_CENA = 40            # colunas da grade de pixels
N_CANAIS = 4                 # R, G, B e brilho (reflexo especular do material)
LADO_RECORTE = 7             # janela quadrada centrada em cada objeto
DIM_RECORTE = LADO_RECORTE * LADO_RECORTE * N_CANAIS
MAX_OBJETOS = 5              # objetos por cena, no máximo
COLUNAS_SLOT = (5, 15, 25, 35)   # centros possíveis (x) — bem separados
LINHAS_SLOT = (4, 14)            # centros possíveis (y)
RUIDO_PIXEL = 0.22           # ruído gaussiano por pixel do objeto
JITTER_COR = 0.09            # variação de tonalidade de objeto para objeto
LIMIAR_RELACAO = 3           # distância mínima (em pixels) para valer uma relação

# --- tamanho dos conjuntos -------------------------------------------------
CENAS_TREINO = 500
CENAS_TESTE = 120
PERGUNTAS_POR_CENA = 4
TAMANHOS_DE_TREINO = (25, 50, 100, 250, 500)   # curva de eficiência de dados

# --- image parser (MLP multi-cabeça) --------------------------------------
OCULTA_PERCEPTOR = 48
EPOCAS_LICOES = (10, 12, 12)   # currículo: uma quantidade de épocas por lição
LOTE_PERCEPTOR = 32
TAXA_PERCEPTOR = 0.05

# --- question parser (modelos lineares esparsos) --------------------------
MAX_CARACTERISTICAS_SACO = 400
JANELA_ETIQUETADOR = (-2, -1, 0, 1, 2, 3)
EPOCAS_PARSER = (10, 16)        # currículo do parser: perguntas simples, depois todas
LOTE_PARSER = 64
TAXA_PARSER = 0.50
ABANDONO_PARSER = 0.25         # fração da janela apagada ao acaso no treino

# --- linha de base ponta a ponta ------------------------------------------
OCULTA_PONTA_A_PONTA = 48
EPOCAS_PONTA_A_PONTA = 25
LOTE_PONTA_A_PONTA = 128
TAXA_PONTA_A_PONTA = 0.15

# --- vocabulário do mundo --------------------------------------------------
ATRIBUTOS = ("forma", "cor", "tamanho", "material")
VALORES = {
    "forma": ("esfera", "cubo", "cilindro"),
    "cor": ("vermelho", "azul", "verde", "amarelo"),
    "tamanho": ("grande", "pequeno"),
    "material": ("metal", "fosco"),
}
INDICE_VALOR = {a: {v: i for i, v in enumerate(vs)} for a, vs in VALORES.items()}
ATRIBUTO_DO_VALOR = {v: a for a, vs in VALORES.items() for v in vs}
TODOS_OS_VALORES = tuple(v for a in ATRIBUTOS for v in VALORES[a])   # 11 valores
RELACOES = ("esquerda", "direita", "frente", "atras")
RELACOES_EM_PERGUNTAS = ("esquerda", "direita", "frente")

# Respostas possíveis (o vocabulário de saída da rede ponta a ponta).
RESPOSTAS = (("sim", "nao")
             + tuple(str(i) for i in range(MAX_OBJETOS + 1))
             + TODOS_OS_VALORES)
INDICE_RESPOSTA = {r: i for i, r in enumerate(RESPOSTAS)}

CORES_RGB = {
    "vermelho": (0.90, 0.16, 0.14),
    "azul": (0.18, 0.34, 0.92),
    "verde": (0.16, 0.74, 0.30),
    "amarelo": (0.94, 0.86, 0.16),
}

# Combinações (tipo de pergunta x valor de atributo) retidas do treino no
# experimento de generalização composicional.
COMBINACOES_RETIDAS = (
    ("CONTAGEM", "cilindro"),
    ("CONSULTA", "metal"),
    ("EXISTENCIA", "pequeno"),
)


# ===========================================================================
# PARTE 1 — O MUNDO DE CENAS: RENDERIZAÇÃO E SEGMENTAÇÃO
# ===========================================================================

def pertence_a_forma(forma, dx, dy, raio):
    """Diz se o pixel deslocado de (dx, dy) do centro faz parte da silhueta.

    cubo     : quadrado cheio de lado 2*raio+1;
    esfera   : disco de raio `raio`;
    cilindro : retângulo mais estreito que alto, com os quatro cantos cortados
               (é o que sobra do contorno de um cilindro visto de lado).
    """
    if forma == "cubo":
        return True
    if forma == "esfera":
        return dx * dx + dy * dy <= raio * raio
    return abs(dx) <= raio - 1 and not (abs(dy) == raio and abs(dx) == raio - 1)


SILHUETAS = {}


def silhueta(forma, tamanho):
    """Deslocamentos (dy, dx) dos pixels de um objeto, calculados uma única vez."""
    chave = (forma, tamanho)
    if chave not in SILHUETAS:
        raio = (7 if tamanho == "grande" else 5) // 2
        pontos = [(dy, dx)
                  for dy in range(-raio, raio + 1)
                  for dx in range(-raio, raio + 1)
                  if pertence_a_forma(forma, dx, dy, raio)]
        SILHUETAS[chave] = (np.array([p[0] for p in pontos]),
                            np.array([p[1] for p in pontos]))
    return SILHUETAS[chave]


def desenhar_objeto(grade, objeto, rng):
    """Pinta um objeto na grade de pixels da cena.

    O canal 3 é o 'brilho': objetos de metal têm base alta e um realce especular
    no quadrante superior esquerdo; objetos foscos são quase pretos nesse canal.
    Ruído por pixel e variação de tonalidade impedem que o perceptor decore
    valores exatos — ele precisa mesmo aprender a reconhecer os atributos.
    """
    dys, dxs = silhueta(objeto["forma"], objeto["tamanho"])
    ys, xs = objeto["cy"] + dys, objeto["cx"] + dxs
    eh_metal = objeto["material"] == "metal"
    rgb = (np.array(CORES_RGB[objeto["cor"]]) * (1.03 if eh_metal else 0.94)
           + rng.normal(0.0, JITTER_COR, 3))
    pixels = np.empty((len(ys), N_CANAIS))
    pixels[:, :3] = rgb
    pixels[:, 3] = 0.28 if eh_metal else 0.05
    if eh_metal:
        pixels[:, 3] += np.where((dys < 0) & (dxs < 0), 0.28, 0.0)
    pixels += rng.normal(0.0, RUIDO_PIXEL, pixels.shape)
    # O piso 0.02 garante que todo pixel de objeto seja > 0: o fundo é
    # exatamente zero, então a segmentação por conectividade é exata.
    grade[ys, xs] = np.clip(pixels, 0.02, 1.2)


def gerar_cena(rng, n_objetos):
    """Sorteia uma cena com `n_objetos` objetos em posições bem separadas.

    Os centros saem de uma grade de 8 posições possíveis (4 colunas x 2 linhas),
    com um tremor de +-1 pixel. A separação garante três coisas: os objetos nunca
    se encostam (logo a segmentação é exata), as relações esquerda/direita e
    frente/atrás nunca são ambíguas, e o recorte de 7x7 centrado em um objeto
    cabe inteiro dentro da imagem.
    """
    grade = np.zeros((ALTURA_CENA, LARGURA_CENA, N_CANAIS), dtype=np.float64)
    slots = [(x, y) for x in COLUNAS_SLOT for y in LINHAS_SLOT]
    escolhidos = rng.permutation(len(slots))[:n_objetos]
    objetos = []
    for indice in escolhidos:
        x, y = slots[int(indice)]
        objeto = {a: VALORES[a][int(rng.integers(len(VALORES[a])))] for a in ATRIBUTOS}
        objeto["cx"] = int(x + rng.integers(-1, 2))
        objeto["cy"] = int(y + rng.integers(-1, 2))
        desenhar_objeto(grade, objeto, rng)
        objetos.append(objeto)
    objetos.sort(key=lambda o: (o["cx"], o["cy"]))
    return {"grade": grade, "objetos": objetos}


def rotular_componentes(mascara):
    """Rotulagem de componentes conexas (4-vizinhança), escrita à mão.

    É o substituto didático do Mask R-CNN do NSCL: sem scipy, sem biblioteca de
    visão. Uma busca em profundidade com pilha explícita percorre cada mancha de
    pixels acesos e lhe atribui um rótulo inteiro.

    Retorna (matriz_de_rotulos, n_componentes).
    """
    altura, largura = mascara.shape
    rotulos = np.zeros((altura, largura), dtype=np.int32)
    n_componentes = 0
    for i, j in zip(*np.nonzero(mascara)):
        if rotulos[i, j]:
            continue
        n_componentes += 1
        rotulos[i, j] = n_componentes
        pilha = [(int(i), int(j))]
        while pilha:
            y, x = pilha.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < altura and 0 <= xx < largura
                        and mascara[yy, xx] and rotulos[yy, xx] == 0):
                    rotulos[yy, xx] = n_componentes
                    pilha.append((yy, xx))
    return rotulos, n_componentes


def segmentar_cena(cena):
    """Segmenta a cena e prepara tudo o que o resto do sistema consome.

    Acrescenta ao dicionário da cena:
      componentes : lista de dicionários com centro, recorte 7x7x4 e nº de pixels;
      recortes    : matriz (n_objetos, DIM_RECORTE) — a entrada do perceptor;
      rotulos     : matriz (n_objetos, 4) com os índices verdadeiros dos atributos;
      relacoes    : dicionário de matrizes n x n com as relações espaciais;
      mapa        : matriz de rótulos, usada para desenhar a cena em ASCII.

    Repare que o recorte de um objeto contém APENAS os pixels da sua componente
    conexa: um vizinho que invada a janela é apagado. É a 'máscara de objeto' do
    Mask R-CNN, aqui obtida de graça pela segmentação.
    """
    grade = cena["grade"]
    mascara = grade.max(axis=2) > 0.0
    mapa, n_componentes = rotular_componentes(mascara)
    meio = LADO_RECORTE // 2
    componentes = []
    for rotulo in range(1, n_componentes + 1):
        ys, xs = np.nonzero(mapa == rotulo)
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
        janela = np.zeros((LADO_RECORTE, LADO_RECORTE, N_CANAIS), dtype=np.float64)
        dy, dx = ys - cy + meio, xs - cx + meio
        dentro = (dy >= 0) & (dy < LADO_RECORTE) & (dx >= 0) & (dx < LADO_RECORTE)
        janela[dy[dentro], dx[dentro]] = grade[ys[dentro], xs[dentro]]
        componentes.append({"cx": cx, "cy": cy, "n_pixels": int(len(ys)),
                            "recorte": janela.ravel()})
    componentes.sort(key=lambda c: (c["cx"], c["cy"]))
    # A grade de posições garante ao menos um pixel de folga entre objetos, de
    # modo que cada objeto vira exatamente uma componente conexa.
    assert n_componentes == len(cena["objetos"]), "objetos encostados na cena"
    cena["componentes"] = componentes
    cena["mapa"] = mapa
    cena["recortes"] = np.array([c["recorte"] for c in componentes], dtype=np.float64)
    cena["rotulos"] = np.array(
        [[INDICE_VALOR[a][o[a]] for a in ATRIBUTOS] for o in cena["objetos"]],
        dtype=np.int64)
    cena["relacoes"] = relacoes_dos_centros([(c["cx"], c["cy"]) for c in componentes])
    return cena


def relacoes_dos_centros(centros):
    """Matrizes de relação espacial. `rel[r][i, j] = 1` <=> 'i está à r de j'.

    As relações NÃO são aprendidas: saem da geometria dos centróides devolvidos
    pela segmentação — o mesmo que o NSCL faz ao derivar relações das caixas
    delimitadoras. Linha maior significa mais perto do observador ('frente').
    """
    n = len(centros)
    rel = {r: np.zeros((n, n), dtype=np.float64) for r in RELACOES}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = centros[i][0] - centros[j][0]
            dy = centros[i][1] - centros[j][1]
            if dx <= -LIMIAR_RELACAO:
                rel["esquerda"][i, j] = 1.0
            if dx >= LIMIAR_RELACAO:
                rel["direita"][i, j] = 1.0
            if dy >= LIMIAR_RELACAO:
                rel["frente"][i, j] = 1.0
            if dy <= -LIMIAR_RELACAO:
                rel["atras"][i, j] = 1.0
    return rel


def desenhar_cena_em_ascii(cena):
    """Devolve a cena como linhas de texto: cada objeto aparece com seu número."""
    mapa = cena["mapa"]
    # `mapa` usa a ordem em que o flood fill encontrou as manchas; queremos a
    # ordem canônica (esquerda para a direita), que é a dos componentes.
    ordem = {}
    for indice, comp in enumerate(cena["componentes"], start=1):
        ordem[int(mapa[comp["cy"], comp["cx"]])] = indice
    linhas = []
    for i in range(ALTURA_CENA):
        linha = "".join("." if mapa[i, j] == 0 else str(ordem.get(int(mapa[i, j]), 0))
                        for j in range(LARGURA_CENA))
        linhas.append(linha)
    return linhas


# ===========================================================================
# PARTE 2 — DUAS REDES ESCRITAS DO ZERO (a única "matemática pesada" do script)
# ===========================================================================

def softmax(z):
    """Softmax estável por linha."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def gradiente_de_cabecas(Z, Y, fatias):
    """Gradiente da entropia cruzada somada sobre várias cabeças softmax.

    `Y[:, k] = -1` marca um rótulo AUSENTE: a cabeça k simplesmente não recebe
    gradiente naquele exemplo. É esse detalhe que permite o currículo — na
    primeira lição, tamanho e material ainda não são supervisionados.
    """
    dZ = np.zeros_like(Z)
    perda = 0.0
    for k, (ini, fim) in enumerate(fatias):
        P = softmax(Z[:, ini:fim])
        rotulos = Y[:, k]
        valido = rotulos >= 0
        seguros = np.where(valido, rotulos, 0)
        linhas = np.arange(len(rotulos))
        perda -= float(np.log(np.maximum(P[linhas, seguros], 1e-12))[valido].sum())
        P[linhas, seguros] -= 1.0
        dZ[:, ini:fim] = P * valido[:, None]
    return dZ, perda


def montar_fatias(tamanhos):
    """Converte [3, 4, 2, 2] em [(0,3), (3,7), (7,9), (9,11)]."""
    fatias, inicio = [], 0
    for t in tamanhos:
        fatias.append((inicio, inicio + t))
        inicio += t
    return fatias


class RedeDensa:
    """MLP com (opcionalmente) uma camada oculta ReLU e várias cabeças softmax.

    Serve a dois papéis no exercício: é o IMAGE PARSER (entrada = recorte de
    pixels, quatro cabeças de atributo) e é a LINHA DE BASE PONTA A PONTA
    (entrada = cena inteira + pergunta, uma cabeça sobre o vocabulário de
    respostas). Treino por descida de gradiente estocástica com momento.
    """

    def __init__(self, n_entrada, n_oculta, tamanhos_cabecas, rng):
        self.n_oculta = n_oculta
        self.fatias = montar_fatias(tamanhos_cabecas)
        total = sum(tamanhos_cabecas)
        if n_oculta:
            self.W1 = rng.normal(0.0, math.sqrt(2.0 / n_entrada), (n_entrada, n_oculta))
            self.b1 = np.zeros(n_oculta)
            self.W2 = rng.normal(0.0, math.sqrt(1.0 / n_oculta), (n_oculta, total))
        else:
            self.W1 = self.b1 = None
            self.W2 = np.zeros((n_entrada, total))
        self.b2 = np.zeros(total)
        self._zerar_velocidades()

    def _zerar_velocidades(self):
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        if self.n_oculta:
            self.vW1 = np.zeros_like(self.W1)
            self.vb1 = np.zeros_like(self.b1)

    def propagar(self, X):
        """Retorna (ativações ocultas, logits)."""
        if self.n_oculta:
            H = np.maximum(0.0, X @ self.W1 + self.b1)
            return H, H @ self.W2 + self.b2
        return None, X @ self.W2 + self.b2

    def distribuicoes(self, X):
        """Lista de matrizes de probabilidade, uma por cabeça."""
        _, Z = self.propagar(X)
        return [softmax(Z[:, ini:fim]) for ini, fim in self.fatias]

    def treinar(self, X, Y, epocas, taxa, lote, rng, momento=0.9):
        """SGD com momento. `Y` é uma matriz (n, n_cabecas) de inteiros."""
        n = len(X)
        if n == 0:
            return
        for _ in range(epocas):
            ordem = rng.permutation(n)
            for inicio in range(0, n, lote):
                idx = ordem[inicio:inicio + lote]
                xb, yb = X[idx], Y[idx]
                H, Z = self.propagar(xb)
                dZ, _ = gradiente_de_cabecas(Z, yb, self.fatias)
                dZ /= max(1, len(idx))
                base = H if self.n_oculta else xb
                gW2, gb2 = base.T @ dZ, dZ.sum(axis=0)
                if self.n_oculta:
                    dH = (dZ @ self.W2.T) * (H > 0)
                    gW1, gb1 = xb.T @ dH, dH.sum(axis=0)
                    self.vW1 = momento * self.vW1 - taxa * gW1
                    self.vb1 = momento * self.vb1 - taxa * gb1
                    self.W1 += self.vW1
                    self.b1 += self.vb1
                self.vW2 = momento * self.vW2 - taxa * gW2
                self.vb2 = momento * self.vb2 - taxa * gb2
                self.W2 += self.vW2
                self.b2 += self.vb2


class ModeloEsparso:
    """Modelo linear multi-cabeça sobre características binárias esparsas.

    Cada exemplo é uma LISTA DE ÍNDICES de características ativas, preenchida
    com o índice 0 até um comprimento fixo. O peso da característica 0 é mantido
    em zero, de modo que o preenchimento (e qualquer palavra nunca vista) não
    contribui para nada. Os gradientes são acumulados com `np.bincount`, o que
    torna o treino barato mesmo com dezenas de milhares de exemplos.

    É o coração do QUESTION PARSER: nada de casamento de padrões, apenas pesos
    aprendidos a partir de pares (pergunta, programa).
    """

    def __init__(self, n_caracteristicas, tamanhos_cabecas):
        self.fatias = montar_fatias(tamanhos_cabecas)
        total = sum(tamanhos_cabecas)
        self.W = np.zeros((n_caracteristicas, total))
        self.b = np.zeros(total)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def pontuar(self, IDX):
        """Logits: soma das linhas de peso das características ativas."""
        return self.W[IDX].sum(axis=1) + self.b

    def distribuicoes(self, IDX):
        Z = self.pontuar(IDX)
        return [softmax(Z[:, ini:fim]) for ini, fim in self.fatias]

    def treinar(self, IDX, Y, epocas, taxa, lote, rng, momento=0.9, decaimento=1e-5,
                abandono=0.0):
        """`abandono` apaga características ao acaso a cada passo.

        É o truque que faz o parser sobreviver a um fraseado novo: se durante o
        treino parte da janela de contexto some, o modelo não pode depender de
        que TODAS as palavras vizinhas sejam conhecidas — e é exatamente isso
        que acontece diante de uma pergunta escrita de outro jeito.
        """
        n = len(IDX)
        if n == 0:
            return
        n_caract = self.W.shape[0]
        comprimento = IDX.shape[1]
        gW = np.empty_like(self.W)
        for _ in range(epocas):
            ordem = rng.permutation(n)
            for inicio in range(0, n, lote):
                sel = ordem[inicio:inicio + lote]
                idx, yb = IDX[sel], Y[sel]
                if abandono > 0.0:
                    idx = np.where(rng.random(idx.shape) < abandono, 0, idx)
                Z = self.pontuar(idx)
                dZ, _ = gradiente_de_cabecas(Z, yb, self.fatias)
                dZ /= max(1, len(sel))
                planos = idx.ravel()
                for c in range(self.W.shape[1]):
                    gW[:, c] = np.bincount(planos,
                                           weights=np.repeat(dZ[:, c], comprimento),
                                           minlength=n_caract)
                gW += decaimento * self.W
                self.vW = momento * self.vW - taxa * gW
                self.vb = momento * self.vb - taxa * (dZ.sum(axis=0) + decaimento * self.b)
                self.W += self.vW
                self.b += self.vb
                self.W[0] = 0.0     # a característica de preenchimento nunca pesa


# ===========================================================================
# PARTE 3 — IMAGE PARSER: O SUB-MODELO NEURAL QUE ENXERGA OS OBJETOS
# ===========================================================================

def objetos_de_cenas(cenas):
    """Empilha recortes e rótulos de atributo de uma lista de cenas."""
    if not cenas:
        return np.zeros((0, DIM_RECORTE)), np.zeros((0, len(ATRIBUTOS)), dtype=np.int64)
    X = np.concatenate([c["recortes"] for c in cenas], axis=0)
    Y = np.concatenate([c["rotulos"] for c in cenas], axis=0)
    return X, Y


def avaliar_perceptor(perceptor, cenas):
    """Acurácia por atributo do perceptor sobre uma lista de cenas."""
    X, Y = objetos_de_cenas(cenas)
    if len(X) == 0:
        return {a: float("nan") for a in ATRIBUTOS}
    probs = perceptor.distribuicoes(X)
    return {a: float(np.mean(np.argmax(probs[k], axis=1) == Y[:, k]))
            for k, a in enumerate(ATRIBUTOS)}


def treinar_perceptor_com_curriculo(cenas, rng, cenas_validacao=None, registro=None):
    """Treina o image parser em três lições, do simples para o complexo.

    Lição 1 : só cenas com 1 objeto e só os conceitos 'forma' e 'cor'
              (tamanho e material entram como rótulo ausente, -1).
    Lição 2 : cenas com 2 ou 3 objetos, agora com os quatro atributos.
    Lição 3 : cenas cheias, com 4 ou 5 objetos — vizinhos invadem a janela de
              recorte e a máscara da componente conexa precisa fazer seu papel.

    É a 'estratégia de aprendizado incremental' descrita no capítulo: conceitos
    simples primeiro, cenas povoadas e conceitos relacionais depois. Se `registro`
    for uma lista, cada lição acrescenta a ela uma linha de diagnóstico.
    """
    perceptor = RedeDensa(DIM_RECORTE, OCULTA_PERCEPTOR,
                          [len(VALORES[a]) for a in ATRIBUTOS], rng)
    licoes = [
        ("1", "forma e cor (1 objeto)", lambda k: k == 1, (0, 1)),
        ("2", "+ tamanho e material (2-3)", lambda k: k in (2, 3), (0, 1, 2, 3)),
        ("3", "+ cenas cheias (4-5)", lambda k: k >= 4, (0, 1, 2, 3)),
    ]
    for (nome, descricao, filtro, cabecas), epocas in zip(licoes, EPOCAS_LICOES):
        subconjunto = [c for c in cenas if filtro(len(c["objetos"]))]
        X, Y = objetos_de_cenas(subconjunto)
        if len(X):
            Y = Y.copy()
            for k in range(len(ATRIBUTOS)):
                if k not in cabecas:
                    Y[:, k] = -1        # atributo ainda não ensinado
            perceptor.treinar(X, Y, epocas, TAXA_PERCEPTOR, LOTE_PERCEPTOR, rng)
        if registro is not None:
            acuracias = (avaliar_perceptor(perceptor, cenas_validacao)
                         if cenas_validacao else {})
            registro.append({"licao": nome, "descricao": descricao,
                             "cenas": len(subconjunto), "objetos": len(X),
                             "acuracias": acuracias})
    return perceptor


# ===========================================================================
# PARTE 4 — A BASE DE CONHECIMENTO PROBABILÍSTICA (ninguém a escreveu)
# ===========================================================================

class BaseDeConhecimento:
    """Fatos sobre uma cena, cada um com uma probabilidade.

    `dist[atributo]` é uma matriz (n_objetos, n_valores) cujas linhas somam 1.
    `rel[relacao][i, j] = 1` significa 'o objeto i está à <relacao> do objeto j'.
    """

    def __init__(self, distribuicoes, relacoes, n_objetos):
        self.dist = distribuicoes
        self.rel = relacoes
        self.n = n_objetos

    def prob_valor(self, valor):
        """Vetor com P(objeto tem o valor) para cada objeto da cena."""
        atributo = ATRIBUTO_DO_VALOR[valor]
        return self.dist[atributo][:, INDICE_VALOR[atributo][valor]]


def base_verdadeira(cena):
    """Base de conhecimento montada a partir da anotação real da cena.

    Serve de ORÁCULO: é ela que produz a resposta correta de cada pergunta, do
    mesmo modo que o CLEVR gera as respostas a partir da descrição da cena que
    ele mesmo renderizou. O sistema híbrido nunca a vê.
    """
    if "base" not in cena:
        n = len(cena["objetos"])
        dist = {}
        for k, atributo in enumerate(ATRIBUTOS):
            M = np.zeros((n, len(VALORES[atributo])))
            M[np.arange(n), cena["rotulos"][:, k]] = 1.0
            dist[atributo] = M
        cena["base"] = BaseDeConhecimento(dist, cena["relacoes"], n)
    return cena["base"]


def base_percebida(cena, perceptor):
    """Base de conhecimento GERADA PELA REDE a partir dos pixels.

    Este é o passo (2) da arquitetura híbrida da Figura 5.3: a rede neural
    constrói automaticamente a base de conhecimento que, no Exercício 1, um
    humano tinha de escrever à mão, fato por fato.
    """
    probs = perceptor.distribuicoes(cena["recortes"])
    dist = {a: probs[k] for k, a in enumerate(ATRIBUTOS)}
    return BaseDeConhecimento(dist, cena["relacoes"], len(cena["componentes"]))


# ===========================================================================
# PARTE 5 — A DSL E O EXECUTOR QUASE-SIMBÓLICO
# ===========================================================================

def distribuicao_ponderada(base, atributo, mascara):
    """Distribuição do atributo no conjunto (suave) descrito pela máscara."""
    peso = float(mascara.sum())
    k = len(VALORES[atributo])
    if peso <= 1e-9:
        return np.full(k, 1.0 / k)
    return (mascara[:, None] * base.dist[atributo]).sum(axis=0) / peso


def executar_programa(programa, base):
    """Executa um programa da DSL sobre a base probabilística.

    Todas as operações são DIFERENCIÁVEIS e trabalham com máscaras suaves em
    [0,1] — cada objeto carrega a probabilidade de pertencer ao conjunto
    corrente, exatamente como descreve o capítulo:

      SCENE          : máscara de uns (todos os objetos)
      FILTER(valor)  : conjunção por produto — máscara * P(valor)
      RELATE(r)      : os objetos que têm algum objeto da âncora à sua direção r
      QUERY(atr)     : média das distribuições do atributo, ponderada pela máscara
      COUNT          : soma da máscara
      EXIST          : máximo da máscara
      AEQUERY(atr)   : produto interno das distribuições da âncora e do alvo

    Retorna (resposta, rastro), em que o rastro guarda a máscara depois de cada
    operação — o 'rastro lógico' que a rede ponta a ponta não sabe produzir.
    """
    n = base.n
    mascara = np.ones(n)
    ancora = mascara.copy()
    resposta, rastro = None, []
    for nome, argumento in programa:
        detalhe = None
        if nome == "SCENE":
            mascara = np.ones(n)
        elif nome == "FILTER":
            mascara = mascara * base.prob_valor(argumento)
        elif nome == "RELATE":
            ancora = mascara.copy()
            mascara = np.max(ancora[:, None] * base.rel[argumento], axis=0)
        elif nome == "QUERY":
            d = distribuicao_ponderada(base, argumento, mascara)
            resposta = VALORES[argumento][int(np.argmax(d))]
            detalhe = float(d.max())
        elif nome == "COUNT":
            detalhe = float(mascara.sum())
            resposta = str(min(MAX_OBJETOS, int(round(detalhe))))
        elif nome == "EXIST":
            detalhe = float(mascara.max()) if n else 0.0
            resposta = "sim" if detalhe > 0.5 else "nao"
        elif nome == "AEQUERY":
            d1 = distribuicao_ponderada(base, argumento, ancora)
            d2 = distribuicao_ponderada(base, argumento, mascara)
            detalhe = float(np.dot(d1, d2))
            resposta = "sim" if detalhe > 0.5 else "nao"
        else:
            raise ValueError("operação desconhecida: " + nome)
        rastro.append({"operacao": nome, "argumento": argumento,
                       "mascara": mascara.copy(), "detalhe": detalhe})
    return resposta, rastro


def formatar_programa(programa):
    """Escreve o programa como no capítulo, juntando FILTERs consecutivos."""
    partes = []
    for nome, argumento in programa:
        if nome == "FILTER" and partes and partes[-1].startswith("FILTER("):
            partes[-1] = partes[-1][:-1] + ", " + argumento + ")"
        elif argumento is None:
            partes.append(nome)
        else:
            partes.append("%s(%s)" % (nome, argumento))
    return " -> ".join(partes)


# ===========================================================================
# PARTE 6 — AS PERGUNTAS EM PORTUGUÊS (gabaritos, concordância, anotação)
# ===========================================================================

TIPOS = ("CONTAGEM", "EXISTENCIA", "CONSULTA", "CONSULTA_RELACIONAL", "COMPARACAO")
ATRIBUTOS_E_NENHUM = ATRIBUTOS + ("nenhum",)
# Ordem em que os FILTERs aparecem no programa — cor antes de forma, como na
# Figura 5.4 do capítulo: FILTER(vermelho, cubo).
ORDEM_FILTROS = ("cor", "forma", "tamanho", "material")
RELACOES_E_NENHUMA = RELACOES_EM_PERGUNTAS + ("nenhuma",)

GENERO = {"esfera": "f", "cubo": "m", "cilindro": "m", "objeto": "m"}
PLURAL_NOME = {"esfera": "esferas", "cubo": "cubos", "cilindro": "cilindros",
               "objeto": "objetos"}
FLEXOES = {   # (masc. sing., fem. sing., masc. pl., fem. pl.)
    "vermelho": ("vermelho", "vermelha", "vermelhos", "vermelhas"),
    "azul": ("azul", "azul", "azuis", "azuis"),
    "verde": ("verde", "verde", "verdes", "verdes"),
    "amarelo": ("amarelo", "amarela", "amarelos", "amarelas"),
    "grande": ("grande", "grande", "grandes", "grandes"),
    "pequeno": ("pequeno", "pequena", "pequenos", "pequenas"),
    "metal": ("metálico", "metálica", "metálicos", "metálicas"),
    "fosco": ("fosco", "fosca", "foscos", "foscas"),
}
ARTIGO_DEFINIDO = {("m", False): "o", ("f", False): "a",
                   ("m", True): "os", ("f", True): "as"}
ARTIGO_INDEFINIDO = {"m": "um", "f": "uma"}
QUANTOS = {"m": "quantos", "f": "quantas"}
ARTIGO_ATRIBUTO = {"forma": "a", "cor": "a", "tamanho": "o", "material": "o"}
MESMO = {"forma": "a mesma", "cor": "a mesma",
         "tamanho": "o mesmo", "material": "o mesmo"}
IGUAL = {"forma": "igual à", "cor": "igual à",
         "tamanho": "igual ao", "material": "igual ao"}

# Palavra flexionada -> valor canônico do atributo. Serve só para ANOTAR os
# pares (pergunta, programa) de treino; o parser não recebe esta tabela.
MAPA_INFLEXOES = {}
for _valor, _formas in FLEXOES.items():
    for _f in _formas:
        MAPA_INFLEXOES[_f] = _valor
for _forma in VALORES["forma"]:
    MAPA_INFLEXOES[_forma] = _forma
    MAPA_INFLEXOES[PLURAL_NOME[_forma]] = _forma


def genero_do_filtro(filtros):
    return GENERO[filtros.get("forma", "objeto")]


def sintagma(filtros, plural=False, artigo="definido"):
    """Monta um sintagma nominal com concordância: 'as esferas azuis metálicas'."""
    nucleo = filtros.get("forma", "objeto")
    g = GENERO[nucleo]
    partes = [PLURAL_NOME[nucleo] if plural else nucleo]
    for chave in ("tamanho", "cor", "material"):
        if chave in filtros:
            i = (0 if g == "m" else 1) + (2 if plural else 0)
            partes.append(FLEXOES[filtros[chave]][i])
    texto = " ".join(partes)
    if artigo == "definido":
        return ARTIGO_DEFINIDO[(g, plural)] + " " + texto
    if artigo == "indefinido":
        return ARTIGO_INDEFINIDO[g] + " " + texto
    return texto


def contrair(sintagma_texto):
    """'o cubo azul' -> 'do cubo azul'; 'a esfera' -> 'da esfera'."""
    if sintagma_texto.startswith("o "):
        return "do " + sintagma_texto[2:]
    if sintagma_texto.startswith("a "):
        return "da " + sintagma_texto[2:]
    return "de " + sintagma_texto


def frase_relacional(p):
    return "à sua " + p["rel"]


def tokenizar(texto):
    """Minúsculas, sem pontuação, separado por espaços."""
    limpo = "".join(" " if c in "?.,;:!" else c for c in texto.lower())
    return limpo.split()


# Cada gabarito é (nome, é_de_teste, função). Os gabaritos marcados com True
# ficam FORA do treino: é neles que se mede a generalização de superfície.
GABARITOS = {
    "CONTAGEM": [
        ("CONT-1", False, lambda p: [(QUANTOS[genero_do_filtro(p["fa"])], "O"),
                                     (sintagma(p["fa"], True, None), "A"),
                                     ("há na cena?", "O")]),
        ("CONT-2", False, lambda p: [("conte", "O"),
                                     (sintagma(p["fa"], True, "definido"), "A"),
                                     ("presentes na imagem.", "O")]),
        ("CONT-3", False, lambda p: [("qual é o número de", "O"),
                                     (sintagma(p["fa"], True, None), "A"),
                                     ("na imagem?", "O")]),
        ("CONT-4", True, lambda p: [("diga", "O"),
                                    (QUANTOS[genero_do_filtro(p["fa"])], "O"),
                                    (sintagma(p["fa"], True, None), "A"),
                                    ("aparecem nesta cena.", "O")]),
    ],
    "EXISTENCIA": [
        ("EXI-1", False, lambda p: [("existe", "O"),
                                    (sintagma(p["fa"], False, "indefinido"), "A"),
                                    ("na cena?", "O")]),
        ("EXI-2", False, lambda p: [("há", "O"),
                                    (sintagma(p["fa"], False, "indefinido"), "A"),
                                    ("na imagem?", "O")]),
        ("EXI-3", False, lambda p: [("a cena contém", "O"),
                                    (sintagma(p["fa"], False, "indefinido"), "A"),
                                    ("?", "O")]),
        ("EXI-4", True, lambda p: [("podemos encontrar", "O"),
                                   (sintagma(p["fa"], False, "indefinido"), "A"),
                                   ("nesta imagem?", "O")]),
    ],
    "CONSULTA": [
        ("CSL-1", False, lambda p: [("qual é", "O"),
                                    (ARTIGO_ATRIBUTO[p["atributo"]], "O"),
                                    (p["atributo"], "O"),
                                    (contrair(sintagma(p["fa"])), "A"), ("?", "O")]),
        ("CSL-2", False, lambda p: [("diga", "O"),
                                    (ARTIGO_ATRIBUTO[p["atributo"]], "O"),
                                    (p["atributo"], "O"),
                                    (contrair(sintagma(p["fa"])), "A"), (".", "O")]),
        ("CSL-3", True, lambda p: [("informe", "O"),
                                   (ARTIGO_ATRIBUTO[p["atributo"]], "O"),
                                   (p["atributo"], "O"),
                                   (contrair(sintagma(p["fa"])), "A"),
                                   (", por favor.", "O")]),
    ],
    "CONSULTA_RELACIONAL": [
        ("REL-1", False, lambda p: [("qual é", "O"),
                                    (ARTIGO_ATRIBUTO[p["atributo"]], "O"),
                                    (p["atributo"], "O"), ("do objeto que tem", "O"),
                                    (sintagma(p["fa"]), "A"),
                                    (frase_relacional(p), "O"), ("?", "O")]),
        ("REL-2", False, lambda p: [("considere o objeto que tem", "O"),
                                    (sintagma(p["fa"]), "A"),
                                    (frase_relacional(p), "O"), (";", "O"),
                                    ("qual é", "O"),
                                    (ARTIGO_ATRIBUTO[p["atributo"]], "O"),
                                    (p["atributo"], "O"), ("dele?", "O")]),
        ("REL-3", True, lambda p: [("para o objeto que tem", "O"),
                                   (sintagma(p["fa"]), "A"),
                                   (frase_relacional(p), "O"), (", qual", "O"),
                                   (ARTIGO_ATRIBUTO[p["atributo"]], "O"),
                                   (p["atributo"], "O"), ("?", "O")]),
    ],
    "COMPARACAO": [
        ("CMP-1", False, lambda p: [(sintagma(p["fb"]), "B"), ("tem", "O"),
                                    (MESMO[p["atributo"]], "O"),
                                    (p["atributo"], "O"),
                                    (contrair(sintagma(p["fa"])), "A"),
                                    (frase_relacional(p), "O"), ("?", "O")]),
        ("CMP-2", False, lambda p: [(ARTIGO_ATRIBUTO[p["atributo"]], "O"),
                                    (p["atributo"], "O"),
                                    (contrair(sintagma(p["fb"])), "B"), ("é", "O"),
                                    (IGUAL[p["atributo"]], "O"),
                                    (contrair(sintagma(p["fa"])), "A"),
                                    (frase_relacional(p), "O"), ("?", "O")]),
        ("CMP-3", False, lambda p: [(sintagma(p["fb"]), "B"), ("possui", "O"),
                                    (MESMO[p["atributo"]], "O"),
                                    (p["atributo"], "O"), ("que", "O"),
                                    (sintagma(p["fa"]), "A"),
                                    (frase_relacional(p), "O"), ("?", "O")]),
        ("CMP-4", True, lambda p: [(sintagma(p["fb"]), "B"), ("e", "O"),
                                   (sintagma(p["fa"]), "A"),
                                   (frase_relacional(p), "O"), ("têm", "O"),
                                   (MESMO[p["atributo"]], "O"),
                                   (p["atributo"], "O"), ("?", "O")]),
    ],
}


def montar_segmentos(segmentos):
    """Junta os segmentos, tokeniza e anota o papel (A/B) de cada palavra."""
    texto = " ".join(s for s, _ in segmentos)
    texto = texto.replace(" ?", "?").replace(" .", ".").replace(" ;", ";")
    texto = texto.replace(" ,", ",")
    tokens, etiquetas = [], []
    for trecho, papel in segmentos:
        for t in tokenizar(trecho):
            valor = MAPA_INFLEXOES.get(t)
            tokens.append(t)
            if valor is None or papel == "O":
                etiquetas.append((0, -1))          # não nomeia atributo nenhum
            else:
                etiquetas.append((1 + TODOS_OS_VALORES.index(valor),
                                  0 if papel == "A" else 1))
    return texto, tokens, etiquetas


def montar_programa(tipo, atributo, relacao, fa, fb):
    """Constrói o programa da DSL a partir das lacunas preenchidas."""
    if atributo == "nenhum":
        atributo = "forma"
    if relacao == "nenhuma":
        relacao = "direita"
    programa = ([("SCENE", None)]
                + [("FILTER", fa[a]) for a in ORDEM_FILTROS if a in fa])
    if tipo == "CONTAGEM":
        programa.append(("COUNT", None))
    elif tipo == "EXISTENCIA":
        programa.append(("EXIST", None))
    elif tipo == "CONSULTA":
        programa.append(("QUERY", atributo))
    elif tipo == "CONSULTA_RELACIONAL":
        programa += [("RELATE", relacao), ("QUERY", atributo)]
    else:
        programa += ([("RELATE", relacao)]
                     + [("FILTER", fb[a]) for a in ORDEM_FILTROS if a in fb]
                     + [("AEQUERY", atributo)])
    return programa


def casa(objeto, filtros):
    return all(objeto[a] == v for a, v in filtros.items())


def amostrar_filtro(objeto, rng, ruido=0.30, obrigatorio=None):
    """Sorteia 1 ou 2 atributos de um objeto; às vezes troca um valor (gera 0/não)."""
    chaves = [ATRIBUTOS[i] for i in rng.permutation(len(ATRIBUTOS))[:1 + int(rng.integers(2))]]
    filtros = {c: objeto[c] for c in chaves}
    if rng.random() < ruido:
        c = chaves[int(rng.integers(len(chaves)))]
        filtros[c] = VALORES[c][int(rng.integers(len(VALORES[c])))]
    if obrigatorio is not None:
        filtros[ATRIBUTO_DO_VALOR[obrigatorio]] = obrigatorio
    return filtros


def tentar_pergunta(cena, indice_cena, rng, tipo, de_teste=False, obrigatorio=None):
    """Sorteia uma pergunta do tipo dado. Devolve None se ela não for bem formada.

    'Bem formada' quer dizer: os sintagmas referem-se a exatamente um objeto onde
    o programa exige unicidade, e o atributo perguntado não está no próprio
    filtro (senão a pergunta responderia a si mesma).
    """
    objetos = cena["objetos"]
    n = len(objetos)
    fonte = objetos[int(rng.integers(n))]
    # Contagem e existência aceitam filtros que não casam com nada (respostas
    # 0 e "não"); nos outros tipos um filtro sem alvo simplesmente invalida.
    ruido = 0.50 if tipo in ("CONTAGEM", "EXISTENCIA") else 0.30
    fa = amostrar_filtro(fonte, rng, ruido=ruido, obrigatorio=obrigatorio)
    fb, relacao, atributo = {}, "nenhuma", "nenhum"
    casam = [i for i, o in enumerate(objetos) if casa(o, fa)]
    livres = [a for a in ATRIBUTOS if a not in fa]

    if tipo in ("CONSULTA", "CONSULTA_RELACIONAL", "COMPARACAO"):
        if len(casam) != 1 or not livres:
            return None
    if tipo == "CONSULTA":
        atributo = livres[int(rng.integers(len(livres)))]
    elif tipo in ("CONSULTA_RELACIONAL", "COMPARACAO"):
        relacao = RELACOES_EM_PERGUNTAS[int(rng.integers(len(RELACOES_EM_PERGUNTAS)))]
        alvos = [j for j in range(n) if cena["relacoes"][relacao][casam[0], j] > 0.5]
        if tipo == "CONSULTA_RELACIONAL":
            if len(alvos) != 1:
                return None
            atributo = ATRIBUTOS[int(rng.integers(len(ATRIBUTOS)))]
        else:
            if not alvos:
                return None
            alvo = alvos[int(rng.integers(len(alvos)))]
            fb = amostrar_filtro(objetos[alvo], rng, ruido=0.0)
            if len([j for j in alvos if casa(objetos[j], fb)]) != 1:
                return None
            livres = [a for a in ATRIBUTOS if a not in fa and a not in fb]
            if not livres:
                return None
            atributo = livres[int(rng.integers(len(livres)))]

    programa = montar_programa(tipo, atributo, relacao, fa, fb)
    resposta, _ = executar_programa(programa, base_verdadeira(cena))
    candidatos = [g for g in GABARITOS[tipo] if g[1] == de_teste]
    nome, _, funcao = candidatos[int(rng.integers(len(candidatos)))]
    partes = {"fa": fa, "fb": fb, "rel": relacao, "atributo": atributo}
    texto, tokens, etiquetas = montar_segmentos(funcao(partes))
    return {"cena": indice_cena, "texto": texto, "tokens": tokens,
            "etiquetas": etiquetas, "tipo": tipo, "atributo": atributo,
            "relacao": relacao, "fa": fa, "fb": fb, "programa": programa,
            "resposta": resposta, "gabarito": nome, "de_teste": de_teste}


def gerar_perguntas(cenas, rng, por_cena=PERGUNTAS_POR_CENA, de_teste=False,
                    tipos=TIPOS, obrigatorio=None, tentativas=4):
    """Gera perguntas bem formadas, percorrendo os tipos em ordem sorteada.

    Percorrer os tipos (em vez de sorteá-los a cada tentativa) mantém o conjunto
    equilibrado: nenhuma cena contribui com quatro perguntas de contagem.
    """
    perguntas = []
    for indice, cena in enumerate(cenas):
        vistos, escolhidas = set(), []
        for posicao in rng.permutation(len(tipos)):
            tipo = tipos[int(posicao)]
            if len(escolhidas) >= por_cena:
                break
            if tipo in ("CONSULTA_RELACIONAL", "COMPARACAO") and len(cena["objetos"]) < 2:
                continue
            for _ in range(tentativas):
                pergunta = tentar_pergunta(cena, indice, rng, tipo, de_teste, obrigatorio)
                if pergunta is not None and pergunta["texto"] not in vistos:
                    vistos.add(pergunta["texto"])
                    escolhidas.append(pergunta)
                    break
        perguntas.extend(escolhidas)
    return perguntas


# ===========================================================================
# PARTE 7 — QUESTION PARSER: DA LINGUAGEM NATURAL PARA A DSL, APRENDIDO
# ===========================================================================

COMPRIMENTO_SACO = 40


def ngramas(tokens):
    """Unigramas e bigramas — o 'saco de n-gramas' que representa a pergunta."""
    return list(tokens) + ["%s_%s" % (a, b) for a, b in zip(tokens, tokens[1:])]


def indices_do_saco(tokens, vocabulario):
    ativos = sorted({vocabulario[g] for g in ngramas(tokens) if g in vocabulario})
    return ativos[:COMPRIMENTO_SACO]


def indices_da_janela(tokens, vocabulario):
    """Uma linha de índices por token: a palavra e seus vizinhos, com posição.

    É o que dá ao etiquetador uma noção de ORDEM: 'vermelho' seguido de
    'à sua direita' pertence ao objeto âncora; 'azul' seguido de 'tem a mesma'
    pertence ao objeto sobre o qual se pergunta.
    """
    n_palavras = len(vocabulario) + 1
    linhas = []
    for i in range(len(tokens)):
        linha = []
        for posicao, deslocamento in enumerate(JANELA_ETIQUETADOR):
            j = i + deslocamento
            indice = vocabulario.get(tokens[j], 0) if 0 <= j < len(tokens) else 0
            linha.append(0 if indice == 0 else 1 + posicao * n_palavras + indice)
        linhas.append(linha)
    return linhas


class ParserSemantico:
    """Traduz perguntas em português para programas da DSL. Tudo aprendido.

    Duas peças, ambas modelos lineares softmax treinados por gradiente:
      * classificador de saco de n-gramas -> tipo de programa, atributo
        perguntado e relação espacial (três cabeças simultâneas);
      * etiquetador de janela -> duas cabeças por palavra: QUE valor de atributo
        ela nomeia (ou nenhum) e em qual lacuna esse valor entra (âncora A ou
        alvo B). A segunda cabeça só recebe gradiente nas palavras que de fato
        nomeiam um atributo.
    Um montador determinístico junta as lacunas no programa final.
    """

    def __init__(self, perguntas, rng):
        contagem = Counter()
        for p in perguntas:
            contagem.update(set(ngramas(p["tokens"])))
        comuns = [g for g, c in sorted(contagem.items()) if c >= 2]
        comuns.sort(key=lambda g: (-contagem[g], g))
        self.vocab_saco = {g: i + 1 for i, g in enumerate(comuns[:MAX_CARACTERISTICAS_SACO])}
        self.vocab_palavras = {w: i + 1 for i, w in
                               enumerate(sorted({t for p in perguntas for t in p["tokens"]}))}
        self.saco = ModeloEsparso(len(self.vocab_saco) + 1,
                                  [len(TIPOS), len(ATRIBUTOS_E_NENHUM),
                                   len(RELACOES_E_NENHUMA)])
        self.etiquetador = ModeloEsparso(
            1 + len(JANELA_ETIQUETADOR) * (len(self.vocab_palavras) + 1),
            [1 + len(TODOS_OS_VALORES), 2])
        self._treinar(perguntas, rng)

    def _dados(self, perguntas):
        IDX = np.zeros((len(perguntas), COMPRIMENTO_SACO), dtype=np.int64)
        Y = np.zeros((len(perguntas), 3), dtype=np.int64)
        janelas, etiquetas = [], []
        for i, p in enumerate(perguntas):
            ativos = indices_do_saco(p["tokens"], self.vocab_saco)
            IDX[i, :len(ativos)] = ativos
            Y[i] = (TIPOS.index(p["tipo"]),
                    ATRIBUTOS_E_NENHUM.index(p["atributo"]),
                    RELACOES_E_NENHUMA.index(p["relacao"]))
            janelas.extend(indices_da_janela(p["tokens"], self.vocab_palavras))
            # Só os programas de comparação têm duas lacunas de filtro; nos
            # demais o papel não é uma decisão e entra como rótulo ausente.
            duas_lacunas = p["tipo"] == "COMPARACAO"
            etiquetas.extend([(v, papel if duas_lacunas else -1)
                              for v, papel in p["etiquetas"]])
        J = np.array(janelas, dtype=np.int64).reshape(-1, len(JANELA_ETIQUETADOR))
        E = np.array(etiquetas, dtype=np.int64).reshape(-1, 2)
        return IDX, Y, J, E

    def _treinar(self, perguntas, rng):
        """Currículo do parser: primeiro perguntas simples, depois as relacionais."""
        simples = [p for p in perguntas
                   if p["tipo"] in ("CONTAGEM", "EXISTENCIA", "CONSULTA")]
        for conjunto, epocas in ((simples, EPOCAS_PARSER[0]),
                                 (perguntas, EPOCAS_PARSER[1])):
            if not conjunto:
                continue
            IDX, Y, J, E = self._dados(conjunto)
            self.saco.treinar(IDX, Y, epocas, TAXA_PARSER, LOTE_PARSER, rng,
                              abandono=ABANDONO_PARSER)
            self.etiquetador.treinar(J, E, epocas, TAXA_PARSER, 512, rng,
                                     abandono=ABANDONO_PARSER)

    def analisar(self, tokens):
        """Pergunta em português -> programa executável da DSL."""
        ativos = indices_do_saco(tokens, self.vocab_saco)
        IDX = np.zeros((1, COMPRIMENTO_SACO), dtype=np.int64)
        IDX[0, :len(ativos)] = ativos
        cabecas = self.saco.distribuicoes(IDX)
        tipo = TIPOS[int(np.argmax(cabecas[0]))]
        atributo = ATRIBUTOS_E_NENHUM[int(np.argmax(cabecas[1]))]
        relacao = RELACOES_E_NENHUMA[int(np.argmax(cabecas[2]))]
        J = np.array(indices_da_janela(tokens, self.vocab_palavras), dtype=np.int64)
        cabeca_valor, cabeca_papel = self.etiquetador.distribuicoes(J)
        valores = np.argmax(cabeca_valor, axis=1)
        papeis = np.argmax(cabeca_papel, axis=1)
        fa, fb = {}, {}
        for indice, papel in zip(valores, papeis):
            if indice == 0:
                continue
            valor = TODOS_OS_VALORES[int(indice) - 1]
            destino = fa if (papel == 0 or tipo != "COMPARACAO") else fb
            destino.setdefault(ATRIBUTO_DO_VALOR[valor], valor)
        return montar_programa(tipo, atributo, relacao, fa, fb)


# ===========================================================================
# PARTE 8 — LINHA DE BASE PONTA A PONTA (a rede do Exercício 2, em versão enxuta)
# ===========================================================================

DIM_CENA = MAX_OBJETOS * DIM_RECORTE


def vetor_de_cena(cena):
    """Cena inteira como um vetor: os recortes dos objetos, lado a lado."""
    if "vetor" not in cena:
        vetor = np.zeros(DIM_CENA)
        achatado = cena["recortes"].ravel()[:DIM_CENA]
        vetor[:len(achatado)] = achatado
        cena["vetor"] = vetor
    return cena["vetor"]


def matriz_ponta_a_ponta(perguntas, cenas, vocabulario):
    """Monta (X, Y): pixels da cena + saco de n-gramas da pergunta -> resposta."""
    X = np.zeros((len(perguntas), DIM_CENA + len(vocabulario) + 1))
    Y = np.zeros((len(perguntas), 1), dtype=np.int64)
    for i, p in enumerate(perguntas):
        X[i, :DIM_CENA] = vetor_de_cena(cenas[p["cena"]])
        for j in indices_do_saco(p["tokens"], vocabulario):
            X[i, DIM_CENA + j] = 1.0
        Y[i, 0] = INDICE_RESPOSTA[p["resposta"]]
    return X, Y


def treinar_ponta_a_ponta(perguntas, cenas, vocabulario, rng):
    """Uma única rede que vai dos pixels e das palavras direto para a resposta."""
    X, Y = matriz_ponta_a_ponta(perguntas, cenas, vocabulario)
    rede = RedeDensa(X.shape[1], OCULTA_PONTA_A_PONTA, [len(RESPOSTAS)], rng)
    rede.treinar(X, Y, EPOCAS_PONTA_A_PONTA, TAXA_PONTA_A_PONTA,
                 LOTE_PONTA_A_PONTA, rng)
    return rede


def avaliar_ponta_a_ponta(rede, perguntas, cenas, vocabulario):
    if not perguntas:
        return float("nan")
    X, Y = matriz_ponta_a_ponta(perguntas, cenas, vocabulario)
    P = rede.distribuicoes(X)[0]
    return float(np.mean(np.argmax(P, axis=1) == Y[:, 0]))


def avaliar_hibrido(perceptor, parser, perguntas, cenas):
    """Acurácia de resposta e de programa do sistema neuro-simbólico."""
    if not perguntas:
        return float("nan"), float("nan")
    bases, certas, programas = {}, 0, 0
    for p in perguntas:
        if p["cena"] not in bases:
            bases[p["cena"]] = base_percebida(cenas[p["cena"]], perceptor)
        programa = parser.analisar(p["tokens"])
        programas += int(programa == p["programa"])
        resposta, _ = executar_programa(programa, bases[p["cena"]])
        certas += int(resposta == p["resposta"])
    return certas / len(perguntas), programas / len(perguntas)


# ===========================================================================
# PARTE 9 — RELATÓRIO
# ===========================================================================

def cabecalho(titulo, caractere="="):
    print()
    print(caractere * LARGURA)
    print(" " + titulo)
    print(caractere * LARGURA)


def subtitulo(titulo):
    print()
    print("-- %s " % titulo + "-" * max(0, LARGURA - len(titulo) - 4))


def paragrafo(texto, recuo=2):
    espaco = " " * recuo
    return textwrap.fill(" ".join(texto.split()), width=LARGURA,
                         initial_indent=espaco, subsequent_indent=espaco)


def cena_da_figura(rng):
    """A cena da Figura 5.4: um cubo azul com um cubo vermelho à sua direita."""
    objetos = [
        {"forma": "cubo", "cor": "azul", "tamanho": "grande",
         "material": "fosco", "cx": 5, "cy": 14},
        {"forma": "cubo", "cor": "vermelho", "tamanho": "grande",
         "material": "metal", "cx": 15, "cy": 14},
        {"forma": "esfera", "cor": "verde", "tamanho": "pequeno",
         "material": "metal", "cx": 25, "cy": 4},
    ]
    grade = np.zeros((ALTURA_CENA, LARGURA_CENA, N_CANAIS))
    for objeto in objetos:
        desenhar_objeto(grade, objeto, rng)
    objetos.sort(key=lambda o: (o["cx"], o["cy"]))
    return segmentar_cena({"grade": grade, "objetos": objetos})


def imprimir_base_de_conhecimento(base, limite=0.02):
    """Imprime os fatos probabilísticos de uma cena, como no capítulo."""
    for i in range(base.n):
        fatos = []
        for atributo in ATRIBUTOS:
            d = base.dist[atributo][i]
            for k in np.argsort(-d):
                if d[k] < limite and fatos and len(fatos) % 2 == 0:
                    continue
                if d[k] < limite:
                    continue
                fatos.append("%s(obj%d, %s) = %.3f"
                             % (atributo, i + 1, VALORES[atributo][k], d[k]))
        for j in range(0, len(fatos), 2):
            print("    " + "   ".join(f.ljust(32) for f in fatos[j:j + 2]).rstrip())
    relacionais = []
    for relacao in RELACOES:
        M = base.rel[relacao]
        for i in range(base.n):
            for j in range(base.n):
                if M[i, j] > 0.5:
                    relacionais.append("a_%s_de(obj%d, obj%d) = 1.0"
                                       % (relacao, i + 1, j + 1))
    for j in range(0, len(relacionais), 2):
        print("    " + "   ".join(f.ljust(32) for f in relacionais[j:j + 2]).rstrip())


def imprimir_rastro(programa, rastro, n_objetos):
    """O RASTRO LÓGICO: a máscara suave depois de cada operação da DSL."""
    largura_op = 26
    print("    %-6s%-*s%s" % ("passo", largura_op, "operação",
                              "".join("   obj%d" % (i + 1) for i in range(n_objetos))))
    print("    " + "-" * (6 + largura_op + 7 * n_objetos))
    for passo, item in enumerate(rastro):
        nome = item["operacao"] if item["argumento"] is None else \
            "%s(%s)" % (item["operacao"], item["argumento"])
        valores = "".join("  %5.3f" % v for v in item["mascara"])
        print("    %-6d%-*s%s" % (passo, largura_op, nome, valores))


def imprimir_tabela(cabecalhos, larguras, linhas, recuo=2):
    """Tabela alinhada em texto puro, com larguras de campo explícitas.

    Largura positiva alinha o campo à direita; largura negativa alinha à
    esquerda (útil para colunas de texto). Em qualquer dos casos o campo
    ocupa exatamente |largura| colunas e reserva um espaço de separação: um
    texto mais comprido é cortado, de modo que duas colunas nunca se encostam
    e a tabela nunca ultrapassa `recuo + soma das larguras` colunas.
    """
    espaco = " " * recuo
    largura_total = sum(abs(w) for w in larguras)

    def campo(valor, w):
        texto = str(valor)
        util = abs(w) - 1               # o espaço que sobra separa as colunas
        if len(texto) > util:
            texto = texto[:util - 1] + "…"
        return (" " + texto.ljust(util)) if w < 0 else texto.rjust(abs(w))

    def formatar(celulas):
        return (espaco + "".join(campo(c, w)
                                 for c, w in zip(celulas, larguras))).rstrip()

    print(formatar(cabecalhos))
    print(espaco + "-" * largura_total)
    for linha in linhas:
        print(formatar(linha))


def main():
    """Executa o exercício completo e imprime o relatório didático."""
    rng = np.random.default_rng(SEMENTE)

    cabecalho("EXERCÍCIO 3 — A MISTURA NEURO-SIMBÓLICA: UM NSCL SIMPLIFICADO")
    print(" Capítulo 5 — Introdução à IA Neuro-Simbólica (Figuras 5.3 e 5.4)")
    print(" A rede constrói a base de conhecimento; a pergunta vira um programa;")
    print(" o executor quase-simbólico responde — e deixa o rastro do raciocínio.")

    # -----------------------------------------------------------------------
    # [1] O mundo de cenas
    # -----------------------------------------------------------------------
    pesos = [0.15, 0.20, 0.25, 0.20, 0.20]
    cenas_treino = [segmentar_cena(gerar_cena(rng, int(rng.choice([1, 2, 3, 4, 5], p=pesos))))
                    for _ in range(CENAS_TREINO)]
    cenas_teste = [segmentar_cena(gerar_cena(rng, int(rng.choice([1, 2, 3, 4, 5], p=pesos))))
                   for _ in range(CENAS_TESTE)]
    demo = cena_da_figura(np.random.default_rng(SEMENTE + 7))

    perguntas_treino = gerar_perguntas(cenas_treino, rng)
    perguntas_teste = gerar_perguntas(cenas_teste, rng)
    perguntas_superficie = gerar_perguntas(cenas_teste, rng, por_cena=2, de_teste=True)

    cabecalho("[1] O MUNDO DE CENAS — um CLEVR de bolso", "-")
    print("  Cenas de treino / teste ....... %d / %d" % (CENAS_TREINO, CENAS_TESTE))
    print("  Grade de pixels ............... %d x %d x %d canais (R, G, B, brilho)"
          % (ALTURA_CENA, LARGURA_CENA, N_CANAIS))
    print(textwrap.fill(
        ", ".join("%s(%d)" % (a, len(VALORES[a])) for a in ATRIBUTOS),
        width=LARGURA, initial_indent="  Atributos por objeto .......... ",
        subsequent_indent=" " * 34))
    print("  Perguntas de treino / teste ... %d / %d"
          % (len(perguntas_treino), len(perguntas_teste)))
    print("  Perguntas com fraseado novo ... %d" % len(perguntas_superficie))
    subtitulo("A cena da Figura 5.4, e a segmentação por componentes conexas")
    print()
    for linha in desenhar_cena_em_ascii(demo):
        print("    " + linha)
    print()
    print("    Cada dígito é uma componente conexa encontrada pelo flood fill —")
    print("    nenhuma biblioteca de visão, apenas 4-vizinhança e uma pilha.")
    print()
    imprimir_tabela(["objeto", "pixels", "centro", "forma", "cor", "tamanho", "material"],
                    [8, 8, 10, 10, 11, 9, 10],
                    [["obj%d" % (i + 1), c["n_pixels"], "(%d,%d)" % (c["cx"], c["cy"]),
                      o["forma"], o["cor"], o["tamanho"], o["material"]]
                     for i, (c, o) in enumerate(zip(demo["componentes"], demo["objetos"]))],
                    recuo=4)

    # -----------------------------------------------------------------------
    # [2] Image parser com currículo
    # -----------------------------------------------------------------------
    cabecalho("[2] IMAGE PARSER — o sub-modelo neural, treinado com currículo", "-")
    registro = []
    rng_perceptor = np.random.default_rng(SEMENTE + 1)
    perceptor = treinar_perceptor_com_curriculo(cenas_treino, rng_perceptor,
                                                cenas_validacao=cenas_teste,
                                                registro=registro)
    print(paragrafo(
        "MLP escrito do zero: %d entradas (o recorte 7x7x4 mascarado do objeto), "
        "%d neurônios ocultos com ReLU e quatro cabeças softmax — uma por "
        "atributo. Não há rótulo de RESPOSTA aqui: o perceptor é supervisionado "
        "apenas com os atributos dos objetos, e nunca vê as perguntas."
        % (DIM_RECORTE, OCULTA_PERCEPTOR)))
    print()
    imprimir_tabela(["Lição", "cenas", "objs", "conceitos introduzidos",
                     "forma", "cor", "tam.", "mat."],
                    [6, 7, 6, -28, 7, 7, 7, 7],
                    [[r["licao"], r["cenas"], r["objetos"], r["descricao"]]
                     + ["%.1f%%" % (100 * r["acuracias"][a]) for a in ATRIBUTOS]
                     for r in registro])
    print()
    print(paragrafo(
        "Na Lição 1 tamanho e material entram como rótulo AUSENTE: as duas "
        "cabeças não recebem gradiente e ficam no chute (perto de 50%). O "
        "currículo do capítulo aparece aqui em estado puro — conceitos simples "
        "em cenas simples primeiro; cenas povoadas, em que vizinhos invadem a "
        "janela de recorte, só no fim."))

    # -----------------------------------------------------------------------
    # [3] Base de conhecimento gerada automaticamente
    # -----------------------------------------------------------------------
    cabecalho("[3] BASE DE CONHECIMENTO — gerada pela rede, não escrita por alguém", "-")
    base_demo = base_percebida(demo, perceptor)
    print(paragrafo(
        "Estes fatos são a saída do perceptor sobre a cena acima. No Exercício 1, "
        "uma base como esta precisava ser digitada à mão, fato por fato, por um "
        "engenheiro do conhecimento. Aqui ela é o passo (2) da Figura 5.3:"))
    print()
    imprimir_base_de_conhecimento(base_demo)
    print()
    print(paragrafo(
        "As relações espaciais saem da geometria dos centróides devolvidos pela "
        "segmentação e por isso valem exatamente 1.0; os atributos saem de uma "
        "rede e carregam a incerteza dela. É uma base de conhecimento "
        "PROBABILÍSTICA — e é isso que exige um motor de inferência diferente."))

    # -----------------------------------------------------------------------
    # [4] Question parser
    # -----------------------------------------------------------------------
    cabecalho("[4] QUESTION PARSER — da linguagem natural para a DSL, aprendendo", "-")
    rng_parser = np.random.default_rng(SEMENTE + 2)
    parser = ParserSemantico(perguntas_treino, rng_parser)
    _, prog_vistos = avaliar_hibrido(perceptor, parser, perguntas_teste, cenas_teste)
    _, prog_novos = avaliar_hibrido(perceptor, parser, perguntas_superficie, cenas_teste)
    print("  Operações da DSL .............. SCENE, FILTER(valor), RELATE(relação),")
    print("                                  QUERY(atributo), AEQUERY(atributo),")
    print("                                  COUNT, EXIST")
    print("  Características do saco ....... %d n-gramas (unigramas + bigramas)"
          % len(parser.vocab_saco))
    print(textwrap.fill(
        "%d (palavra x posição em [%s])"
        % (parser.etiquetador.W.shape[0],
           ",".join(str(p) for p in JANELA_ETIQUETADOR)),
        width=LARGURA, initial_indent="  Características da janela ..... ",
        subsequent_indent=" " * 34))
    print("  Pares (pergunta, programa) .... %d" % len(perguntas_treino))
    print()
    print("  Programa EXATAMENTE correto:")
    print("    gabaritos vistos no treino .. %.1f%% (%d perguntas)"
          % (100 * prog_vistos, len(perguntas_teste)))
    print("    fraseado NOVO, não visto .... %.1f%% (%d perguntas)"
          % (100 * prog_novos, len(perguntas_superficie)))
    subtitulo("Perguntas traduzidas em programas")
    partes_demo = {"fa": {"cor": "vermelho", "forma": "cubo"}, "fb": {"cor": "azul"},
                   "rel": "direita", "atributo": "forma"}
    texto_demo, tokens_demo, _ = montar_segmentos(GABARITOS["COMPARACAO"][0][2](partes_demo))
    exemplos = [(texto_demo, tokens_demo)]
    for pergunta in perguntas_teste[:2] + perguntas_superficie[:2]:
        exemplos.append((pergunta["texto"], pergunta["tokens"]))
    for texto, tokens in exemplos:
        print()
        print(textwrap.fill('"%s"' % texto, width=LARGURA - 4,
                            initial_indent="    ", subsequent_indent="     "))
        print(textwrap.fill(formatar_programa(parser.analisar(tokens)),
                            width=LARGURA - 8, initial_indent="      -> ",
                            subsequent_indent="         "))

    # -----------------------------------------------------------------------
    # [5] Executor quase-simbólico
    # -----------------------------------------------------------------------
    cabecalho("[5] EXECUTOR QUASE-SIMBÓLICO — o rastro lógico, passo a passo", "-")
    programa_demo = parser.analisar(tokens_demo)
    resposta_demo, rastro_demo = executar_programa(programa_demo, base_demo)
    resposta_oraculo, _ = executar_programa(programa_demo, base_verdadeira(demo))
    print(textwrap.fill('Pergunta da Figura 5.4: "%s"' % texto_demo, width=LARGURA - 2,
                        initial_indent="  ", subsequent_indent="  "))
    print()
    print(textwrap.fill("Programa: " + formatar_programa(programa_demo),
                        width=LARGURA - 2, initial_indent="  ", subsequent_indent="    "))
    print()
    imprimir_rastro(programa_demo, rastro_demo, base_demo.n)
    print()
    detalhe = rastro_demo[-1]["detalhe"]
    print("    AEQUERY(forma) = produto interno das distribuições de forma")
    print("                     da âncora e do alvo = %.3f  ->  resposta: %s"
          % (detalhe, resposta_demo.upper()))
    print("    Resposta do oráculo simbólico sobre a cena verdadeira: %s"
          % resposta_oraculo.upper())
    print()
    print(paragrafo(
        "Cada linha da tabela é uma máscara suave: o valor de cada objeto é a "
        "probabilidade de ele pertencer ao conjunto corrente. FILTER multiplica "
        "pela probabilidade do atributo, RELATE propaga a máscara pela matriz de "
        "relações, COUNT soma, EXIST tira o máximo. Nenhuma dessas operações tem "
        "parâmetro treinável e todas são diferenciáveis — é o 'quase-simbólico' "
        "do capítulo."))

    # -----------------------------------------------------------------------
    # [6] Comparação 1 — eficiência de dados
    # -----------------------------------------------------------------------
    cabecalho("[6] COMPARAÇÃO 1 — EFICIÊNCIA DE DADOS", "-")
    linhas_eficiencia = []
    for n_cenas in TAMANHOS_DE_TREINO:
        sub_cenas = cenas_treino[:n_cenas]
        sub_perguntas = [p for p in perguntas_treino if p["cena"] < n_cenas]
        r = np.random.default_rng(SEMENTE + 100 + n_cenas)
        perceptor_n = treinar_perceptor_com_curriculo(sub_cenas, r)
        parser_n = ParserSemantico(sub_perguntas, r)
        acc_hibrido, acc_programa = avaliar_hibrido(perceptor_n, parser_n,
                                                    perguntas_teste, cenas_teste)
        rede_n = treinar_ponta_a_ponta(sub_perguntas, cenas_treino,
                                       parser_n.vocab_saco, r)
        acc_rede = avaliar_ponta_a_ponta(rede_n, perguntas_teste, cenas_teste,
                                         parser_n.vocab_saco)
        linhas_eficiencia.append([n_cenas, len(sub_perguntas),
                                  "%.1f%%" % (100 * acc_programa),
                                  "%.1f%%" % (100 * acc_hibrido),
                                  "%.1f%%" % (100 * acc_rede),
                                  "%+.1f" % (100 * (acc_hibrido - acc_rede))])
    imprimir_tabela(["cenas", "perguntas", "programa ok", "HÍBRIDO",
                     "PONTA A PONTA", "dif. (p.p.)"],
                    [8, 12, 14, 12, 16, 14], linhas_eficiencia)
    print()
    print(paragrafo(
        "O híbrido reparte o problema: o perceptor aprende atributos a partir de "
        "poucos objetos, o parser aprende a estrutura da linguagem a partir de "
        "poucas frases, e o executor NÃO APRENDE NADA — ele já sabe contar, "
        "filtrar e comparar. A rede ponta a ponta precisa descobrir tudo isso do "
        "zero, a partir de exemplos de pergunta e resposta."))

    # -----------------------------------------------------------------------
    # [7] Comparação 2 — generalização composicional
    # -----------------------------------------------------------------------
    cabecalho("[7] COMPARAÇÃO 2 — GENERALIZAÇÃO COMPOSICIONAL", "-")
    rng_comp = np.random.default_rng(SEMENTE + 3)

    def eh_retida(p):
        return any(p["tipo"] == t and v in p["fa"].values()
                   for t, v in COMBINACOES_RETIDAS)

    treino_reduzido = [p for p in perguntas_treino if not eh_retida(p)]
    retidas, vistas = [], []
    for tipo, valor in COMBINACOES_RETIDAS:
        alvo = gerar_perguntas(cenas_teste, rng_comp, por_cena=1, tipos=(tipo,),
                               obrigatorio=valor)
        retidas.append((tipo, valor, [p for p in alvo if eh_retida(p)]))
        vistas.extend([p for p in gerar_perguntas(cenas_teste, rng_comp, por_cena=1,
                                                  tipos=(tipo,)) if not eh_retida(p)])
    perceptor_c = treinar_perceptor_com_curriculo(cenas_treino, rng_comp)
    parser_c = ParserSemantico(treino_reduzido, rng_comp)
    rede_c = treinar_ponta_a_ponta(treino_reduzido, cenas_treino,
                                   parser_c.vocab_saco, rng_comp)
    linhas_comp = []
    medidas = []
    for rotulo, perguntas in ([("%s x %s" % (t, v), ps) for t, v, ps in retidas]
                              + [("combinações vistas", vistas)]):
        acc_h, _ = avaliar_hibrido(perceptor_c, parser_c, perguntas, cenas_teste)
        acc_e = avaliar_ponta_a_ponta(rede_c, perguntas, cenas_teste, parser_c.vocab_saco)
        linhas_comp.append([rotulo, len(perguntas), "%.1f%%" % (100 * acc_h),
                            "%.1f%%" % (100 * acc_e)])
        medidas.append({"rotulo": rotulo, "acc_h": acc_h, "acc_e": acc_e,
                        "perguntas": perguntas})

    # Diagnóstico do caso que FALHA: quais perguntas retidas usam alguma
    # palavra que o treino reduzido nunca mostrou ao parser.
    vocab_reduzido = set()
    for p in treino_reduzido:
        vocab_reduzido.update(p["tokens"])
    for medida in medidas[:len(retidas)]:
        novas = sorted({t for p in medida["perguntas"] for t in p["tokens"]
                        if t not in vocab_reduzido})
        medida["palavras_novas"] = novas
        medida["com_palavra_nova"] = sum(
            1 for p in medida["perguntas"]
            if any(t not in vocab_reduzido for t in p["tokens"]))
    print(paragrafo(
        "Três combinações (tipo de pergunta x valor de atributo) foram RETIRADAS "
        "do treino. Cada peça continua presente separadamente: o modelo viu "
        "perguntas de contagem e viu a palavra 'cilindro' — só nunca as viu "
        "juntas. Restaram %d das %d perguntas de treino."
        % (len(treino_reduzido), len(perguntas_treino))))
    print()
    imprimir_tabela(["combinação", "perguntas", "HÍBRIDO", "PONTA A PONTA"],
                    [30, 12, 12, 16], linhas_comp)
    print()
    print(paragrafo(
        "O programa é COMPOSTO na hora da execução, não memorizado: COUNT não "
        "sabe o que é um cilindro e FILTER(cilindro) não sabe o que é contar. "
        "Para a rede ponta a ponta, cada par (tipo, atributo) é uma região "
        "diferente do espaço de entrada, e as regiões que ela nunca visitou "
        "continuam vazias."))
    print()
    pior = min(medidas[:len(retidas)], key=lambda m: m["acc_h"])
    outras = [m for m in medidas[:len(retidas)] if m is not pior]
    print(paragrafo(
        "Uma das três, porém, NÃO funcionou, e o resultado negativo fica "
        "registrado: em %s o híbrido também erra — %.1f%% de acerto, contra "
        "%.1f%% da rede ponta a ponta. A causa provável não está "
        "no executor, e sim no parser: %d das %d perguntas dessa combinação "
        "usam palavra que sumiu por inteiro do treino reduzido (%s, zero "
        "ocorrências nas %d perguntas restantes). Nas outras duas combinações "
        "retiradas esse contador é %d, e ali o híbrido marca %.1f%% e %.1f%%. "
        "Compor resolve a ESTRUTURA da pergunta; não adivinha o "
        "sentido de uma palavra que o modelo nunca leu."
        % (pior["rotulo"], 100 * pior["acc_h"], 100 * pior["acc_e"],
           pior["com_palavra_nova"], len(pior["perguntas"]),
           ", ".join("'%s'" % w for w in pior["palavras_novas"]),
           len(treino_reduzido),
           sum(m["com_palavra_nova"] for m in outras),
           100 * outras[0]["acc_h"], 100 * outras[1]["acc_h"])))

    # -----------------------------------------------------------------------
    # [8] Comparação 3 — explicabilidade
    # -----------------------------------------------------------------------
    cabecalho("[8] COMPARAÇÃO 3 — EXPLICABILIDADE DE UMA RESPOSTA", "-")
    rng_expl = np.random.default_rng(SEMENTE + 4)
    parser_expl = parser
    rede_expl = treinar_ponta_a_ponta(perguntas_treino, cenas_treino,
                                      parser.vocab_saco, rng_expl)
    alvo = next(p for p in perguntas_teste
                if p["tipo"] == "CONSULTA_RELACIONAL"
                and len(cenas_teste[p["cena"]]["objetos"]) >= 3)
    cena_alvo = cenas_teste[alvo["cena"]]
    base_alvo = base_percebida(cena_alvo, perceptor)
    programa_alvo = parser_expl.analisar(alvo["tokens"])
    resposta_h, rastro_alvo = executar_programa(programa_alvo, base_alvo)
    X_alvo, _ = matriz_ponta_a_ponta([alvo], cenas_teste, parser.vocab_saco)
    P_alvo = rede_expl.distribuicoes(X_alvo)[0][0]
    print(textwrap.fill('Pergunta: "%s"' % alvo["texto"], width=LARGURA - 2,
                        initial_indent="  ", subsequent_indent="  "))
    print("  Resposta correta (oráculo sobre a cena verdadeira): %s"
          % alvo["resposta"].upper())
    subtitulo("O que o sistema NEURO-SIMBÓLICO consegue dizer")
    print("    1. o programa que ele executou:")
    print(textwrap.fill(formatar_programa(programa_alvo), width=LARGURA - 10,
                        initial_indent="       ", subsequent_indent="       "))
    print("    2. a base de conhecimento que ele leu (extrato):")
    for i in range(min(3, base_alvo.n)):
        d = base_alvo.dist["cor"][i]
        f = base_alvo.dist["forma"][i]
        print("       obj%d: forma=%s (%.2f)  cor=%s (%.2f)"
              % (i + 1, VALORES["forma"][int(np.argmax(f))], f.max(),
                 VALORES["cor"][int(np.argmax(d))], d.max()))
    print("    3. a máscara final, objeto por objeto:")
    print("       " + "  ".join("obj%d=%.3f" % (i + 1, v)
                                for i, v in enumerate(rastro_alvo[-1]["mascara"])))
    print("    4. a resposta: %s" % resposta_h.upper())
    subtitulo("O que a rede PONTA A PONTA consegue dizer")
    melhores = np.argsort(-P_alvo)[:3]
    print("    1. a resposta: %s" % RESPOSTAS[int(melhores[0])].upper())
    print("    2. ... e a distribuição de saída, que não é uma explicação:")
    for k in melhores:
        print("       P(%-10s) = %.3f" % (RESPOSTAS[int(k)], P_alvo[k]))
    print()
    print(paragrafo(
        "Não há terceiro item. A rede ponta a ponta não tem, dentro de si, "
        "nenhum objeto identificado, nenhum passo intermediário e nenhuma "
        "afirmação sobre a cena — só um vetor de 19 números. É a caixa-preta do "
        "Capítulo 4, de novo, agora em um problema de raciocínio visual."))

    # -----------------------------------------------------------------------
    # [9] Interpretação
    # -----------------------------------------------------------------------
    cabecalho("[9] INTERPRETAÇÃO — o que você acabou de ver")
    print("""
  1. Os CINCO PASSOS da arquitetura híbrida (Figura 5.3) rodaram inteiros:
     (1) a imagem entrou como pixels; (2) a rede neural CONSTRUIU a base de
     conhecimento, em vez de recebê-la pronta de um engenheiro; (3) a
     pergunta em português foi traduzida em proposições simbólicas — o
     programa da DSL; (4) o motor de inferência executou o programa sobre a
     base; (5) a resposta saiu acompanhada do caminho que a produziu.

  2. O sub-modelo NEURAL faz o que o simbólico nunca soube fazer: olhar
     para pixels ruidosos e decidir que aquilo é um cubo azul metálico. O
     sub-modelo SIMBÓLICO faz o que o neural nunca soube fazer: contar,
     comparar, compor uma pergunta nova a partir de peças conhecidas. As
     forças de cada paradigma atacam exatamente as fraquezas do outro — é a
     Tabela 5.1 do capítulo virando código executável.

  3. A eficiência de dados não é mágica: ela vem de CONHECIMENTO EMBUTIDO.
     O executor já sabe que contar é somar e que comparar é confrontar duas
     distribuições. Esse conhecimento não precisa ser aprendido, e os dados
     ficam livres para ensinar o que realmente varia — a aparência dos
     objetos e o fraseado das perguntas.

  4. A generalização composicional é consequência da MODULARIDADE — e tem
     um limite, que a seção [7] mostrou sem disfarce. Como o significado da
     pergunta é um programa, e programas se montam por composição, uma
     combinação inédita de OPERAÇÕES continua funcionando: foi o caso de
     duas das três. Na terceira, porém, a combinação retirada levou junto
     uma palavra inteira, o parser não teve de onde tirá-la, e o híbrido
     errou tanto quanto a rede ponta a ponta. Compor resolve estrutura,
     não vocabulário.

  5. E a explicabilidade vem de graça, não de um explicador acoplado
     depois. O rastro lógico não é uma aproximação do que o sistema fez:
     ele É o que o sistema fez. Compare com o Exercício 1 do Capítulo 4,
     em que a legibilidade custava acurácia — aqui, boa parte dela veio
     junto com o desempenho.

  6. Guarde as ressalvas, que também são do capítulo: o mundo tem 11
     conceitos, as relações são geométricas e o perceptor recebeu
     supervisão de atributos. O NSCL real aprende os conceitos apenas com
     pares (pergunta, resposta), usando o executor diferenciável para levar
     o gradiente até o perceptor. A arquitetura, porém, é esta — e ela cabe
     em um arquivo de texto.
""")
    print("=" * LARGURA)
    print(" Fim do Exercício 3.")
    print("=" * LARGURA)


if __name__ == "__main__":
    main()
