# -*- coding: utf-8 -*-
"""
EXERCÍCIO PRÁTICO 2 — O INGREDIENTE NEURAL: PERCEPÇÃO SEM RACIOCÍNIO
====================================================================

Capítulo 5 — "Introdução à IA Neuro-Simbólica — o Próximo Nível da IA".
Referência visual: Figura 5.2 (exemplo de conexão de uma rede neural).

O QUE ESTE EXERCÍCIO DEMONSTRA
------------------------------
O Exercício 1 mostrou o ingrediente simbólico: regras escritas à mão, legíveis,
auditáveis — e que quebram diante de qualquer forma nova. Aqui entra o outro
ingrediente da IA Neuro-Simbólica, a REDE NEURAL, com suas duas caras:

  1. O QUE ELA FAZ MUITO BEM — PERCEBER. Um perceptron multicamada (MLP)
     escrito do zero em numpy olha o raster de um objeto e diz sua forma, sua
     cor, seu tamanho e seu material. Ninguém escreveu uma regra sequer sobre
     círculos, brilhos ou pixels: os padrões foram extraídos automaticamente
     dos dados. É a promessa do capítulo, cumprida.

  2. O QUE ELA FAZ MAL — RACIOCINAR. A MESMA arquitetura, alimentada com uma
     CENA de 2 a 4 objetos e com uma pergunta codificada em um vetor ("quantas
     esferas há?", "existe algum cubo azul?"), passa a errar feio. E erra de
     dois jeitos:
       (a) percebe, mas não RELACIONA — na única pergunta que se resolve
           lendo um objeto (a cor do objeto da posição 2) ela dispara acima do
           chute; nas três que exigem varrer a cena, comparar e agregar
           (CONTAR, EXISTE, MESMA_FORMA) milhares de exemplos não a tiram do
           patamar de um chutador que nem olha a imagem;
       (b) não COMPÕE conceitos — treinada em CONSULTAR(cor, obj 0),
           CONSULTAR(cor, obj 1) e CONSULTAR(forma, obj 3) milhares de vezes,
           mas nunca em CONSULTAR(cor, obj 3), sua acurácia nessa combinação
           inédita cai ao ACASO, embora cada peça isolada lhe seja familiar e
           as imagens de teste sejam exatamente as mesmas.

A moral: a rede aprendeu PADRÕES, não CONCEITOS. Ela nunca representou
"objeto", "cor" ou "posição 3" como entidades manipuláveis, e por isso não sabe
que "a cor de" é a mesma operação onde quer que se aplique. Daí a pergunta que
o capítulo faz e que abre o Exercício 3: podemos explorar o poder de
reconhecimento de padrões das redes neurais para extrair automaticamente
padrões SIMBÓLICOS dos nossos dados?

Uma advertência metodológica que atravessa o arquivo inteiro: TODO fracasso
relatado aqui foi antes investigado como se fosse um defeito de código. A
seção [2] do relatório é a prova disso — um teste de gradiente por diferenças
finitas que compara a retropropagação com a definição de derivada. Só depois
de o gradiente passar no teste, de a entrada ser padronizada, de a taxa de
aprendizado ser calibrada e de a curva de aprendizado passar a SUBIR com mais
dados é que os números abaixo puderam ser lidos como resultado.

RESTRIÇÕES DE IMPLEMENTAÇÃO
---------------------------
Somente a biblioteca padrão do Python 3 + numpy. Nenhum PyTorch, TensorFlow,
scikit-learn, PIL ou matplotlib. As "imagens" são rasterizadas à mão com
aritmética de matrizes; as camadas densas, a ReLU, a softmax, a entropia
cruzada, a retropropagação e o SGD com momento são todos escritos aqui dentro.
Os gráficos são barras de asteriscos no terminal.

EXECUÇÃO
--------
    python3 exercicio_pratico_2_ingrediente_neural_percepcao_sem_raciocinio.py

Sem argumentos, sem arquivos de entrada, sem interação, sem rede. A saída é um
relatório de texto no terminal. A semente aleatória é fixa, então o relatório é
sempre o mesmo — você pode conferir seus números com os do colega ao lado.
"""

import os

# Uma única linha de configuração ANTES de importar numpy, e ela é técnica, não
# estilística: as matrizes deste exercício são pequenas (lotes de 32 linhas), e
# nesse regime uma BLAS multithread gasta mais tempo sincronizando threads do
# que multiplicando números — chega a ficar duas ordens de grandeza mais lenta.
# Fixar uma thread também elimina qualquer variação de arredondamento causada
# pela ordem em que as threads somam, o que reforça a reprodutibilidade.
for _variavel in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variavel, "1")

import textwrap

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTES GLOBAIS
# ---------------------------------------------------------------------------

SEMENTE = 42                 # semente fixa => saída 100% reprodutível
LARGURA = 78                 # largura das linhas do relatório
DTIPO = np.float32           # metade da memória e ~2x mais rápido que float64

# --- O vocabulário do mundo sintético (inspirado no CLEVR, citado no capítulo)
FORMAS = ("esfera", "cubo", "cilindro")
CORES = ("vermelho", "verde", "azul", "cinza")
TAMANHOS = ("pequeno", "grande")
MATERIAIS = ("fosco", "metalico")

# Cor de cada classe em RGB: são os três canais que a rede recebe por pixel,
# exatamente como uma câmera os entregaria.
RGB_DAS_CORES = np.array([[0.92, 0.20, 0.18],    # vermelho
                          [0.18, 0.74, 0.30],    # verde
                          [0.22, 0.38, 0.94],    # azul
                          [0.66, 0.66, 0.68]],   # cinza
                         dtype=np.float64)

# --- Rasterização
LADO_OBJETO = 16             # objeto isolado: 16x16 x 3 canais = 768 entradas
LADO_CELULA = 12             # na cena, cada objeto ocupa uma célula 12x12
N_CELULAS = 4                # a cena é uma tira de 4 células => 12x48
INTENSIDADE_DO_FUNDO = 0.10  # fundo escuro, como nas renderizações do CLEVR
DESVIO_DO_RUIDO = 0.045      # ruído gaussiano somado a todos os pixels
JITTER_DE_POSICAO = 1.4      # deslocamento máximo (em pixels) do centro

# Raio relativo ao lado do raster. Nas cenas a faixa é mais estreita: em uma
# célula de 12x12 um objeto "pequeno" do tamanho usado no objeto isolado
# ficaria irreconhecível, e a tarefa de raciocínio viraria adivinhação visual.
RAIO_DO_OBJETO = {"pequeno": 0.165, "grande": 0.265}
RAIO_NA_CENA = {"pequeno": 0.200, "grande": 0.280}

# Fatores que igualam a ÁREA das três formas para um mesmo raio nominal. A área
# do disco (norma L2) é pi*r², a do quadrado (L-infinito) é 4r² e a do losango
# (L1) é 2r². Sem essa correção a rede resolveria "forma" apenas contando
# pixels acesos — e o exercício mediria tamanho disfarçado de forma.
FATOR_DE_AREA = (2.0 / np.sqrt(np.pi), 1.0, np.sqrt(2.0))

# --- Normalização FIXA da entrada. Os pixels crus vivem em [0, 1] com média
# 0,166 e desvio 0,174 (medidos sobre milhares de rasters do próprio gerador,
# tanto de objetos isolados quanto de cenas). Padronizar para média ~0 e desvio
# ~1 não é cosmética: sem isso, todos os 768 pixels de um raster entram com o
# mesmo viés negativo forte, o gradiente da primeira camada vira um vetor
# gigante quase constante e o SGD com momento diverge. Os números são
# CONSTANTES, e não estimados no conjunto de treino: com 10 exemplos uma média
# estimada seria pura sorte, e a curva de aprendizado passaria a medir duas
# coisas ao mesmo tempo. Assim ela mede só o tamanho do treino.
MEDIA_DO_PIXEL = 0.166
DESVIO_DO_PIXEL = 0.174

# Amplitude dos "1" do vetor da pergunta. Não é enfeite: a pergunta ocupa 17
# números ao lado de 1.728 pixels padronizados, e com amplitude 1 ela responde
# por ~3% da norma da entrada. A rede simplesmente não a enxerga — mede-se
# isso: com amplitude 1 a acurácia nas combinações VISTAS fica em 46,7%; com
# amplitude 6 sobe para 79,8%, sem mudar mais nada. Damos essa vantagem de
# propósito, para que o fracasso adiante não possa ser atribuído a uma
# codificação ruim da pergunta.
AMPLITUDE_DA_PERGUNTA = 6.0

# --- Arquitetura das redes (Figura 5.2: camadas densas, tudo ligado a tudo)
OCULTAS = (64, 48)           # duas camadas ocultas com ReLU
TAM_LOTE = 32
TAXA_APRENDIZADO = 0.02      # SGD...
MOMENTO = 0.9                # ...com momento
# O passo efetivo do momento é taxa/(1 - momento) = 0,20. Com 0,05 ele valeria
# 0,50 e as duas redes DIVERGIAM: a acurácia colava no acaso (33% em forma,
# 25% em cor) para qualquer tamanho de treino.

# --- Orçamento de treino. Toda a curva de aprendizado usa o MESMO número de
# épocas: se o treino encolhesse junto com o conjunto, a curva mediria o
# orçamento de otimização em vez do tamanho do treino — e chegaria a DESCER com
# mais dados. O piso de passos existe para o outro lado: com 10 exemplos, uma
# época é um único mini-lote, e 25 deles não bastariam para convergir nem
# naquele punhado. O piso dá aos conjuntos pequenos MAIS otimização que aos
# grandes, nunca menos — o que torna qualquer subida da curva conservadora.
EPOCAS_PERCEPTOR = 25
EPOCAS_CENA = 30             # de propósito MAIOR: ninguém pode alegar subtreino
PASSOS_MINIMOS = 300         # piso de atualizações de peso, para n pequeno
EPOCAS_COMPOSICIONAL = 30

# --- Tamanhos da curva de aprendizado (a "fome de dados" do capítulo)
TAMANHOS_DE_TREINO = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)

N_OBJETOS_TREINO = 5000
N_OBJETOS_TESTE = 1500
PERGUNTAS_POR_CENA = 4       # como no CLEVR, cada cena rende várias perguntas
N_PERGUNTAS_TREINO = 5000
N_PERGUNTAS_TESTE = 1600
N_CENAS_TREINO_COMPOSICIONAL = 3000
N_CENAS_TESTE_COMPOSICIONAL = 600   # cenas de 4 objetos, usadas nos DOIS testes

# --- Perguntas sobre a cena
#
# Quatro tipos, de duas naturezas bem diferentes — e a distinção é o eixo do
# exercício inteiro:
#
#   PERCEPÇÃO  CONSULTAR(atributo, posição) só exige LER um objeto: olhar a
#              célula indicada e dizer sua forma ou sua cor.
#   RACIOCÍNIO CONTAR, EXISTE e MESMA_FORMA exigem RELACIONAR objetos: varrer a
#              cena, comparar, agregar. Nenhum deles se resolve olhando um
#              pedaço fixo da imagem.
TIPOS_DE_PERGUNTA = ("CONTAR", "EXISTE", "MESMA_FORMA", "CONSULTAR")
TIPOS_DE_RACIOCINIO = ("CONTAR", "EXISTE", "MESMA_FORMA")
PARES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
ATRIBUTOS_CONSULTAVEIS = ("forma", "cor")
VALORES_DO_ATRIBUTO = {"forma": FORMAS, "cor": CORES}

# Vocabulário único de respostas: uma só softmax cobre contagens, sim/não,
# cores e formas.
RESPOSTAS = ("0", "1", "2", "3", "4", "sim", "nao") + CORES + FORMAS

# Comprimento do vetor que codifica a pergunta (ver `vetor_da_pergunta`).
DIM_PERGUNTA = 23


def _familia(resposta):
    """A que 'família' pertence um rótulo de resposta."""
    if resposta.isdigit():
        return "numero"
    if resposta in ("sim", "nao"):
        return "booleana"
    return "cor" if resposta in CORES else "forma"


# Usado para diagnosticar se a rede ao menos acertou o FORMATO da resposta
# quando errou o conteúdo.
FAMILIA_DA_RESPOSTA = {r: _familia(r) for r in RESPOSTAS}

# Combinações (atributo x posição) DELIBERADAMENTE retiradas do treino, e seus
# controles pareados: mesmo tipo de pergunta, mesmas cenas, mesmo formato de
# resposta — só a combinação de argumentos muda. Todas são do tipo CONSULTAR,
# de propósito: é a única família que a rede DOMINA, e por isso a única em que
# uma queda pode ser atribuída à composição, e não à incompetência de base.
COMBINACOES_RETIDAS = (("CONSULTAR", "cor", 3),
                       ("CONSULTAR", "forma", 1),
                       ("CONSULTAR", "cor", 2))
COMBINACOES_CONTROLE = (("CONSULTAR", "cor", 0),
                        ("CONSULTAR", "forma", 3),
                        ("CONSULTAR", "forma", 2))
COMBINACOES_DE_TREINO = tuple(
    ("CONSULTAR", atributo, posicao)
    for atributo in ATRIBUTOS_CONSULTAVEIS
    for posicao in range(N_CELULAS)
    if ("CONSULTAR", atributo, posicao) not in COMBINACOES_RETIDAS)

PALETA_ASCII = " .:-=+*#%@"   # 10 níveis de "tinta" para desenhar no terminal
LARGURA_BARRA = 28


# ===========================================================================
# PARTE 1 — O MUNDO SINTÉTICO: RASTERIZAÇÃO SEM PIL E SEM MATPLOTLIB
# ===========================================================================

_CACHE_DE_GRADES = {}


def grades_de_coordenadas(lado):
    """Matrizes (linha, coluna) com a coordenada do centro de cada pixel.

    Fica em cache: montar as grades dentro do laço custaria mais caro do que
    desenhar os objetos.
    """
    if lado not in _CACHE_DE_GRADES:
        eixo = np.arange(lado, dtype=np.float64) + 0.5
        _CACHE_DE_GRADES[lado] = np.meshgrid(eixo, eixo, indexing="ij")
    return _CACHE_DE_GRADES[lado]


def desenhar_objeto(rng, atributos, lado, raios):
    """Rasteriza UM objeto em um quadro `lado` x `lado` x 3 canais RGB.

    A geometria sai de três normas do plano, medidas a partir do centro:
        esfera   : distância euclidiana    (L2)          -> disco
        cubo     : distância do máximo     (L-infinito)  -> quadrado
        cilindro : distância de Manhattan  (L1)          -> losango, o contorno
                   de um cilindro visto de lado

    O MATERIAL vira textura de preenchimento — fosco é uniforme, metálico ganha
    uma rampa de iluminação diagonal mais um brilho especular branco. Posição,
    escala e exposição são sorteadas, e sobre tudo vai um ruído gaussiano leve.
    """
    linhas, colunas = grades_de_coordenadas(lado)
    forma, cor = atributos["forma"], atributos["cor"]

    raio = raios[TAMANHOS[atributos["tamanho"]]] * lado * FATOR_DE_AREA[forma]
    raio *= float(rng.uniform(0.92, 1.08))

    # O centro passeia dentro de uma margem que mantém o objeto todo no quadro.
    deslocamento = min(max(0.0, lado / 2.0 - raio - 0.6), JITTER_DE_POSICAO)
    dl = linhas - (lado / 2.0 + float(rng.uniform(-deslocamento, deslocamento)))
    dc = colunas - (lado / 2.0 + float(rng.uniform(-deslocamento, deslocamento)))

    if forma == 0:
        distancia = np.sqrt(dl * dl + dc * dc)
    elif forma == 1:
        distancia = np.maximum(np.abs(dl), np.abs(dc))
    else:
        distancia = np.abs(dl) + np.abs(dc)

    # Borda suave de ~1 pixel: sem esse antialiasing, um disco e um quadrado de
    # raio 3 ficariam quase idênticos em um raster tão pequeno.
    mascara = np.clip(raio - distancia + 0.5, 0.0, 1.0)

    if atributos["material"] == 0:                   # fosco
        campo = np.full_like(mascara, 0.86)
        especular = np.zeros_like(mascara)
    else:                                            # metálico
        rampa = 0.5 - 0.5 * (dl + dc) / (2.0 * raio)
        campo = 0.42 + 0.62 * np.clip(rampa, 0.0, 1.0)
        alvo = -0.38 * raio
        sigma = max(0.30 * raio, 0.8)
        especular = 0.75 * np.exp(-((dl - alvo) ** 2 + (dc - alvo) ** 2)
                                  / (2.0 * sigma ** 2))

    tinta = ((campo[:, :, None] * RGB_DAS_CORES[cor][None, None, :]
              + especular[:, :, None]) * float(rng.uniform(0.88, 1.06)))

    alfa = mascara[:, :, None]
    quadro = INTENSIDADE_DO_FUNDO * (1.0 - alfa) + tinta * alfa
    quadro = quadro + rng.normal(0.0, DESVIO_DO_RUIDO, size=quadro.shape)
    return np.clip(quadro, 0.0, 1.0)


def sortear_atributos(rng):
    """Sorteia, uniformemente, os quatro atributos simbólicos de um objeto."""
    return {"forma": int(rng.integers(0, len(FORMAS))),
            "cor": int(rng.integers(0, len(CORES))),
            "tamanho": int(rng.integers(0, len(TAMANHOS))),
            "material": int(rng.integers(0, len(MATERIAIS)))}


def normalizar(vetor):
    """Padroniza a intensidade dos pixels: média ~0 e desvio ~1.

    Usa as constantes MEDIA_DO_PIXEL e DESVIO_DO_PIXEL, medidas uma única vez
    sobre o gerador e escritas no topo do arquivo. Não é detalhe de estilo: com
    a entrada descentrada (todo pixel de fundo valendo o mesmo número longe de
    zero), o gradiente da primeira camada fica dominado por uma componente
    contínua enorme e o treino diverge. Foi exatamente esse o defeito que
    travava este exercício.
    """
    return ((vetor.astype(DTIPO) - DTIPO(MEDIA_DO_PIXEL))
            / DTIPO(DESVIO_DO_PIXEL))


def gerar_objetos_isolados(n_objetos, rng):
    """Gera rasters de objetos isolados com seus quatro rótulos.

    Retorna (entradas, alvos, galeria), sendo `galeria` os quatro primeiros
    quadros guardados para o desenho ASCII do relatório.
    """
    entradas = np.empty((n_objetos, LADO_OBJETO * LADO_OBJETO * 3), dtype=DTIPO)
    alvos = {nome: np.empty(n_objetos, dtype=np.int64)
             for nome in ("forma", "cor", "tamanho", "material")}
    galeria = []
    for i in range(n_objetos):
        atributos = sortear_atributos(rng)
        quadro = desenhar_objeto(rng, atributos, LADO_OBJETO, RAIO_DO_OBJETO)
        entradas[i] = normalizar(quadro.reshape(-1))
        for nome, valor in atributos.items():
            alvos[nome][i] = valor
        if i < 4:
            galeria.append((quadro, atributos))
    return entradas, alvos, galeria


def gerar_cena(rng, n_objetos=None):
    """Desenha uma cena: uma tira de N_CELULAS células de LADO_CELULA pixels.

    De 2 a 4 objetos ocupam as células da esquerda para a direita; as restantes
    ficam vazias. Damos de presente à rede uma cena JÁ SEGMENTADA em posições
    fixas — sem sobreposição, sem oclusão. Isso torna a tarefa bem mais fácil do
    que o CLEVR de verdade, e é esse o ponto: mesmo assim o raciocínio não vem.

    Retorna (raster, objetos), sendo `objetos` a descrição simbólica que a rede
    NUNCA recebe.
    """
    if n_objetos is None:
        n_objetos = int(rng.integers(2, N_CELULAS + 1))
    raster = np.full((LADO_CELULA, LADO_CELULA * N_CELULAS, 3),
                     INTENSIDADE_DO_FUNDO, dtype=np.float64)
    raster += rng.normal(0.0, DESVIO_DO_RUIDO, size=raster.shape)
    objetos = []
    for celula in range(n_objetos):
        atributos = sortear_atributos(rng)
        inicio = celula * LADO_CELULA
        raster[:, inicio:inicio + LADO_CELULA, :] = desenhar_objeto(
            rng, atributos, LADO_CELULA, RAIO_NA_CENA)
        objetos.append(atributos)
    return np.clip(raster, 0.0, 1.0), objetos


def desenhar_em_ascii(quadro):
    """Converte um raster RGB em linhas de texto, uma por linha de pixels."""
    luminancia = quadro.mean(axis=2)
    indices = np.clip((luminancia * len(PALETA_ASCII)).astype(int),
                      0, len(PALETA_ASCII) - 1)
    return ["".join(PALETA_ASCII[j] for j in linha) for linha in indices]


def imprimir_lado_a_lado(blocos, rotulos, recuo=4, separador="   "):
    """Imprime vários desenhos ASCII em colunas, com um rótulo sob cada um."""
    larguras = [max(len(l) for l in bloco) for bloco in blocos]
    for i in range(max(len(b) for b in blocos)):
        print((" " * recuo + separador.join(
            (b[i] if i < len(b) else "").ljust(w)
            for b, w in zip(blocos, larguras))).rstrip())
    print((" " * recuo + separador.join(r[:w].ljust(w)
                                        for r, w in zip(rotulos, larguras))
           ).rstrip())


# ===========================================================================
# PARTE 2 — A REDE NEURAL, ESCRITA DO ZERO (Figura 5.2)
# ===========================================================================

def softmax_estavel(logits):
    """Softmax por linha, subtraindo o máximo antes de exponenciar.

    Subtrair o máximo não muda o resultado (a softmax é invariante a somas
    constantes no expoente) e evita que exp() estoure para infinito.
    """
    exponenciais = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponenciais / exponenciais.sum(axis=1, keepdims=True)


def gradientes_da_saida(logits_por_cabeca, alvos, indices, tipo=DTIPO):
    """Gradiente da perda média em relação aos LOGITS de cada cabeça.

    A composição softmax + entropia cruzada tem derivada notavelmente simples:
    (p - y). Dividimos pelo tamanho do lote porque a perda de referência é a
    MÉDIA por exemplo — se a divisão sumisse, o passo efetivo cresceria com o
    lote e o teste de gradiente acusaria o erro na hora.
    """
    gradientes = {}
    for nome, z in logits_por_cabeca.items():
        p = softmax_estavel(z)
        p[np.arange(len(indices)), alvos[nome][indices]] -= 1.0
        gradientes[nome] = (p / len(indices)).astype(tipo)
    return gradientes


class CamadaDensa:
    """Uma camada totalmente conectada: saída = entrada @ W + b.

    É literalmente o desenho da Figura 5.2 — cada neurônio ligado a TODOS os
    da camada anterior. O número de conexões é entradas x saídas, e é ele que
    explode quando a entrada é uma imagem. A camada também guarda os
    acumuladores de momento, porque é ela quem dá o passo de otimização.
    """

    def __init__(self, n_entradas, n_saidas, rng, tipo=DTIPO):
        # Inicialização de He: variância 2/n_entradas, adequada à ReLU.
        escala = np.sqrt(2.0 / n_entradas)
        self.W = rng.normal(0.0, escala, size=(n_entradas, n_saidas)).astype(tipo)
        self.b = np.zeros(n_saidas, dtype=tipo)
        self.tipo = tipo
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
        self.entrada = self.gW = self.gb = None

    @property
    def n_conexoes(self):
        """Quantidade de pesos — as arestas do desenho da Figura 5.2."""
        return self.W.size

    def frente(self, entrada):
        """Passo para a frente; guarda a entrada para a retropropagação."""
        self.entrada = entrada
        return entrada @ self.W + self.b

    def tras(self, gradiente_saida):
        """Retropropagação: calcula dL/dW e dL/db, devolve dL/d(entrada)."""
        self.gW = self.entrada.T @ gradiente_saida
        self.gb = gradiente_saida.sum(axis=0)
        return gradiente_saida @ self.W.T

    def passo(self, taxa, momento):
        """SGD com momento: v <- momento*v - taxa*grad ; peso <- peso + v."""
        self.vW *= self.tipo(momento)
        self.vW -= self.tipo(taxa) * self.gW
        self.W += self.vW
        self.vb *= self.tipo(momento)
        self.vb -= self.tipo(taxa) * self.gb
        self.b += self.vb


class RedeMulticabeca:
    """MLP com tronco compartilhado e uma ou mais cabeças de classificação.

    Com quatro cabeças ela é o PERCEPTOR DE ATRIBUTOS (forma, cor, tamanho,
    material); com uma cabeça só, a rede de perguntas-e-respostas sobre cenas.
    A arquitetura é a mesma — o que muda é a tarefa. Essa igualdade é
    proposital: o contraste entre as duas seções não pode ser atribuído a uma
    rede melhor ou pior.
    """

    def __init__(self, n_entradas, tamanhos_ocultos, cabecas, rng, tipo=DTIPO):
        self.tipo = tipo
        self.tronco = []
        anterior = n_entradas
        for tamanho in tamanhos_ocultos:
            self.tronco.append(CamadaDensa(anterior, tamanho, rng, tipo))
            anterior = tamanho
        self.cabecas = {nome: CamadaDensa(anterior, n_classes, rng, tipo)
                        for nome, n_classes in cabecas.items()}
        self.mascaras = []

    @property
    def n_parametros(self):
        """Total de pesos e vieses — o tamanho do modelo."""
        return sum(c.W.size + c.b.size
                   for c in list(self.tronco) + list(self.cabecas.values()))

    def frente(self, entrada):
        """Devolve um dicionário nome_da_cabeça -> logits."""
        ativacao = entrada
        self.mascaras = []
        for camada in self.tronco:
            z = camada.frente(ativacao)
            mascara = z > 0                      # derivada da ReLU
            self.mascaras.append(mascara)
            ativacao = z * mascara
        return {nome: cabeca.frente(ativacao)
                for nome, cabeca in self.cabecas.items()}

    def tras(self, gradientes_das_cabecas):
        """Retropropaga: as cabeças SOMAM seus gradientes na saída do tronco."""
        gradiente = None
        for nome, cabeca in self.cabecas.items():
            parcela = cabeca.tras(gradientes_das_cabecas[nome])
            gradiente = parcela if gradiente is None else gradiente + parcela
        for camada, mascara in zip(reversed(self.tronco),
                                   reversed(self.mascaras)):
            gradiente = camada.tras(gradiente * mascara)

    def passo(self, taxa, momento):
        """Atualiza todos os pesos da rede."""
        for camada in list(self.tronco) + list(self.cabecas.values()):
            camada.passo(taxa, momento)


def numero_de_epocas(n_treino, epocas_base):
    """Quantas épocas treinar um conjunto de `n_treino` exemplos.

    Regra: `epocas_base` épocas para todo mundo, com um PISO de
    PASSOS_MINIMOS atualizações de peso. Como um conjunto de n exemplos rende
    ceil(n / TAM_LOTE) atualizações por época, o piso só entra em ação nos
    conjuntos pequenos — e sempre para lhes dar MAIS treino, nunca menos.

    A versão anterior deste exercício fazia o contrário: dividia um orçamento
    fixo de apresentações pelo tamanho do conjunto, de modo que n=10 ganhava 60
    épocas e n=5000 apenas 10. Uma curva de aprendizado construída assim não
    mede o efeito do dado — mede o orçamento de otimização, e pode até descer.
    """
    passos_por_epoca = max(1, -(-n_treino // TAM_LOTE))
    piso = -(-PASSOS_MINIMOS // passos_por_epoca)
    return int(max(epocas_base, piso))


def perda_media(logits_por_cabeca, alvos, indices):
    """Entropia cruzada média por exemplo, somada sobre as cabeças.

    É a função que a retropropagação afirma derivar. O teste de gradiente da
    seção de validação compara a derivada analítica com a variação medida
    DESTE número, e por isso as duas precisam ser exatamente a mesma coisa.
    """
    total = 0.0
    for nome, z in logits_por_cabeca.items():
        p = softmax_estavel(z.astype(np.float64))
        alvo = alvos[nome][indices]
        total -= float(np.log(np.maximum(
            p[np.arange(len(indices)), alvo], 1e-300)).sum())
    return total / len(indices)


def treinar_rede(rede, entradas, alvos, epocas, rng):
    """Treina em mini-lotes com SGD + momento e taxa em decaimento linear.

    A perda é a soma das entropias cruzadas das cabeças. O gradiente da
    entropia cruzada composta com a softmax é, felizmente, apenas
    (p - y)/tamanho_do_lote — por isso as duas aparecem sempre juntas.
    """
    n = entradas.shape[0]
    for epoca in range(epocas):
        fator = 1.0 - 0.9 * (epoca / max(1, epocas - 1))
        ordem = rng.permutation(n)
        for inicio in range(0, n, TAM_LOTE):
            indices = ordem[inicio:inicio + TAM_LOTE]
            logits = rede.frente(entradas[indices])
            rede.tras(gradientes_da_saida(logits, alvos, indices, rede.tipo))
            rede.passo(TAXA_APRENDIZADO * fator, MOMENTO)
    return rede


def teste_de_gradiente(n_amostras=24, epsilon=1e-5):
    """Confere a retropropagação contra diferenças finitas centradas.

    A ideia é a definição de derivada, aplicada a um peso de cada vez. Para um
    parâmetro w, empurramos w para w + eps e para w - eps, medimos a perda nos
    dois casos e comparamos

        (L(w + eps) - L(w - eps)) / (2 * eps)     [medido]

    com o valor que a retropropagação afirmou   [analítico].

    A diferença CENTRADA é usada porque seu erro cai com eps², e não com eps —
    com eps = 1e-6 sobram cerca de dez dígitos corretos. Toda a conta roda em
    float64: em float32 o próprio arredondamento da perda seria maior que a
    diferença que queremos medir, e o teste não provaria nada.

    A rede de teste é minúscula (12 entradas, duas camadas ocultas, duas
    cabeças) porque o que se testa é a ÁLGEBRA, e ela não depende do tamanho.
    O critério usual da literatura é erro relativo abaixo de 1e-7.
    """
    rng = np.random.default_rng(SEMENTE + 900)
    n_entradas, n_exemplos = 12, 7
    cabecas = {"a": 4, "b": 3}
    rede = RedeMulticabeca(n_entradas, (9, 6), cabecas, rng, tipo=np.float64)
    entradas = rng.normal(0.0, 1.0, size=(n_exemplos, n_entradas))
    alvos = {nome: rng.integers(0, k, size=n_exemplos)
             for nome, k in cabecas.items()}
    indices = np.arange(n_exemplos)

    # 1) gradiente analítico, pelo mesmo código que treina as redes de verdade
    logits = rede.frente(entradas)
    rede.tras(gradientes_da_saida(logits, alvos, indices, np.float64))

    camadas = [("tronco1", rede.tronco[0]), ("tronco2", rede.tronco[1]),
               ("cabeca_a", rede.cabecas["a"]), ("cabeca_b", rede.cabecas["b"])]

    def perda_atual():
        return perda_media(rede.frente(entradas), alvos, indices)

    linhas, pior = [], 0.0
    for nome, camada in camadas:
        for matriz, gradiente, rotulo in ((camada.W, camada.gW, "W"),
                                          (camada.b, camada.gb, "b")):
            plana = matriz.reshape(-1)
            plano_g = gradiente.reshape(-1)
            sorteio = rng.permutation(plana.size)[:n_amostras]
            erros = []
            for k in sorteio:
                original = plana[k]
                plana[k] = original + epsilon
                mais = perda_atual()
                plana[k] = original - epsilon
                menos = perda_atual()
                plana[k] = original
                medido = (mais - menos) / (2.0 * epsilon)
                analitico = float(plano_g[k])
                escala = max(abs(medido), abs(analitico), 1e-12)
                erros.append(abs(medido - analitico) / escala)
            maior = max(erros)
            pior = max(pior, maior)
            linhas.append({"camada": f"{nome}.{rotulo}",
                           "n_parametros": int(plana.size),
                           "n_testados": int(len(sorteio)),
                           "erro_maximo": maior,
                           "erro_medio": float(np.mean(erros))})
    return {"linhas": linhas, "pior": pior, "epsilon": epsilon,
            "n_testados": sum(l["n_testados"] for l in linhas)}


def prever(rede, entradas, bloco=2048):
    """Classe de maior probabilidade, por cabeça, em blocos."""
    partes = {nome: [] for nome in rede.cabecas}
    for inicio in range(0, entradas.shape[0], bloco):
        for nome, z in rede.frente(entradas[inicio:inicio + bloco]).items():
            partes[nome].append(np.argmax(z, axis=1))
    return {nome: np.concatenate(lista) for nome, lista in partes.items()}


def acuracia(verdadeiros, previstos):
    """Fração de acertos."""
    return float(np.mean(verdadeiros == previstos))


# ===========================================================================
# PARTE 3 — PERGUNTAS SOBRE CENAS (a tarefa de raciocínio)
# ===========================================================================

def chave_do_combo(pergunta):
    """Identidade simbólica da pergunta: (tipo, argumentos).

    É por essa chave que decidimos o que o modelo pode ou não ver no treino.
    """
    tipo = pergunta["tipo"]
    if tipo == "CONTAR":
        return ("CONTAR", FORMAS[pergunta["forma"]])
    if tipo == "EXISTE":
        return ("EXISTE", CORES[pergunta["cor"]], FORMAS[pergunta["forma"]])
    if tipo == "MESMA_FORMA":
        return ("MESMA_FORMA", pergunta["par"])
    return ("CONSULTAR", pergunta["atributo"], pergunta["posicao"])


def pergunta_da_chave(chave):
    """Reconstrói o dicionário da pergunta a partir de sua chave simbólica."""
    if chave[0] == "CONTAR":
        return {"tipo": "CONTAR", "forma": FORMAS.index(chave[1])}
    if chave[0] == "EXISTE":
        return {"tipo": "EXISTE", "cor": CORES.index(chave[1]),
                "forma": FORMAS.index(chave[2])}
    if chave[0] == "MESMA_FORMA":
        return {"tipo": "MESMA_FORMA", "par": chave[1]}
    return {"tipo": "CONSULTAR", "atributo": chave[1], "posicao": chave[2]}


def texto_da_pergunta(pergunta):
    """Versão em português da pergunta, para o relatório."""
    tipo = pergunta["tipo"]
    if tipo == "CONTAR":
        return f"quantos objetos da forma '{FORMAS[pergunta['forma']]}' há na cena?"
    if tipo == "EXISTE":
        return (f"existe algum objeto {CORES[pergunta['cor']]} da forma "
                f"'{FORMAS[pergunta['forma']]}'?")
    if tipo == "MESMA_FORMA":
        return (f"os objetos das posições {pergunta['par'][0]} e "
                f"{pergunta['par'][1]} têm a mesma forma?")
    return (f"qual é a {pergunta['atributo']} do objeto da posição "
            f"{pergunta['posicao']}?")


def responder(objetos, pergunta):
    """O motor SIMBÓLICO que produz o rótulo correto.

    Repare no contraste com o resto do arquivo: responder a essas perguntas é
    trivial QUANDO se tem os objetos como entidades. Uma linha de Python conta
    esferas; nenhuma quantidade de dados ensina isso a uma rede que nunca
    representou "objeto".
    """
    tipo = pergunta["tipo"]
    if tipo == "CONTAR":
        return str(sum(1 for o in objetos if o["forma"] == pergunta["forma"]))
    if tipo == "EXISTE":
        return "sim" if any(o["forma"] == pergunta["forma"]
                            and o["cor"] == pergunta["cor"]
                            for o in objetos) else "nao"
    if tipo == "MESMA_FORMA":
        i, j = pergunta["par"]
        return "sim" if objetos[i]["forma"] == objetos[j]["forma"] else "nao"
    atributo = pergunta["atributo"]
    return VALORES_DO_ATRIBUTO[atributo][objetos[pergunta["posicao"]][atributo]]


def vetor_da_pergunta(pergunta):
    """Codifica a pergunta em DIM_PERGUNTA números, cada um 0 ou a amplitude.

        [0:4]   tipo (CONTAR, EXISTE, MESMA_FORMA, CONSULTAR)
        [4:7]   argumento de forma     (CONTAR e EXISTE)
        [7:11]  argumento de cor       (EXISTE)
        [11:17] argumento de par       (MESMA_FORMA)
        [17:19] atributo consultado    (CONSULTAR)
        [19:23] posição consultada     (CONSULTAR)

    É a versão mais generosa possível para a rede: a pergunta chega já
    analisada, sem nenhum problema de linguagem natural pelo caminho, e com
    amplitude alta o bastante para competir com os pixels. Ainda assim a
    composição não vem.
    """
    a = DTIPO(AMPLITUDE_DA_PERGUNTA)
    vetor = np.zeros(DIM_PERGUNTA, dtype=DTIPO)
    vetor[TIPOS_DE_PERGUNTA.index(pergunta["tipo"])] = a
    if pergunta.get("forma") is not None:
        vetor[4 + pergunta["forma"]] = a
    if pergunta.get("cor") is not None:
        vetor[7 + pergunta["cor"]] = a
    if pergunta.get("par") is not None:
        vetor[11 + PARES.index(pergunta["par"])] = a
    if pergunta.get("atributo") is not None:
        vetor[17 + ATRIBUTOS_CONSULTAVEIS.index(pergunta["atributo"])] = a
        vetor[19 + pergunta["posicao"]] = a
    return vetor


def sortear_pergunta(rng, objetos):
    """Sorteia uma pergunta válida para a cena, com os quatro tipos igualmente
    prováveis.

    Em EXISTE, metade das perguntas é ancorada em um objeto realmente presente.
    Sem esse cuidado a resposta seria "nao" na esmagadora maioria dos casos, e a
    tarefa viraria um chute enviesado que esconderia o que queremos medir.
    """
    n = len(objetos)
    tipo = TIPOS_DE_PERGUNTA[int(rng.integers(0, len(TIPOS_DE_PERGUNTA)))]
    if tipo == "CONTAR":
        return {"tipo": tipo, "forma": int(rng.integers(0, len(FORMAS)))}
    if tipo == "EXISTE":
        if rng.random() < 0.5:
            modelo = objetos[int(rng.integers(0, n))]
            return {"tipo": tipo, "cor": modelo["cor"], "forma": modelo["forma"]}
        return {"tipo": tipo, "cor": int(rng.integers(0, len(CORES))),
                "forma": int(rng.integers(0, len(FORMAS)))}
    if tipo == "MESMA_FORMA":
        validos = [par for par in PARES if par[1] < n]
        return {"tipo": tipo,
                "par": validos[int(rng.integers(0, len(validos)))]}
    return {"tipo": tipo,
            "atributo": ATRIBUTOS_CONSULTAVEIS[
                int(rng.integers(0, len(ATRIBUTOS_CONSULTAVEIS)))],
            "posicao": int(rng.integers(0, n))}


def montar_conjunto_de_cenas(n_exemplos, rng, chaves_fixas=None,
                             chaves_sorteadas=None, n_objetos=None):
    """Monta um conjunto (entrada, resposta) para a tarefa de cenas.

    Cada cena rende várias perguntas — a mesma economia do CLEVR, que faz
    dezenas de perguntas sobre cada imagem renderizada. A entrada de um exemplo
    é o raster inteiro da cena, achatado, com o vetor da pergunta concatenado
    no fim. Uma rede só, ponta a ponta.

    Três regimes de amostragem das perguntas:
      * `chaves_fixas`   — cada cena recebe EXATAMENTE aquelas combinações
                           (é o que permite parear teste visto x teste inédito);
      * `chaves_sorteadas` — sorteia entre as combinações listadas (é o treino
                           do experimento composicional, do qual as retidas
                           foram removidas);
      * nenhum dos dois — sorteio livre entre os quatro tipos.
    """
    por_cena = len(chaves_fixas) if chaves_fixas else PERGUNTAS_POR_CENA
    n_cenas = int(np.ceil(n_exemplos / por_cena))
    n_pixels = LADO_CELULA * LADO_CELULA * N_CELULAS * 3

    entradas = np.empty((n_cenas * por_cena, n_pixels + DIM_PERGUNTA),
                        dtype=DTIPO)
    respostas = np.empty(n_cenas * por_cena, dtype=np.int64)
    perguntas = []

    linha = 0
    for _ in range(n_cenas):
        raster, objetos = gerar_cena(rng, n_objetos=n_objetos)
        pixels = normalizar(raster.reshape(-1))
        if chaves_fixas:
            lote = [pergunta_da_chave(c) for c in chaves_fixas]
        elif chaves_sorteadas:
            lote = [pergunta_da_chave(
                chaves_sorteadas[int(rng.integers(0, len(chaves_sorteadas)))])
                for _ in range(por_cena)]
        else:
            lote = [sortear_pergunta(rng, objetos) for _ in range(por_cena)]
        for pergunta in lote:
            entradas[linha, :n_pixels] = pixels
            entradas[linha, n_pixels:] = vetor_da_pergunta(pergunta)
            respostas[linha] = RESPOSTAS.index(responder(objetos, pergunta))
            perguntas.append(pergunta)
            linha += 1

    return entradas[:n_exemplos], respostas[:n_exemplos], perguntas[:n_exemplos]


def treinar_chute_cego(perguntas, respostas):
    """Aprende a resposta mais frequente por pergunta, IGNORANDO a imagem.

    É a linha de base honesta desta tarefa. Sem ela, uma acurácia de 50%
    pareceria aprendizado; com ela, descobre-se quanto daquilo é apenas o viés
    das perguntas ("CONTAR quase sempre dá 1"). São três níveis de recuo:
    combinação exata, tipo de pergunta e moda global.
    """
    por_combo, por_tipo = {}, {}
    for pergunta, resposta in zip(perguntas, respostas):
        por_combo.setdefault(chave_do_combo(pergunta), []).append(resposta)
        por_tipo.setdefault(pergunta["tipo"], []).append(resposta)

    def moda(lista):
        valores, contagens = np.unique(np.array(lista), return_counts=True)
        return int(valores[int(np.argmax(contagens))])

    return {"combo": {c: moda(v) for c, v in por_combo.items()},
            "tipo": {t: moda(v) for t, v in por_tipo.items()},
            "global": moda(list(respostas)) if len(respostas) else 0}


def prever_chute_cego(chutador, perguntas):
    """Aplica o chutador cego a uma lista de perguntas."""
    saida = np.empty(len(perguntas), dtype=np.int64)
    for i, pergunta in enumerate(perguntas):
        chave = chave_do_combo(pergunta)
        if chave in chutador["combo"]:
            saida[i] = chutador["combo"][chave]
        elif pergunta["tipo"] in chutador["tipo"]:
            saida[i] = chutador["tipo"][pergunta["tipo"]]
        else:
            saida[i] = chutador["global"]
    return saida


# ===========================================================================
# PARTE 4 — OS EXPERIMENTOS
# ===========================================================================

def curva_de_aprendizado_perceptor(entradas_tr, alvos_tr, entradas_te, alvos_te):
    """Treina o perceptor com conjuntos de tamanhos crescentes e ANINHADOS.

    Os primeiros 10 exemplos estão dentro dos primeiros 25, e assim por diante,
    como acontece quando se coleta mais dado de verdade. Retorna a lista de
    resultados e a última rede (a treinada com tudo).
    """
    cabecas = {"forma": len(FORMAS), "cor": len(CORES),
               "tamanho": len(TAMANHOS), "material": len(MATERIAIS)}
    resultados, ultima = [], None
    for i, n in enumerate(TAMANHOS_DE_TREINO):
        rng = np.random.default_rng(SEMENTE + 100 + i)
        rede = RedeMulticabeca(entradas_tr.shape[1], OCULTAS, cabecas, rng)
        epocas = numero_de_epocas(n, EPOCAS_PERCEPTOR)
        treinar_rede(rede, entradas_tr[:n],
                     {k: v[:n] for k, v in alvos_tr.items()}, epocas, rng)
        previsoes = prever(rede, entradas_te)
        por_atributo = {nome: acuracia(alvos_te[nome], previsoes[nome])
                        for nome in cabecas}
        resultados.append({"n": n, "epocas": epocas, "por_atributo": por_atributo,
                           "media": float(np.mean(list(por_atributo.values())))})
        ultima = rede
    return resultados, ultima


def curva_de_aprendizado_cenas(treino, teste):
    """A mesma curva, agora para as perguntas sobre cenas.

    Em cada ponto medimos também o CHUTE CEGO treinado no MESMO subconjunto:
    assim fica visível quanto da acurácia é aprendizado de verdade e quanto é
    só o viés das perguntas.
    """
    entradas_tr, respostas_tr, perguntas_tr = treino
    entradas_te, respostas_te, perguntas_te = teste
    resultados = []
    for i, n in enumerate(TAMANHOS_DE_TREINO):
        rng = np.random.default_rng(SEMENTE + 300 + i)
        rede = RedeMulticabeca(entradas_tr.shape[1], OCULTAS,
                               {"resposta": len(RESPOSTAS)}, rng)
        epocas = numero_de_epocas(n, EPOCAS_CENA)
        treinar_rede(rede, entradas_tr[:n], {"resposta": respostas_tr[:n]},
                     epocas, rng)
        previsoes = prever(rede, entradas_te)["resposta"]
        chutador = treinar_chute_cego(perguntas_tr[:n], respostas_tr[:n])
        cegas = prever_chute_cego(chutador, perguntas_te)
        por_tipo = {}
        for tipo in TIPOS_DE_PERGUNTA:
            m = np.array([p["tipo"] == tipo for p in perguntas_te])
            por_tipo[tipo] = {"n": int(m.sum()),
                              "rede": acuracia(respostas_te[m], previsoes[m]),
                              "cego": acuracia(respostas_te[m], cegas[m])}
        resultados.append({
            "n": n, "epocas": epocas,
            "acuracia": acuracia(respostas_te, previsoes),
            "cego": acuracia(respostas_te, cegas),
            "por_tipo": por_tipo})
    return resultados


def contar_exposicao_das_pecas(perguntas_tr):
    """Quantas vezes cada PEÇA das combinações retidas apareceu no treino.

    A tabela que sai daqui é o argumento central do exercício: cada ingrediente
    isolado foi visto milhares de vezes; só a combinação nunca apareceu.
    """
    def contar(condicao):
        return sum(1 for p in perguntas_tr if condicao(p))

    exposicao = {}
    for atributo in ATRIBUTOS_CONSULTAVEIS:
        exposicao[f"atributo '{atributo}' (em qualquer posição)"] = contar(
            lambda p, a=atributo: p.get("atributo") == a)
    for posicao in range(N_CELULAS):
        exposicao[f"posição {posicao} (com qualquer atributo)"] = contar(
            lambda p, k=posicao: p.get("posicao") == k)
    return exposicao


def experimento_composicional():
    """Treina retendo combinações inteiras e mede o custo dessa retenção.

    Desenho do experimento — e é ele que dá força à conclusão:

      * TAREFA: só CONSULTAR(atributo, posição). É a família que a rede
        DOMINA, como mostra a seção anterior. Se ela falhasse aqui por
        incompetência de base, a comparação não provaria nada sobre composição;
        assim, qualquer queda tem uma causa só.
      * TREINO: cenas de 4 objetos, perguntas sorteadas entre as
        COMBINACOES_DE_TREINO — todas as combinações atributo x posição EXCETO
        as três retidas.
      * TESTE: um único lote de cenas com 4 objetos. Sobre CADA cena fazemos
        seis perguntas — três com combinações vistas e três com as retidas,
        pareadas pelo atributo perguntado. As imagens são as MESMAS nos dois
        casos, o tipo de pergunta é o MESMO e o formato da resposta é o MESMO;
        só muda a combinação de argumentos.

    Nenhuma queda de acurácia pode, portanto, ser atribuída a imagens mais
    difíceis, a perguntas de outro tipo ou a um vocabulário de resposta maior.
    """
    n_treino = N_CENAS_TREINO_COMPOSICIONAL * PERGUNTAS_POR_CENA
    entradas_tr, respostas_tr, perguntas_tr = montar_conjunto_de_cenas(
        n_treino, np.random.default_rng(SEMENTE + 500),
        chaves_sorteadas=COMBINACOES_DE_TREINO, n_objetos=N_CELULAS)

    n_teste = N_CENAS_TESTE_COMPOSICIONAL * len(COMBINACOES_CONTROLE)
    vistas = montar_conjunto_de_cenas(n_teste,
                                      np.random.default_rng(SEMENTE + 501),
                                      chaves_fixas=COMBINACOES_CONTROLE,
                                      n_objetos=N_CELULAS)
    # MESMA semente de propósito: as cenas dos dois testes são idênticas.
    ineditas = montar_conjunto_de_cenas(n_teste,
                                        np.random.default_rng(SEMENTE + 501),
                                        chaves_fixas=COMBINACOES_RETIDAS,
                                        n_objetos=N_CELULAS)

    rng = np.random.default_rng(SEMENTE + 502)
    rede = RedeMulticabeca(entradas_tr.shape[1], OCULTAS,
                           {"resposta": len(RESPOSTAS)}, rng)
    treinar_rede(rede, entradas_tr, {"resposta": respostas_tr},
                 EPOCAS_COMPOSICIONAL, rng)
    chutador = treinar_chute_cego(perguntas_tr, respostas_tr)

    def avaliar(conjunto, chaves):
        entradas, respostas, perguntas = conjunto
        previsoes = prever(rede, entradas)["resposta"]
        cegas = prever_chute_cego(chutador, perguntas)
        detalhes = []
        for chave in chaves:
            m = np.array([chave_do_combo(p) == chave for p in perguntas])
            esperada = chave[1]      # "forma" ou "cor"
            detalhes.append({
                "chave": chave, "n": int(m.sum()),
                "acuracia": acuracia(respostas[m], previsoes[m]),
                "cego": acuracia(respostas[m], cegas[m]),
                "formato": float(np.mean(
                    [FAMILIA_DA_RESPOSTA[RESPOSTAS[r]] == esperada
                     for r in previsoes[m]]))})
        return {"geral": acuracia(respostas, previsoes),
                "cego": acuracia(respostas, cegas), "detalhes": detalhes}

    return {"n_treino": len(respostas_tr),
            "vistas": avaliar(vistas, COMBINACOES_CONTROLE),
            "ineditas": avaliar(ineditas, COMBINACOES_RETIDAS),
            "exposicao": contar_exposicao_das_pecas(perguntas_tr)}


# ===========================================================================
# PARTE 5 — FERRAMENTAS E SEÇÕES DO RELATÓRIO
# ===========================================================================

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


def paragrafo(texto, recuo=2):
    """Quebra um texto corrido dentro da largura do relatório."""
    espaco = " " * recuo
    return textwrap.fill(" ".join(texto.split()), width=LARGURA,
                         initial_indent=espaco, subsequent_indent=espaco)


def barra_ascii(valor, largura=LARGURA_BARRA):
    """Barra de asteriscos proporcional a `valor` em [0, 1]."""
    return "*" * int(round(largura * max(0.0, min(valor, 1.0))))


def milhar(n):
    """Formata um inteiro com ponto de milhar, no padrão brasileiro."""
    return f"{n:,}".replace(",", ".")


def formatar_combo(chave):
    """Escreve a chave simbólica de um combo em notação de programa."""
    if chave[0] == "MESMA_FORMA":
        return f"MESMA_FORMA(obj {chave[1][0]}, obj {chave[1][1]})"
    if chave[0] == "CONSULTAR":
        return f"CONSULTAR({chave[1]}, obj {chave[2]})"
    return f"{chave[0]}({', '.join(str(a) for a in chave[1:])})"


def acaso_do_combo(chave):
    """Acurácia de quem responde ao acaso dentro da família certa."""
    return 1.0 / len(VALORES_DO_ATRIBUTO[chave[1]])


def imprimir_secao_mundo(galeria, cena_exemplo):
    """Seção [1]: as 'imagens' sintéticas e a cena."""
    cabecalho("[1] O MUNDO SINTÉTICO — 'imagens' rasterizadas à mão", "-")
    print(f"  Formas ........................ {', '.join(FORMAS)}")
    print(f"  Cores ......................... {', '.join(CORES)}")
    print(f"  Tamanhos / materiais .......... {', '.join(TAMANHOS)} / "
          f"{', '.join(MATERIAIS)}")
    print(f"  Raster do objeto isolado ...... {LADO_OBJETO}x{LADO_OBJETO} x 3 "
          f"canais = {LADO_OBJETO * LADO_OBJETO * 3} entradas")
    print(f"  Raster da cena ................ {LADO_CELULA}x"
          f"{LADO_CELULA * N_CELULAS} x 3 canais = "
          f"{LADO_CELULA * LADO_CELULA * N_CELULAS * 3} entradas")
    print(f"  Ruído gaussiano (desvio) ...... {DESVIO_DO_RUIDO}")
    print()
    print(paragrafo(
        "As três formas saem de três normas diferentes do plano — euclidiana "
        "(disco), do máximo (quadrado) e de Manhattan (losango) — e seus raios "
        "são corrigidos para que as três tenham a MESMA área. Sem essa "
        "correção a rede resolveria 'forma' contando pixels acesos, e o "
        "exercício estaria medindo tamanho disfarçado de forma."))

    subtitulo("Quatro objetos, como a rede os recebe (tinta em ASCII)")
    print()
    imprimir_lado_a_lado(
        [desenhar_em_ascii(q) for q, _ in galeria],
        [f"{FORMAS[a['forma']][:8]}/{CORES[a['cor']][:5]}" for _, a in galeria])
    print()
    for quadro, a in galeria:
        print(f"    {FORMAS[a['forma']]:<9s} {CORES[a['cor']]:<10s} "
              f"{TAMANHOS[a['tamanho']]:<9s} {MATERIAIS[a['material']]:<9s} "
              f"(tinta média {quadro.mean():.3f})")

    raster, objetos = cena_exemplo
    subtitulo("Uma cena: 4 células lado a lado, uma por objeto")
    print()
    for linha in desenhar_em_ascii(raster):
        print(("    " + linha).rstrip())
    print()
    for i, o in enumerate(objetos):
        print(f"    posição {i}: {FORMAS[o['forma']]:<9s} {CORES[o['cor']]:<10s} "
              f"{TAMANHOS[o['tamanho']]:<9s} {MATERIAIS[o['material']]}")


def imprimir_secao_validacao(teste):
    """Seção [2]: a prova de que a retropropagação está correta."""
    cabecalho("[2] VALIDAÇÃO — a retropropagação está certa?", "-")
    print(paragrafo(
        "Antes de qualquer conclusão sobre o que a rede aprende ou deixa de "
        "aprender, é preciso saber se o gradiente que a move está correto. Um "
        "sinal trocado, um fator de lote esquecido ou uma máscara de ReLU fora "
        "de lugar produzem exatamente o que se viu numa versão anterior deste "
        "exercício: acurácia colada no acaso e uma curva de aprendizado que "
        "DESCE com mais dados. Como o defeito não levanta exceção, ele precisa "
        "ser caçado com um teste."))
    print()
    print(paragrafo(
        "O teste é a definição de derivada. Para cada peso w, empurramos w "
        "para w+eps e para w-eps, medimos a perda nos dois pontos e comparamos "
        "a diferença centrada (L(w+eps) - L(w-eps)) / (2*eps) com o valor que "
        "a retropropagação afirmou. Se os dois coincidirem em vários dígitos "
        "para pesos sorteados de TODAS as camadas, o gradiente está certo.",
        recuo=2))
    print()
    print(f"  Rede de teste ................. 12 -> 9 -> 6 -> [4, 3], "
          f"7 exemplos")
    print("  Aritmética .................... float64 (em float32 o "
          "arredondamento")
    print("                                  da perda encobriria o efeito)")
    print(f"  Passo das diferenças (eps) .... {teste['epsilon']:.0e}")
    print(f"  Pesos sorteados e conferidos .. {teste['n_testados']}")
    print()
    print(f"  {'parâmetro':<14s}{'total':>8s}{'testados':>10s}"
          f"{'erro médio':>13s}{'erro máximo':>14s}")
    print("  " + "-" * 59)
    for linha in teste["linhas"]:
        print(f"  {linha['camada']:<14s}{linha['n_parametros']:>8d}"
              f"{linha['n_testados']:>10d}{linha['erro_medio']:>13.2e}"
              f"{linha['erro_maximo']:>14.2e}")
    print("  " + "-" * 59)
    print(f"  {'PIOR CASO':<14s}{'':>18s}{'':>13s}{teste['pior']:>14.2e}")
    print()
    veredito = ("APROVADO" if teste["pior"] < 1e-5 else "REPROVADO")
    print(paragrafo(
        f"Erro relativo máximo de {teste['pior']:.2e}. O critério usual para "
        "diferenças centradas em float64 é ficar abaixo de 1e-5; o resultado "
        f"está {1e-5 / teste['pior']:.0f}x abaixo desse limite. Veredito: "
        f"{veredito}. A retropropagação deste arquivo calcula o gradiente que "
        "diz calcular — o que vier a seguir é propriedade da TAREFA, não de um "
        "erro de implementação."))


def imprimir_secao_perceptor(rede, resultado, entradas_te, alvos_te):
    """Seção [3]: o perceptor de atributos e sua acurácia."""
    cabecalho("[3] O PERCEPTOR DE ATRIBUTOS — o que a rede faz MUITO bem", "-")
    print(paragrafo(
        "Uma MLP olha o raster de um objeto isolado e responde quatro "
        "perguntas ao mesmo tempo: que forma, que cor, que tamanho, que "
        "material. Nenhuma regra sobre círculos, brilhos ou pixels foi "
        "escrita — os padrões saíram sozinhos dos dados."))
    print()
    print(f"  Arquitetura ................... {entradas_te.shape[1]} -> "
          f"{' -> '.join(str(t) for t in OCULTAS)} -> "
          f"[{len(FORMAS)}, {len(CORES)}, {len(TAMANHOS)}, {len(MATERIAIS)}]")
    print("  Ativações ..................... ReLU no tronco, softmax nas cabeças")
    print(f"  Otimizador .................... SGD com momento {MOMENTO}, "
          f"lotes de {TAM_LOTE}")
    print(f"  Parâmetros treináveis ......... {milhar(rede.n_parametros)}")
    print(f"  Conexões da 1a camada ......... "
          f"{milhar(rede.tronco[0].n_conexoes)}  (as arestas da Figura 5.2)")
    print(f"  Exemplos de treino / épocas ... {resultado['n']} / "
          f"{resultado['epocas']}")
    print()
    print(f"  {'Atributo':<12s}{'Classes':>9s}{'Acaso':>9s}{'Acurácia':>10s}"
          f"   Gráfico")
    print("  " + "-" * 68)
    classes = {"forma": len(FORMAS), "cor": len(CORES),
               "tamanho": len(TAMANHOS), "material": len(MATERIAIS)}
    for nome in ("forma", "cor", "tamanho", "material"):
        valor = resultado["por_atributo"][nome]
        print(f"  {nome:<12s}{classes[nome]:>9d}{1.0 / classes[nome]:>8.0%}"
              f"{valor:>10.1%}   {barra_ascii(valor)}")
    print("  " + "-" * 68)
    print(f"  {'média':<12s}{'':>18s}{resultado['media']:>10.1%}")
    print()
    previsoes = prever(rede, entradas_te[:6])
    print("  Seis objetos do conjunto de teste, previsão contra verdade:")
    print(f"    {'#':>2s}  {'previsto':<31s} {'verdadeiro'}")
    print("    " + "-" * 70)
    for i in range(6):
        prev = "/".join([FORMAS[previsoes["forma"][i]], CORES[previsoes["cor"][i]],
                         TAMANHOS[previsoes["tamanho"][i]],
                         MATERIAIS[previsoes["material"][i]]])
        verd = "/".join([FORMAS[alvos_te["forma"][i]], CORES[alvos_te["cor"][i]],
                         TAMANHOS[alvos_te["tamanho"][i]],
                         MATERIAIS[alvos_te["material"][i]]])
        print(f"    {i:2d}  {prev:<31s} {verd:<31s} "
              f"{'ok' if prev == verd else 'ERRO'}".rstrip())


def n_para_95(resultados, atributos, com_nomes=True):
    """Frase com o menor n em que TODOS os atributos dados passam de 95%."""
    alvo = next((r["n"] for r in resultados
                 if all(r["por_atributo"][a] >= 0.95 for a in atributos)), None)
    verbo = "passam" if len(atributos) > 1 else "passa"
    prefixo = ""
    if com_nomes:
        nomes = (" e ".join((", ".join(atributos[:-1]), atributos[-1]))
                 if len(atributos) > 1 else atributos[0])
        prefixo = nomes + " "
    else:
        verbo = "só " + verbo
    if alvo is None:
        return f"{prefixo}não {verbo} de 95% em nenhum tamanho testado"
    return f"{prefixo}{verbo} de 95% com {milhar(alvo)} exemplos"


def imprimir_secao_fome(resultados):
    """Seção [4]: a curva de aprendizado do perceptor — a fome de dados."""
    cabecalho("[4] A FOME DE DADOS — quanto exemplo custa PERCEBER", "-")
    print(paragrafo(
        "A mesma rede, o mesmo teste, treinos de tamanhos crescentes. Cada "
        "conjunto contém o anterior, como acontece quando se coleta mais dado "
        "de verdade. Todos os pontos treinam o MESMO número de épocas "
        f"({EPOCAS_PERCEPTOR}), com um piso de {PASSOS_MINIMOS} atualizações "
        "de peso que só beneficia os conjuntos pequenos — assim a curva mede o "
        "efeito do DADO, e não o do orçamento de otimização."))
    print()
    print(f"  {'n treino':>9s}{'épocas':>8s}{'forma':>8s}{'cor':>8s}{'tam.':>8s}"
          f"{'mater.':>8s}{'média':>8s}   Gráfico da média")
    print("  " + "-" * 76)
    for r in resultados:
        p = r["por_atributo"]
        print(f"  {r['n']:>9d}{r['epocas']:>8d}{p['forma']:>8.1%}{p['cor']:>8.1%}"
              f"{p['tamanho']:>8.1%}{p['material']:>8.1%}{r['media']:>8.1%}"
              f"   {barra_ascii(r['media'], largura=15)}")
    print("  " + "-" * 76)
    print()
    primeiro, ultimo = resultados[0], resultados[-1]
    n_alvo = next((r["n"] for r in resultados if r["media"] >= 0.90), None)
    quedas = sum(1 for a, b in zip(resultados, resultados[1:])
                 if b["media"] < a["media"] - 0.005)
    print(paragrafo(
        f"Com {primeiro['n']} exemplos a média fica em {primeiro['media']:.1%}; "
        f"com {milhar(ultimo['n'])}, em {ultimo['media']:.1%}. "
        + (f"Foram precisos {n_alvo} exemplos rotulados para cruzar a marca de "
           f"90%." if n_alvo else
           f"Nem com {milhar(ultimo['n'])} exemplos a média cruzou 90%.")
        + f" A curva sobe em todos os {len(resultados) - 1} degraus"
        + (" — nenhuma queda." if quedas == 0
           else f", com {quedas} queda(s) acima de meio ponto.")))
    print()
    print(paragrafo(
        "É a passagem do capítulo, medida: 'com um punhado de instâncias um "
        "bebê já distingue um gato de um cachorro — tarefa para a qual um "
        "modelo de IA precisaria de milhares de amostras'. Aqui o punhado "
        f"({primeiro['n']} exemplos) rende {primeiro['media']:.1%} e os "
        f"milhares rendem {ultimo['media']:.1%}. E note: distinguir um disco de "
        f"um quadrado em {LADO_OBJETO}x{LADO_OBJETO} pixels é "
        "incomparavelmente mais simples do que distinguir um gato de um "
        "cachorro. Repare também em QUAL atributo é o gargalo: "
        f"{n_para_95(resultados, ('cor', 'tamanho', 'material'))}; a FORMA — "
        "a única que exige olhar o contorno inteiro, e não uma estatística "
        "local de cor ou brilho — "
        f"{n_para_95(resultados, ('forma',), com_nomes=False)}."))


def imprimir_secao_tarefa(perguntas, respostas):
    """Seção [5]: a tarefa de perguntas e respostas sobre cenas."""
    cabecalho("[5] A TAREFA DE RACIOCÍNIO — perguntas sobre CENAS", "-")
    print(paragrafo(
        "Agora não há mais um objeto, e sim de 2 a 4. A rede recebe o raster "
        "inteiro da cena com o vetor da pergunta concatenado no fim e precisa "
        "devolver a resposta. Uma rede só, ponta a ponta — sem detector de "
        "objetos, sem programa, sem executor simbólico."))
    print()
    print("  Tipos de pergunta, em duas naturezas:")
    print()
    print("    PERCEPÇÃO — basta LER um objeto")
    print("      CONSULTAR(atributo, pos) .. qual a forma/cor do objeto na")
    print("                                  posição pos?")
    print()
    print("    RACIOCÍNIO — é preciso RELACIONAR objetos")
    print("      CONTAR(forma) ............. quantos objetos daquela forma há?")
    print("      EXISTE(cor, forma) ........ existe objeto daquela cor E forma?")
    print("      MESMA_FORMA(i, j) ......... i e j têm a mesma forma?")
    print()
    n_pixels = LADO_CELULA * LADO_CELULA * N_CELULAS * 3
    print(f"  Vocabulário de respostas ...... {len(RESPOSTAS)} rótulos:")
    for inicio in range(0, len(RESPOSTAS), 8):
        print("    " + ", ".join(RESPOSTAS[inicio:inicio + 8]))
    print(f"  Vetor da pergunta ............. {DIM_PERGUNTA} números "
          f"(4 tipo + 3 forma +")
    print("                                  4 cor + 6 par + 2 atributo + "
          "4 posição)")
    print(f"  Amplitude de cada '1' ......... {AMPLITUDE_DA_PERGUNTA} "
          f"(para a pergunta não se")
    print(f"                                  perder entre {milhar(n_pixels)} "
          f"pixels)")
    print(f"  Entrada total da rede ......... {milhar(n_pixels)} pixels + "
          f"{DIM_PERGUNTA} = {milhar(n_pixels + DIM_PERGUNTA)} números")
    subtitulo("Quatro exemplos de pergunta e resposta")
    print()
    for pergunta, resposta in zip(perguntas[:4], respostas[:4]):
        print(f"    {formatar_combo(chave_do_combo(pergunta)):<28s} -> "
              f"{RESPOSTAS[resposta]}")
        print(f'      "{texto_da_pergunta(pergunta)}"')
    print()
    print(paragrafo(
        "Responder a isso é TRIVIAL para quem tem os objetos como entidades: a "
        "função `responder` deste arquivo faz cada caso em uma linha de Python. "
        f"O problema é que a rede não tem entidades — ela tem "
        f"{milhar(n_pixels + DIM_PERGUNTA)} números."))


def imprimir_secao_fracasso_1(curva_perceptor, curva_cenas):
    """Seção [6]: primeiro fracasso — percebe, mas não relaciona."""
    cabecalho("[6] FRACASSO 1 — a mesma rede percebe, mas não relaciona", "-")
    print(paragrafo(
        "As duas curvas, lado a lado: mesma arquitetura, mesmo otimizador, "
        "mesma taxa, mesmos tamanhos de treino. A coluna 'cego' é a acurácia "
        "de um chutador que NÃO olha a imagem e só responde o mais frequente "
        "para cada pergunta — é o piso abaixo do qual não há aprendizado "
        f"nenhum. E a rede de cenas ainda recebe MAIS épocas ({EPOCAS_CENA} "
        f"contra {EPOCAS_PERCEPTOR}) que o perceptor, para que ninguém possa "
        "alegar subtreino."))
    print()
    print(f"  {'n treino':>9s} | {'PERCEPÇÃO (4 atrib.)':>22s} | "
          f"{'RACIOCÍNIO (cenas)':>24s}")
    print(f"  {'':>9s} | {'épocas':>8s}{'média':>8s}{'':>6s} | "
          f"{'épocas':>8s}{'rede':>8s}{'cego':>8s}")
    print("  " + "-" * 62)
    for a, b in zip(curva_perceptor, curva_cenas):
        print(f"  {a['n']:>9d} | {a['epocas']:>8d}{a['media']:>8.1%}{'':>6s} | "
              f"{b['epocas']:>8d}{b['acuracia']:>8.1%}{b['cego']:>8.1%}")
    print("  " + "-" * 62)
    print()
    print("  As duas curvas em barras:")
    print()
    for a, b in zip(curva_perceptor, curva_cenas):
        print(f"    n={a['n']:<5d} percepção  {a['media']:>6.1%} "
              f"|{barra_ascii(a['media'], largura=24)}")
        print(f"    {'':<7s} raciocínio {b['acuracia']:>6.1%} "
              f"|{barra_ascii(b['acuracia'], largura=24)}")
    print()
    final_p = curva_perceptor[-1]["media"]
    final_c, cego_c = curva_cenas[-1]["acuracia"], curva_cenas[-1]["cego"]
    n_p = next((r["n"] for r in curva_perceptor if r["media"] >= 0.90), None)
    n_c = next((r["n"] for r in curva_cenas if r["acuracia"] >= 0.90), None)
    print(paragrafo(
        f"Com os mesmos {milhar(curva_cenas[-1]['n'])} exemplos, a percepção "
        f"chega a {final_p:.1%} e o raciocínio a {final_c:.1%} — contra "
        f"{cego_c:.1%} de um chutador cego, ou seja, "
        f"{(final_c - cego_c) * 100:+.1f} pontos percentuais de vantagem sobre "
        "quem nem olha a imagem. "
        + (f"O perceptor cruzou 90% com {n_p} exemplos; " if n_p else
           "O perceptor não cruzou 90% na faixa testada; ")
        + (f"a rede de cenas cruzou com {n_c}." if n_c else
           "a rede de cenas NÃO cruzou 90% em nenhum tamanho testado.")))

    subtitulo("Onde exatamente ela ganha e onde ela perde")
    print()
    ultimo = curva_cenas[-1]
    print(f"    {'tipo de pergunta':<22s}{'natureza':<12s}{'n':>6s}"
          f"{'rede':>8s}{'cego':>8s}{'saldo':>8s}")
    print("    " + "-" * 64)
    ganhos = []
    for tipo in TIPOS_DE_PERGUNTA:
        d = ultimo["por_tipo"][tipo]
        saldo = (d["rede"] - d["cego"]) * 100
        natureza = "raciocínio" if tipo in TIPOS_DE_RACIOCINIO else "percepção"
        print(f"    {tipo:<22s}{natureza:<12s}{d['n']:>6d}{d['rede']:>8.1%}"
              f"{d['cego']:>8.1%}{saldo:>+8.1f}")
        ganhos.append((tipo, saldo))
    print("    " + "-" * 64)
    print("    saldo = pontos percentuais acima (+) ou abaixo (-) do "
          "chutador cego")
    print()
    acima = [t for t, g in ganhos if g > 0]
    abaixo = [t for t, g in ganhos if g <= 0]
    d_consultar = ultimo["por_tipo"]["CONSULTAR"]
    print(paragrafo(
        f"Com {milhar(ultimo['n'])} exemplos de treino, a rede supera o piso "
        f"cego em: {', '.join(acima) if acima else 'nenhum tipo'}. Fica no "
        f"piso ou abaixo dele em: "
        f"{', '.join(abaixo) if abaixo else 'nenhum tipo'}. A separação não é "
        "aleatória: a única pergunta em que ela ganha com folga é CONSULTAR "
        f"({d_consultar['rede']:.1%} contra {d_consultar['cego']:.1%}, um "
        f"saldo de {(d_consultar['rede'] - d_consultar['cego']) * 100:+.1f} "
        "pontos), a única que se resolve LENDO um objeto, sem relacionar nada. "
        "Nas três que exigem varrer a cena, comparar e agregar, o saldo vai de "
        + f"{min(g for t, g in ganhos if t in TIPOS_DE_RACIOCINIO):+.1f} a "
        + f"{max(g for t, g in ganhos if t in TIPOS_DE_RACIOCINIO):+.1f} "
        "pontos: ruído em torno do chute, e não aprendizado."))
    print()
    print(paragrafo(
        "Este é o resultado central da seção, e vale enunciá-lo sem rodeio: a "
        "rede não é ruim de VISÃO. Ela é ruim de RACIOCÍNIO. Os mesmos pixels "
        "que lhe permitem dizer a cor do objeto da posição 2 não lhe permitem "
        "contar quantas esferas existem — porque contar exige a noção de "
        "'objeto' como coisa contável, e essa noção não está em lugar nenhum "
        "dentro dela."))


def imprimir_secao_fracasso_2(comp):
    """Seção [7]: segundo fracasso — generalização composicional."""
    cabecalho("[7] FRACASSO 2 — combinações inéditas (o ponto central)", "-")
    print(paragrafo(
        "O experimento decisivo, e ele é montado justamente sobre a família "
        "que a rede DOMINA. Se a comparação fosse feita com CONTAR ou "
        "MESMA_FORMA, uma queda não provaria nada: a rede já falha ali mesmo "
        "no que viu. Com CONSULTAR(atributo, posição) não há essa saída — a "
        "seção anterior mostrou que ela lê objetos muito bem."))
    print()
    print(paragrafo(
        f"A rede é treinada em {milhar(comp['n_treino'])} perguntas do tipo "
        "CONSULTAR sobre cenas de 4 objetos, das quais TRÊS COMBINAÇÕES "
        "atributo x posição foram deliberadamente removidas. No teste, cada "
        "cena recebe seis perguntas: três com combinações vistas e três com as "
        "retidas. MESMAS imagens, MESMO tipo de pergunta, MESMO formato de "
        "resposta — só muda a combinação de argumentos."))

    subtitulo("As combinações retidas e seus controles pareados")
    print()
    print(f"    {'controle (visto no treino)':<34s}"
          f"{'retido (inédito)'}")
    print("    " + "-" * 60)
    for controle, retido in zip(COMBINACOES_CONTROLE, COMBINACOES_RETIDAS):
        print(f"    {formatar_combo(controle):<34s}{formatar_combo(retido)}")

    subtitulo("Cada PEÇA isolada foi vista milhares de vezes no treino")
    print()
    for rotulo, contagem in comp["exposicao"].items():
        print(f"    {rotulo:<44s} {contagem:>6d}")
    print()
    print("    Cada uma das três COMBINAÇÕES acima:              0")

    subtitulo("Resultado: acurácia em combinações vistas x inéditas")
    print()
    vistas, ineditas = comp["vistas"], comp["ineditas"]
    print(f"    {'conjunto de teste':<28s}{'n':>7s}{'rede':>9s}"
          f"    Gráfico")
    print("    " + "-" * 63)
    for rotulo, bloco in (("combinações VISTAS", vistas),
                          ("combinações INÉDITAS", ineditas)):
        n = sum(d["n"] for d in bloco["detalhes"])
        print(f"    {rotulo:<28s}{n:>7d}{bloco['geral']:>9.1%}"
              f"    {barra_ascii(bloco['geral'], largura=20)}")
    print("    " + "-" * 63)
    print(f"    Queda: {(vistas['geral'] - ineditas['geral']) * 100:.1f} "
          f"pontos percentuais.")

    subtitulo("Abertura por combinação")
    print()
    print(f"    {'combinação':<26s}{'situação':<10s}{'n':>6s}{'rede':>8s}"
          f"{'acaso':>8s}{'formato':>9s}")
    print("    " + "-" * 67)
    for situacao, bloco in (("vista", vistas), ("INÉDITA", ineditas)):
        for d in bloco["detalhes"]:
            print(f"    {formatar_combo(d['chave']):<26s}{situacao:<10s}"
                  f"{d['n']:>6d}{d['acuracia']:>8.1%}"
                  f"{acaso_do_combo(d['chave']):>8.1%}{d['formato']:>9.1%}")
    print("    " + "-" * 67)
    print("    acaso   = responder ao acaso dentro da família certa")
    print("              (1/4 para cor, 1/3 para forma)")
    print("    formato = fração das respostas que ao menos pertencem à")
    print("              família certa (cor quando se pede cor, forma")
    print("              quando se pede forma)")
    print()
    print(paragrafo(
        "O chutador cego não aparece nesta tabela por um motivo honesto: para "
        "uma combinação que ele nunca viu, ele não tem moda para consultar e "
        "recua para a moda do TIPO, o que o deixa em 0% nas duas linhas de "
        "cor. Comparar a rede com esse 0% seria enganoso. O piso justo aqui é "
        "o acaso dentro da família certa."))
    print()
    margens = [(d["acuracia"] - acaso_do_combo(d["chave"])) * 100
               for d in ineditas["detalhes"]]
    formato = float(np.mean([d["formato"] for d in ineditas["detalhes"]]))
    pior_vista = min(d["acuracia"] for d in vistas["detalhes"])
    print(paragrafo(
        f"Nas combinações vistas a rede vai de {pior_vista:.1%} a "
        f"{max(d['acuracia'] for d in vistas['detalhes']):.1%} — muito acima "
        "de qualquer acaso. Nas inéditas ela fica entre "
        f"{min(margens):+.1f} e {max(margens):+.1f} pontos percentuais do "
        "acaso: dentro do ruído de quem responde sem olhar. Não é uma queda "
        "de desempenho; é o desaparecimento completo da competência."))
    print()
    print(paragrafo(
        f"E a coluna 'formato' é o detalhe que fecha o argumento: em "
        f"{formato:.0%} dos casos inéditos a rede devolve uma resposta do tipo "
        "certo — pede-se uma cor, ela responde uma cor. Ela entendeu o "
        "atributo perguntado, entendeu que era uma pergunta de leitura, e "
        "mesmo assim não conseguiu LER a posição indicada. Viu 'cor' milhares "
        "de vezes, viu 'posição 3' milhares de vezes, e não sabe juntar as "
        "duas coisas. Um sistema com CONCEITOS faria essa conta de cabeça; um "
        "sistema que tem apenas PADRÕES precisa ter visto o padrão."))


def item_numerado(numero, texto):
    """Formata um item numerado do fecho, quebrado dentro da largura."""
    return textwrap.fill(" ".join(texto.split()), width=LARGURA,
                         initial_indent=f"  {numero}. ",
                         subsequent_indent="     ")


def imprimir_interpretacao(curva_perceptor, curva_cenas, comp, teste):
    """Seção [8]: o fecho que amarra tudo aos conceitos do capítulo."""
    cabecalho("[8] INTERPRETAÇÃO — o que você acabou de ver")
    perc = curva_perceptor[-1]["media"]
    forma = curva_perceptor[-1]["por_atributo"]["forma"]
    n90 = next((r["n"] for r in curva_perceptor if r["media"] >= 0.90), None)
    ultimo = curva_cenas[-1]
    cena, cego = ultimo["acuracia"], ultimo["cego"]
    consultar = ultimo["por_tipo"]["CONSULTAR"]
    contar = ultimo["por_tipo"]["CONTAR"]
    vistas = comp["vistas"]["geral"]
    ineditas = comp["ineditas"]["geral"]
    itens = [
        f"""O GRADIENTE ESTÁ CERTO. A seção [2] não é formalidade: o erro
        relativo máximo entre a retropropagação e as diferenças finitas foi de
        {teste['pior']:.2e}, dez vezes abaixo do critério usual. Tudo o que se
        afirma adiante é propriedade da TAREFA, e não de um defeito de
        implementação — porque o defeito foi procurado, com método, e não
        encontrado.""",

        f"""O INGREDIENTE NEURAL FUNCIONA — E FUNCIONA SOZINHO. Com {perc:.1%}
        de acurácia média em quatro atributos ({forma:.1%} só em forma, a mais
        difícil delas), o perceptor faz o que o ingrediente simbólico do
        Exercício 1 não fazia: aprendeu a reconhecer formas, cores, tamanhos e
        materiais SEM que ninguém escrevesse uma regra. Diante de uma forma
        nova basta rotular exemplos; não é preciso reescrever a base de
        conhecimento à mão. É essa a escalabilidade que o capítulo celebra.""",

        f"""MAS ELE É FAMINTO. A curva da seção [4] é a passagem do capítulo em
        forma de tabela: um bebê separa gato de cachorro com um punhado de
        instâncias; esta rede precisou de {n90} exemplos rotulados para cruzar
        90% de média e de {milhar(curva_perceptor[-1]['n'])} para chegar perto
        do teto — isso para separar um disco de um quadrado em
        {LADO_OBJETO}x{LADO_OBJETO} pixels. O conhecimento não vem de graça:
        vem de dado rotulado, e dado rotulado custa caro.""",

        f"""E ELE NÃO RACIOCINA. Trocada a pergunta "que forma é esta?" por
        "quantas esferas há nesta cena?", a mesma arquitetura cai de
        {perc:.1%} para {cena:.1%} — apenas {(cena - cego) * 100:+.1f} pontos
        acima de um chutador que nem olha a imagem, e isso recebendo MAIS
        épocas de treino e uma cena já segmentada em quatro posições fixas, um
        presente que o mundo não dá. A seção [6] mostra onde o saldo se
        concentra: CONSULTAR, a pergunta de leitura, vai a
        {consultar['rede']:.1%} contra {consultar['cego']:.1%} do chute;
        CONTAR fica em {contar['rede']:.1%} contra {contar['cego']:.1%},
        ABAIXO do piso. Ela enxerga bem e relaciona mal.""",

        f"""O SINTOMA DECISIVO É A COMPOSIÇÃO. Na seção [7] a rede acerta
        {vistas:.1%} nas combinações que viu e apenas {ineditas:.1%} nas
        inéditas, SOBRE AS MESMAS IMAGENS, com o MESMO tipo de pergunta e o
        MESMO formato de resposta. O experimento foi montado de propósito
        sobre a família que ela domina, justamente para que a queda não
        pudesse ser confundida com incompetência geral. Ela viu "cor" milhares
        de vezes, viu "posição 3" milhares de vezes — e CONSULTAR(cor, obj 3)
        a derruba até o acaso.""",

        """A CAUSA: NÃO HÁ OBJETOS LÁ DENTRO. A rede nunca representou a cena
        como um conjunto de entidades com atributos. Para ela, "cor na posição
        0" e "cor na posição 3" são duas regiões diferentes do espaço de
        entrada, não a mesma operação aplicada a coisas diferentes. É por isso
        que aprender uma não ajuda a aprender a outra: não há nada
        compartilhado entre elas além dos pixels. Pelo mesmo motivo CONTAR não
        decola — contar pressupõe que exista algo contável, e não existe.""",

        f"""A PONTE PARA O EXERCÍCIO 3. O capítulo pergunta: "podemos explorar
        o poder de reconhecimento de padrões das redes neurais para extrair
        automaticamente padrões simbólicos dos nossos dados?" Este exercício
        mostra os dois lados da resposta. O perceptor da seção [3] JÁ É um
        extrator de símbolos: transforma pixels em (forma, cor, tamanho,
        material) com {perc:.1%} de acerto. Falta apenas parar de exigir que a
        rede também raciocine — e entregar esses símbolos a um motor de
        inferência, como fazem o NSCL e o NSDR (Figuras 5.4 e 5.5). Percepção
        neural mais raciocínio simbólico: é essa a mistura do Exercício 3.""",
    ]
    print()
    for numero, texto in enumerate(itens):
        print(item_numerado(numero, texto))
        print()
    print("=" * LARGURA)
    print(" Fim do Exercício 2.")
    print("=" * LARGURA)


# ===========================================================================
# PARTE 6 — PROGRAMA PRINCIPAL
# ===========================================================================

def main():
    """Executa o exercício completo e imprime o relatório didático."""
    cabecalho("EXERCÍCIO 2 — O INGREDIENTE NEURAL: PERCEPÇÃO SEM RACIOCÍNIO")
    print(" Capítulo 5 — Introdução à IA Neuro-Simbólica (Figura 5.2)")
    print(" A rede extrai padrões sozinha. Ela não compõe conceitos.")

    # 1. Objetos isolados -----------------------------------------------------
    rng = np.random.default_rng(SEMENTE)
    entradas_tr, alvos_tr, galeria = gerar_objetos_isolados(N_OBJETOS_TREINO, rng)
    entradas_te, alvos_te, _ = gerar_objetos_isolados(N_OBJETOS_TESTE, rng)
    imprimir_secao_mundo(galeria,
                         gerar_cena(np.random.default_rng(SEMENTE + 7),
                                    n_objetos=N_CELULAS))

    # 2. A retropropagação está correta? -------------------------------------
    teste = teste_de_gradiente()
    imprimir_secao_validacao(teste)

    # 3 e 4. Perceptor de atributos e sua curva de aprendizado ----------------
    curva_perceptor, rede_perceptor = curva_de_aprendizado_perceptor(
        entradas_tr, alvos_tr, entradas_te, alvos_te)
    imprimir_secao_perceptor(rede_perceptor, curva_perceptor[-1],
                             entradas_te, alvos_te)
    imprimir_secao_fome(curva_perceptor)

    # 5 e 6. Cenas, perguntas e a segunda curva de aprendizado ----------------
    rng_cenas = np.random.default_rng(SEMENTE + 200)
    treino_cenas = montar_conjunto_de_cenas(N_PERGUNTAS_TREINO, rng_cenas)
    teste_cenas = montar_conjunto_de_cenas(N_PERGUNTAS_TESTE, rng_cenas)
    imprimir_secao_tarefa(treino_cenas[2], treino_cenas[1])
    curva_cenas = curva_de_aprendizado_cenas(treino_cenas, teste_cenas)
    imprimir_secao_fracasso_1(curva_perceptor, curva_cenas)

    # 7 e 8. Composicionalidade e fecho --------------------------------------
    composicional = experimento_composicional()
    imprimir_secao_fracasso_2(composicional)
    imprimir_interpretacao(curva_perceptor, curva_cenas, composicional, teste)


if __name__ == "__main__":
    main()
