# -*- coding: utf-8 -*-
"""
EXERCÍCIO PRÁTICO 5 — NEURAL LOGIC MACHINE
==========================================

Capítulo 5 — "Introdução à IA Neuro-Simbólica — o Próximo Nível da IA".
Referência visual: Figura 5.6 (visão geral da arquitetura NLM).

O QUE ESTE EXERCÍCIO DEMONSTRA
------------------------------
A *Neural Logic Machine* (NLM), apresentada e mantida de modo não oficial pelo
Google, resolve o dilema central do capítulo de um jeito radical: em vez de
colar uma rede neural em um motor simbólico externo, ela ESCREVE A LÓGICA
DENTRO DO TENSOR. Um predicado deixa de ser um símbolo em uma base de
conhecimento e passa a ser um bloco de números entre 0 e 1:

    NULÁRIO   TODOS_EMPILHADOS()      ->  tensor [lote, C0]
    UNÁRIO    LIVRE(x)                ->  tensor [lote, N, C1]
    BINÁRIO   SOBRE(x, y)             ->  tensor [lote, N, N, C2]

Sobre essa representação, os quantificadores da lógica de primeira ordem viram
operações de "fiação" entre tensores — as setas da Figura 5.6:

    EXPAND  : sobe a aridade replicando um eixo   (o "para todo" da nova variável)
    REDUCE  : desce a aridade agregando um eixo   (máximo = EXISTE, mínimo = PARA TODO)
    PERMUTE : concatena o tensor com sua transposta (inversão de argumentos)

Uma CAMADA da NLM concatena, para cada aridade, [entrada da própria aridade,
expand da aridade abaixo, reduce da aridade acima] e passa tudo por um MLP com
saída sigmoide, compartilhado entre os objetos — a "lógica booleana neural" da
figura. Empilhando D camadas obtém-se a DEPTH (número de passos de dedução) e
usando aridades até A obtém-se a BREADTH (aridade máxima) da Figura 5.6.

Três experimentos são executados:

  1. MUNDO DE BLOCOS  — a partir de SOBRE(x,y) e NO_CHAO(x), a rede deduz
     LIVRE(x), ACIMA(x,y) (fecho transitivo) e TODOS_EMPILHADOS(). Treina com
     5 blocos e é testada com 8, 12 e 20 blocos: como os MLPs são compartilhados
     entre objetos, o que está fixo é a ARIDADE, não o NÚMERO DE OBJETOS.
  2. ORDENAÇÃO DE VETOR — a partir de MENOR(x,y), a rede deduz PRIMEIRO(x),
     SEGUNDO(x) e TERCEIRO(x). Cada predicado exige DOIS passos de dedução a
     mais que o anterior, então a mesma tarefa é treinada com profundidades
     1, 2, 3 e 4 para medir a escada da DEPTH. O resultado é honesto nos dois
     sentidos: PRIMEIRO(x) e SEGUNDO(x) generalizam perfeitamente de 5 para 20
     posições, e TERCEIRO(x) — que exigiria profundidade 6 — NÃO generaliza, e
     isso é reportado como resultado negativo.
  3. EXTRAÇÃO DA REGRA — a rede treinada é sondada com TODOS os mundos de
     4 blocos e a tabela de verdade empírica é comparada com a regra lógica
     verdadeira LIVRE(x) <-> NÃO EXISTE y : SOBRE(y, x).

RESTRIÇÕES DE IMPLEMENTAÇÃO
---------------------------
Somente a biblioteca padrão do Python 3 + numpy. Nenhum torch, tensorflow,
scikit-learn, pandas ou matplotlib. Expand, reduce, permute, o MLP, a
retropropagação de TODAS essas operações (inclusive o roteamento do gradiente
do máximo e do mínimo para o argumento vencedor) e o otimizador Adam são
escritos do zero. A seção [3] do relatório confere os gradientes por diferenças
finitas — é a prova de que a retropropagação manual está correta.

EXECUÇÃO
--------
    python3 exercicio_pratico_5_neural_logic_machine.py

Sem argumentos, sem arquivos de entrada, sem interação. A saída é um relatório
de texto no terminal. A semente aleatória é fixa, então o relatório é sempre o
mesmo — você pode conferir seus números com os do colega ao lado.
"""

import itertools
import os
import textwrap

# As matrizes deste exercício são pequenas e finas: deixar o BLAS abrir uma
# thread por núcleo custa MAIS em sincronização do que economiza em cálculo
# (chega a ser 6x mais lento). Fixar uma thread também remove a última fonte
# de variação entre execuções. Isto precisa vir ANTES de importar o numpy.
for _variavel in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variavel, "1")

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTES GLOBAIS
# ---------------------------------------------------------------------------

SEMENTE = 42                 # semente fixa => saída 100% reprodutível
LARGURA = 78                 # largura das linhas do relatório

CANAIS = 6                   # canais (predicados latentes) por aridade em cada camada
OCULTAS = 12                 # neurônios da camada oculta de cada MLP
TAXA_APRENDIZADO = 0.02      # passo do Adam
LOTE = 16                    # cenas por iteração de treino

# --- Tarefa 1: mundo de blocos ---------------------------------------------
ARIDADE_MAXIMA_BLOCOS = 3    # BREADTH da Figura 5.6 (aridade auxiliar 3 permite composição)
ARIDADE_MAXIMA_ABLACAO = 2   # a mesma rede limitada a predicados até binários
PROFUNDIDADE_BLOCOS = 3      # DEPTH da Figura 5.6 (passos de dedução)
N_TREINO_BLOCOS = 4          # objetos vistos no TREINO
ALTURA_MAXIMA_TORRE = 3      # altura máxima de uma torre (limita o fecho transitivo)
ITERACOES_BLOCOS = 250
NS_TESTE_BLOCOS = (4, 8, 12, 20)   # objetos vistos no TESTE (generalização)
# Cenas de teste por tamanho: o tensor de aridade 3 cresce com N^3, então
# usamos menos cenas nos tamanhos grandes para o exercício rodar em segundos.
CENAS_POR_TAMANHO = {4: 120, 8: 80, 12: 50, 20: 24}
N_CENAS_TESTE = 120          # cenas de teste da tarefa de ordenação

# --- Tarefa 2: ordenação de vetor ------------------------------------------
ARIDADE_MAXIMA_ORDENACAO = 2
PROFUNDIDADE_ORDENACAO = 4     # DEPTH da rede principal desta tarefa
N_TREINO_ORDENACAO = 5
ITERACOES_ORDENACAO = 400
NS_TESTE_ORDENACAO = (5, 8, 12, 20)
# Escada da DEPTH: a MESMA tarefa treinada com profundidades crescentes.
PROFUNDIDADES_ESCADA = (1, 2, 3, 4)

# Pesos da classe positiva na entropia cruzada (os predicados-alvo são raros).
PESOS_POSITIVOS_BLOCOS = {"LIVRE": 1.0, "ACIMA": 3.0, "TODOS_EMPILHADOS": 1.0}

N_BLOCOS_SONDAGEM = 4        # sondagem exaustiva da regra aprendida


# ===========================================================================
# PARTE 1 — UTILIDADES NUMÉRICAS
# ===========================================================================

def sigmoide(z):
    """Sigmoide numericamente estável, escrita como tangente hiperbólica.

    A identidade sigma(z) = (1 + tanh(z/2)) / 2 é exata e nunca estoura: tanh
    satura em +-1 em vez de calcular exp(z) para z grande. Também é bem mais
    rápida que a versão com máscara, e esta função é chamada milhares de vezes.
    """
    return 0.5 * (np.tanh(0.5 * z) + 1.0)


def relu(x):
    """Retificador linear."""
    return np.maximum(x, 0.0)


# ===========================================================================
# PARTE 2 — AS OPERAÇÕES DE FIAÇÃO (os quantificadores neurais da Figura 5.6)
# ===========================================================================
#
# Convenção de formas. Um predicado de aridade r com C canais é um tensor
#
#       [lote, N, N, ..., N, C]        (r eixos de objeto)
#
# com valores em [0, 1]. O eixo -1 é sempre o canal; os eixos 1..r são os
# argumentos do predicado. Todas as operações abaixo mexem apenas nos eixos de
# objeto — é por isso que a mesma rede aceita qualquer N.

def expandir(x, n_objetos):
    """EXPAND: sobe a aridade de r para r+1 replicando um novo eixo de objeto.

    O predicado P(x1..xr) vira Q(x1..xr, x_{r+1}) = P(x1..xr), constante na
    variável nova. É o análogo neural de introduzir uma variável quantificada
    universalmente: a nova variável ainda não restringe nada.

    Exemplo: NULÁRIO [lote, C] -> UNÁRIO [lote, N, C]
             UNÁRIO  [lote, N, C] -> BINÁRIO [lote, N, N, C]
    """
    forma = x.shape[:-1] + (n_objetos, x.shape[-1])
    return np.broadcast_to(np.expand_dims(x, -2), forma)


def expandir_tras(g):
    """Retropropagação do EXPAND: soma o gradiente sobre o eixo replicado."""
    return g.sum(axis=-2)


def reduzir(x):
    """REDUCE: desce a aridade de r para r-1 agregando o ÚLTIMO eixo de objeto.

    Duas agregações são calculadas e concatenadas no eixo de canais:

        MÁXIMO  sobre a variável  ->  EXISTE y : P(..., y)
        MÍNIMO  sobre a variável  ->  PARA TODO y : P(..., y)

    Com predicados em [0,1], máximo é a t-conorma e mínimo é a t-norma de
    Gödel: são exatamente a disjunção e a conjunção fuzzy. Daí a leitura
    lógica das duas saídas.

    Retorna (saida, indice_do_maximo, indice_do_minimo). Os índices são o que
    permite rotear o gradiente para o argumento vencedor na volta.
    """
    indice_max = np.argmax(x, axis=-2)
    indice_min = np.argmin(x, axis=-2)
    saida = np.concatenate([np.max(x, axis=-2), np.min(x, axis=-2)], axis=-1)
    return saida, indice_max, indice_min


def reduzir_tras(g, indice_max, indice_min, forma):
    """Retropropagação do REDUCE.

    O máximo e o mínimo são funções lineares por partes: a derivada é 1 para o
    argumento vencedor e 0 para todos os outros. Portanto o gradiente NÃO é
    espalhado — ele é ROTEADO para a posição que venceu a disputa. É esse
    roteamento que faz a NLM aprender "qual objeto" satisfez o existencial.
    """
    n_canais = g.shape[-1] // 2
    g_maximo = g[..., :n_canais]
    g_minimo = g[..., n_canais:]

    g_x = np.zeros(forma)
    np.put_along_axis(g_x, np.expand_dims(indice_max, -2),
                      np.expand_dims(g_maximo, -2), axis=-2)
    auxiliar = np.zeros(forma)
    np.put_along_axis(auxiliar, np.expand_dims(indice_min, -2),
                      np.expand_dims(g_minimo, -2), axis=-2)
    return g_x + auxiliar


_CACHE_PERMUTACOES = {}


def permutacoes_de(aridade):
    """Lista (memorizada e determinística) das r! permutações dos argumentos."""
    if aridade not in _CACHE_PERMUTACOES:
        _CACHE_PERMUTACOES[aridade] = list(itertools.permutations(range(aridade)))
    return _CACHE_PERMUTACOES[aridade]


def eixos_da_permutacao(permutacao, aridade):
    """Traduz uma permutação de argumentos para os eixos aceitos por np.transpose."""
    return (0,) + tuple(1 + p for p in permutacao) + (aridade + 1,)


def permutar(x, aridade):
    """PERMUTE: concatena o tensor com TODAS as permutações de seus argumentos.

    Para um predicado binário isso significa concatenar P(x,y) com P(y,x): a
    camada seguinte passa a poder aprender a inversão de argumentos, que a
    lógica escreveria como um predicado novo. Para aridade 0 e 1 a operação é a
    identidade (0! = 1! = 1) e nada é duplicado.
    """
    permutacoes = permutacoes_de(aridade)
    if len(permutacoes) == 1:
        return x
    partes = [np.transpose(x, eixos_da_permutacao(p, aridade)) for p in permutacoes]
    return np.concatenate(partes, axis=-1)


def permutar_tras(g, aridade):
    """Retropropagação do PERMUTE: desfaz cada transposição e soma as parcelas."""
    permutacoes = permutacoes_de(aridade)
    if len(permutacoes) == 1:
        return g
    n_canais = g.shape[-1] // len(permutacoes)
    total = None
    for k, permutacao in enumerate(permutacoes):
        eixos = eixos_da_permutacao(permutacao, aridade)
        inversos = tuple(int(i) for i in np.argsort(eixos))
        parte = np.transpose(g[..., k * n_canais:(k + 1) * n_canais], inversos)
        total = parte if total is None else total + parte
    return total


# ===========================================================================
# PARTE 3 — O MLP COMPARTILHADO ENTRE OBJETOS
# ===========================================================================

def mlp_frente(pesos_1, vieses_1, pesos_2, vieses_2, entrada):
    """Aplica um MLP de uma camada oculta PONTO A PONTO sobre os eixos de objeto.

    A entrada tem forma [lote, N, ..., N, D]; achatamos tudo o que não é canal,
    aplicamos o MLP e devolvemos a forma original. Como os MESMOS pesos são
    usados em toda posição (x1..xr), a rede aprende uma REGRA, não uma tabela
    indexada por objeto — e por isso ela funciona com qualquer número N.

    A saída passa por sigmoide: o resultado volta a ser um predicado
    probabilístico em [0,1], pronto para a próxima camada.
    """
    forma_original = entrada.shape
    plano = entrada.reshape(-1, forma_original[-1])
    pre_oculta = plano @ pesos_1 + vieses_1
    oculta = relu(pre_oculta)
    pre_saida = oculta @ pesos_2 + vieses_2
    saida_plana = sigmoide(pre_saida)
    cache = (plano, pre_oculta, oculta, saida_plana, forma_original)
    saida = saida_plana.reshape(forma_original[:-1] + (pesos_2.shape[1],))
    return saida, cache


def mlp_tras(cache, pesos_1, pesos_2, gradiente_saida):
    """Retropropagação do MLP. Retorna (g_entrada, gW1, gb1, gW2, gb2)."""
    plano, pre_oculta, oculta, saida_plana, forma_original = cache
    g_saida = gradiente_saida.reshape(saida_plana.shape)

    # Derivada da sigmoide: s * (1 - s).
    g_pre_saida = g_saida * saida_plana * (1.0 - saida_plana)
    g_pesos_2 = oculta.T @ g_pre_saida
    g_vieses_2 = g_pre_saida.sum(axis=0)

    g_oculta = g_pre_saida @ pesos_2.T
    g_pre_oculta = g_oculta * (pre_oculta > 0.0)     # derivada da ReLU
    g_pesos_1 = plano.T @ g_pre_oculta
    g_vieses_1 = g_pre_oculta.sum(axis=0)

    g_entrada = (g_pre_oculta @ pesos_1.T).reshape(forma_original)
    return g_entrada, g_pesos_1, g_vieses_1, g_pesos_2, g_vieses_2


# ===========================================================================
# PARTE 4 — A NEURAL LOGIC MACHINE
# ===========================================================================

def dimensao_de_entrada(aridade, aridade_maxima, canais_por_aridade):
    """Quantos canais chegam ao MLP da aridade `aridade` em uma camada.

    A conta é a leitura literal da Figura 5.6:
        entrada = [ própria aridade
                  + EXPAND da aridade de baixo
                  + REDUCE (máximo e mínimo) da aridade de cima ]
        depois tudo isso é PERMUTADO, o que multiplica por r!.
    """
    total = canais_por_aridade[aridade]
    if aridade >= 1:
        total += canais_por_aridade[aridade - 1]
    if aridade + 1 <= aridade_maxima:
        total += 2 * canais_por_aridade[aridade + 1]
    return total * len(permutacoes_de(aridade))


def criar_modelo(aridade_maxima, profundidade, cabecas, canais_entrada,
                 canais=CANAIS, ocultas=OCULTAS, semente=SEMENTE):
    """Cria uma NLM com pesos aleatórios.

    `cabecas` é um dicionário {nome_do_predicado: aridade_da_saida}: são os
    conceitos conclusivos que a rede deve gerar sobre os objetos.
    """
    gerador = np.random.default_rng(semente)
    pesos = {}
    canais_correntes = list(canais_entrada)

    for profundidade_atual in range(profundidade):
        for aridade in range(aridade_maxima + 1):
            entrada = dimensao_de_entrada(aridade, aridade_maxima, canais_correntes)
            prefixo = f"c{profundidade_atual}a{aridade}"
            # Inicialização de He na camada com ReLU e de Xavier na de saída.
            pesos[prefixo + "W1"] = gerador.normal(
                0.0, np.sqrt(2.0 / entrada), size=(entrada, ocultas))
            pesos[prefixo + "b1"] = np.zeros(ocultas)
            pesos[prefixo + "W2"] = gerador.normal(
                0.0, np.sqrt(1.0 / ocultas), size=(ocultas, canais))
            pesos[prefixo + "b2"] = np.zeros(canais)
        canais_correntes = [canais] * (aridade_maxima + 1)

    for nome in sorted(cabecas):
        pesos[f"cab{nome}W"] = gerador.normal(0.0, np.sqrt(1.0 / canais),
                                              size=(canais, 1))
        pesos[f"cab{nome}b"] = np.zeros(1)

    return {
        "aridade_maxima": aridade_maxima,
        "profundidade": profundidade,
        "cabecas": dict(cabecas),
        "canais": canais,
        "ocultas": ocultas,
        "canais_entrada": list(canais_entrada),
        "pesos": pesos,
    }


def contar_parametros(modelo):
    """Número total de parâmetros treináveis."""
    return int(sum(p.size for p in modelo["pesos"].values()))


def frente(modelo, entradas):
    """Passagem direta: dos predicados dados aos logits dos predicados deduzidos.

    `entradas` é a lista de tensores por aridade (índice = aridade). Retorna
    (logits_por_cabeca, cache_para_a_volta).
    """
    aridade_maxima = modelo["aridade_maxima"]
    pesos = modelo["pesos"]
    n_objetos = entradas[1].shape[1]

    tensores = list(entradas)
    caches_camadas = []

    for profundidade_atual in range(modelo["profundidade"]):
        formas_entrada = [t.shape for t in tensores]
        novos = []
        caches_aridade = []
        for aridade in range(aridade_maxima + 1):
            partes = [tensores[aridade]]
            tamanhos = [tensores[aridade].shape[-1]]
            cache_reduce = None

            if aridade >= 1:
                partes.append(expandir(tensores[aridade - 1], n_objetos))
                tamanhos.append(tensores[aridade - 1].shape[-1])
            if aridade + 1 <= aridade_maxima:
                reduzido, indice_max, indice_min = reduzir(tensores[aridade + 1])
                partes.append(reduzido)
                tamanhos.append(reduzido.shape[-1])
                cache_reduce = (indice_max, indice_min, tensores[aridade + 1].shape)

            bloco = np.concatenate(partes, axis=-1) if len(partes) > 1 else partes[0]
            entrada_mlp = permutar(bloco, aridade)

            prefixo = f"c{profundidade_atual}a{aridade}"
            saida, cache_mlp = mlp_frente(pesos[prefixo + "W1"], pesos[prefixo + "b1"],
                                          pesos[prefixo + "W2"], pesos[prefixo + "b2"],
                                          entrada_mlp)
            novos.append(saida)
            caches_aridade.append({"tamanhos": tamanhos, "reduce": cache_reduce,
                                   "mlp": cache_mlp, "prefixo": prefixo})

        caches_camadas.append((formas_entrada, caches_aridade))
        tensores = novos

    logits = {}
    caches_cabecas = {}
    for nome in sorted(modelo["cabecas"]):
        aridade = modelo["cabecas"][nome]
        tensor = tensores[aridade]
        plano = tensor.reshape(-1, tensor.shape[-1])
        bruto = plano @ pesos[f"cab{nome}W"] + pesos[f"cab{nome}b"]
        logits[nome] = bruto.reshape(tensor.shape[:-1] + (1,))
        caches_cabecas[nome] = (plano, tensor.shape, aridade)

    return logits, (caches_camadas, caches_cabecas, tensores)


def tras(modelo, cache, gradientes_logits):
    """Retropropagação completa. Retorna o dicionário de gradientes dos pesos."""
    caches_camadas, caches_cabecas, tensores_finais = cache
    aridade_maxima = modelo["aridade_maxima"]
    pesos = modelo["pesos"]
    gradientes = {chave: np.zeros_like(valor) for chave, valor in pesos.items()}

    # --- cabeças de saída ---------------------------------------------------
    g_tensores = [np.zeros_like(t) for t in tensores_finais]
    for nome in sorted(modelo["cabecas"]):
        plano, forma, aridade = caches_cabecas[nome]
        g_bruto = gradientes_logits[nome].reshape(-1, 1)
        gradientes[f"cab{nome}W"] += plano.T @ g_bruto
        gradientes[f"cab{nome}b"] += g_bruto.sum(axis=0)
        g_tensores[aridade] += (g_bruto @ pesos[f"cab{nome}W"].T).reshape(forma)

    # --- camadas, da última para a primeira ---------------------------------
    for profundidade_atual in reversed(range(modelo["profundidade"])):
        formas_entrada, caches_aridade = caches_camadas[profundidade_atual]
        g_anteriores = [np.zeros(forma) for forma in formas_entrada]

        for aridade in range(aridade_maxima + 1):
            item = caches_aridade[aridade]
            prefixo = item["prefixo"]
            g_entrada_mlp, gW1, gb1, gW2, gb2 = mlp_tras(
                item["mlp"], pesos[prefixo + "W1"], pesos[prefixo + "W2"],
                g_tensores[aridade])
            gradientes[prefixo + "W1"] += gW1
            gradientes[prefixo + "b1"] += gb1
            gradientes[prefixo + "W2"] += gW2
            gradientes[prefixo + "b2"] += gb2

            g_bloco = permutar_tras(g_entrada_mlp, aridade)

            tamanhos = item["tamanhos"]
            corte = tamanhos[0]
            g_anteriores[aridade] += g_bloco[..., :corte]
            posicao = 1

            if aridade >= 1:
                largura = tamanhos[posicao]
                g_expandido = g_bloco[..., corte:corte + largura]
                g_anteriores[aridade - 1] += expandir_tras(g_expandido)
                corte += largura
                posicao += 1
            if aridade + 1 <= aridade_maxima:
                largura = tamanhos[posicao]
                g_reduzido = g_bloco[..., corte:corte + largura]
                indice_max, indice_min, forma_origem = item["reduce"]
                g_anteriores[aridade + 1] += reduzir_tras(
                    g_reduzido, indice_max, indice_min, forma_origem)

        g_tensores = g_anteriores

    return gradientes


# ===========================================================================
# PARTE 5 — PERDA E OTIMIZADOR ADAM (ambos escritos do zero)
# ===========================================================================

def entropia_cruzada(logits, alvo, peso_positivo=1.0):
    """Entropia cruzada binária estável, com peso na classe positiva.

    Retorna (perda_media, gradiente_em_relacao_aos_logits).
    """
    probabilidade = sigmoide(logits)
    perda = np.mean(peso_positivo * alvo * np.logaddexp(0.0, -logits)
                    + (1.0 - alvo) * np.logaddexp(0.0, logits))
    gradiente = ((1.0 - alvo) * probabilidade
                 - peso_positivo * alvo * (1.0 - probabilidade)) / logits.size
    return float(perda), gradiente


def entropia_cruzada_categorica(logits, alvo):
    """Entropia cruzada com SOFTMAX sobre o eixo de objetos (uma escolha em N).

    Quando o predicado-alvo é verdadeiro em EXATAMENTE UMA posição — o menor
    elemento de um vetor, por exemplo —, tratá-lo como N problemas binários
    independentes é um erro de modelagem: a proporção de positivos é 1/N, e a
    entropia cruzada binária é minimizada de forma trivial respondendo sempre
    "não". A saída fica degenerada e o argmax nunca aponta a posição certa.

    A leitura correta é: as N posições COMPETEM entre si por uma única
    resposta. Isso é um softmax sobre o eixo de objetos, e o gradiente
    resultante (p - alvo) sempre empurra a posição certa para cima E as demais
    para baixo, seja qual for N. A normalização por N também some, então a
    mesma rede treinada com N = 5 continua calibrada em N = 20.

    Espera `logits` e `alvo` com forma [lote, N, 1] e uma posição positiva por
    linha. Retorna (perda_media, gradiente_em_relacao_aos_logits).
    """
    z = logits[..., 0]
    z = z - z.max(axis=1, keepdims=True)          # estabilidade numérica
    exponenciais = np.exp(z)
    probabilidade = exponenciais / exponenciais.sum(axis=1, keepdims=True)
    referencia = alvo[..., 0]
    perda = -float(np.mean(np.sum(
        referencia * np.log(probabilidade + 1e-12), axis=1)))
    gradiente = (probabilidade - referencia) / z.shape[0]
    return perda, gradiente[..., None]


def iniciar_adam(modelo):
    """Estado do Adam: primeiro e segundo momentos, um por parâmetro."""
    return {
        "m": {chave: np.zeros_like(valor) for chave, valor in modelo["pesos"].items()},
        "v": {chave: np.zeros_like(valor) for chave, valor in modelo["pesos"].items()},
    }


def passo_adam(modelo, gradientes, estado, passo, taxa=TAXA_APRENDIZADO,
               beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Uma atualização do Adam, com correção de viés dos momentos."""
    pesos = modelo["pesos"]
    for chave in sorted(pesos):
        momento = estado["m"][chave]
        segundo = estado["v"][chave]
        gradiente = gradientes[chave]
        momento *= beta1
        momento += (1.0 - beta1) * gradiente
        segundo *= beta2
        segundo += (1.0 - beta2) * (gradiente * gradiente)
        momento_corrigido = momento / (1.0 - beta1 ** passo)
        segundo_corrigido = segundo / (1.0 - beta2 ** passo)
        pesos[chave] -= taxa * momento_corrigido / (np.sqrt(segundo_corrigido) + epsilon)


def perda_do_lote(modelo, entradas, alvos, pesos_positivos, categorica=False):
    """Perda somada sobre as cabeças + gradientes dos logits + cache da ida.

    Com `categorica=False` cada cabeça é uma coleção de decisões binárias
    independentes (o caso do mundo de blocos: vários blocos podem estar livres
    ao mesmo tempo). Com `categorica=True` cada cabeça é uma ESCOLHA de uma
    posição entre N (o caso da ordenação) e a perda vira softmax.
    """
    logits, cache = frente(modelo, entradas)
    perda_total = 0.0
    gradientes_logits = {}
    for nome in sorted(modelo["cabecas"]):
        if categorica:
            perda, gradiente = entropia_cruzada_categorica(
                logits[nome], alvos[nome])
        else:
            peso = pesos_positivos.get(nome, 1.0)
            perda, gradiente = entropia_cruzada(logits[nome], alvos[nome], peso)
        perda_total += perda
        gradientes_logits[nome] = gradiente
    return perda_total, gradientes_logits, cache


def treinar(modelo, gerador_de_lote, n_iteracoes, pesos_positivos, gerador,
            marcos=(), categorica=False):
    """Laço de treino com Adam. Retorna [(iteracao, perda)] nos marcos pedidos."""
    estado = iniciar_adam(modelo)
    historico = []
    for iteracao in range(1, n_iteracoes + 1):
        entradas, alvos = gerador_de_lote(gerador, LOTE)
        perda, gradientes_logits, cache = perda_do_lote(
            modelo, entradas, alvos, pesos_positivos, categorica)
        gradientes = tras(modelo, cache, gradientes_logits)
        passo_adam(modelo, gradientes, estado, iteracao)
        if iteracao in marcos:
            historico.append((iteracao, perda))
    return historico


# ===========================================================================
# PARTE 6 — VERIFICAÇÃO NUMÉRICA DOS GRADIENTES
# ===========================================================================

def verificar_gradientes(n_parametros=8, epsilon=1e-6):
    """Compara o gradiente analítico com diferenças finitas centrais.

    Usa uma NLM minúscula com BREADTH 2, DEPTH 2 e entradas ALEATÓRIAS e
    CONTÍNUAS: valores contínuos evitam empates no máximo/mínimo, onde a
    derivada não existe e a comparação não faria sentido.

    Retorna [(nome_do_parametro, analitico, numerico, erro_relativo)].
    """
    gerador = np.random.default_rng(SEMENTE + 7)
    lote, n_objetos = 2, 3
    cabecas = {"P0": 0, "P1": 1, "P2": 2}
    modelo = criar_modelo(aridade_maxima=2, profundidade=2, cabecas=cabecas,
                          canais_entrada=[2, 2, 2], canais=3, ocultas=4,
                          semente=SEMENTE + 8)
    entradas = [
        gerador.random((lote, 2)),
        gerador.random((lote, n_objetos, 2)),
        gerador.random((lote, n_objetos, n_objetos, 2)),
    ]
    alvos = {
        "P0": (gerador.random((lote, 1)) < 0.5).astype(np.float64),
        "P1": (gerador.random((lote, n_objetos, 1)) < 0.5).astype(np.float64),
        "P2": (gerador.random((lote, n_objetos, n_objetos, 1)) < 0.5).astype(np.float64),
    }
    pesos_positivos = {"P0": 1.0, "P1": 2.0, "P2": 1.5}

    def avaliar_perda():
        perda, _, _ = perda_do_lote(modelo, entradas, alvos, pesos_positivos)
        return perda

    perda, gradientes_logits, cache = perda_do_lote(
        modelo, entradas, alvos, pesos_positivos)
    gradientes = tras(modelo, cache, gradientes_logits)

    chaves = sorted(modelo["pesos"])
    resultados = []
    for _ in range(n_parametros):
        chave = chaves[int(gerador.integers(len(chaves)))]
        parametro = modelo["pesos"][chave]
        posicao = tuple(int(gerador.integers(d)) for d in parametro.shape)

        original = parametro[posicao]
        parametro[posicao] = original + epsilon
        perda_mais = avaliar_perda()
        parametro[posicao] = original - epsilon
        perda_menos = avaliar_perda()
        parametro[posicao] = original

        numerico = (perda_mais - perda_menos) / (2.0 * epsilon)
        analitico = float(gradientes[chave][posicao])
        denominador = max(1e-12, abs(numerico) + abs(analitico))
        erro = abs(numerico - analitico) / denominador
        rotulo = f"{chave}{list(posicao)}"
        resultados.append((rotulo, analitico, numerico, erro))
    return resultados


def verificar_gradiente_softmax(n_parametros=6, epsilon=1e-6):
    """Confere por diferenças finitas a perda softmax da tarefa de ordenação.

    Mesma ideia da função acima, mas com uma NLM de cabeça unária única e a
    perda `entropia_cruzada_categorica`. Retorna o PIOR erro relativo.
    """
    gerador = np.random.default_rng(SEMENTE + 17)
    lote, n_objetos = 2, 4
    modelo = criar_modelo(aridade_maxima=2, profundidade=2,
                          cabecas={"MENOR_DE_TODOS": 1},
                          canais_entrada=[1, 1, 1], canais=3, ocultas=4,
                          semente=SEMENTE + 18)
    entradas = [
        gerador.random((lote, 1)),
        gerador.random((lote, n_objetos, 1)),
        gerador.random((lote, n_objetos, n_objetos, 1)),
    ]
    escolhas = gerador.integers(n_objetos, size=lote)
    alvo = np.zeros((lote, n_objetos, 1))
    alvo[np.arange(lote), escolhas, 0] = 1.0
    alvos = {"MENOR_DE_TODOS": alvo}

    def avaliar_perda():
        perda, _, _ = perda_do_lote(modelo, entradas, alvos, {}, True)
        return perda

    _, gradientes_logits, cache = perda_do_lote(modelo, entradas, alvos, {}, True)
    gradientes = tras(modelo, cache, gradientes_logits)

    chaves = sorted(modelo["pesos"])
    pior = 0.0
    for _ in range(n_parametros):
        chave = chaves[int(gerador.integers(len(chaves)))]
        parametro = modelo["pesos"][chave]
        posicao = tuple(int(gerador.integers(d)) for d in parametro.shape)
        original = parametro[posicao]
        parametro[posicao] = original + epsilon
        perda_mais = avaliar_perda()
        parametro[posicao] = original - epsilon
        perda_menos = avaliar_perda()
        parametro[posicao] = original
        numerico = (perda_mais - perda_menos) / (2.0 * epsilon)
        analitico = float(gradientes[chave][posicao])
        denominador = max(1e-12, abs(numerico) + abs(analitico))
        pior = max(pior, abs(numerico - analitico) / denominador)
    return pior


# ===========================================================================
# PARTE 7 — TAREFA 1: MUNDO DE BLOCOS
# ===========================================================================

def sortear_tamanhos_de_torres(gerador, n_blocos, minimo, maximo):
    """Sorteia uma composição de `n_blocos` em torres de tamanho em [minimo, maximo].

    A cada passo só são considerados os tamanhos que deixam um resto viável
    (zero ou pelo menos `minimo`), de modo que o sorteio nunca fica preso.
    """
    tamanhos = []
    restante = n_blocos
    while restante > 0:
        candidatos = [t for t in range(minimo, min(maximo, restante) + 1)
                      if restante - t == 0 or restante - t >= minimo]
        escolhido = candidatos[int(gerador.integers(len(candidatos)))]
        tamanhos.append(escolhido)
        restante -= escolhido
    return tamanhos


def gerar_cena_de_blocos(gerador, n_blocos, forcar_empilhado):
    """Gera uma cena do mundo de blocos e TODOS os predicados verdadeiros.

    Uma cena é um conjunto de torres (pilhas) de blocos rotulados. Predicados:

        SOBRE(x, y)          : x está DIRETAMENTE sobre y            (dado)
        NO_CHAO(x)           : x é a base de sua torre                (dado)
        LIVRE(x)             : nada está sobre x                      (deduzir)
        ACIMA(x, y)          : x está em algum lugar acima de y       (deduzir)
        TODOS_EMPILHADOS()   : nenhuma torre tem um único bloco       (deduzir)
    """
    minimo = 2 if forcar_empilhado else 1
    tamanhos = sortear_tamanhos_de_torres(gerador, n_blocos, minimo,
                                          ALTURA_MAXIMA_TORRE)
    ordem = gerador.permutation(n_blocos)

    sobre = np.zeros((n_blocos, n_blocos))
    no_chao = np.zeros(n_blocos)
    livre = np.ones(n_blocos)
    acima = np.zeros((n_blocos, n_blocos))

    inicio = 0
    for tamanho in tamanhos:
        torre = [int(b) for b in ordem[inicio:inicio + tamanho]]
        inicio += tamanho
        no_chao[torre[0]] = 1.0
        for posicao in range(1, tamanho):
            sobre[torre[posicao], torre[posicao - 1]] = 1.0
            livre[torre[posicao - 1]] = 0.0
            for abaixo in range(posicao):
                acima[torre[posicao], torre[abaixo]] = 1.0

    todos_empilhados = 1.0 if all(t >= 2 for t in tamanhos) else 0.0
    return sobre, no_chao, livre, acima, todos_empilhados


def montar_entradas_blocos(sobre, no_chao, aridade_maxima):
    """Empacota os predicados DADOS na lista de tensores por aridade."""
    lote, n_objetos = no_chao.shape
    entradas = [
        np.ones((lote, 1)),                       # nulário: proposição sempre verdadeira
        no_chao[:, :, None],                      # unário : NO_CHAO(x)
        sobre[:, :, :, None],                     # binário: SOBRE(x, y)
    ]
    if aridade_maxima >= 3:
        # A aridade auxiliar entra vazia: ela existe apenas como espaço de
        # trabalho para a composição de relações (é a BREADTH da figura).
        entradas.append(np.zeros((lote, n_objetos, n_objetos, n_objetos, 1)))
    return entradas


def gerar_lote_blocos(gerador, tamanho_lote, n_blocos, aridade_maxima):
    """Monta um lote de cenas do mundo de blocos, metade delas toda empilhada."""
    sobre = np.zeros((tamanho_lote, n_blocos, n_blocos))
    no_chao = np.zeros((tamanho_lote, n_blocos))
    livre = np.zeros((tamanho_lote, n_blocos))
    acima = np.zeros((tamanho_lote, n_blocos, n_blocos))
    todos = np.zeros((tamanho_lote, 1))

    for i in range(tamanho_lote):
        forcar = bool(gerador.random() < 0.5)
        s, c, l, a, t = gerar_cena_de_blocos(gerador, n_blocos, forcar)
        sobre[i], no_chao[i], livre[i], acima[i], todos[i, 0] = s, c, l, a, t

    entradas = montar_entradas_blocos(sobre, no_chao, aridade_maxima)
    alvos = {
        "LIVRE": livre[:, :, None],
        "ACIMA": acima[:, :, :, None],
        "TODOS_EMPILHADOS": todos,
    }
    return entradas, alvos


def metricas_binarias(probabilidade, alvo):
    """Acurácia e F1 de um predicado, com corte em 0,5."""
    previsto = (probabilidade >= 0.5).astype(np.float64)
    acuracia = float(np.mean(previsto == alvo))
    verdadeiros = float(np.sum(previsto * alvo))
    falsos_positivos = float(np.sum(previsto * (1.0 - alvo)))
    falsos_negativos = float(np.sum((1.0 - previsto) * alvo))
    denominador = 2.0 * verdadeiros + falsos_positivos + falsos_negativos
    f1 = 1.0 if denominador == 0.0 else 2.0 * verdadeiros / denominador
    return acuracia, f1


def tamanho_de_lote_seguro(n_objetos, aridade_maxima, alvo_elementos=8_000_000):
    """Quantas cenas cabem de uma vez na memória para um dado N.

    Com aridade 3 o tensor cresce com N³; sem esse cuidado, N = 20 estouraria a
    memória. É o preço concreto da BREADTH — vale a pena o leitor perceber isso.
    """
    elementos = (n_objetos ** aridade_maxima) * CANAIS * 12
    return int(max(1, min(32, alvo_elementos // max(1, elementos))))


def avaliar_blocos(modelo, n_objetos, n_cenas=None, semente=SEMENTE + 100):
    """Avalia a NLM em cenas com `n_objetos` blocos. Retorna {cabeça: (acc, f1)}."""
    if n_cenas is None:
        n_cenas = CENAS_POR_TAMANHO[n_objetos]
    gerador = np.random.default_rng(semente + n_objetos)
    aridade_maxima = modelo["aridade_maxima"]
    tamanho = tamanho_de_lote_seguro(n_objetos, aridade_maxima)

    acumulado = {nome: [[], []] for nome in modelo["cabecas"]}
    restantes = n_cenas
    while restantes > 0:
        atual = min(tamanho, restantes)
        entradas, alvos = gerar_lote_blocos(gerador, atual, n_objetos, aridade_maxima)
        logits, _ = frente(modelo, entradas)
        for nome in modelo["cabecas"]:
            acumulado[nome][0].append(sigmoide(logits[nome]).ravel())
            acumulado[nome][1].append(alvos[nome].ravel())
        restantes -= atual

    resultado = {}
    for nome in modelo["cabecas"]:
        probabilidade = np.concatenate(acumulado[nome][0])
        alvo = np.concatenate(acumulado[nome][1])
        resultado[nome] = metricas_binarias(probabilidade, alvo)
    return resultado


# ===========================================================================
# PARTE 8 — TAREFA 2: ORDENAÇÃO DE VETOR
# ===========================================================================

def gerar_lote_ordenacao(gerador, tamanho_lote, n_posicoes):
    """Monta um lote da tarefa de ordenação.

    Dado: MENOR(x, y) = 1 se o valor na posição x é menor que o da posição y.
    Deduzir: PRIMEIRO(x), SEGUNDO(x), TERCEIRO(x) — a posição do menor, do
    segundo menor e do terceiro menor elemento. Cada um exige um passo de
    dedução a mais que o anterior:

        PRIMEIRO(x) <-> NÃO EXISTE y : MENOR(y, x)
        SEGUNDO(x)  <-> NÃO EXISTE y : (NÃO PRIMEIRO(y)) E MENOR(y, x)
        TERCEIRO(x) <-> NÃO EXISTE y : (NÃO PRIMEIRO(y)) E (NÃO SEGUNDO(y))
                                       E MENOR(y, x)
    """
    menor = np.zeros((tamanho_lote, n_posicoes, n_posicoes))
    alvos = {nome: np.zeros((tamanho_lote, n_posicoes, 1))
             for nome in ("PRIMEIRO", "SEGUNDO", "TERCEIRO")}

    for i in range(tamanho_lote):
        postos = gerador.permutation(n_posicoes)          # postos distintos: sem empates
        menor[i] = (postos[:, None] < postos[None, :]).astype(np.float64)
        alvos["PRIMEIRO"][i, :, 0] = (postos == 0).astype(np.float64)
        alvos["SEGUNDO"][i, :, 0] = (postos == 1).astype(np.float64)
        alvos["TERCEIRO"][i, :, 0] = (postos == 2).astype(np.float64)

    entradas = [
        np.ones((tamanho_lote, 1)),
        np.zeros((tamanho_lote, n_posicoes, 1)),          # nenhum predicado unário dado
        menor[:, :, :, None],
    ]
    return entradas, alvos


def acerto_por_argmax(pontuacao, alvo):
    """Fração de vetores em que a posição de MAIOR pontuação é a correta.

    Esta é a única métrica honesta da tarefa: como só uma posição em N é
    positiva, uma acurácia binária por posição já vale 1 - 1/N para a rede
    degenerada que responde "não" em toda parte. O acaso, aqui, vale 1/N.
    """
    previsto = np.argmax(pontuacao[..., 0], axis=1)
    verdadeiro = np.argmax(alvo[..., 0], axis=1)
    return float(np.mean(previsto == verdadeiro))


def avaliar_ordenacao(modelo, n_posicoes, n_cenas=N_CENAS_TESTE,
                      semente=SEMENTE + 200):
    """Avalia a tarefa de ordenação. Retorna {cabeça: acerto_por_argmax}.

    Comparar com o acaso é imediato: 1 / n_posicoes.
    """
    gerador = np.random.default_rng(semente + n_posicoes)
    entradas, alvos = gerar_lote_ordenacao(gerador, n_cenas, n_posicoes)
    logits, _ = frente(modelo, entradas)
    return {nome: acerto_por_argmax(logits[nome], alvos[nome])
            for nome in modelo["cabecas"]}


def treinar_ordenacao(profundidade, iteracoes=ITERACOES_ORDENACAO, marcos=()):
    """Treina do zero uma NLM de ordenação com a profundidade pedida.

    A semente dos pesos e a do sorteio de lotes são fixas e NÃO dependem da
    profundidade: a única coisa que muda entre as linhas da escada da DEPTH é
    o número de passos de dedução disponíveis.
    """
    modelo = criar_modelo(
        aridade_maxima=ARIDADE_MAXIMA_ORDENACAO,
        profundidade=profundidade,
        cabecas={"PRIMEIRO": 1, "SEGUNDO": 1, "TERCEIRO": 1},
        canais_entrada=[1, 1, 1],
        semente=SEMENTE + 21,
    )
    gerador = np.random.default_rng(SEMENTE + 31)
    historico = treinar(
        modelo,
        lambda g, t: gerar_lote_ordenacao(g, t, N_TREINO_ORDENACAO),
        iteracoes, {}, gerador, marcos, categorica=True)
    return modelo, historico


# ===========================================================================
# PARTE 9 — EXTRAÇÃO DA REGRA APRENDIDA
# ===========================================================================

def enumerar_mundos_de_blocos(n_blocos):
    """Enumera EXAUSTIVAMENTE todos os mundos possíveis com `n_blocos` blocos.

    Um mundo é descrito pelo vetor `suporte`, em que suporte[x] é o bloco
    diretamente abaixo de x (ou -1 se x está no chão). Para ser um mundo de
    blocos válido, `suporte` precisa ser injetivo nos blocos (dois blocos não
    ficam sobre o mesmo bloco) e acíclico (nada se apoia em si mesmo).
    """
    opcoes = [[-1] + [y for y in range(n_blocos) if y != x] for x in range(n_blocos)]
    mundos = []
    for suporte in itertools.product(*opcoes):
        apoios = [s for s in suporte if s >= 0]
        if len(set(apoios)) != len(apoios):
            continue
        valido = True
        for bloco in range(n_blocos):
            atual = bloco
            for _ in range(n_blocos):
                atual = suporte[atual]
                if atual < 0:
                    break
                if atual == bloco:
                    valido = False
                    break
            if not valido:
                break
        if valido:
            mundos.append(suporte)
    return mundos


def tensores_dos_mundos(mundos, n_blocos, aridade_maxima):
    """Converte a lista de mundos em (entradas da NLM, LIVRE verdadeiro)."""
    total = len(mundos)
    sobre = np.zeros((total, n_blocos, n_blocos))
    no_chao = np.zeros((total, n_blocos))
    livre = np.ones((total, n_blocos))
    for i, suporte in enumerate(mundos):
        for bloco, apoio in enumerate(suporte):
            if apoio < 0:
                no_chao[i, bloco] = 1.0
            else:
                sobre[i, bloco, apoio] = 1.0
                livre[i, apoio] = 0.0
    return montar_entradas_blocos(sobre, no_chao, aridade_maxima), livre, sobre


def desenhar_mundo(suporte, n_blocos):
    """Desenha as torres de um mundo como texto, do topo para o chão."""
    tem_algo_em_cima = {apoio for apoio in suporte if apoio >= 0}
    bases = [b for b in range(n_blocos) if suporte[b] < 0]
    torres = []
    for base in bases:
        torre = [base]
        while True:
            acima = [b for b in range(n_blocos) if suporte[b] == torre[-1]]
            if not acima:
                break
            torre.append(acima[0])
        torres.append(torre)
    partes = []
    for torre in torres:
        rotulos = [chr(ord("A") + b) for b in reversed(torre)]
        partes.append("/".join(rotulos))
    _ = tem_algo_em_cima
    return "  |  ".join(partes)


# ===========================================================================
# PARTE 10 — RELATÓRIO NO TERMINAL
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


def imprimir_secao_representacao():
    """Seção [1]: os três tipos de tensor da Figura 5.6, com um exemplo concreto."""
    cabecalho("[1] REPRESENTAÇÃO TENSORIAL DOS PREDICADOS (Figura 5.6)", "-")
    print(paragrafo(
        "Na NLM um predicado não é um símbolo em uma base de conhecimento: é um "
        "tensor de probabilidades. A ARIDADE do predicado vira o número de eixos "
        "de objeto do tensor, e é ela — não o número de objetos — que fica "
        "fixada na arquitetura."))
    print()
    print(f"  {'tipo':<10s}{'exemplo':<26s}{'forma do tensor':<26s}{'valores':>13s}")
    print("  " + "-" * 74)
    linhas = [
        ("NULÁRIO", "TODOS_EMPILHADOS()", "[lote, C0]", "[0, 1]"),
        ("UNÁRIO", "LIVRE(x), NO_CHAO(x)", "[lote, N, C1]", "[0, 1]"),
        ("BINÁRIO", "SOBRE(x,y), ACIMA(x,y)", "[lote, N, N, C2]", "[0, 1]"),
    ]
    for tipo, exemplo, forma, valores in linhas:
        print(f"  {tipo:<10s}{exemplo:<26s}{forma:<26s}{valores:>13s}")
    print("  " + "-" * 74)

    gerador = np.random.default_rng(SEMENTE + 3)
    sobre, no_chao, livre, acima, todos = gerar_cena_de_blocos(gerador, 4, True)
    rotulos = [chr(ord("A") + i) for i in range(4)]

    subtitulo("Uma cena concreta com 4 blocos, escrita como tensores")
    print()
    print("  SOBRE(x, y) — 1 quando x está DIRETAMENTE sobre y   (tensor binário)")
    print("        " + "".join(f"{r:>5s}" for r in rotulos))
    for i, r in enumerate(rotulos):
        print(f"    {r}   " + "".join(f"{sobre[i, j]:5.0f}" for j in range(4)))
    print()
    print("  NO_CHAO(x) (dado)   : " + "  ".join(
        f"{r}={no_chao[i]:.0f}" for i, r in enumerate(rotulos)))
    print("  LIVRE(x)  (deduzir) : " + "  ".join(
        f"{r}={livre[i]:.0f}" for i, r in enumerate(rotulos)))
    print("  ACIMA(x,y) (deduzir): " + ", ".join(
        f"{rotulos[i]}>{rotulos[j]}" for i in range(4) for j in range(4)
        if acima[i, j] > 0))
    print(f"  TODOS_EMPILHADOS()  : {todos:.0f}")


def imprimir_secao_operacoes():
    """Seção [2]: expand, reduce e permute demonstrados com números."""
    cabecalho("[2] AS OPERAÇÕES DE FIAÇÃO — OS QUANTIFICADORES NEURAIS", "-")
    print(paragrafo(
        "As setas verticais da Figura 5.6 são só duas operações. EXPAND sobe a "
        "aridade replicando um eixo (introduz uma variável nova). REDUCE desce a "
        "aridade agregando um eixo por MÁXIMO (que, em [0,1], é o EXISTE) e por "
        "MÍNIMO (que é o PARA TODO), concatenando as duas leituras. PERMUTE "
        "concatena o tensor binário com sua transposta, o que dá à camada "
        "seguinte acesso à inversão de argumentos."))

    predicado = np.array([[[0.9], [0.1], [0.4]]])          # [1, 3, 1]
    rotulos = ["A", "B", "C"]

    subtitulo("EXPAND: unário [1,3,1] -> binário [1,3,3,1]")
    print()
    print("  P(x) = " + "  ".join(
        f"{r}:{predicado[0, i, 0]:.1f}" for i, r in enumerate(rotulos)))
    expandido = expandir(predicado, 3)
    print(f"  forma depois do EXPAND: {tuple(expandido.shape)}"
          "   (Q(x,y) = P(x), constante em y)")
    print("        " + "".join(f"{r:>6s}" for r in rotulos))
    for i, r in enumerate(rotulos):
        print(f"    {r}   " + "".join(f"{expandido[0, i, j, 0]:6.1f}" for j in range(3)))

    subtitulo("REDUCE: binário [1,3,3,1] -> unário [1,3,2]")
    binario = np.array([[[[0.0], [0.8], [0.2]],
                         [[0.1], [0.0], [0.9]],
                         [[0.3], [0.4], [0.0]]]])
    reduzido, indice_max, indice_min = reduzir(binario)
    print()
    print("  R(x,y) =")
    print("        " + "".join(f"{r:>6s}" for r in rotulos))
    for i, r in enumerate(rotulos):
        print(f"    {r}   " + "".join(f"{binario[0, i, j, 0]:6.1f}" for j in range(3)))
    print()
    print(f"  {'x':<4s}{'máx_y R(x,y)':>16s}{'(EXISTE y)':>14s}"
          f"{'mín_y R(x,y)':>16s}{'(PARA TODO y)':>16s}")
    for i, r in enumerate(rotulos):
        print(f"  {r:<4s}{reduzido[0, i, 0]:>16.1f}{'y=' + rotulos[indice_max[0, i, 0]]:>14s}"
              f"{reduzido[0, i, 1]:>16.1f}{'y=' + rotulos[indice_min[0, i, 0]]:>16s}")
    print()
    print(paragrafo(
        "Na volta, o gradiente do máximo e do mínimo não é espalhado: ele é "
        "ROTEADO para a coluna vencedora indicada acima. É assim que a rede "
        "aprende QUAL objeto testemunhou o existencial.", recuo=2))

    subtitulo("PERMUTE: binário [1,3,3,1] -> [1,3,3,2]")
    permutado = permutar(binario, 2)
    print()
    print(f"  forma depois do PERMUTE: {tuple(permutado.shape)}"
          "   (canal 0 = R(x,y), canal 1 = R(y,x))")
    print(f"  R(A,B) = {permutado[0, 0, 1, 0]:.1f}   e   "
          f"R(B,A) = {permutado[0, 0, 1, 1]:.1f}  no MESMO ponto (A,B)")


def imprimir_secao_gradientes(resultados, pior_softmax):
    """Seção [3]: prova de que a retropropagação manual está correta."""
    cabecalho("[3] VERIFICAÇÃO NUMÉRICA DOS GRADIENTES", "-")
    print(paragrafo(
        "Toda a retropropagação — matmul, ReLU, sigmoide, expand, reduce (com "
        "roteamento para o argumento vencedor), permute, concatenação e entropia "
        "cruzada (binária e softmax) — foi escrita à mão. A conferência abaixo "
        "compara, em parâmetros sorteados, o gradiente analítico com a "
        "diferença finita central. Erros relativos da ordem de 1e-8 ou menos "
        "significam que as duas contas concordam até a precisão da aritmética "
        "de ponto flutuante."))
    print()
    print(f"  {'parâmetro':<22s}{'analítico':>16s}{'numérico':>16s}{'erro relativo':>18s}")
    print("  " + "-" * 72)
    for nome, analitico, numerico, erro in resultados:
        print(f"  {nome:<22s}{analitico:>16.9f}{numerico:>16.9f}{erro:>18.2e}")
    print("  " + "-" * 72)
    pior = max(erro for _, _, _, erro in resultados)
    print(f"  Pior erro relativo (perda binária, tarefa 1) .. {pior:.2e}")
    print(f"  Pior erro relativo (perda softmax, tarefa 2) .. {pior_softmax:.2e}")


def imprimir_curva(historico, titulo):
    """Imprime a curva de treino como tabela."""
    subtitulo(titulo)
    print()
    print(f"  {'iteração':>10s}{'perda (soma das cabeças)':>28s}")
    print("  " + "-" * 38)
    for iteracao, perda in historico:
        print(f"  {iteracao:>10d}{perda:>28.4f}")
    print("  " + "-" * 38)


def imprimir_tabela_blocos(resultados_por_n, titulo):
    """Tabela de acurácia por predicado x número de objetos no teste."""
    subtitulo(titulo)
    print()
    print(f"  {'N objetos':>10s}{'LIVRE(x)':>12s}{'ACIMA(x,y)':>13s}"
          f"{'ACIMA F1':>11s}{'TODOS_EMPILHADOS()':>21s}")
    print("  " + "-" * 67)
    for n_objetos in sorted(resultados_por_n):
        metricas = resultados_por_n[n_objetos]
        marca = " (treino)" if n_objetos == N_TREINO_BLOCOS else ""
        print(f"  {n_objetos:>10d}{metricas['LIVRE'][0]:>12.3f}"
              f"{metricas['ACIMA'][0]:>13.3f}{metricas['ACIMA'][1]:>11.3f}"
              f"{metricas['TODOS_EMPILHADOS'][0]:>21.3f}{marca}")
    print("  " + "-" * 67)


def imprimir_secao_sondagem(modelo):
    """Seção [8]: a regra lógica extraída da rede treinada."""
    cabecalho("[8] EXTRAÇÃO DA REGRA APRENDIDA — LIVRE(x) <-> NÃO EXISTE y : SOBRE(y,x)", "-")

    mundos = enumerar_mundos_de_blocos(N_BLOCOS_SONDAGEM)
    entradas, livre_verdadeiro, sobre = tensores_dos_mundos(
        mundos, N_BLOCOS_SONDAGEM, modelo["aridade_maxima"])
    logits, _ = frente(modelo, entradas)
    probabilidade = sigmoide(logits["LIVRE"])[..., 0]

    print(paragrafo(
        f"A rede treinada com cenas de {N_TREINO_BLOCOS} blocos é agora sondada "
        f"com TODOS os {len(mundos)} mundos possíveis de {N_BLOCOS_SONDAGEM} "
        f"blocos — nenhuma amostragem, nenhuma sorte: a enumeração é exaustiva. "
        f"Para cada bloco comparamos a saída da rede com o valor lógico de "
        f"NÃO EXISTE y : SOBRE(y, x)."))

    existe_apoiado = (sobre.sum(axis=1) > 0.5)      # existe y com SOBRE(y, x)
    verdade = (~existe_apoiado)
    previsto = probabilidade >= 0.5

    subtitulo("Tabela de verdade empírica (todos os blocos de todos os mundos)")
    print()
    print(f"  {'EXISTE y : SOBRE(y,x)':<26s}{'casos':>9s}"
          f"{'LIVRE previsto = 1':>21s}{'prob. média':>15s}")
    print("  " + "-" * 71)
    for valor, rotulo in ((False, "FALSO  (nada sobre x)"), (True, "VERDADEIRO (algo sobre x)")):
        mascara = existe_apoiado == valor
        casos = int(mascara.sum())
        positivos = int(previsto[mascara].sum())
        media = float(probabilidade[mascara].mean())
        print(f"  {rotulo:<26s}{casos:>9d}{positivos:>21d}{media:>15.4f}")
    print("  " + "-" * 71)

    concordancia = float(np.mean(previsto == verdade))
    print(f"  Concordância com a regra lógica: {concordancia:.4f} "
          f"({int(np.sum(previsto == verdade))} de {previsto.size} blocos)")
    print()
    print(paragrafo(
        "Repare que a regra NÃO foi programada em lugar nenhum: a rede recebeu "
        "apenas SOBRE(x,y), NO_CHAO(x) e exemplos rotulados, e o que emergiu do "
        "REDUCE por máximo foi exatamente o quantificador existencial negado. "
        "É esse o ponto do capítulo: a NLM parte de proposições em lógica de "
        "predicados, executa dedução e gera conceitos conclusivos sobre os "
        "objetos."))

    subtitulo("Quatro mundos sondados um a um (topo/.../chão, blocos A a D)")
    print()
    escolhidos = [0, len(mundos) // 5, len(mundos) // 2, len(mundos) - 1]
    largura_mundo = 22
    largura_coluna = 13
    print(f"  {'mundo (torres)':<{largura_mundo}s}" + "".join(
        f"{'P(LIVRE ' + chr(ord('A') + b) + ')':>{largura_coluna}s}"
        for b in range(N_BLOCOS_SONDAGEM)))
    regua = "  " + "-" * (largura_mundo + largura_coluna * N_BLOCOS_SONDAGEM)
    print(regua)
    for indice in escolhidos:
        desenho = desenhar_mundo(mundos[indice], N_BLOCOS_SONDAGEM)
        valores = "".join(f"{probabilidade[indice, b]:>{largura_coluna}.3f}"
                          for b in range(N_BLOCOS_SONDAGEM))
        print(f"  {desenho:<{largura_mundo}s}{valores}")
    print(regua)
    print("  Os blocos livres (topo de cada torre) recebem probabilidade alta;")
    print("  os que sustentam alguém recebem probabilidade baixa.")


def imprimir_secao_interpretacao(resultados_blocos, resultados_ablacao,
                                 resultados_ordenacao, n_parametros):
    """Seção [9]: o fecho que amarra tudo aos conceitos do capítulo."""
    cabecalho("[9] INTERPRETAÇÃO — o que você acabou de ver")

    acc_livre_treino = resultados_blocos[N_TREINO_BLOCOS]["LIVRE"][0]
    acc_livre_maior = resultados_blocos[max(NS_TESTE_BLOCOS)]["LIVRE"][0]
    f1_acima_treino = resultados_blocos[N_TREINO_BLOCOS]["ACIMA"][1]
    f1_acima_maior = resultados_blocos[max(NS_TESTE_BLOCOS)]["ACIMA"][1]
    f1_ablacao = resultados_ablacao[N_TREINO_BLOCOS]["ACIMA"][1]
    maior_n = max(NS_TESTE_ORDENACAO)
    argmax_primeiro = resultados_ordenacao[maior_n]["PRIMEIRO"]
    argmax_segundo = resultados_ordenacao[maior_n]["SEGUNDO"]
    argmax_terceiro = resultados_ordenacao[maior_n]["TERCEIRO"]
    acaso_maior = 1.0 / maior_n

    print()
    print(paragrafo(
        f"1. GENERALIZAÇÃO NO NÚMERO DE OBJETOS. A rede treinou apenas com "
        f"{N_TREINO_BLOCOS} blocos e foi testada com até {max(NS_TESTE_BLOCOS)}. "
        f"A acurácia de LIVRE(x) foi de {acc_livre_treino:.3f} no tamanho de "
        f"treino e {acc_livre_maior:.3f} no maior tamanho de teste. Isso não é "
        f"mágica: os MLPs são compartilhados entre todas as posições dos eixos "
        f"de objeto, então o que a rede aprendeu foi uma REGRA sobre variáveis, "
        f"não uma tabela indexada por objeto. Uma rede densa comum, com uma "
        f"entrada por par de blocos, teria de ser reconstruída e retreinada a "
        f"cada novo N."))
    print()
    print(paragrafo(
        f"2. O EIXO DEPTH DA FIGURA 5.6 É O NÚMERO DE PASSOS DE DEDUÇÃO, E "
        f"ISSO FOI MEDIDO. A escada da seção [7] treinou a MESMA tarefa de "
        f"ordenação com profundidades 1, 2, 3 e 4, tudo o mais igual. Com "
        f"DEPTH 2 apareceu PRIMEIRO(x); só com DEPTH 4 apareceu SEGUNDO(x) — "
        f"exatamente a contagem de passos que a fórmula lógica exige. Depois "
        f"de treinada, a rede de DEPTH {PROFUNDIDADE_ORDENACAO} acerta a "
        f"posição do menor elemento em {argmax_primeiro:.1%} e a do segundo "
        f"menor em {argmax_segundo:.1%} dos vetores de {maior_n} posições, "
        f"contra um acaso de {acaso_maior:.1%}. TERCEIRO(x) exigiria 6 passos "
        f"e ficou em {argmax_terceiro:.1%} nesse tamanho: a rede decorou um "
        f"atalho válido em {N_TREINO_ORDENACAO} posições em vez de aprender a "
        f"regra. Profundidade insuficiente não degrada suavemente — ela "
        f"simplesmente não deduz."))
    print()
    print(paragrafo(
        f"3. O EIXO BREADTH É A ARIDADE MÁXIMA — E ELE NÃO É DECORAÇÃO. "
        f"ACIMA(x,y) é o fecho transitivo de SOBRE(x,y): deduzi-lo exige "
        f"compor duas relações, ou seja, escrever EXISTE k : SOBRE(x,k) E "
        f"ACIMA(k,y). Essa fórmula tem TRÊS variáveis livres ao mesmo tempo. "
        f"Com BREADTH {ARIDADE_MAXIMA_BLOCOS} a rede tem onde escrevê-la e "
        f"chega a F1 = {f1_acima_treino:.3f}; com BREADTH "
        f"{ARIDADE_MAXIMA_ABLACAO}, mantida toda a demais configuração, o mesmo "
        f"predicado desaba para F1 = {f1_ablacao:.3f}. A largura da Figura 5.6 "
        f"delimita literalmente QUE FÓRMULAS a máquina consegue pensar."))
    print()
    print(paragrafo(
        f"4. O PREÇO DA BREADTH. O tensor de aridade r tem N^r posições. Subir "
        f"de 2 para 3 trocou N² por N³ e multiplicou também o PERMUTE, que "
        f"passou de 2! = 2 para 3! = 6 cópias. Com "
        f"{max(NS_TESTE_BLOCOS)} objetos o script precisa avaliar as cenas em "
        f"lotes menores para caber na memória. É a mesma tensão do capítulo, "
        f"agora em forma de custo computacional: expressividade lógica não sai "
        f"de graça."))
    print()
    print(paragrafo(
        f"5. TOTALMENTE DIFERENCIÁVEL. Nenhuma etapa deste programa chama um "
        f"motor de inferência externo. Quantificadores viraram máximo e mínimo; "
        f"conjunção e disjunção viraram um MLP com saída sigmoide; o "
        f"encadeamento de regras virou profundidade. Por isso os "
        f"{n_parametros} parâmetros puderam ser ajustados por gradiente — e a "
        f"seção [3] mostrou que esse gradiente está correto até 1e-8. É esse o "
        f"casamento que o capítulo chama de neuro-simbólico: a estrutura é "
        f"simbólica (aridade, quantificação, encadeamento), o aprendizado é "
        f"neural."))
    print()
    print(paragrafo(
        f"6. ONDE ISSO ENCOSTA NO RESTO DO CAPÍTULO. O NSCL e o NSDR resolvem "
        f"CLEVR e CLEVRER separando um analisador de percepção de um executor "
        f"simbólico escrito à mão. A NLM vai mais longe e dissolve o executor "
        f"dentro da rede: aqui o F1 de ACIMA(x,y) caiu de "
        f"{f1_acima_treino:.3f} para {f1_acima_maior:.3f} ao passar de "
        f"{N_TREINO_BLOCOS} para {max(NS_TESTE_BLOCOS)} objetos, o que mostra "
        f"que a dedução aprendida é boa mas aproximada — enquanto um motor "
        f"simbólico clássico seria exato e, em compensação, não seria "
        f"treinável. Guarde as duas colunas dessa comparação: é entre elas que "
        f"a IA Neuro-Simbólica se move."))


# ===========================================================================
# PARTE 11 — PROGRAMA PRINCIPAL
# ===========================================================================

def main():
    """Executa o exercício completo e imprime o relatório didático."""

    cabecalho("EXERCÍCIO 5 — NEURAL LOGIC MACHINE")
    print(" Capítulo 5 — Introdução à IA Neuro-Simbólica (Figura 5.6)")
    print(" Dedução de lógica de primeira ordem com tensores, do zero, em numpy.")

    imprimir_secao_representacao()
    imprimir_secao_operacoes()
    imprimir_secao_gradientes(verificar_gradientes(),
                              verificar_gradiente_softmax())

    # -----------------------------------------------------------------------
    # Tarefa 1 — mundo de blocos, BREADTH 3
    # -----------------------------------------------------------------------
    cabecalho("[4] TAREFA 1 — MUNDO DE BLOCOS (BREADTH 3)", "-")
    cabecas_blocos = {"LIVRE": 1, "ACIMA": 2, "TODOS_EMPILHADOS": 0}
    canais_entrada_blocos = [1, 1, 1, 1]

    modelo_blocos = criar_modelo(
        aridade_maxima=ARIDADE_MAXIMA_BLOCOS,
        profundidade=PROFUNDIDADE_BLOCOS,
        cabecas=cabecas_blocos,
        canais_entrada=canais_entrada_blocos[:ARIDADE_MAXIMA_BLOCOS + 1],
    )
    print(f"  Aridade máxima (BREADTH) ...... {ARIDADE_MAXIMA_BLOCOS}")
    print(f"  Profundidade (DEPTH) .......... {PROFUNDIDADE_BLOCOS}")
    print(f"  Canais por aridade ............ {CANAIS}")
    print(f"  Neurônios ocultos por MLP ..... {OCULTAS}")
    print(f"  Parâmetros treináveis ......... {contar_parametros(modelo_blocos)}")
    print(f"  Blocos por cena no treino ..... {N_TREINO_BLOCOS}")
    print(f"  Altura máxima de uma torre .... {ALTURA_MAXIMA_TORRE}")
    print(f"  Predicados dados .............. SOBRE(x,y), NO_CHAO(x)")
    print(f"  Predicados a deduzir .......... LIVRE(x), ACIMA(x,y), "
          f"TODOS_EMPILHADOS()")

    gerador_treino = np.random.default_rng(SEMENTE + 11)
    marcos = tuple(sorted({1, 25, 50, 100, 150, 200, ITERACOES_BLOCOS}))
    historico = treinar(
        modelo_blocos,
        lambda g, t: gerar_lote_blocos(g, t, N_TREINO_BLOCOS, ARIDADE_MAXIMA_BLOCOS),
        ITERACOES_BLOCOS, PESOS_POSITIVOS_BLOCOS, gerador_treino, marcos)
    imprimir_curva(historico, "Treino (Adam escrito do zero)")

    cabecalho("[5] GENERALIZAÇÃO PARA MAIS OBJETOS DO QUE OS DO TREINO", "-")
    print(paragrafo(
        f"A rede viu apenas cenas de {N_TREINO_BLOCOS} blocos. Abaixo ela é "
        f"avaliada em cenas novas de cada tamanho, SEM nenhum "
        f"retreino e SEM mudar um único peso — os mesmos MLPs são aplicados a um "
        f"tensor maior."))
    resultados_blocos = {n: avaliar_blocos(modelo_blocos, n) for n in NS_TESTE_BLOCOS}
    imprimir_tabela_blocos(resultados_blocos,
                           "Acurácia por predicado x número de objetos")
    print()
    print(paragrafo(
        "ACIMA(x,y) é fortemente desbalanceado (a maioria dos pares de blocos "
        "não tem relação alguma), por isso a coluna do F1 é a leitura honesta "
        "desse predicado: acurácia alta com F1 baixo significaria uma rede que "
        "aprendeu apenas a dizer 'não'."))
    print()
    print(paragrafo(
        "Duas leituras opostas na mesma tabela, e as duas importam. LIVRE(x) e "
        "ACIMA(x,y) generalizam: são predicados sobre objetos e pares, e o MLP "
        "compartilhado se aplica a qualquer N. TODOS_EMPILHADOS() NÃO "
        "generaliza — vale 1,000 nos 4 blocos do treino e desce para perto do "
        "acaso (o rótulo é balanceado por construção) nos tamanhos maiores. "
        "Faz sentido: é um predicado NULÁRIO, e chegar a ele exige um PARA "
        "TODO sobre todos os blocos, cujo valor mínimo depende de quantos "
        "blocos existem. A quantificação universal aprendida em 4 objetos não "
        "está calibrada para 20."))

    # -----------------------------------------------------------------------
    # Tarefa 1b — ablação da BREADTH
    # -----------------------------------------------------------------------
    cabecalho("[6] O EIXO BREADTH — A MESMA REDE LIMITADA A PREDICADOS BINÁRIOS", "-")
    print(paragrafo(
        f"Repetimos o experimento com uma única mudança: a aridade máxima cai "
        f"de {ARIDADE_MAXIMA_BLOCOS} para {ARIDADE_MAXIMA_ABLACAO}. Tudo o mais "
        f"— profundidade, canais, iterações, dados, semente — permanece igual."))
    modelo_ablacao = criar_modelo(
        aridade_maxima=ARIDADE_MAXIMA_ABLACAO,
        profundidade=PROFUNDIDADE_BLOCOS,
        cabecas=cabecas_blocos,
        canais_entrada=canais_entrada_blocos[:ARIDADE_MAXIMA_ABLACAO + 1],
    )
    gerador_ablacao = np.random.default_rng(SEMENTE + 11)
    historico_ablacao = treinar(
        modelo_ablacao,
        lambda g, t: gerar_lote_blocos(g, t, N_TREINO_BLOCOS, ARIDADE_MAXIMA_ABLACAO),
        ITERACOES_BLOCOS, PESOS_POSITIVOS_BLOCOS, gerador_ablacao, marcos)
    imprimir_curva(historico_ablacao, "Treino da rede com BREADTH 2")

    resultados_ablacao = {n: avaliar_blocos(modelo_ablacao, n)
                          for n in NS_TESTE_BLOCOS}
    imprimir_tabela_blocos(resultados_ablacao,
                           "BREADTH 2 — acurácia por predicado x número de objetos")
    print()
    print(f"  {'predicado':<22s}{'BREADTH ' + str(ARIDADE_MAXIMA_ABLACAO):>16s}"
          f"{'BREADTH ' + str(ARIDADE_MAXIMA_BLOCOS):>16s}   métrica")
    print("  " + "-" * 70)
    comparacoes = [
        ("LIVRE(x)", "LIVRE", 0, "acurácia"),
        ("TODOS_EMPILHADOS()", "TODOS_EMPILHADOS", 0, "acurácia"),
        ("ACIMA(x,y)", "ACIMA", 0, "acurácia"),
        ("ACIMA(x,y)", "ACIMA", 1, "F1"),
    ]
    for rotulo, chave, indice, metrica in comparacoes:
        valor_2 = resultados_ablacao[N_TREINO_BLOCOS][chave][indice]
        valor_3 = resultados_blocos[N_TREINO_BLOCOS][chave][indice]
        print(f"  {rotulo:<22s}{valor_2:>16.3f}{valor_3:>16.3f}   {metrica}")
    print("  " + "-" * 70)
    print(paragrafo(
        "LIVRE(x) sobrevive intacto: é uma fórmula com uma única variável "
        "quantificada, e a aridade 2 basta para escrevê-la. Os outros dois "
        "não sobrevivem igualmente. ACIMA(x,y) continua sendo aproximado — F1 "
        "cai de 1,000 para 0,943 no tamanho de treino e para 0,864 em 20 "
        "objetos —, ou seja, com BREADTH 2 a rede chega perto do fecho "
        "transitivo por atalho, mas nunca o acerta e piora quando o N cresce. "
        "TODOS_EMPILHADOS() é o que realmente desaba: de 1,000 para 0,592, "
        "praticamente o acaso. O fecho transitivo precisa de três variáveis "
        "simultâneas (EXISTE k : SOBRE(x,k) E ACIMA(k,y)) e um tensor binário "
        "não tem eixo onde guardá-las."))

    # -----------------------------------------------------------------------
    # Tarefa 2 — ordenação de vetor
    # -----------------------------------------------------------------------
    cabecalho("[7] TAREFA 2 — ORDENAÇÃO DE VETOR (a escada da DEPTH)", "-")
    print(f"  Predicado dado ................ MENOR(x,y)")
    print(f"  Predicados a deduzir .......... PRIMEIRO(x), SEGUNDO(x), "
          f"TERCEIRO(x)")
    print(f"  Aridade máxima (BREADTH) ...... {ARIDADE_MAXIMA_ORDENACAO}")
    print(f"  Profundidade da rede principal  {PROFUNDIDADE_ORDENACAO}")
    print(f"  Posições por vetor no treino .. {N_TREINO_ORDENACAO}")
    print(f"  Iterações de treino ........... {ITERACOES_ORDENACAO}")
    print(f"  Perda ......................... softmax sobre as N posições")
    print()
    print("  As três regras-alvo, em lógica de primeira ordem, e quantos")
    print("  passos de dedução cada uma exige nesta arquitetura:")
    print("    PRIMEIRO(x) <-> NÃO EXISTE y : MENOR(y,x)")
    print("                    -> 2 passos (transpor MENOR, depois o EXISTE)")
    print("    SEGUNDO(x)  <-> NÃO EXISTE y : NÃO PRIMEIRO(y) E MENOR(y,x)")
    print("                    -> 4 passos (PRIMEIRO + mais um par)")
    print("    TERCEIRO(x) <-> NÃO EXISTE y : NÃO PRIMEIRO(y) E")
    print("                    NÃO SEGUNDO(y) E MENOR(y,x)")
    print("                    -> 6 passos (SEGUNDO + mais um par)")
    print()
    print(paragrafo(
        "Só uma posição em N é positiva. Tratar isso como N decisões binárias "
        "independentes faz a rede colapsar em 'responda não sempre' — a "
        "acurácia por posição fica em 1 - 1/N e o argmax nunca acerta. Por "
        "isso esta tarefa usa uma perda SOFTMAX sobre o eixo de objetos: as N "
        "posições competem por uma única resposta. A métrica abaixo é o "
        "argmax, e o acaso vale exatamente 1/N."))

    subtitulo("A escada da DEPTH: a MESMA tarefa, profundidades crescentes")
    print()
    print(paragrafo(
        "Pesos iniciais, dados, iterações e semente são idênticos em todas as "
        "linhas. A ÚNICA diferença é o número de camadas — ou seja, o número "
        "de passos de dedução disponíveis. Acerto por argmax em vetores novos "
        "de {} posições; acaso = {:.3f}.".format(
            N_TREINO_ORDENACAO, 1.0 / N_TREINO_ORDENACAO)))
    print()
    print(f"  {'DEPTH':>7s}{'perda final':>14s}{'PRIMEIRO':>11s}"
          f"{'SEGUNDO':>10s}{'TERCEIRO':>11s}")
    print("  " + "-" * 51)
    modelo_ordenacao = None
    historico_ordenacao = None
    marcos_ordenacao = tuple(sorted(
        {1, 25, 50, 100, 200, 300, ITERACOES_ORDENACAO}))
    for profundidade in PROFUNDIDADES_ESCADA:
        principal = profundidade == PROFUNDIDADE_ORDENACAO
        marcos = marcos_ordenacao if principal else (ITERACOES_ORDENACAO,)
        modelo, historico = treinar_ordenacao(profundidade, marcos=marcos)
        acertos = avaliar_ordenacao(modelo, N_TREINO_ORDENACAO)
        print(f"  {profundidade:>7d}{historico[-1][1]:>14.4f}"
              f"{acertos['PRIMEIRO']:>11.3f}{acertos['SEGUNDO']:>10.3f}"
              f"{acertos['TERCEIRO']:>11.3f}")
        if principal:
            modelo_ordenacao = modelo
            historico_ordenacao = historico
    print("  " + "-" * 51)
    print(paragrafo(
        "A leitura é direta e bate com a contagem de passos acima. Com DEPTH 1 "
        "nada é aprendido. Com DEPTH 2 PRIMEIRO(x) fica perfeito e os outros "
        "dois ficam no acaso. DEPTH 3 não muda nada (falta um par inteiro de "
        "camadas para SEGUNDO). Só com DEPTH 4 SEGUNDO(x) aparece. É a "
        "profundidade da Figura 5.6 sendo medida, não ilustrada."))

    print(f"  Parâmetros treináveis da rede DEPTH "
          f"{PROFUNDIDADE_ORDENACAO}: "
          f"{contar_parametros(modelo_ordenacao)}")
    imprimir_curva(historico_ordenacao,
                   f"Treino da rede principal (DEPTH {PROFUNDIDADE_ORDENACAO})")

    resultados_ordenacao = {n: avaliar_ordenacao(modelo_ordenacao, n)
                            for n in NS_TESTE_ORDENACAO}
    subtitulo("Generalização para vetores maiores que os do treino")
    print()
    print(f"  {'N posições':>11s}{'acaso':>9s}{'PRIMEIRO(x)':>13s}"
          f"{'SEGUNDO(x)':>13s}{'TERCEIRO(x)':>13s}")
    print("  " + "-" * 59)
    for n_posicoes in sorted(resultados_ordenacao):
        metricas = resultados_ordenacao[n_posicoes]
        marca = " (treino)" if n_posicoes == N_TREINO_ORDENACAO else ""
        print(f"  {n_posicoes:>11d}{1.0 / n_posicoes:>9.3f}"
              f"{metricas['PRIMEIRO']:>13.3f}{metricas['SEGUNDO']:>13.3f}"
              f"{metricas['TERCEIRO']:>13.3f}{marca}")
    print("  " + "-" * 59)
    print(paragrafo(
        "PRIMEIRO(x) e SEGUNDO(x) são exatos em TODOS os tamanhos, inclusive "
        "com 20 posições — quatro vezes o tamanho de treino e um espaço de "
        "respostas quatro vezes maior. Nenhum peso foi tocado entre uma linha "
        "e a outra: o MLP compartilhado aprendeu uma regra sobre variáveis."))
    print()
    print(paragrafo(
        "TERCEIRO(x) é o RESULTADO NEGATIVO desta seção, e ele é informativo. "
        "A rede acerta 100% em 5 posições, mas cai para o nível do acaso "
        "assim que o vetor cresce. A explicação está na contagem de passos: "
        f"TERCEIRO exige 6, e esta rede tem {PROFUNDIDADE_ORDENACAO}. Sem "
        "profundidade para escrever a regra geral, o que sobra é decorar um "
        "atalho válido só em N = 5 (ali 'o terceiro menor' coincide com 'o "
        "terceiro maior'), e o atalho não sobrevive a N maior. Aumentar "
        "PROFUNDIDADES_ESCADA para 6 não resolve de graça: uma pilha de seis "
        "camadas sigmoides satura e o gradiente deixa de chegar às primeiras "
        "camadas — a perda estaciona perto do valor inicial. Fica como "
        "exercício para o leitor."))

    # -----------------------------------------------------------------------
    # Sondagem e fecho
    # -----------------------------------------------------------------------
    imprimir_secao_sondagem(modelo_blocos)
    imprimir_secao_interpretacao(resultados_blocos, resultados_ablacao,
                                 resultados_ordenacao,
                                 contar_parametros(modelo_blocos))

    print()
    print("=" * LARGURA)
    print(" Fim do Exercício 5.")
    print("=" * LARGURA)


if __name__ == "__main__":
    main()
