# -*- coding: utf-8 -*-
"""
EXERCÍCIO PRÁTICO 1 — O INGREDIENTE SIMBÓLICO
=============================================
Base de conhecimento e motor de inferência.

Capítulo 5 — "Introdução à IA Neuro-Simbólica — o Próximo Nível da IA".
Referência visual: Figura 5.1 (base de conhecimento + motor de inferência).

O QUE ESTE EXERCÍCIO DEMONSTRA
------------------------------
O lado SIMBÓLICO da NSAI, inteiro e funcionando: um mundo 3D no estilo CLEVR
descrito não por pixels, mas por PROPOSIÇÕES — forma(obj1, esfera),
cor(obj1, azul), sobre(obj5, obj4) —; uma BASE DE CONHECIMENTO escrita à mão em
cláusulas de Horn, com fechamento transitivo; um MOTOR DE INFERÊNCIA por
encadeamento para frente (*forward chaining*) com unificação de variáveis, que
itera até o PONTO FIXO registrando, para cada fato derivado, a regra e as
premissas que o produziram — o que permite imprimir a ÁRVORE DE DERIVAÇÃO, a
prova de qualquer conclusão; e um MOTOR DE CONSULTA no estilo CLEVR que devolve
cada resposta com o rastro lógico inteiro.

E, principalmente, as DUAS FRAQUEZAS FATAIS que o capítulo aponta: o PROCESSO
MANUAL — basta uma cena com uma forma nova (pirâmide) e uma cor nova (verde)
para o sistema errar EM SILÊNCIO — e a AUSÊNCIA DE ESCALABILIDADE, quantificada
aqui pela explosão combinatória de cobrir mundos cada vez maiores, até o CLEVR
real. As duas têm a MESMA causa: o ser humano no meio do processo. É isso que
motiva o ingrediente neural do Exercício 2.

RESTRIÇÕES DE IMPLEMENTAÇÃO
---------------------------
Somente a biblioteca padrão do Python 3 + numpy. Nenhum motor Prolog, nenhuma
biblioteca de lógica: fatos, unificação, encadeamento para frente, registro das
provas e DSL de consulta são implementados do zero — é aí que mora o valor
didático.

EXECUÇÃO
--------
    python3 exercicio_pratico_1_base_de_conhecimento_simbolica.py

Sem argumentos, sem arquivos de entrada, sem interação. A saída é um relatório
de texto no terminal, e a semente é fixa: o relatório é sempre o mesmo, e você
pode conferir seus números com os do colega ao lado.
"""

import textwrap

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTES GLOBAIS
# ---------------------------------------------------------------------------

SEMENTE = 42                 # semente fixa => saída 100% reprodutível
LARGURA = 78                 # largura das linhas do relatório

# O vocabulário: é EXATAMENTE isto que a base sabe nomear, e cada constante
# custará ao menos uma regra escrita à mão.
FORMAS = ("esfera", "cilindro", "cubo")
CORES = ("vermelho", "azul", "amarelo", "cinza")
TAMANHOS = ("grande", "pequeno")
MATERIAIS = ("metálico", "fosco")

# O que o mundo NÃO tem, e que a cena da fragilidade vai apresentar (seção 5).
FORMA_NOVA = "pirâmide"
COR_NOVA = "verde"

N_OBJETOS_CENA_PRINCIPAL = 6     # cena usada nas seções [1] a [4]
N_OBJETOS_CENAS_EXTRAS = (4, 3)  # cenas de apoio, para as estatísticas
ALTURA_MAXIMA_PILHA = 3          # objetos empilháveis na mesma coluna
PROB_EMPILHAR = 0.45             # chance de empilhar em vez de abrir coluna
MAX_TENTATIVAS = 20000           # limite da amostragem por rejeição

N_RELACOES_CLEVR = 4             # esquerda, direita, frente e atrás
N_CENAS_CLEVR = 100000           # tamanho do conjunto CLEVR citado no capítulo

# "Orçamento humano": traduz a explosão combinatória em anos de anotação.
SEGUNDOS_POR_PROPOSICAO = 10
HORAS_UTEIS_POR_DIA = 8
DIAS_UTEIS_POR_ANO = 250

# Rótulos editoriais, usados só para agrupar a impressão das regras.
GRUPO_NOMEACAO = "Nomeação do vocabulário (uma regra por constante)"
GRUPO_UNIVERSO = "Pertinência ao universo de objetos"
GRUPO_RELACOES = "Relações espaciais e comparativas"
GRUPO_COMPOSTOS = "Conceitos compostos"


# ===========================================================================
# PARTE 1 — REPRESENTAÇÃO LÓGICA E MOTOR DE INFERÊNCIA
# ===========================================================================
#
# Um LITERAL é uma tupla ("predicado", termo1, ...); um termo iniciado por
# maiúscula é uma VARIÁVEL, qualquer outro é uma CONSTANTE; um FATO é um
# literal sem variáveis. Tuplas são imutáveis e ORDENÁVEIS, o que mantém a
# saída do programa idêntica em toda execução.

def eh_variavel(termo):
    """Verdadeiro se o termo é uma variável lógica (inicial maiúscula)."""
    return isinstance(termo, str) and termo[:1].isupper()


def formatar_literal(literal):
    """Escreve um literal na notação usual: predicado(arg1, arg2)."""
    return f"{literal[0]}({', '.join(literal[1:])})"


class Regra:
    """Uma cláusula de Horn. As `restricoes` são desigualdades
    ("!=", "X", "Y") verificadas sobre as ligações de variáveis — é o que
    impede um objeto de ser declarado "similar a si mesmo"."""

    def __init__(self, nome, cabeca, corpo, restricoes=(), grupo=""):
        self.nome, self.cabeca, self.grupo = nome, cabeca, grupo
        self.corpo, self.restricoes = tuple(corpo), tuple(restricoes)

    def texto(self):
        """A regra escrita como o leitor a escreveria em um caderno."""
        partes = [formatar_literal(literal) for literal in self.corpo]
        partes += [f"{e} {op} {d}" for op, e, d in self.restricoes]
        return f"{formatar_literal(self.cabeca)} :- {', '.join(partes)}."


def casar_literal(literal, fato, ligacoes):
    """Casa um literal (com variáveis) contra um fato fechado: devolve novas
    ligações, ou None. Uma variável já ligada só casa com o mesmo valor — é
    assim que o F repetido em `forma(X, F), forma(Y, F)` iguala as formas."""
    if literal[0] != fato[0] or len(literal) != len(fato):
        return None
    novas = dict(ligacoes)
    for termo, valor in zip(literal[1:], fato[1:]):
        if eh_variavel(termo):
            if termo in novas:
                if novas[termo] != valor:
                    return None
            else:
                novas[termo] = valor
        elif termo != valor:
            return None
    return novas


def satisfaz_restricoes(restricoes, ligacoes):
    """Confere as desigualdades cujas duas variáveis já estão ligadas."""
    for _operador, esquerda, direita in restricoes:
        valor_esq = ligacoes.get(esquerda, esquerda)
        valor_dir = ligacoes.get(direita, direita)
        if eh_variavel(valor_esq) or eh_variavel(valor_dir):
            continue          # ainda não dá para decidir; verifica-se depois
        if valor_esq == valor_dir:
            return False
    return True


def resolver_corpo(corpo, restricoes, indice, ligacoes=None, apoios=()):
    """Produz as ligações que satisfazem o corpo de uma regra, por busca em
    profundidade. Cada resultado é um par (ligacoes, apoios), em que `apoios`
    são os fatos usados — a matéria-prima da prova."""
    if ligacoes is None:
        ligacoes = {}
    if not corpo:
        if satisfaz_restricoes(restricoes, ligacoes):
            yield ligacoes, apoios
        return
    literal, resto = corpo[0], corpo[1:]
    for fato in indice.get(literal[0], ()):
        novas = casar_literal(literal, fato, ligacoes)
        if novas is None or not satisfaz_restricoes(restricoes, novas):
            continue          # não casa, ou já viola a restrição X != Y
        yield from resolver_corpo(resto, restricoes, indice, novas,
                                  apoios + (fato,))


def instanciar(literal, ligacoes):
    """Substitui as variáveis do literal pelos valores ligados."""
    return tuple([literal[0]] + [ligacoes.get(t, t) for t in literal[1:]])


def indexar(fatos):
    """Agrupa os fatos por predicado. Usa a lista ORDENADA: um `set` não tem
    ordem estável entre execuções, e ela vazaria para o relatório."""
    indice = {}
    for fato in sorted(fatos):
        indice.setdefault(fato[0], []).append(fato)
    return indice


def encadear_para_frente(fatos_observados, regras):
    """Aplica todas as regras repetidamente até o ponto fixo: cada ciclo casa
    TODAS as regras contra TUDO o que já se sabe, e os fatos novos podem
    disparar outras regras no ciclo seguinte — é assim que o fechamento
    transitivo de `acima_de` sobe a pilha. Em `justificativas`, cada fato
    aponta para None (observado) ou para (regra, apoios)."""
    base = set(fatos_observados)
    justificativas = {fato: None for fato in sorted(base)}
    historico, n_ciclos = [], 0
    while True:
        n_ciclos += 1
        indice = indexar(base)
        novos = {}
        for regra in regras:
            for ligacoes, apoios in resolver_corpo(regra.corpo,
                                                   regra.restricoes, indice):
                cabeca = instanciar(regra.cabeca, ligacoes)
                if cabeca in base or cabeca in novos:
                    continue
                novos[cabeca] = (regra, apoios)
        if not novos:
            break             # ponto fixo: nenhuma regra tem mais o que dizer
        for fato in sorted(novos):
            base.add(fato)
            justificativas[fato] = novos[fato]
        historico.append((n_ciclos, len(novos), len(base)))
    return base, justificativas, historico, n_ciclos


def fatos_orfaos(justificativas):
    """Fatos observados que NÃO participaram de nenhuma derivação: a medida
    numérica da ignorância do sistema — proposições bem formadas sobre as quais
    nenhuma regra escrita à mão tem coisa alguma a dizer."""
    usados = set()
    for justificativa in justificativas.values():
        if justificativa is not None:
            usados.update(justificativa[1])
    return sorted(f for f, j in justificativas.items()
                  if j is None and f not in usados)  # nunca serviu de premissa


def imprimir_prova(fato, justificativas, recuo=6, marcador="", visitados=None):
    """Imprime a árvore de derivação de um fato: cada linha traz o fato e a
    regra que o produziu, e os filhos são as premissas. É aqui que a
    explicabilidade simbólica se materializa — isto é a prova."""
    if visitados is None:
        visitados = set()
    inicio = " " * recuo + marcador
    texto = formatar_literal(fato)
    largura_nome = max(12, 52 - len(inicio))
    justificativa = justificativas.get(fato)
    if justificativa is None:
        print(f"{inicio}{texto:<{largura_nome}s} [observado na cena]")
        return
    regra, apoios = justificativa
    print(f"{inicio}{texto:<{largura_nome}s} [{regra.nome}]")
    if fato in visitados:
        print(f"{' ' * (recuo + 3)}+- (prova já exibida acima)")
        return
    visitados.add(fato)
    for apoio in apoios:
        imprimir_prova(apoio, justificativas, recuo + 3, "+- ", visitados)


# ===========================================================================
# PARTE 2 — A BASE DE CONHECIMENTO ESCRITA À MÃO
# ===========================================================================

def construir_base_de_conhecimento():
    """Devolve a lista de regras de Horn — o "cérebro" simbólico do sistema.
    TODA regra abaixo foi digitada por um humano: nada de aprendizado, de dados
    ou de ajuste de parâmetros. O sistema saberá exatamente o que estiver
    escrito aqui, nem uma vírgula a mais."""
    regras = []

    def adicionar(cabeca, corpo, restricoes=(), grupo=""):
        regras.append(Regra(f"R{len(regras) + 1:02d}", cabeca, corpo,
                            restricoes, grupo))

    # Grupo A — nomeação: uma regra por constante do vocabulário. É o trabalho
    # mais tedioso, e cresce com |F| + |C| + |T| + |M|.
    for atributo, valores in (("forma", FORMAS), ("cor", CORES),
                              ("tamanho", TAMANHOS), ("material", MATERIAIS)):
        for valor in valores:
            adicionar((valor, "X"), [(atributo, "X", valor)],
                      grupo=GRUPO_NOMEACAO)

    # Grupo B — o que conta como "um objeto". Horn não tem disjunção no corpo,
    # então "esfera OU cilindro OU cubo" vira três regras com a mesma cabeça.
    # Guarde a decisão: é aqui que a pirâmide da seção [5] some do mundo.
    for forma in FORMAS:
        adicionar(("objeto_conhecido", "X"), [(forma, "X")],
                  grupo=GRUPO_UNIVERSO)

    # Grupo C — relações, incluindo o fechamento transitivo de `acima_de`, que
    # obriga o motor a iterar mais de um ciclo.
    adicionar(("a_direita_de", "X", "Y"), [("a_esquerda_de", "Y", "X")],
              grupo=GRUPO_RELACOES)
    adicionar(("acima_de", "X", "Y"), [("sobre", "X", "Y")],
              grupo=GRUPO_RELACOES)
    adicionar(("acima_de", "X", "Z"),
              [("sobre", "X", "Y"), ("acima_de", "Y", "Z")],
              grupo=GRUPO_RELACOES)
    adicionar(("similar", "X", "Y"),
              [("forma", "X", "F"), ("forma", "Y", "F")],
              restricoes=[("!=", "X", "Y")], grupo=GRUPO_RELACOES)
    adicionar(("mesma_cor", "X", "Y"), [("cor", "X", "C"), ("cor", "Y", "C")],
              restricoes=[("!=", "X", "Y")], grupo=GRUPO_RELACOES)

    # Grupo D — conceitos compostos, construídos sobre os anteriores.
    adicionar(("destaque", "X"), [("metálico", "X"), ("grande", "X")],
              grupo=GRUPO_COMPOSTOS)
    adicionar(("par_contrastante", "X", "Y"),
              [("similar", "X", "Y"), ("metálico", "X"), ("fosco", "Y")],
              grupo=GRUPO_COMPOSTOS)

    return regras


def imprimir_base_de_conhecimento(regras):
    """Imprime a base agrupada, no formato `cabeça :- corpo.`"""
    grupo_atual = None
    for regra in regras:
        if regra.grupo != grupo_atual:
            grupo_atual = regra.grupo
            print()
            print(f"  -- {grupo_atual} "
                  + "-" * max(0, LARGURA - len(grupo_atual) - 6))
        print(f"  {regra.nome}  {regra.texto()}")


# ===========================================================================
# PARTE 3 — O MUNDO SIMBÓLICO NO ESTILO CLEVR
# ===========================================================================
#
# Cada cena é uma lista de objetos. A geometria é mínima de propósito: um
# objeto ocupa uma COLUNA (da esquerda para a direita) e uma ALTURA nessa
# coluna. Isso basta para as duas famílias de relação que interessam ao
# capítulo: uma ordem total (esquerda/direita) e uma transitiva (sobre/acima).

def sortear(rng, opcoes):
    """Sorteio uniforme e reprodutível de um elemento de uma tupla."""
    return opcoes[int(rng.integers(len(opcoes)))]


def gerar_cena(rng, nome, n_objetos, altura_maxima):
    """Gera uma cena sintética com n_objetos distribuídos em colunas."""
    objetos = []
    coluna, altura = 0, 0
    for i in range(n_objetos):
        if i > 0:
            if altura + 1 < altura_maxima and rng.random() < PROB_EMPILHAR:
                altura += 1                     # empilha sobre o anterior
            else:
                coluna, altura = coluna + 1, 0  # abre uma coluna nova
        objetos.append({"id": f"obj{i + 1}", "coluna": coluna,
                        "altura": altura, "forma": sortear(rng, FORMAS),
                        "cor": sortear(rng, CORES),
                        "tamanho": sortear(rng, TAMANHOS),
                        "material": sortear(rng, MATERIAIS)})
    return {"nome": nome, "objetos": objetos, "tentativas": 1}


def fatos_da_cena(cena):
    """Traduz a cena em proposições fechadas: quatro fatos de atributo por
    objeto, mais sobre(A, B) e a_esquerda_de(A, B)."""
    fatos = []
    for objeto in cena["objetos"]:
        for atributo in ("forma", "cor", "tamanho", "material"):
            fatos.append((atributo, objeto["id"], objeto[atributo]))
    for a in cena["objetos"]:
        for b in cena["objetos"]:
            if a is b:
                continue
            if a["coluna"] == b["coluna"] and a["altura"] == b["altura"] + 1:
                fatos.append(("sobre", a["id"], b["id"]))
            if a["coluna"] < b["coluna"]:
                fatos.append(("a_esquerda_de", a["id"], b["id"]))
    return sorted(set(fatos))


def cena_e_didatica(cena):
    """Filtro de qualidade: exige uma cena em que as consultas da seção [4]
    tenham resposta sem ambiguidade — o mesmo tipo de restrição que os autores
    do CLEVR impuseram ao gerador de cenas deles."""
    objetos = cena["objetos"]
    azuis = [o for o in objetos
             if o["forma"] == "esfera" and o["cor"] == "azul"]
    if len(azuis) != 1:
        return False
    cilindros = [o for o in objetos if o["forma"] == "cilindro"
                 and o["coluna"] > azuis[0]["coluna"]]
    cubo_grande = any(o["forma"] == "cubo" and o["tamanho"] == "grande"
                      and o["material"] == "metálico" for o in objetos)
    formas = [o["forma"] for o in objetos]
    return (len(cilindros) == 1 and cubo_grande
            and max(o["altura"] for o in objetos) >= ALTURA_MAXIMA_PILHA - 1
            and len(set(formas)) < len(formas))


def gerar_cena_principal(rng):
    """Amostragem por rejeição até sair uma cena que passe no filtro."""
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        cena = gerar_cena(rng, "Cena 1", N_OBJETOS_CENA_PRINCIPAL,
                          ALTURA_MAXIMA_PILHA)
        if cena_e_didatica(cena):
            cena["tentativas"] = tentativa
            return cena
    raise RuntimeError("nenhuma cena satisfez o filtro didático")


def montar_cena_da_fragilidade():
    """A cena que quebra o sistema, escrita à mão: quatro objetos dentro do
    vocabulário e um quinto com forma e cor novas. Nada nele é malformado — é
    apenas um objeto sobre o qual ninguém escreveu regra alguma."""
    descricoes = [("esfera", "vermelho", "grande", "metálico"),
                  ("cubo", "cinza", "pequeno", "fosco"),
                  ("esfera", "azul", "pequeno", "metálico"),
                  ("cilindro", "amarelo", "grande", "fosco"),
                  (FORMA_NOVA, COR_NOVA, "grande", "metálico")]
    objetos = [{"id": f"obj{i + 1}", "forma": f, "cor": c, "tamanho": t,
                "material": m, "coluna": i, "altura": 0}
               for i, (f, c, t, m) in enumerate(descricoes)]
    return {"nome": "Cena 4", "objetos": objetos, "tentativas": 1}


def descrever_cena(cena):
    """Uma linha resumindo a geometria da cena."""
    objetos = cena["objetos"]
    n_colunas = len({o["coluna"] for o in objetos})
    altura = max(o["altura"] for o in objetos) + 1
    return (f"{len(objetos)} objetos em {n_colunas} "
            f"{plural(n_colunas, 'coluna', 'colunas')}, pilha de {altura}")


def imprimir_tabela_de_objetos(cena, recuo=4):
    """Imprime a cena como uma tabela legível por humanos."""
    titulo = (f"{'id':<6s}{'forma':<11s}{'cor':<11s}{'tamanho':<10s}"
              f"{'material':<11s}{'coluna':>7s}{'altura':>8s}")
    print(" " * recuo + titulo + "\n" + " " * recuo + "-" * len(titulo))
    for o in cena["objetos"]:
        print(" " * recuo + f"{o['id']:<6s}{o['forma']:<11s}{o['cor']:<11s}"
                            f"{o['tamanho']:<10s}{o['material']:<11s}"
                            f"{o['coluna']:>7d}{o['altura']:>8d}")


# ===========================================================================
# PARTE 4 — O MOTOR DE CONSULTA (DSL NO ESTILO CLEVR)
# ===========================================================================
#
# Os operadores são os mesmos que o capítulo cita ao descrever o NSCL: Filtrar
# (FILTER), Relacionar (RELATE), Contar/Existe (COUNT/EXIST) e Consultar
# (QUERY). Todos operam sobre a base SATURADA: um conceito que não foi
# derivado não existe para a consulta.

def filtrar(indice, conceitos, candidatos=None):
    """FILTER encadeado: objetos que satisfazem todos os conceitos unários.
    Retorna (lista ordenada de objetos, lista ordenada de fatos de apoio)."""
    selecionados, apoios = candidatos, {}
    for conceito in conceitos:
        atuais = {f[1]: f for f in indice.get(conceito, ())
                  if selecionados is None or f[1] in selecionados}
        selecionados = set(atuais)
        for objeto, fato in atuais.items():
            apoios.setdefault(objeto, []).append(fato)
    selecionados = selecionados or set()
    return (sorted(selecionados),
            sorted({f for o in selecionados for f in apoios.get(o, [])}))


def relacionar(indice, relacao, referencias):
    """RELATE: objetos ligados a alguma referência pela relação dada.
    `a_direita_de(X, Y)` lê-se "X está à direita de Y": passando as
    referências em Y, colhemos os X."""
    encontrados = {}
    for fato in indice.get(relacao, ()):
        if fato[2] in referencias:
            encontrados.setdefault(fato[1], []).append(fato)
    return sorted(encontrados), sorted({f for lista in encontrados.values()
                                        for f in lista})


def consultar_atributo(indice, objeto, conceitos_possiveis):
    """QUERY: qual conceito de uma família vale para um objeto. Devolve
    (None, None) quando nenhum se aplica — o caso de um objeto verde."""
    for conceito in conceitos_possiveis:
        for fato in indice.get(conceito, ()):
            if fato[1] == objeto:
                return conceito, fato
    return None, None


# ===========================================================================
# PARTE 5 — O CUSTO COMBINATÓRIO DE COBRIR UM MUNDO À MÃO
# ===========================================================================

def custos_do_mundo(n_formas, n_cores, n_tamanhos, n_materiais,
                    n_relacoes, n_objetos):
    """Contas de guardanapo: descrições completas de objeto (|F|x|C|x|T|x|M|),
    regras de nomeação (|F|+|C|+|T|+|M|), conceitos conjuntivos nomeáveis como
    "cubo metálico grande" ((|F|+1)(|C|+1)(|T|+1)(|M|+1)-1) e proposições
    atômicas por cena (4N de atributo + R x N x (N-1) relacionais)."""
    voc = np.array([n_formas, n_cores, n_tamanhos, n_materiais], np.int64)
    pares = int(n_objetos * (n_objetos - 1))
    return {"vocabulario": voc, "descricoes": int(np.prod(voc)),
            "nomeacao": int(voc.sum()), "conceitos": int(np.prod(voc + 1) - 1),
            "n_objetos": int(n_objetos), "n_relacoes": int(n_relacoes),
            "total_cena": int(len(voc) * n_objetos + n_relacoes * pares)}


def mundos_de_referencia():
    """Os mundos comparados na tabela de explosão combinatória."""
    return [("exercício", custos_do_mundo(
                len(FORMAS), len(CORES), len(TAMANHOS), len(MATERIAIS),
                N_RELACOES_CLEVR, N_OBJETOS_CENA_PRINCIPAL)),
            ("CLEVR real", custos_do_mundo(3, 8, 2, 2, N_RELACOES_CLEVR, 10)),
            ("CLEVR+3 formas",
             custos_do_mundo(6, 8, 2, 2, N_RELACOES_CLEVR, 10)),
            ("cozinha", custos_do_mundo(20, 12, 3, 6, 6, 25)),
            ("mundo aberto", custos_do_mundo(120, 30, 5, 12, 10, 60))]


# ===========================================================================
# PARTE 6 — UTILIDADES DE IMPRESSÃO DO RELATÓRIO
# ===========================================================================

def plural(n, singular, plural_):
    """Concordância de número — evita as formas feias do tipo 'coluna(s)'."""
    return singular if n == 1 else plural_


def formatar_milhar(valor):
    """Formata um inteiro com ponto de milhar, ao gosto brasileiro."""
    return f"{int(round(valor)):,}".replace(",", ".")


def paragrafo(texto, recuo=2):
    """Quebra um texto corrido dentro da largura do relatório."""
    return textwrap.fill(" ".join(texto.split()), width=LARGURA,
                         initial_indent=" " * recuo,
                         subsequent_indent=" " * recuo)


def item_numerado(numero, texto):
    """Um item numerado da seção de interpretação, com recuo pendurado."""
    print()
    print(textwrap.fill(" ".join(texto.split()), width=LARGURA,
                        initial_indent=f"  {numero}. ",
                        subsequent_indent="     "))


def cabecalho(titulo, caractere="="):
    """Imprime um cabeçalho de seção."""
    print(f"\n{caractere * LARGURA}\n {titulo}\n{caractere * LARGURA}")


def subtitulo(titulo):
    """Imprime um separador de subseção."""
    print(f"\n-- {titulo} " + "-" * max(0, LARGURA - len(titulo) - 4))


def imprimir_em_colunas(itens, n_colunas=2, recuo=6, largura_item=32):
    """Imprime uma lista de textos curtos em colunas alinhadas."""
    for i in range(0, len(itens), n_colunas):
        print(" " * recuo + "".join(f"{item:<{largura_item}s}" for item
                                    in itens[i:i + n_colunas]).rstrip())


def imprimir_consulta(numero, pergunta, programa, resposta, apoios,
                      justificativas, limite=4, nota=""):
    """Imprime uma consulta, sua resposta e a prova que a sustenta."""
    print()
    print(f'  Consulta {numero} — "{pergunta}"')
    print(textwrap.fill(f"programa: {programa}", width=LARGURA,
                        initial_indent="    ", subsequent_indent=" " * 14))
    print(f"    resposta: {resposta}")
    if not apoios:
        print("    prova: nenhum fato da base sustenta uma resposta")
        return
    print("    prova:")
    for fato in apoios[:limite]:
        imprimir_prova(fato, justificativas, recuo=6)
    restantes = len(apoios) - limite
    if restantes > 0:
        print(f"      ... e mais {restantes} "
              f"{plural(restantes, 'fato', 'fatos')} de apoio com a mesma "
              f"estrutura")
    if nota:
        print(paragrafo(nota, recuo=4))


# ===========================================================================
# PARTE 7 — AS SEÇÕES DO RELATÓRIO
# ===========================================================================

def imprimir_secao_mundo(cena, fatos, outras, fatos_das_outras):
    """Seção [1]: o mundo simbólico e a cena principal."""
    cabecalho("[1] O MUNDO SIMBÓLICO — CENAS COMO CONJUNTOS DE PROPOSIÇÕES",
              "-")
    for rotulo, valores in (("Formas", FORMAS), ("Cores", CORES),
                            ("Tamanhos", TAMANHOS), ("Materiais", MATERIAIS)):
        print(f"  {rotulo + ' ':.<14s} {', '.join(valores)}")
    print()
    print(paragrafo(
        "Não há imagem, não há pixel, não há tensor: uma cena É o conjunto de "
        "proposições abaixo. É o ponto de partida da IA simbólica — e sua "
        "primeira suposição forte, porque alguém converteu o mundo assim."))
    subtitulo(f"Cena 1, a principal — {descrever_cena(cena)}")
    print()
    imprimir_tabela_de_objetos(cena)
    print()
    print(paragrafo(
        f"Ela foi sorteada com semente fixa e submetida a um filtro de "
        f"qualidade — uma esfera azul única, um cilindro à direita dela, "
        f"um cubo metálico grande, uma pilha de três —, o que exigiu "
        f"{cena['tentativas']} sorteios."))
    atributos = [f for f in fatos if f[0] not in ("sobre", "a_esquerda_de")]
    relacionais = [f for f in fatos if f[0] in ("sobre", "a_esquerda_de")]
    subtitulo(f"Os {len(fatos)} fatos observados desta cena")
    for rotulo, grupo in (("de atributo", atributos),
                          ("relacionais", relacionais)):
        print(f"\n    Fatos {rotulo} ({len(grupo)}):")
        imprimir_em_colunas([formatar_literal(f) for f in grupo])
    subtitulo("As demais cenas do catálogo")
    print()
    for cena_extra, fatos_extra in zip(outras, fatos_das_outras):
        print(f"    {cena_extra['nome']}: {descrever_cena(cena_extra)} "
              f"-> {len(fatos_extra)} fatos observados")


def imprimir_secao_inferencia(base, justificativas, historico, n_ciclos):
    """Seção [3]: o encadeamento para frente até o ponto fixo."""
    cabecalho("[3] O MOTOR DE INFERÊNCIA — ENCADEAMENTO PARA FRENTE", "-")
    print(paragrafo(
        "O motor aplica todas as regras contra tudo o que se sabe; o que for "
        "novo pode disparar outras regras no ciclo seguinte. Quando um ciclo "
        "inteiro não produz nada, a base está saturada — é o ponto fixo."))
    print()
    n_observados = sum(1 for j in justificativas.values() if j is None)
    print(f"    {'ciclo':>7s}{'fatos novos':>14s}{'total na base':>16s}")
    print("    " + "-" * 37)
    for ciclo, novos, total in historico:
        print(f"    {ciclo:>7d}{novos:>14d}{total:>16d}")
    print("    " + "-" * 37)
    print(f"    Ponto fixo confirmado no ciclo {n_ciclos}: nada novo.")
    print()
    print(f"  Fatos observados na cena ....... {n_observados}")
    print(f"  Fatos derivados pelas regras ... {len(base) - n_observados}")
    print(f"  Base saturada .................. {len(base)} proposições")

    subtitulo("A prova de uma conclusão transitiva")
    print()
    print(paragrafo(
        "Ninguém escreveu que o objeto do topo está acima do da base: isso "
        "saiu da regra transitiva R17 aplicada sobre o resultado de R16, e a "
        "cadeia inteira pode ser exibida.", recuo=4))
    print()
    for fato in sorted(base):
        justificativa = justificativas.get(fato)
        if (fato[0] == "acima_de" and justificativa
                and len(justificativa[1]) == 2):
            imprimir_prova(fato, justificativas)
            break


def imprimir_secao_consultas(indice, justificativas):
    """Seção [4]: cinco perguntas no estilo CLEVR, cada uma com sua prova."""
    cabecalho("[4] O MOTOR DE CONSULTA — PERGUNTAS NO ESTILO CLEVR", "-")
    print(paragrafo(
        "Quatro operadores — Filtrar, Relacionar, Contar/Existe e Consultar — "
        "operam sobre a base já saturada, e toda resposta vem com o rastro "
        "lógico que a produziu."))

    esferas, apoios = filtrar(indice, ["esfera"])
    imprimir_consulta(1, "Quantas esferas há na cena?",
                      "Objetos -> Filtrar[esfera] -> Contar",
                      f"{len(esferas)}   ({', '.join(esferas)})",
                      apoios, justificativas, limite=3)

    cubos, apoios = filtrar(indice, ["cubo", "metálico", "grande"])
    imprimir_consulta(2, "Existe algum cubo metálico grande?",
                      "Objetos -> Filtrar[cubo] -> Filtrar[metálico] "
                      "-> Filtrar[grande] -> Existe",
                      f"{'SIM' if cubos else 'NÃO'}   ({', '.join(cubos)})",
                      apoios, justificativas, limite=3)

    referencia, apoio_ref = filtrar(indice, ["esfera", "azul"])
    a_direita, apoio_rel = relacionar(indice, "a_direita_de", set(referencia))
    cilindros, apoio_cil = filtrar(indice, ["cilindro"], set(a_direita))
    alvo = cilindros[0]
    valor, fato_cor = consultar_atributo(indice, alvo, CORES)
    imprimir_consulta(3, "Qual a cor do cilindro à direita da esfera azul?",
                      "Objetos -> Filtrar[esfera] -> Filtrar[azul] "
                      "-> Relacionar[a_direita_de] -> Filtrar[cilindro] "
                      "-> Consultar[cor]",
                      f"{valor}   (o objeto é {alvo})",
                      apoio_ref + [f for f in apoio_rel if f[1] == alvo]
                      + apoio_cil + [fato_cor], justificativas, limite=5)

    pares = sorted(indice.get("similar", ()))
    imprimir_consulta(4, "Há dois objetos com a mesma forma?",
                      "Pares -> Filtrar[similar] -> Existe",
                      f"{'SIM' if pares else 'NÃO'}   "
                      f"({len(pares)} pares ordenados)",
                      pares, justificativas, limite=2)

    objetos, apoios = filtrar(indice, ["objeto_conhecido"])
    imprimir_consulta(5, "Quantos objetos há na cena?",
                      "Objetos -> Filtrar[objeto_conhecido] -> Contar",
                      f"{len(objetos)}   ({', '.join(objetos)})",
                      apoios, justificativas, limite=2,
                      nota="Repare na prova: um objeto só é objeto porque uma "
                           "regra à mão diz que esferas, cilindros e cubos "
                           "são objetos. Correto — desta vez.")


def imprimir_secao_fragilidade(regras, cena, justificativas, indice):
    """Seção [5]: a cena com forma e cor novas — o sistema erra em silêncio."""
    cabecalho("[5] A FRAGILIDADE — UMA FORMA NOVA E UMA COR NOVA", "-")
    print(paragrafo(
        f"A cena abaixo é impecável: cinco objetos bem formados, todos os "
        f"atributos preenchidos. O quinto, porém, é uma {FORMA_NOVA} "
        f"{COR_NOVA} — duas constantes que ninguém previu."))
    print()
    imprimir_tabela_de_objetos(cena)

    objetos = cena["objetos"]
    alvo, referencia = objetos[-1], objetos[-2]
    def conta(campo, valor):
        return str(sum(1 for o in objetos if o[campo] == valor))
    n_objetos = len(filtrar(indice, ["objeto_conhecido"])[0])
    n_metalicos = len(filtrar(indice, ["metálico"])[0])
    valor_cor, _fato = consultar_atributo(indice, alvo["id"], CORES)
    linhas = [
        ("Quantos objetos há na cena?", str(n_objetos), str(len(objetos))),
        ("Quantas esferas há na cena?",
         str(len(filtrar(indice, ["esfera"])[0])), conta("forma", "esfera")),
        (f"Existe algum objeto {COR_NOVA}?",
         "SIM" if indice.get(COR_NOVA) else "NÃO", "SIM"),
        (f"Qual a cor do objeto à direita de {referencia['id']}?",
         valor_cor if valor_cor else "sem resposta", alvo["cor"]),
        ("Quantos objetos metálicos há?", str(n_metalicos),
         conta("material", "metálico")),
        ("Há dois objetos com a mesma forma?",
         "SIM" if indice.get("similar") else "NÃO", "SIM"),
    ]
    subtitulo("O que o sistema responde vs. o que a cena de fato contém")
    print()
    print(f"    {'pergunta':<41s}{'sistema':>12s}{'verdade':>9s}  veredito")
    print("    " + "-" * 72)
    for pergunta, do_sistema, da_verdade in linhas:
        print(f"    {pergunta:<41s}{do_sistema:>12s}{da_verdade:>9s}"
              f"  {'ok' if do_sistema == da_verdade else 'ERRADO'}")
    print("    " + "-" * 72)
    print()
    print(paragrafo(
        f"Repare na incoerência interna: o sistema diz que há {n_metalicos} "
        f"objetos metálicos e que a cena tem {n_objetos} objetos — e um dos "
        f"metálicos não está entre eles. A regra "
        f"`metálico(X) :- material(X, metálico)` continua valendo para a "
        f"{FORMA_NOVA}, porque `metálico` está no vocabulário; as regras que "
        f"ENUMERAM as formas nada têm a dizer sobre ela.", recuo=4))

    orfaos = fatos_orfaos(justificativas)
    subtitulo("Fatos órfãos — proposições sobre as quais nenhuma regra fala")
    print()
    for fato in orfaos:
        print(f"      {formatar_literal(fato)}")
    print()
    print(paragrafo(
        f"São {len(orfaos)} proposições bem formadas que não participaram de "
        f"derivação alguma (na cena principal esse número era 0): a medida "
        f"numérica da ignorância do sistema.", recuo=4))

    subtitulo("A única correção possível: um humano escrever mais regras")
    print()
    novas = [f"{FORMA_NOVA}(X) :- forma(X, {FORMA_NOVA}).",
             f"{COR_NOVA}(X) :- cor(X, {COR_NOVA}).",
             f"objeto_conhecido(X) :- {FORMA_NOVA}(X)."]
    for i, texto in enumerate(novas, start=1):
        print(f"      R{len(regras) + i:02d}  {texto}")
    print()
    print(paragrafo(
        "Três regras a mais para UMA forma e UMA cor novas. Não há treino, "
        "não há adaptação, não há generalização: alguém precisa abrir o "
        "arquivo, entender a base inteira, escrever as regras e revisar os "
        "conceitos compostos que dependem delas. Multiplique pelo número de "
        "coisas que existem no mundo: é a segunda fraqueza fatal.", recuo=4))
    return orfaos


def imprimir_secao_custo(regras, base_principal):
    """Seção [6]: a explosão combinatória de cobrir o mundo à mão."""
    cabecalho("[6] O CUSTO DO CRESCIMENTO — A EXPLOSÃO COMBINATÓRIA", "-")
    print(paragrafo(
        "Com |F| formas, |C| cores, |T| tamanhos e |M| materiais, há "
        "|F|x|C|x|T|x|M| descrições completas de objeto. Cada constante custa "
        "uma regra de nomeação; cada conceito conjuntivo nomeável — 'cubo "
        "metálico grande' — é candidato a virar outra regra; e uma cena de N "
        "objetos com R relações exige 4N + R x N x (N-1) proposições."))
    print()
    print(f"    {'mundo (F/C/T/M)':<27s}{'descrições':>10s}{'nomeação':>9s}"
          f"{'conceitos':>10s}{'N':>3s}{'R':>3s}{'átomos':>9s}")
    print("    " + "-" * 71)
    for nome, custo in mundos_de_referencia():
        f, c, t, m = custo["vocabulario"]
        print(f"    {f'{nome} ({f}/{c}/{t}/{m})':<27s}"
              f"{formatar_milhar(custo['descricoes']):>10s}"
              f"{custo['nomeacao']:>9d}"
              f"{formatar_milhar(custo['conceitos']):>10s}"
              f"{custo['n_objetos']:>3d}{custo['n_relacoes']:>3d}"
              f"{formatar_milhar(custo['total_cena']):>9s}")
    print("    " + "-" * 71)
    custo_clevr = dict(mundos_de_referencia())["CLEVR real"]
    total = custo_clevr["total_cena"] * N_CENAS_CLEVR
    anos = (total * SEGUNDOS_POR_PROPOSICAO
            / (HORAS_UTEIS_POR_DIA * 3600 * DIAS_UTEIS_POR_ANO))
    print()
    print(paragrafo(
        f"A cena principal foi coberta por {len(regras)} regras e chegou a "
        f"{len(base_principal)} proposições. A coluna 'átomos' conta pares "
        f"ORDENADOS: crescem com o QUADRADO do número de objetos — 120 em uma "
        f"cena de 6, 35.400 em uma de 60. E o CLEVR tem "
        f"{formatar_milhar(N_CENAS_CLEVR)} cenas: a "
        f"{custo_clevr['total_cena']} proposições cada, descrevê-lo à mão "
        f"exigiria {formatar_milhar(total)} proposições atômicas — a "
        f"{SEGUNDOS_POR_PROPOSICAO} s cada, {HORAS_UTEIS_POR_DIA} horas "
        f"por dia, {DIAS_UTEIS_POR_ANO} dias por ano, são "
        f"{formatar_milhar(anos)} anos-pessoa de digitação só para DESCREVER "
        f"as cenas, sem uma única regra de raciocínio.", recuo=4))


def imprimir_secao_interpretacao(regras, base_principal, orfaos):
    """Seção [7]: o fecho que amarra o experimento ao capítulo."""
    cabecalho("[7] INTERPRETAÇÃO — O QUE VOCÊ ACABOU DE VER")
    item_numerado(1, f"""
        O SISTEMA FUNCIONA. Com {len(regras)} regras escritas à mão e ZERO
        exemplos de treino, ele respondeu a cinco perguntas no estilo CLEVR e
        provou cada resposta: é a Figura 5.1 do capítulo em execução.
        """)
    item_numerado(2, f"""
        A EXPLICABILIDADE É DE GRAÇA E É TOTAL. Cada uma das
        {len(base_principal)} proposições da base saturada carrega a regra e as
        premissas que a produziram — não é atribuição de importância nem
        aproximação local: é a prova. Compare com a caixa-preta do Capítulo 4.
        """)
    item_numerado(3, f"""
        PRIMEIRA FRAQUEZA FATAL — O PROCESSO É MANUAL. Tudo o que o sistema
        sabe alguém digitou. Bastou uma pirâmide verde para ele contar objetos
        errado, negar uma cor que está bem ali e deixar {len(orfaos)}
        proposições órfãs — e o erro é SILENCIOSO: sem exceção, sem aviso, sem
        incerteza declarada.
        """)
    item_numerado(4, """
        SEGUNDA FRAQUEZA FATAL — NÃO ESCALA. O vocabulário custa uma regra por
        constante, os conceitos conjuntivos crescem como um produto e as
        relações crescem com N x (N-1). Descrever à mão o CLEVR inteiro seria
        trabalho de décadas — e o CLEVR é um mundo de brinquedo.
        """)
    item_numerado(5, """
        É AQUI QUE O CAPÍTULO PERGUNTA: "podemos explorar a IA simbólica e
        melhorar suas limitações?" As duas fraquezas têm a MESMA causa — o ser
        humano no meio do processo —, e o raciocínio simbólico em si não tem
        defeito: é exato, barato e auditável. A saída não é abandonar o
        símbolo, é AUTOMATIZAR A ENTRADA: se alguma coisa OLHASSE a cena e
        produzisse sozinha os fatos forma(obj5, pirâmide) e cor(obj5, verde),
        esta base voltaria a funcionar sem o gargalo humano. Essa coisa é uma
        rede neural — o tema do Exercício 2.
        """)
    print()
    print("=" * LARGURA)
    print(" Fim do Exercício 1.")
    print("=" * LARGURA)


def main():
    """Executa o exercício completo e imprime o relatório didático."""
    cabecalho("EXERCÍCIO 1 — O INGREDIENTE SIMBÓLICO")
    print(" Capítulo 5 — Introdução à IA Neuro-Simbólica (Figura 5.1)")
    print(" Base de conhecimento escrita à mão + motor de inferência.")

    rng = np.random.default_rng(SEMENTE)
    cenas = [gerar_cena_principal(rng)] + [
        gerar_cena(rng, f"Cena {i}", n, ALTURA_MAXIMA_PILHA)
        for i, n in enumerate(N_OBJETOS_CENAS_EXTRAS, start=2)]
    fatos_por_cena = [fatos_da_cena(cena) for cena in cenas]
    imprimir_secao_mundo(cenas[0], fatos_por_cena[0], cenas[1:],
                         fatos_por_cena[1:])

    regras = construir_base_de_conhecimento()
    cabecalho("[2] A BASE DE CONHECIMENTO ESCRITA À MÃO", "-")
    print(paragrafo(
        f"São {len(regras)} cláusulas de Horn, todas digitadas por um humano. "
        f"Repare em quantas existem só para dar NOME às constantes: esse é o "
        f"custo linear de cobrir um mundo à mão."))
    imprimir_base_de_conhecimento(regras)

    base, justificativas, historico, ciclos = encadear_para_frente(
        fatos_por_cena[0], regras)
    imprimir_secao_inferencia(base, justificativas, historico, ciclos)
    imprimir_secao_consultas(indexar(base), justificativas)

    quebrada = montar_cena_da_fragilidade()
    base_q, just_q, _h, _c = encadear_para_frente(fatos_da_cena(quebrada),
                                                  regras)
    orfaos = imprimir_secao_fragilidade(regras, quebrada, just_q,
                                        indexar(base_q))
    imprimir_secao_custo(regras, base)
    imprimir_secao_interpretacao(regras, base, orfaos)


if __name__ == "__main__":
    main()
