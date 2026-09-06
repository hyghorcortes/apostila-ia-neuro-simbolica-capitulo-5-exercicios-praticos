# Exercício Prático 5 — Neural Logic Machine: dedução de primeira ordem feita de tensores

**Capítulo 5 — Apresentando a IA Neuro-Simbólica — o Próximo Nível da IA**
Reproduz computacionalmente a Figura 5.6 (a arquitetura da Neural Logic Machine e seus dois eixos:
*breadth*, a aridade máxima dos predicados, e *depth*, o número de passos de dedução) e **mede**
as duas dimensões em vez de apenas ilustrá-las.

---

## O que este exercício demonstra

Nos exercícios anteriores o motor simbólico era um programa à parte: exato, auditável e **não
treinável**. Aqui ele é dissolvido dentro da rede. Um predicado deixa de ser um símbolo e vira um
**tensor de probabilidades**; a aridade vira o número de eixos de objeto; os quantificadores viram
máximo e mínimo; o encadeamento de regras vira profundidade. Tudo diferenciável, tudo escrito do
zero em numpy — inclusive a retropropagação e o Adam.

| Peça simbólica | O que vira na NLM | Onde aparece na saída |
| --- | --- | --- |
| aridade do predicado | número de eixos de objeto do tensor | seção [1] |
| `EXISTE y` | REDUCE por **máximo** sobre um eixo | seção [2] |
| `PARA TODO y` | REDUCE por **mínimo** sobre um eixo | seção [2] |
| introduzir/inverter variável | EXPAND (replica um eixo) e PERMUTE (transposta) | seção [2] |
| conjunção e disjunção | MLP compartilhado com saída sigmoide | seções [4] e [7] |
| encadeamento de regras | uma camada por passo de dedução (DEPTH) | seção [7] |

O coração do exercício são três medidas, e nenhuma delas é decorativa:

| Medida | Resultado real da execução |
| --- | --- |
| Regra emergente `LIVRE(x) <-> NÃO EXISTE y : SOBRE(y,x)` | concordância **1,0000** em **292 de 292** blocos, sondando exaustivamente os **73** mundos possíveis de 4 blocos |
| A escada da DEPTH (ordenação de vetores) | DEPTH 1 no acaso (0,233 contra 0,200); DEPTH 2 e 3 acertam só `PRIMEIRO(x)`; DEPTH 4 acerta os três |
| Generalização para N maior que o do treino | `PRIMEIRO` e `SEGUNDO` seguem em 1,000 até 20 posições; `TERCEIRO` cai para 0,300 já em 8 |

A escada da DEPTH é o experimento central: **mesma tarefa, mesmos pesos iniciais, mesma semente,
mesmos dados** — a única variável é o número de camadas, e a curva bate com a contagem de passos.

## Como executar

Requisitos: **Python 3** e **numpy**. Nada além disso — não há PyTorch, TensorFlow nem biblioteca
de autodiferenciação. Matmul, ReLU, sigmoide, expand, reduce (com roteamento do gradiente para o
argumento vencedor do máximo), permute, entropia cruzada, softmax e o Adam são implementados à
mão: a seção [3] confere o gradiente analítico contra a diferença finita antes de treinar nada.

```bash
python3 exercicio_pratico_5_neural_logic_machine.py
```

Sem argumentos, sem entrada interativa, sem escrita de arquivos, sem rede. A execução leva cerca
de **13 segundos** e produz **382 linhas**, todas dentro de 78 colunas. A semente é fixa
(`SEMENTE = 42`), portanto **a saída é byte a byte a mesma** em execuções sucessivas.

## O que esperar da saída

O relatório sai no terminal em nove seções:

1. **Representação tensorial dos predicados** — a tabela de tipos: nulário `[lote, C0]`, unário
   `[lote, N, C1]`, binário `[lote, N, N, C2]`, valores em `[0, 1]`. Depois, uma cena de 4 blocos
   escrita como tensores: a matriz `SOBRE(x,y)`, o dado `NO_CHAO(x) = A:0 B:1 C:1 D:0` e os alvos
   `LIVRE(x) = A:1 B:0 C:0 D:1`, `ACIMA(x,y)` e `TODOS_EMPILHADOS() = 1`.
2. **As operações de fiação** — EXPAND leva um unário `[1,3,1]` a um binário `[1,3,3,1]`
   constante em `y`; REDUCE leva `[1,3,3,1]` a `[1,3,2]` concatenando máximo (EXISTE) e mínimo
   (PARA TODO), com a testemunha explícita (para `A`, máximo 0.8 em `y=B`); PERMUTE leva a
   `[1,3,3,2]`, pondo `R(A,B) = 0.8` e `R(B,A) = 0.1` no mesmo ponto.
3. **Verificação numérica dos gradientes** — oito parâmetros sorteados, analítico contra
   numérico. Pior erro relativo: **7,25e-09** na perda binária e **3,39e-09** na softmax.
4. **Tarefa 1 — mundo de blocos (BREADTH 3)** — rede com DEPTH 3, 6 canais por aridade, 12
   neurônios ocultos e **5.313 parâmetros treináveis**, em cenas de 4 blocos com torres de até 3.
   A perda cai de **2,1126** na iteração 1 para **0,0226** na 250.
5. **Generalização para mais objetos do que os do treino** — a rede vê 4 blocos e é avaliada em
   4, 8, 12 e 20, sem retreino e sem mudar um peso:

   | N objetos | LIVRE(x) | ACIMA(x,y) | ACIMA F1 | TODOS_EMPILHADOS() |
   | ---: | ---: | ---: | ---: | ---: |
   | 4 (treino) | 1,000 | 1,000 | 1,000 | 1,000 |
   | 8 | 1,000 | 1,000 | 1,000 | 0,588 |
   | 12 | 1,000 | 1,000 | 1,000 | 0,600 |
   | 20 | 1,000 | 1,000 | 1,000 | 0,458 |

   Duas leituras opostas. `LIVRE` e `ACIMA` generalizam — o F1 está ali porque `ACIMA` é
   desbalanceado e a acurácia sozinha mentiria. `TODOS_EMPILHADOS()` **não**: é nulário e depende
   de um PARA TODO cujo mínimo muda de calibração conforme o número de blocos.
6. **O eixo BREADTH** — a mesma rede com aridade máxima 2, tudo o mais igual. A perda estaciona
   em **0,7396** (contra 0,0226). `LIVRE(x)` sobrevive intacto em **1,000** em todos os tamanhos:
   uma variável quantificada cabe em aridade 2. `ACIMA(x,y)` vira aproximação — F1 **0,943** em 4
   objetos e **0,864** em 20 —, e `TODOS_EMPILHADOS()` desaba de 1,000 para **0,592**. O fecho
   transitivo pede três variáveis simultâneas; um tensor binário não tem eixo para elas.
7. **Tarefa 2 — ordenação de vetor: a escada da DEPTH** — dado `MENOR(x,y)`, deduzir
   `PRIMEIRO(x)`, `SEGUNDO(x)` e `TERCEIRO(x)`, que exigem 2, 4 e 6 passos. A perda é softmax
   sobre as N posições (como binários independentes a rede colapsaria em "responda não sempre"),
   a métrica é o argmax e o acaso em 5 posições vale 0,200:

   | DEPTH | perda final | PRIMEIRO | SEGUNDO | TERCEIRO |
   | ---: | ---: | ---: | ---: | ---: |
   | 1 | 4,1594 | 0,233 | 0,233 | 0,217 |
   | 2 | 2,2005 | **1,000** | 0,283 | 0,300 |
   | 3 | 2,2041 | **1,000** | 0,283 | 0,300 |
   | 4 | 0,0444 | **1,000** | **1,000** | **1,000** |

   DEPTH 1 fica no acaso. DEPTH 2 acerta `PRIMEIRO(x)` e deixa os outros dois no acaso. DEPTH 3
   não muda nada — falta um **par** inteiro de camadas para `SEGUNDO`. Só com DEPTH 4 (3.609
   parâmetros) os três chegam a 1,000: a *Profundidade* da Figura 5.6 sendo medida. Em seguida a
   mesma rede DEPTH 4, treinada com 5 posições, é avaliada em vetores maiores:

   | N posições | acaso | PRIMEIRO(x) | SEGUNDO(x) | TERCEIRO(x) |
   | ---: | ---: | ---: | ---: | ---: |
   | 5 (treino) | 0,200 | 1,000 | 1,000 | 1,000 |
   | 8 | 0,125 | 1,000 | 1,000 | 0,300 |
   | 12 | 0,083 | 1,000 | 1,000 | 0,075 |
   | 20 | 0,050 | 1,000 | 1,000 | 0,083 |

   `TERCEIRO(x)` é o **resultado negativo explícito** do exercício: 1,000 em 5 posições e nível
   de acaso a partir de 8. A causa é a contagem de passos — exige 6, a rede tem 4 —, então o que
   sobra é decorar um atalho válido só em N = 5, onde "o terceiro menor" coincide com "o terceiro
   maior". Subir a profundidade para 6 **não** resolve de graça: a pilha de sigmoides satura.
8. **Extração da regra aprendida** — a rede é sondada com os **73** mundos possíveis de 4 blocos,
   por enumeração exaustiva, e comparada com `NÃO EXISTE y : SOBRE(y,x)`. Nos **136** casos em que
   nada está sobre `x` ela prevê `LIVRE = 1` nas 136 vezes (média **0,9967**); nos **156** casos
   em que há algo sobre `x`, prevê 1 zero vezes (média **0,0030**). Concordância: **1,0000 (292 de
   292 blocos)** — e a regra não foi programada em lugar nenhum, emergiu do REDUCE por máximo.
   Fecha sondando quatro mundos um a um (`A | B | C | D`, `D/B/A | C`, `A/B | D/C`, `B/C/A/D`):
   todo bloco livre recebe 0,997, todo bloco que sustenta alguém recebe 0,003.
9. **Interpretação** — os seis pontos que amarram tudo ao capítulo: generalização em N, DEPTH
   como número de passos de dedução, BREADTH como aridade máxima, o preço da largura (N³ e o
   PERMUTE com 3! = 6 cópias, obrigando lotes menores em 20 objetos), a diferenciabilidade dos
   5.313 parâmetros e a comparação com NSCL e NSDR, que separam percepção de um executor
   simbólico escrito à mão — enquanto a NLM dissolve o executor dentro da rede.

## Sugestões de extensão para o leitor

1. **Suba a escada mais um degrau — e apanhe.** Troque `PROFUNDIDADES_ESCADA = (1, 2, 3, 4)` por
   `(4, 5, 6)` e rode de novo. Pela contagem de passos, `TERCEIRO(x)` precisaria de DEPTH 6 — mas
   a perda estaciona perto do valor inicial, porque a pilha de sigmoides satura. Confirme na
   coluna "perda final" e tente consertar: normalização entre camadas, conexões residuais ou uma
   `TAXA_APRENDIZADO` menor. É um resultado negativo aberto, não um enfeite.

2. **Meça o preço da BREADTH.** Mude `ARIDADE_MAXIMA_BLOCOS` de 3 para 4 e cronometre. O tensor
   de aridade *r* tem N^r posições e o PERMUTE passa de 3! = 6 para 4! = 24 cópias. Anote os
   parâmetros treináveis (eram 5.313), o tempo e as acurácias da seção [5]: o ganho lógico compensa
   o custo, ou a rede só ficou mais cara e mais difícil de treinar?

3. **Force o fecho transitivo a ficar mais longo.** Aumente `ALTURA_MAXIMA_TORRE` de 3 para 5 e
   observe `ACIMA(x,y)`, hoje em F1 = 1,000 em todos os tamanhos. Cada bloco a mais na torre é uma
   composição a mais na cadeia `SOBRE`, e a rede tem `PROFUNDIDADE_BLOCOS = 3` passos para
   percorrê-la. Em que altura o F1 começa a cair, e isso bate com a contagem de passos?

4. **Ataque o predicado que não generaliza.** `TODOS_EMPILHADOS()` cai de 1,000 para 0,458 em 20
   blocos. Amplie `N_TREINO_BLOCOS` de 4 para 6 ou 8, ou treine com tamanhos mistos, e veja se a
   quantificação universal fica calibrada para N maior. Se não ficar, a conclusão da seção [5] se
   fortalece: a fragilidade é do predicado nulário e do PARA TODO, não da falta de dados.
