# Exercício Prático 3 — A mistura neuro-simbólica: um NSCL simplificado

**Capítulo 5 — Apresentando a IA Neuro-Simbólica — o Próximo Nível da IA**
Reproduz computacionalmente as Figuras 5.3 (os cinco passos da arquitetura híbrida) e 5.4
(a visão geral do NSCL: image parser, base de conhecimento, question parser e executor
quase-simbólico).

---

## O que este exercício demonstra

O Exercício 1 mostrou o ingrediente **simbólico** — uma base de conhecimento escrita à mão, que
raciocina com elegância e quebra diante da primeira forma nova. O Exercício 2 mostrou o
ingrediente **neural** — uma rede que aprende dos pixels, não precisa de ninguém escrevendo
regras, e não sabe raciocinar. Aqui os dois são **misturados**, na escala de uma sala de aula,
seguindo os sub-modelos do NSCL:

| Sub-modelo (Figura 5.4) | O que é aqui | Quem escreveu |
| --- | --- | --- |
| Image parser | flood fill de 4-vizinhança + MLP 196 → 48 → 4 cabeças softmax | aprendido dos pixels |
| Base de conhecimento | fatos probabilísticos: `forma(obj1, cubo) = 1.000` | ninguém — saiu da rede |
| Question parser | saco de 376 n-gramas + etiquetador de janela (457 características) | aprendido de 1.727 pares |
| Executor quase-simbólico | máscaras suaves em [0,1]: produto, soma, máximo | **zero** parâmetros treináveis |

A base que o Exercício 1 exigia digitada fato a fato é agora **produzida pela rede**; o
raciocínio que o Exercício 2 não sabia fazer é agora **executado sobre ela**. O script põe esse
híbrido contra uma rede **ponta a ponta** (mesmos pixels, mesma pergunta, uma resposta) em três
comparações, todas com os números da execução real:

| Comparação | Híbrido | Ponta a ponta |
| --- | ---: | ---: |
| Eficiência de dados — 25 cenas, 84 perguntas | **62,2%** | 44,3% |
| Eficiência de dados — 500 cenas, 1.727 perguntas | **99,3%** | 55,0% |
| Composicional — CONSULTA x metal (retirada do treino) | **99,0%** | 48,5% |
| Composicional — EXISTENCIA x pequeno (retirada do treino) | **99,2%** | 53,3% |
| Composicional — CONTAGEM x cilindro (retirada do treino) | 29,2% | 22,5% |
| Explicabilidade de uma resposta | 4 itens auditáveis | 1 vetor de 19 números |

A linha **CONTAGEM x cilindro é um resultado negativo**, e o relatório a trata como tal: ali o
híbrido *também* falha. O script diagnostica a causa provável com números, não com retórica —
as 120 perguntas dessa combinação usam a palavra `cilindros`, que sumiu por inteiro do treino
reduzido (zero ocorrências nas 1.477 perguntas restantes). Compor resolve a **estrutura** da
pergunta; não adivinha o sentido de uma palavra que o modelo nunca leu.

## Como executar

Requisitos: **Python 3** e **numpy**. Nada além disso — a segmentação por componentes conexas, o
MLP multi-cabeça, os modelos lineares do parser, o executor da DSL e a rede ponta a ponta são
escritos do zero, com gradientes à mão.

```bash
python3 exercicio_pratico_3_nscl_simplificado.py
```

Sem argumentos, sem entrada interativa, sem escrita de arquivos, sem rede. A execução leva cerca
de **10 segundos** numa máquina ociosa — quase tudo é treino — e produz **280 linhas**. A semente
é fixa (`SEMENTE = 42`) e todo modelo recebe um gerador derivado dela, portanto **duas execuções
seguidas produzem saídas byte a byte idênticas**. Nenhuma linha do relatório passa de 78 colunas.

## O que esperar da saída

O relatório sai no terminal em nove seções:

1. **O mundo de cenas** — um CLEVR de bolso: **500 cenas de treino** e **120 de teste** numa
   grade de **19 x 40 x 4 canais**, com `forma(3), cor(4), tamanho(2), material(2)`, das quais
   saem **1.727 perguntas de treino** e **418 de teste**, mais **240 perguntas com fraseado
   novo**. Em seguida, a cena da Figura 5.4 desenhada em ASCII e segmentada por flood fill em
   três componentes: **49 pixels** em (5,14) — o cubo azul grande fosco —, **49 pixels** em
   (15,14) — o cubo vermelho grande de metal — e **13 pixels** em (25,4) — a esfera verde
   pequena de metal.
2. **Image parser** — o MLP de **196 entradas** (o recorte 7x7x4 mascarado) e **48 neurônios
   ocultos**, treinado em três lições. A tabela do currículo mostra o efeito de ensinar na
   ordem certa: a Lição 1 vê **92 cenas / 92 objetos** e chega a **96,6%** em forma e **99,7%**
   em cor, com tamanho (**41,8%**) e material (**51,9%**) ainda no chute, porque entram como
   rótulo ausente; a Lição 2 (**218 cenas / 557 objetos**) já leva os quatro atributos a
   **100,0% / 99,7% / 100,0% / 97,4%**; a Lição 3 (**190 cenas / 861 objetos**), com cenas
   cheias, fecha em **100,0% / 99,7% / 100,0% / 98,2%**.
3. **A base de conhecimento** — os **22 fatos** que a rede produziu sobre a cena da figura: 12
   de atributo, entre `0.998` e `1.000`, e 10 relacionais, todos `1.0`. As relações valem
   exatamente 1 porque saem da geometria dos centróides; os atributos carregam a incerteza da
   rede. É uma base **probabilística** — e é por isso que ela exige outro motor de inferência.
4. **Question parser** — **376 n-gramas** no saco, **457 características** de janela
   (palavra x posição em [-2,-1,0,1,2,3]) e **1.727 pares** (pergunta, programa). O programa
   sai **exatamente** correto em **99,8%** das 418 perguntas de fraseado visto e em **95,8%**
   das 240 de fraseado novo. Cinco perguntas aparecem traduzidas, entre elas *"o objeto azul tem
   a mesma forma do cubo vermelho à sua direita?"* → `SCENE -> FILTER(vermelho, cubo) ->
   RELATE(direita) -> FILTER(azul) -> AEQUERY(forma)`.
5. **Executor quase-simbólico** — o **rastro lógico** dessa pergunta: seis linhas (0 a 5), cada
   uma a máscara suave de cada objeto. `FILTER(vermelho)` isola obj2, `FILTER(cubo)` o mantém em
   0.999, `RELATE(direita)` transporta a máscara para obj1 e `AEQUERY(forma)` devolve
   **0.999 → SIM** — a mesma resposta do oráculo simbólico sobre a cena verdadeira.
6. **Comparação 1 — eficiência de dados** — cinco tamanhos de treino, do menor ao maior:

   | cenas | perguntas | programa ok | HÍBRIDO | PONTA A PONTA | dif. (p.p.) |
   | ---: | ---: | ---: | ---: | ---: | ---: |
   | 25 | 84 | 25,8% | 62,2% | 44,3% | **+17,9** |
   | 50 | 171 | 56,9% | 80,6% | 44,5% | **+36,1** |
   | 100 | 344 | 89,0% | 95,2% | 52,2% | **+43,1** |
   | 250 | 852 | 97,1% | 98,3% | 56,5% | **+41,9** |
   | 500 | 1.727 | 99,8% | 99,3% | 55,0% | **+44,3** |

   O híbrido vence em toda a curva, de **+17,9** a **+44,3 pontos percentuais**. Com 100 cenas
   ele já está em 95,2%; a rede ponta a ponta, com 500, não passa de 55,0%.
7. **Comparação 2 — generalização composicional** — três pares (tipo de pergunta x valor de
   atributo) são retirados do treino, que cai de **1.727 para 1.477 perguntas**. Duas
   combinações inéditas funcionam (**99,0%** e **99,2%**, contra 48,5% e 53,3% da rede) e as
   combinações vistas ficam em **99,6%** contra **62,4%**. A terceira, **CONTAGEM x cilindro**,
   **falha nos dois sistemas**: 29,2% do híbrido contra 22,5% da rede. O parágrafo seguinte
   registra o fracasso e sua causa provável, medida no próprio script: **120 das 120 perguntas**
   dessa combinação trazem a palavra `cilindros`, ausente das 1.477 perguntas restantes,
   enquanto nas outras duas combinações esse contador é **0**.
8. **Comparação 3 — explicabilidade** — a pergunta *"qual é o material do objeto que tem o
   objeto amarelo à sua esquerda?"*, cuja resposta verdadeira é **FOSCO**. O sistema
   neuro-simbólico responde FOSCO e mostra **quatro itens**: o programa executado, o extrato da
   base que ele leu, a máscara final objeto por objeto (`obj5=1.000`) e a resposta. A rede ponta
   a ponta responde **METAL** — errado — e só sabe oferecer a distribuição de saída
   (`P(metal) = 0.948`, `P(fosco) = 0.051`). Não há terceiro item: é a caixa-preta do
   Capítulo 4 de novo, agora em raciocínio visual.
9. **Interpretação** — seis pontos que amarram tudo: os cinco passos da Figura 5.3 rodaram
   inteiros; cada paradigma cobriu a fraqueza do outro (a Tabela 5.1 virando código); a
   eficiência de dados vem de conhecimento embutido no executor; a composicionalidade tem
   limite, e o limite apareceu; a explicabilidade veio de graça; e ficam as ressalvas — 11
   conceitos, relações geométricas e um perceptor supervisionado por atributos, enquanto o
   NSCL real aprende os conceitos só com pares (pergunta, resposta).

## Sugestões de extensão para o leitor

1. **Desmonte o currículo.** `EPOCAS_LICOES = (10, 12, 12)` distribui as épocas entre as três
   lições. Troque por `(0, 0, 34)` — mesmo orçamento total, mas sem currículo: o perceptor vê
   direto as cenas cheias, em que vizinhos invadem a janela de recorte. As acurácias da tabela
   da seção [2] sobrevivem? E se você inverter para `(12, 12, 10)`, dando mais tempo ao fácil?

2. **Ache o ponto em que o híbrido também quebra.** `TAMANHOS_DE_TREINO = (25, 50, 100, 250,
   500)` define a curva da seção [6]. Acrescente 5 e 10 no começo da tupla e observe a coluna
   *programa ok*: a vantagem do híbrido depende de o parser semântico acertar o programa, e
   abaixo de certo número de frases ele deixa de acertar. Em que tamanho a diferença de
   +17,9 p.p. vira zero — ou negativa?

3. **Teste a explicação do resultado negativo.** `COMBINACOES_RETIDAS` guarda os três pares
   retirados do treino. A hipótese do script é que CONTAGEM x cilindro falhou porque levou
   junto a palavra `cilindros`. Troque `("CONTAGEM", "cilindro")` por `("CONSULTA", "cilindro")`
   — em perguntas de consulta a palavra aparece no singular, que continua no treino — e depois
   por `("CONTAGEM", "cubo")`, que deve levar embora `cubos`. Se a hipótese estiver certa, a
   falha acompanha o plural, não o tipo de pergunta.

4. **Suje a imagem até a base ficar incerta.** `RUIDO_PIXEL = 0.22` é o desvio do ruído
   gaussiano por pixel do objeto. Suba para 0.40 e 0.60 e acompanhe três coisas ao mesmo tempo:
   os fatos da seção [3], que deixam de valer 1.000; o rastro da seção [5], em que as máscaras
   suaves param de ser quase binárias; e a resposta final. Onde exatamente a incerteza da
   percepção passa a contaminar a conclusão do raciocínio?
