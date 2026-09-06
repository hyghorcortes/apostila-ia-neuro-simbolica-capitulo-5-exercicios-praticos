# Exercício Prático 1 — O ingrediente simbólico: base de conhecimento e motor de inferência

**Capítulo 5 — Apresentando a IA Neuro-Simbólica — o Próximo Nível da IA**
Reproduz computacionalmente a Figura 5.1 (o arcabouço de uma IA simbólica: uma base de
conhecimento escrita por humanos alimentando um motor de inferência).

---

## O que este exercício demonstra

Este é o **primeiro dos dois ingredientes** da IA Neuro-Simbólica, implementado inteiro e
funcionando. Um mundo 3D no estilo **CLEVR** — esferas, cilindros e cubos, de quatro cores, dois
tamanhos e dois materiais — é descrito não por pixels, mas por **proposições lógicas**:
`forma(obj1, esfera)`, `cor(obj3, azul)`, `a_esquerda_de(obj1, obj2)`, `sobre(obj5, obj4)`.

| Peça | O que é aqui | Custo |
| --- | --- | --- |
| Base de conhecimento | 21 cláusulas de Horn escritas à mão | 100% trabalho humano |
| Motor de inferência | encadeamento para frente até o ponto fixo | 3 ciclos, 64 fatos derivados |
| Motor de consulta | DSL com Filtrar / Relacionar / Contar / Consultar | zero dados de treino |
| Explicação | árvore de derivação completa de cada resposta | de graça, exata, auditável |

Das 21 regras, **11 existem só para dar nome às constantes** (`esfera(X) :- forma(X, esfera).`),
3 dizem o que conta como objeto, 5 são relacionais — inclusive o fechamento transitivo
`acima_de(X, Z) :- sobre(X, Y), acima_de(Y, Z).` — e 2 são conceitos compostos.

A **força** aparece de imediato: sem um único exemplo de treino, o sistema responde a perguntas do
tipo CLEVR e **prova** cada resposta, linha a linha. Não é atribuição de importância nem
aproximação local, como no Capítulo 4: é a prova.

As **duas fraquezas fatais** aparecem logo em seguida:

| Fraqueza | Como o exercício a expõe |
| --- | --- |
| Processo manual | uma cena com uma **pirâmide verde** faz o sistema errar em silêncio |
| Não escala | tabela da explosão combinatória: 40.000.000 de proposições para o CLEVR |

Diante da pirâmide, o sistema não levanta exceção nem declara incerteza: ele responde com a
mesma confiança de sempre — e responde **errado**. A pirâmide simplesmente não existe para ele.

## Como executar

Requisitos: **Python 3** e **numpy**. Nada além disso — não há Prolog, nem biblioteca de lógica,
nem solucionador pronto. A representação dos fatos, a unificação de variáveis, o encadeamento
para frente, o registro das provas e a DSL de consulta são implementados do zero, e é aí que mora
o valor didático: você vê a máquina simbólica por dentro.

```bash
python3 exercicio_pratico_1_base_de_conhecimento_simbolica.py
```

Sem argumentos, sem entrada interativa, sem escrita de arquivos. A execução leva **menos de
1 segundo** e produz **316 linhas**, todas dentro de 78 colunas. A semente aleatória é fixa
(`SEMENTE = 42`), portanto **a saída é sempre a mesma** — inclusive entre máquinas, porque toda
varredura de fatos é feita sobre listas ordenadas.

## O que esperar da saída

O relatório sai no terminal em sete seções:

1. **O mundo simbólico** — o vocabulário (3 formas, 4 cores, 2 tamanhos, 2 materiais) e a cena
   principal: 6 objetos em 4 colunas, com uma pilha de 3. A cena é sorteada e passa por um filtro
   de qualidade (uma esfera azul única, um cilindro à direita dela, um cubo metálico grande, uma
   pilha de três); foram precisos **48 sorteios** até uma cena aprovada — o mesmo tipo de
   restrição que o gerador do CLEVR real enfrenta. Em seguida vêm os **38 fatos observados**
   (24 de atributo e 14 relacionais), listados um a um, mais as cenas 2 e 3 do catálogo, com
   22 e 14 fatos.
2. **A base de conhecimento** — as 21 regras impressas em notação de Horn (`cabeça :- corpo.`),
   agrupadas em nomeação, pertinência, relações e conceitos compostos, numeradas de R01 a R21.
3. **O motor de inferência** — a tabela do ponto fixo: o ciclo 1 deriva **54** fatos novos
   (92 na base), o ciclo 2 deriva mais **10** (102) e o ciclo 3 confirma a saturação sem derivar
   nada. Total: 38 observados + **64 derivados** = **102 proposições**. Logo abaixo, a prova de
   `acima_de(obj6, obj4)`, que ninguém escreveu: ela sai de R17 aplicada sobre o resultado de R16.
4. **O motor de consulta** — cinco perguntas, cada uma com o programa na DSL, a resposta e a prova:
   *3 esferas*; *SIM, existe um cubo metálico grande* (obj2); *a cor do cilindro à direita da
   esfera azul é vermelho* (obj6); *SIM, há objetos com a mesma forma* (8 pares ordenados);
   *6 objetos na cena*.
5. **A fragilidade** — a mesma base recebe uma cena com uma **pirâmide** (forma nova) **verde**
   (cor nova). A tabela compara a resposta do sistema com a verdade da cena:

   | Pergunta | Sistema | Verdade | |
   | --- | --- | --- | --- |
   | Quantos objetos há na cena? | 4 | 5 | **ERRADO** |
   | Quantas esferas há na cena? | 2 | 2 | ok |
   | Existe algum objeto verde? | NÃO | SIM | **ERRADO** |
   | Qual a cor do objeto à direita de obj4? | sem resposta | verde | **ERRADO** |
   | Quantos objetos metálicos há? | 3 | 3 | ok |
   | Há dois objetos com a mesma forma? | SIM | SIM | ok |

   Repare na **incoerência interna**: o sistema afirma que há 3 objetos metálicos e que a cena tem
   4 objetos — e um dos metálicos não está entre eles. `metálico(X) :- material(X, metálico)`
   continua valendo para a pirâmide, porque *metálico* está no vocabulário; já as regras que
   **enumeram** as formas nada têm a dizer sobre ela. A seção lista ainda os **2 fatos órfãos**
   (`cor(obj5, verde)` e `forma(obj5, pirâmide)`), proposições bem formadas que não participaram
   de derivação alguma — na cena principal esse contador era 0 —, e termina com as **3 regras**
   (R22 a R24) que um humano teria de escrever à mão para consertar o caso.
6. **O custo do crescimento** — a tabela da explosão combinatória, do mundo do exercício até um
   mundo aberto:

   | Mundo (F/C/T/M) | Descrições | Nomeação | Conceitos | N | R | Átomos/cena |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | exercício (3/4/2/2) | 48 | 11 | 179 | 6 | 4 | 144 |
   | CLEVR real (3/8/2/2) | 96 | 15 | 323 | 10 | 4 | 400 |
   | CLEVR+3 formas (6/8/2/2) | 192 | 18 | 566 | 10 | 4 | 400 |
   | cozinha (20/12/3/6) | 4.320 | 41 | 7.643 | 25 | 6 | 3.700 |
   | mundo aberto (120/30/5/12) | 216.000 | 167 | 292.577 | 60 | 10 | 35.640 |

   Os átomos relacionais são **pares ordenados**: crescem com N x (N-1), ou seja, com o quadrado
   do número de objetos — 120 numa cena de 6 objetos, 35.400 numa de 60. Como o CLEVR tem 100.000
   cenas a 400 proposições cada, descrevê-lo à mão custaria **40.000.000 de proposições atômicas**
   — a 10 segundos cada, 8 h/dia, 250 dias/ano, **56 anos-pessoa** só para *descrever* as cenas,
   sem escrever uma única regra de raciocínio sobre elas.
7. **Interpretação** — o fecho que amarra tudo à pergunta do capítulo: *podemos explorar a IA
   simbólica e melhorar suas limitações?* As duas fraquezas têm a mesma causa — o ser humano no
   meio do processo — e o raciocínio simbólico em si não tem defeito. A saída não é abandonar o
   símbolo: é **automatizar a entrada**, e isso é o ingrediente neural do Exercício 2.

## Sugestões de extensão para o leitor

1. **Faça o mundo crescer e conte o trabalho.** Acrescente `"roxo"` e `"laranja"` a `CORES` e
   `"cone"` a `FORMAS` no topo do script. Rode de novo: quantas regras a base passou a ter?
   Quantos fatos a mais o motor derivou? Agora repita mentalmente a operação para um robô
   doméstico, que precisa reconhecer alguns milhares de objetos. Em que ponto a base deixa de
   caber na cabeça de quem a mantém?

2. **Meça a explosão relacional na prática.** Aumente `N_OBJETOS_CENA_PRINCIPAL` de 6 para 8, 10 e
   12 (relaxe o filtro em `cena_e_didatica` se nenhuma cena passar) e anote os fatos observados,
   os derivados e o tempo de execução. Compare a curva observada com a previsão N x (N-1) da
   seção [6]. O motor de inferência é exato — mas é barato?

3. **Adicione uma relação nova e veja o efeito em cadeia.** Implemente `atras_de(X, Y)` (por
   exemplo, comparando um novo campo de profundidade em `gerar_cena`) e escreva as regras
   correspondentes, inclusive a inversa `a_frente_de`. Quantas regras foram necessárias? Quais
   conceitos compostos existentes tiveram de ser revisados? É exatamente esse retrabalho que a
   Figura 5.1 esconde atrás da caixa "base de conhecimento".

4. **Tente consertar a pirâmide — e depois desista de propósito.** Acrescente à
   `construir_base_de_conhecimento` as três regras R22–R24 que a seção [5] sugere e confirme que
   as respostas voltam a ficar corretas. Em seguida, mude `FORMA_NOVA` para `"toro"` e
   `COR_NOVA` para `"turquesa"` e rode outra vez: o sistema quebra de novo, exatamente do mesmo
   jeito. Escrever regras corrige um caso, nunca a classe de casos — e é essa constatação que
   leva ao ingrediente neural do próximo exercício.
