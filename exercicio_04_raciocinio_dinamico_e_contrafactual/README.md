# Exercício Prático 4 — Raciocínio dinâmico e inferência contrafactual (NSDR)

**Capítulo 5 — Apresentando a IA Neuro-Simbólica — o Próximo Nível da IA**
Reproduz computacionalmente a Figura 5.5 (o arcabouço do NSDR: video parser, dynamics
predictor, question parser e executor simbólico), fechando o arco que começou na Figura 5.1.

---

## O que este exercício demonstra

O NSCL do Exercício 3 olhava para uma **imagem parada** e descrevia relações espaciais. O NSDR
olha para um **vídeo** e responde perguntas sobre **causalidade**. A diferença arquitetural é
uma peça só — o **Dynamics Predictor**, uma rede de relações no estilo PropNet — e é ela que
transforma um sistema que descreve em um sistema que tenta prever, explicar e imaginar.

O mundo é um CLEVRER sintético: 3 a 5 discos coloridos numa arena de 7,0 x 4,5, 25 quadros de
0,2 s, colisões elásticas de massas iguais, e um **campo de visão menor que a arena** — daí saem
os eventos ENTRA e SAI, e a oclusão.

| Peça | O que é aqui | Papel na Figura 5.5 |
| --- | --- | --- |
| Video parser | detector com ruído de 0,008 por eixo, 775 quadros processados | percepção quadro a quadro |
| Dynamics predictor | rede de relações, **2.378 parâmetros**, Adam, 60 épocas | o motor de física aprendido |
| Executor simbólico | DSL de 12 operações sobre listas discretas de eventos | raciocínio 100% simbólico |
| Verdade de referência | o simulador exato, re-rodado sem o disco cinza | auditoria do contrafactual |

O que **funciona**, e está medido:

| Resultado | Número da execução |
| --- | --- |
| Física aprendida vence a inércia em 24 passos | erro **2,647** contra **7,323** |
| Rastreamento sob oclusão (11 quadros sem parser) | erro **0,317** no reaparecimento |
| Pergunta preditiva do clipe de demonstração | acertou `COLIDE(amarelo, verde) @ 13` |
| Explicação contrafactual | achou o quadro de divergência (**5**) e o objeto desviado |

O que **não funciona** — e o relatório diz isso com todas as letras na seção [6]: das **76**
colisões que a física verdadeira produz nos mundos sem o disco cinza, o preditor reencontra
**15** (revocação **19,7%**), deixa passar **61** e inventa **21**. Nos eventos de campo o placar
é **64** acertos, **196** perdidos e **56** falsos (revocação **24,6%**). Este é um **resultado
negativo**, e ele está no exercício de propósito: o mundo contrafactual é imaginado por **24
passos em malha aberta**, sem uma única observação para corrigir o rumo, e o erro de Δv de cada
passo é integrado na posição. Como eventos são **discretos**, um desvio menor que um raio de
disco já decide entre "colidiu" e "não colidiu" — e uma colisão fora de lugar troca velocidades
na hora errada e reescreve todo o resto do vídeo imaginado.

O mecanismo da Figura 5.5 é o que sobrevive ao teste; a exatidão do resultado, não.

## Como executar

Requisitos: **Python 3** e **numpy**. Nada além disso — a física, a detecção de eventos, a rede
de relações, o otimizador Adam com retropropagação escrita à mão e o executor simbólico são
implementados do zero.

```bash
python3 exercicio_pratico_4_raciocinio_dinamico_e_contrafactual.py
```

Sem argumentos, sem entrada interativa, sem escrita de arquivos. A execução leva cerca de
**14 segundos** e produz **473 linhas**, todas dentro de 78 colunas. A semente é fixa
(`SEMENTE = 42`), portanto duas execuções seguidas são byte a byte idênticas.

## O que esperar da saída

O relatório sai no terminal em sete seções:

1. **O mundo** — a configuração do CLEVRER sintético (25 quadros x 0,2 s = 5,0 s; arena
   [0,0, 7,0] x [0,0, 4,5]; campo de visão [0,9, 6,1] x [0,6, 3,9]) e o clipe de demonstração:
   **5 discos** (verde, amarelo, cinza, vermelho, azul) e a lista verdadeira de **25 eventos**
   registrada pelo simulador, de `COLIDE(amarelo, azul) @ quadro 02` a
   `SAI(vermelho) @ quadro 22`.
2. **Video parser** — o extrator aplicado a cada quadro, não ao vídeo: **775** quadros
   processados, **2.506** detecções e **544** objetos ocultos. Vem o mapa de visibilidade do
   clipe (o disco azul, por exemplo, some do quadro 16 em diante) e o erro médio de localização,
   **0,0100** unidade.
3. **Dynamics predictor** — a rede de relações: `f_relacao` 9→24→24, `f_objeto` 12→24, `f_saida`
   48→24→2, **2.378** parâmetros treináveis. O treino usa **8.208** transições de 90 clipes, das
   quais só **1.492** (18,2%) têm impulso — daí o reforço de 3 cópias extras. A perda cai de
   **0,85313** (época 1) para **0,06719** (época 60).
4. **Avaliação do motor de física** — a tabela de horizontes (rede contra inércia): 0,0559 x
   0,0791 em 1 passo; 0,3775 x 0,6258 em 5; 1,1503 x 2,3517 em 10; **2,6470 x 7,3227** em 24
   passos, contra uma diagonal de arena de 8,32 (31,8% dela). Depois, a rolagem sob oclusão do
   disco amarelo do clipe #7, que fica **11 quadros** invisível e reaparece a **0,317** unidade
   da posição real. Por fim, as colisões futuras previstas a partir do quadro 12: **24** acertos,
   **16** falsos, **36** perdidos — precisão **60,0%**, revocação **40,0%**, com a leitura honesta
   logo abaixo (6 em cada 10 colisões futuras passam batido).
5. **Executor simbólico** — as quatro famílias de pergunta, cada uma com o rastro passo a passo
   do programa funcional. *Descritiva*: **3** colisões até o quadro 12, e o parceiro do cinza é
   **amarelo**. *Preditiva*: `COLIDE(amarelo, verde) @ quadro 13`, que bate com a física
   verdadeira. *Contrafactual*: o programa da Figura 5.5 (`OBJETOS -> FILTRAR_COR(cinza) ->
   OBTER_CONTRAFACTUAIS -> PERTENCE_A`) responde as 4 alternativas — e o texto avisa que **3 dos
   4 acertos são respostas negativas**, que são baratas. *Explicativa*: a divergência das
   trajetórias começa no **quadro 5**, quando o cinza bate no amarelo; a distância mínima entre
   amarelo e vermelho é **0,907** no mundo real (folga de 0,112 sobre os 0,796 de soma de raios)
   contra **1,329** no mundo sem o cinza.
6. **Validação do contrafactual** — a seção que **falha**, e a mais importante do exercício.
   **26** dos 30 clipes de teste têm disco cinza; para cada um, dois mundos alternativos são
   produzidos e comparados. A tabela por clipe mostra o estrago (clipe 0: **1** acerto, **2**
   falsas, **6** perdidas), e o agregado confirma: COLIDE **15/21/61** (41,7% / 19,7%), ENTRA-SAI
   **64/56/196** (53,3% / 24,6%), erro final médio de **2,492** unidade. Seguem quatro parágrafos
   explicando por que o erro se acumula em rolagens longas, e o lado a lado dos dois mundos do
   clipe de demonstração: **6** eventos imaginados contra **12** eventos verdadeiros.
7. **Interpretação** — os seis pontos de fecho, do "NSDR = NSCL + motor de física" ao
   reconhecimento explícito de que o contrafactual, medido, entrega **42%** de precisão e **20%**
   de revocação. O exercício demonstra o **mecanismo** do raciocínio dinâmico; o motor que o move
   é de brinquedo, com alcance útil de poucos quadros.

## Sugestões de extensão para o leitor

1. **Encurte o horizonte e veja o contrafactual melhorar.** Reduza `N_QUADROS` de 25 para 15 e
   depois para 10 (ajuste `QUADRO_ATUAL` para uns 40% disso). A rolagem contrafactual da seção
   [6] passa a ter menos passos em malha aberta: a revocação de COLIDE tem de subir dos atuais
   19,7%. Se não subir, a hipótese do relatório está errada e a causa é outra — anote qual.

2. **Dê mais rede ao problema.** Aumente `UNIDADE_OCULTA` de 24 para 48 ou 64 e `EPOCAS` de 60
   para 150. A perda final (0,06719) cai bastante, mas acompanhe o que interessa: o erro de 24
   passos (2,647) e a revocação da seção [6]. Quanta capacidade é preciso comprar para ganhar um
   ponto de revocação? Essa razão é a pergunta central de quem constrói NSDR de verdade.

3. **Mude a dieta de treino, não o modelo.** Suba `N_CLIPES_COLISAO` de 50 para 150 e
   `FATOR_REFORCO_COLISAO` de 3 para 6: mais exemplos do evento raro que é uma colisão. Compare
   o erro de 1 passo (0,0559) com o de 24 passos. Melhorar o passo curto melhora o longo na mesma
   proporção — ou o erro composto engole o ganho?

4. **Ataque a fronteira discreta.** Aumente `TOLERANCIA_QUADROS` de 2 para 4 e `LIMIAR_CONTATO`
   de 0,30 para 0,50, e rode de novo. Os placares das seções [4] e [6] melhoram — mas o que você
   comprou foi raciocínio melhor ou um critério mais frouxo? Depois faça o contrário
   (`TOLERANCIA_QUADROS = 1`) e veja quanto do resultado positivo do exercício sobrevive. É esta
   a lição do capítulo: quando o símbolo se apoia num substrato aprendido, a definição do evento
   passa a fazer parte do resultado.
