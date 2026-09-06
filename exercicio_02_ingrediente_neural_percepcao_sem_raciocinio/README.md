# Exercício Prático 2 — O ingrediente neural: percepção sem raciocínio

**Capítulo 5 — Introdução à IA Neuro-Simbólica — o Próximo Nível da IA**
Reproduz computacionalmente a Figura 5.2 (uma rede neural densa, cada neurônio ligado a todos os
da camada anterior) e prepara as Figuras 5.4 e 5.5, retomadas no Exercício 3.

---

## O que este exercício demonstra

O Exercício 1 mostrou o ingrediente simbólico: regras escritas à mão, exatas, auditáveis — e que
quebram diante de qualquer forma nova. Aqui entra o **segundo ingrediente**. Um perceptron
multicamada escrito do zero em numpy — camadas densas, ReLU, softmax, entropia cruzada,
retropropagação e SGD com momento — percebe e raciocina com a **mesma arquitetura**.

| Tarefa | Entrada | Resultado com 5.000 exemplos |
| --- | --- | --- |
| Perceber um objeto | raster 16x16x3 = 768 números | **99,5%** de média em 4 atributos |
| Responder sobre a cena | 12x48x3 + pergunta = 1.751 números | **54,4%**, contra 50,2% do chute cego |

A **força** é imediata: forma 98,1%, cor 100,0%, tamanho 100,0% e material 100,0%, sem que
ninguém escrevesse uma regra sobre círculos, brilhos ou pixels — exatamente o que faltava ao
Exercício 1. As **fraquezas** vêm logo atrás, e são três, medidas separadamente:

| Fraqueza | Como o exercício a expõe |
| --- | --- |
| Fome de dados | 10 exemplos rendem 66,2%; foram precisos 250 para cruzar 90% |
| Percebe mas não relaciona | saldo sobre o chute cego: **+24,0** pontos em CONSULTAR, mas −3,8 em CONTAR e −6,2 em MESMA_FORMA |
| Não compõe conceitos | 79,1% nas combinações vistas, **25,3%** nas inéditas — o próprio acaso |

A terceira é o coração do exercício, e foi montada de propósito sobre `CONSULTAR(atributo,
posição)` — a única família de perguntas que a rede **domina**. Medida em `CONTAR`, a queda não
provaria nada: ali a rede já falha no que viu. Ela viu `cor` 4.783 vezes e `posição 3` 2.444
vezes, e `CONSULTAR(cor, obj 3)` a derruba de 99,3% (no controle `CONSULTAR(cor, obj 0)`) para
26,2% — contra 25,0% de quem responde uma cor ao acaso.

Antes de tudo isso, a seção [2] **prova que a retropropagação está certa**, por diferenças
finitas centradas. Não é ornamento: uma versão anterior deste exercício tinha entrada não
padronizada e taxa alta demais, e produzia uma curva de aprendizado que **descia** com mais
dados.

## Como executar

Requisitos: **Python 3** e **numpy**. Nada além disso — sem PyTorch, TensorFlow, scikit-learn,
scipy, PIL ou matplotlib. As "imagens" saem de três normas do plano (euclidiana para o disco, do
máximo para o quadrado, de Manhattan para o losango) e os gráficos são barras de asteriscos.

```bash
python3 exercicio_pratico_2_ingrediente_neural_percepcao_sem_raciocinio.py
```

Sem argumentos, sem entrada interativa, sem escrita de arquivos, sem rede. A execução leva cerca
de **50 segundos** e imprime 439 linhas. A semente é fixa (`SEMENTE = 42`) e a álgebra linear
roda em uma única thread, portanto **a saída é byte a byte idêntica entre execuções**.

## O que esperar da saída

O relatório sai no terminal em oito seções:

1. **O mundo sintético** — o vocabulário (3 formas, 4 cores, 2 tamanhos, 2 materiais), os dois
   rasters (objeto isolado 16x16x3 = **768 entradas**; cena 12x48x3 = **1.728 entradas**) e o
   ruído gaussiano de desvio 0,045. Quatro objetos e uma cena de quatro células saem desenhados
   em ASCII, com a descrição simbólica que a rede **nunca recebe**.
2. **Validação** — o teste de gradiente. Numa rede minúscula (12 → 9 → 6 → [4, 3], 7 exemplos,
   float64), **112 pesos** de todas as camadas têm sua derivada comparada com a diferença
   centrada de passo 1e-05: erro médio entre 8,9e-11 e 4,1e-08 por camada e **pior caso
   9,81e-07**, dez vezes abaixo do critério usual de 1e-5. APROVADO.
3. **O perceptor de atributos** — arquitetura 768 → 64 → 48 → [3, 4, 2, 2], **52.875 parâmetros**,
   dos quais **49.152** são as conexões da primeira camada (as arestas da Figura 5.2). Com 5.000
   exemplos e 25 épocas: forma **98,1%** (acaso 33%), cor, tamanho e material **100,0%**, média
   **99,5%**. Saem impressos seis objetos de teste, todos com os quatro atributos corretos.
4. **A fome de dados** — a curva de aprendizado, com o **mesmo** número de épocas em todos os
   pontos e um piso de 300 atualizações que só favorece os conjuntos pequenos:

   | n treino | 10 | 25 | 50 | 100 | 250 | 500 | 1.000 | 2.500 | 5.000 |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | média | 66,2% | 75,4% | 82,5% | 87,6% | 93,4% | 96,0% | 97,5% | 99,2% | 99,5% |
   | forma | 32,7% | 41,5% | 51,7% | 60,8% | 76,3% | 85,6% | 90,3% | 96,9% | 98,1% |

   A curva **sobe nos oito degraus, sem uma única queda**, e cruza 90% de média com 250 exemplos.
   O gargalo tem nome: cor, tamanho e material passam de 95% com 250; a **forma**, a única que
   exige olhar o contorno inteiro, só passa com 2.500.
5. **A tarefa de raciocínio** — quatro tipos de pergunta em duas naturezas: de **percepção**,
   `CONSULTAR(atributo, pos)`; de **raciocínio**, `CONTAR(forma)`, `EXISTE(cor, forma)` e
   `MESMA_FORMA(i, j)`. A resposta sai de um vocabulário único de **14 rótulos**; a pergunta chega
   já analisada, em 23 números de amplitude 6,0 (para não se perder entre 1.728 pixels).
6. **Fracasso 1** — as duas curvas lado a lado: com 5.000 exemplos a percepção chega a 99,5% e o
   raciocínio a 54,4%, **+4,1 pontos** sobre o chutador cego (50,2%). A abertura por tipo mostra
   de onde vem o saldo:

   | tipo | natureza | rede | cego | saldo |
   | --- | --- | ---: | ---: | ---: |
   | CONTAR | raciocínio | 39,9% | 43,7% | **−3,8** |
   | EXISTE | raciocínio | 65,3% | 62,6% | +2,7 |
   | MESMA_FORMA | raciocínio | 57,0% | 63,2% | **−6,2** |
   | CONSULTAR | percepção | 53,8% | 29,8% | **+24,0** |

   A rede supera o piso cego em EXISTE e CONSULTAR; fica no piso ou **abaixo** dele em CONTAR e
   MESMA_FORMA. A única vitória com folga é a pergunta que se resolve **lendo** um objeto: a rede
   não é ruim de visão, é ruim de raciocínio.
7. **Fracasso 2** — o experimento decisivo, com 12.000 perguntas `CONSULTAR` sobre cenas de 4
   objetos, das quais três combinações foram removidas. Cada peça isolada apareceu milhares de
   vezes (`forma` 7.217, `cor` 4.783, `posição 0` 4.689, `posição 1` 2.443, `posição 2` 2.424,
   `posição 3` 2.444); cada **combinação** retida, zero vezes.

   | combinação | situação | rede | acaso | formato |
   | --- | --- | ---: | ---: | ---: |
   | CONSULTAR(cor, obj 0) | vista | 99,3% | 25,0% | 100,0% |
   | CONSULTAR(forma, obj 3) | vista | 69,2% | 33,3% | 100,0% |
   | CONSULTAR(forma, obj 2) | vista | 68,8% | 33,3% | 100,0% |
   | CONSULTAR(cor, obj 3) | **INÉDITA** | 26,2% | 25,0% | 98,0% |
   | CONSULTAR(forma, obj 1) | **INÉDITA** | 27,7% | 33,3% | 83,8% |
   | CONSULTAR(cor, obj 2) | **INÉDITA** | 22,0% | 25,0% | 97,8% |

   No agregado: **79,1%** nas vistas contra **25,3%** nas inéditas — **53,8 pontos** de queda
   sobre as **mesmas imagens**, com o mesmo tipo de pergunta e o mesmo formato de resposta. Nas
   inéditas a rede fica entre −5,7 e +1,2 pontos do acaso: não é queda de desempenho, é o
   desaparecimento da competência. A coluna *formato* fecha o argumento — em 93% dos casos
   inéditos ela devolve resposta do **tipo certo**: entendeu o atributo e que era leitura, e
   ainda assim não leu a posição pedida.
8. **Interpretação** — sete itens, do item 0 (o gradiente está certo, logo o fracasso é da
   tarefa) até a ponte para o Exercício 3: o perceptor da seção [3] **já é** um extrator de
   símbolos com 99,5% de acerto — falta entregá-los a um motor de inferência, como o NSCL faz.

## Sugestões de extensão para o leitor

1. **Reproduza o bug original — e conserte-o de novo.** Mude `TAXA_APRENDIZADO` de `0.02` para
   `0.05`. O passo efetivo do momento salta de 0,20 para 0,50, o treino diverge em parte dos
   pontos e a curva **perde a monotonia**: a média vai a 42,5% com 10 exemplos, sobe a 92,4% com
   250, cai a 70,2% com 500 e termina em 84,7% com 5.000 — abaixo do que conseguia com 250; a
   forma bate no acaso (33,8%) em dois pontos. Reponha 0,02 e troque `MEDIA_DO_PIXEL` por `0.0`
   e `DESVIO_DO_PIXEL` por `1.0`: qual dos dois defeitos é mais difícil de flagrar na tabela?

2. **Falsifique a curva de aprendizado.** Faça `EPOCAS_PERCEPTOR` valer `10` e `PASSOS_MINIMOS`
   valer `6000`: os pequenos treinarão centenas de épocas e os grandes, dez — o orçamento
   invertido da versão quebrada. A curva desce? Onde? Ela só mede o efeito do dado quando tudo o
   mais fica constante, e é fácil violar isso sem perceber.

3. **Meça o custo de esconder a pergunta.** Reduza `AMPLITUDE_DA_PERGUNTA` de `6.0` para `1.0`.
   A acurácia nas combinações vistas cai de 79,1% para **63,2%**, e `CONSULTAR(forma, obj 3)`
   despenca de 69,2% para 45,7%: com amplitude 1, os 23 números da pergunta valem cerca de 3% da
   norma de uma entrada de 1.751. Foi preciso **ajudar** a rede para que o fracasso composicional
   pudesse ser atribuído à composição, e não à codificação da pergunta.

4. **Escolha outras combinações para reter.** Troque `COMBINACOES_RETIDAS` e
   `COMBINACOES_CONTROLE` por outros pares — por exemplo, retendo `("CONSULTAR", "forma", 0)` com
   controle `("CONSULTAR", "cor", 0)`. A queda se mantém quando a posição retida é a mais fácil,
   aquela em que o controle acerta 99,3%? E se você retirar **duas** posições do mesmo atributo,
   deixando uma só? Cada variação testa a mesma hipótese: a rede aprendeu pares (atributo,
   posição) como padrões, e não a operação "ler o atributo A do objeto P".
