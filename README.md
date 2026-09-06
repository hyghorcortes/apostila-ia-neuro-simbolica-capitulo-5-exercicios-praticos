# Capítulo 5 — Exercícios Práticos de IA Neuro-Simbólica

Este repositório reúne **5 exercícios práticos** prontos para importação direta no GitHub.

Os exercícios acompanham o Capítulo 5 — *Apresentando a IA Neuro-Simbólica — o Próximo Nível da
IA* — e reproduzem computacionalmente as figuras do capítulo, do arcabouço simbólico da
Figura 5.1 até a arquitetura da Neural Logic Machine da Figura 5.6.

## Estrutura

- `exercicio_01_base_de_conhecimento_simbolica/`: o ingrediente simbólico — base de conhecimento em cláusulas de Horn, encadeamento para frente e árvore de derivação (Figura 5.1)
- `exercicio_02_ingrediente_neural_percepcao_sem_raciocinio/`: o ingrediente neural — rede densa treinada do zero, percepção sem raciocínio (Figura 5.2)
- `exercicio_03_nscl_simplificado/`: a mistura neuro-simbólica — um NSCL simplificado, com image parser, question parser e executor quase-simbólico (Figuras 5.3 e 5.4)
- `exercicio_04_raciocinio_dinamico_e_contrafactual/`: raciocínio dinâmico e inferência contrafactual — o arcabouço do NSDR, com video parser e dynamics predictor (Figura 5.5)
- `exercicio_05_neural_logic_machine/`: Neural Logic Machine — dedução de primeira ordem feita de tensores, medindo os eixos *breadth* e *depth* (Figura 5.6)

## Como executar

Requisitos: **Python 3** e **numpy**. Nada além disso — não há PyTorch, TensorFlow, Prolog nem
biblioteca de autodiferenciação. A unificação de variáveis, o encadeamento para frente, a
retropropagação e o otimizador Adam são implementados do zero, e é aí que mora o valor didático.

```bash
pip install -r requirements.txt
```

Exemplo:

```bash
python3 exercicio_01_base_de_conhecimento_simbolica/exercicio_pratico_1_base_de_conhecimento_simbolica.py
```

Os scripts rodam sem argumentos, sem entrada interativa, sem escrita de arquivos e sem rede. A
semente aleatória é fixa (`SEMENTE = 42`), portanto **a saída é sempre a mesma** entre execuções e
entre máquinas. Nenhuma linha de relatório passa de 78 colunas.

| Exercício | Execução | Relatório |
| --- | ---: | ---: |
| 1 — o ingrediente simbólico | menos de 1 s | 316 linhas |
| 2 — o ingrediente neural | ~1 min | 439 linhas |
| 3 — o NSCL simplificado | ~10 s | 280 linhas |
| 4 — raciocínio dinâmico e contrafactual | ~6 s | 473 linhas |
| 5 — Neural Logic Machine | ~13 s | 382 linhas |

Os tempos são de uma máquina ociosa e variam com o hardware; as contagens de linhas, não.

## Objetivo didático

Os exercícios foram organizados para mostrar, de forma prática, conceitos centrais do Capítulo 5,
como:

- base de conhecimento simbólica, motor de inferência e prova auditável
- as duas fraquezas fatais da IA simbólica: o processo manual e a fragilidade fora do vocabulário
- percepção neural robusta e sua incapacidade de raciocinar composicionalmente
- a arquitetura híbrida que combina os dois ingredientes: NSCL e executor quase-simbólico
- raciocínio dinâmico, predição de trajetórias e inferência contrafactual (NSDR)
- dedução de primeira ordem em tensores e os eixos de *breadth* e *depth* da Neural Logic Machine

Cada exercício tem seu próprio `README.md`, com o detalhamento do que é demonstrado, como
executar, o que esperar da saída e sugestões de extensão para o leitor.

## Observação

Este pacote contém **apenas a parte do repositório GitHub**, sem arquivos do capítulo da apostila.
