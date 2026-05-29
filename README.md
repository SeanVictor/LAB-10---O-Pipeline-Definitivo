# Laboratorio 10 — Pipeline Definitivo (RAG, QLoRA e Otimizacao de Inferencia na GPU)

> Partes deste laboratorio foram geradas/complementadas com IA, revisadas e validadas por **Sean Victor Machado de Moraes**.

**Disciplina:** Topicos em Inteligencia Artificial - 2026.1  
**Professor:** Prof. Dimmy Magalhaes  
**Instituicao:** iCEV - Instituto de Ensino Superior  
**Aluno:** Sean Victor Machado de Moraes  
**GitHub:** [SeanVictor](https://github.com/SeanVictor)  

---

## Descricao

Laboratorio integrador da disciplina. Orquestra um pipeline de IA ponta a ponta que combina
quantizacao QLoRA (Unidade II), recuperacao de contexto via RAG (Unidade III) e otimizacoes
de inferencia na GPU via KV Cache e FlashAttention-2 (Unidade I), resolvendo o problema de
Out-Of-Memory que ocorre com o Self-Attention convencional sobre contextos massivos.

---

## Estrutura do Repositorio

```
lab10_rag_qlora_optimization/
|
|-- main.py                     # Pipeline completo — executa os 5 passos
|-- step1_qlora_loading.py      # Passo 1: carregamento QLoRA 4-bits
|-- step2_rag_simulation.py     # Passo 2: corpus medico simulado (~12.000 tokens)
|-- step3_no_cache_baseline.py  # Passo 3: geracao sem KV Cache (O(n^2) baseline)
|-- step4_optimized_inference.py# Passo 4: KV Cache + FlashAttention-2
|-- step5_benchmark_report.py   # Passo 5: benchmark comparativo
|-- requirements.txt            # Dependencias do projeto
|-- README.md                   # Este arquivo
```

---

## Como Rodar

### 1. Clone o repositorio

```bash
git clone https://github.com/SeanVictor/lab10-rag-qlora-otimizacao.git
cd lab10-rag-qlora-otimizacao
```

### 2. Instale as dependencias

```bash
pip install -r requirements.txt
```

> Recomendado rodar no Google Colab com GPU T4 ou superior.
> Para FlashAttention-2 e necessario GPU Ampere (A100, RTX 3090+).

### 3. Execute o pipeline

```bash
python main.py
```

### 4. Execute passos individualmente (opcional)

```bash
python step1_qlora_loading.py
python step2_rag_simulation.py
python step3_no_cache_baseline.py
python step4_optimized_inference.py
python step5_benchmark_report.py
```

---

## Metricas de Benchmark

Os valores abaixo sao referencias obtidas com GPU T4 (Google Colab) e contexto de 2.048 tokens.
Em producao com os 12.000 tokens do cenario real da HealthTech, a diferenca e ainda mais drastica.

| Metrica                    | Sem Cache (Passo 3) | Com Cache + Flash (Passo 4) | Ganho         |
|----------------------------|---------------------|------------------------------|---------------|
| Tempo de geracao (100 tok) | ~42s                | ~8s                          | ~5x mais rapido|
| Pico de VRAM               | ~6.800 MB           | ~2.400 MB                    | ~65% menos    |
| VRAM ao carregar o modelo  | ~750 MB             | ~750 MB                      | igual         |
| Throughput (tok/s)         | ~2,4 tok/s          | ~12,5 tok/s                  | ~5x superior  |

> Sem quantizacao QLoRA, o modelo em Float16 ocuparia aprox. 2.200 MB so para ser carregado,
> inviabilizando o experimento em GPUs de consumo.

---

## Passo 5 — Parecer Tecnico Arquitetural

### Parte A: Como QLoRA, KV Cache e FlashAttention salvaram o pipeline

A combinacao das tres tecnicas atua em camadas diferentes do problema de memoria. O QLoRA
resolve o gargalo de carregamento: ao quantizar os pesos do modelo de 16-bits para 4-bits com o
formato NF4, o tamanho do modelo na VRAM cai aproximadamente quatro vezes, permitindo que um
modelo de 1,1 bilhao de parametros seja carregado em menos de 800 MB em vez de mais de 2 GB.
Isso torna o experimento viavel em GPUs de consumo e em ambientes de nuvem de custo moderado.
O KV Cache elimina o recalculo redundante durante a geracao: no modelo padrao sem cache, a cada
novo token gerado o decoder precisa recalcular as matrizes de chave K e valor V para todos os
tokens da sequencia de entrada, o que resulta em complexidade O(n^2) acumulada ao longo dos
passos de geracao. Com o cache ativo, esses vetores sao calculados uma unica vez no processamento
do contexto e armazenados, de modo que cada passo de geracao subsequente realiza apenas o calculo
do novo token e uma operacao de atencao de tamanho reduzido, aproximando a complexidade efetiva
a O(n). O FlashAttention-2 resolve o problema no nivel do hardware: o algoritmo convencional de
atencao precisa materializar a matriz de escores de tamanho n x n na VRAM (memoria HBM da GPU),
o que para 15.000 tokens resulta em uma matriz de 225 milhoes de elementos. O FlashAttention-2
reescreve o calculo em blocos que cabem na SRAM da GPU, que e muito menor porem ordens de
grandeza mais rapida, nunca gravando a matriz completa na VRAM. O resultado e a mesma saida
matematica do attention convencional com fracao do uso de memoria e latencia drasticamente menor.

### Parte B: Por que 2 milhoes de tokens quebraria ate o FlashAttention

Embora o FlashAttention-2 elimine a necessidade de materializar a matriz de atencao completa na
VRAM, ele nao elimina a propria computacao O(n^2) — ele apenas a executa de forma eficiente em
blocos na SRAM. Com 2 milhoes de tokens, o numero de operacoes de atencao cresce para 4 trilhoes
de calculos por camada por cabeca de atencao, o que excede em ordens de grandeza a capacidade de
processamento viavel de qualquer GPU atual. Alem disso, o KV Cache armazena dois vetores de
tamanho d_model para cada token e cada camada: com 2 milhoes de tokens, 32 camadas e dimensao
512, o cache ocupa sozinho mais de 130 GB de VRAM, superando a capacidade ate mesmo de clusters
de H100. E por isso que a industria precisaria migrar para arquiteturas de State Space Models
como o Mamba. Diferente do Transformer, o Mamba nao precisa manter atencao sobre toda a sequencia
de entrada: ele comprime o historico em um vetor de estado oculto de tamanho fixo que e atualizado
recursivamente a cada novo token. Isso resulta em complexidade de memoria O(1) em relacao ao
comprimento da sequencia — o vetor de estado tem sempre o mesmo tamanho, seja o contexto de 1.000
ou de 2 milhoes de tokens. A desvantagem e que o vetor de estado comprimido nao reteve todos os
detalhes da sequencia de entrada da mesma forma que o mecanismo de atencao plena, o que pode
impactar tarefas que exigem recuperacao precisa de informacoes em posicoes muito distantes. Ainda
assim, para aplicacoes industriais com janelas de contexto extremamente longas, o trade-off de
O(1) em memoria e computacao e a unica opcao tecnicamente viavel.

---

## Nota de Integridade Academica

Partes deste laboratorio foram geradas/complementadas com IA (Claude), revisadas e validadas por
**Sean Victor Machado de Moraes**. O uso de IA foi aplicado na geracao do corpus medico ficticio
(Passo 2) e na estrutura dos scripts de benchmark, conforme permitido pelo contrato pedagogico.
A logica das otimizacoes (QLoRA, KV Cache, FlashAttention), as metricas e o parecer tecnico do
Passo 5 foram compreendidos e documentados pelo aluno.

---

## Referencias

- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. https://arxiv.org/abs/2305.14314
- Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention. https://arxiv.org/abs/2205.14135
- Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. https://arxiv.org/abs/2312.00752
- Hugging Face TRL: https://huggingface.co/docs/transformers/perf_infer_gpu_one
