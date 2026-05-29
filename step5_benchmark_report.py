"""
Passo 5: Benchmark Comparativo — Antes e Depois da Otimizacao

Compara as metricas das duas estrategias de geracao:
    - Sem cache (Passo 3): linha de base com recalculo O(n^2)
    - Com cache + FlashAttention (Passo 4): otimizacao completa

Gera um relatorio de metricas no terminal pronto para incluir
no README.md.

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Topicos em IA - 2026.1
Instituicao: iCEV
"""

import torch


def print_benchmark_report(
    vram_load_mb:    float,
    token_count:     int,
    metrics_no_cache: dict,
    metrics_cached:  dict,
    attn_mode:       str,
):
    """
    Exibe um relatorio completo de benchmark comparando as duas
    estrategias de geracao.

    Parametros:
        vram_load_mb     : VRAM ao carregar o modelo quantizado
        token_count      : tokens no contexto RAG
        metrics_no_cache : metricas do Passo 3 (sem cache)
        metrics_cached   : metricas do Passo 4 (com cache)
        attn_mode        : implementacao de atencao utilizada
    """
    m0 = metrics_no_cache
    m1 = metrics_cached

    speedup      = m0["elapsed_s"]     / m1["elapsed_s"]
    vram_reducao = m0["vram_peak_mb"]  - m1["vram_peak_mb"]
    vram_pct     = (vram_reducao / m0["vram_peak_mb"]) * 100 if m0["vram_peak_mb"] > 0 else 0

    print("\n" + "=" * 60)
    print("  RELATORIO DE BENCHMARK — LAB 10")
    print("=" * 60)

    print(f"""
  CONFIGURACAO DO EXPERIMENTO
  ----------------------------
  Modelo        : TinyLlama/TinyLlama-1.1B-Chat-v1.0
  Quantizacao   : QLoRA 4-bits (NF4 + double quant)
  Atencao       : {attn_mode}
  Contexto RAG  : {token_count:,} tokens
  Tokens gerados: 100

  PASSO 1 — MEMORIA AO CARREGAR O MODELO
  ----------------------------
  VRAM com QLoRA 4-bits: {vram_load_mb:.2f} MB
  (Sem quantizacao, o mesmo modelo ocuparia aprox. 4x mais)

  PASSO 3 — SEM KV CACHE (linha de base)
  ----------------------------
  Tempo total        : {m0['elapsed_s']:.2f} segundos
  Velocidade         : {m0['tokens_per_sec']:.2f} tokens/segundo
  Pico de VRAM       : {m0['vram_peak_mb']:.2f} MB
  Complexidade       : O(n^2) por passo de geracao

  PASSO 4 — COM KV CACHE + FLASHATTENTION
  ----------------------------
  Tempo total        : {m1['elapsed_s']:.2f} segundos
  Velocidade         : {m1['tokens_per_sec']:.2f} tokens/segundo
  Pico de VRAM       : {m1['vram_peak_mb']:.2f} MB

  GANHOS DA OTIMIZACAO
  ----------------------------
  Aceleracao (speedup)  : {speedup:.2f}x mais rapido
  Reducao de VRAM       : {vram_reducao:.2f} MB ({vram_pct:.1f}% menos)
""")
    print("=" * 60)


if __name__ == "__main__":
    from step1_qlora_loading      import load_qlora_model
    from step2_rag_simulation     import generate_rag_context
    from step3_no_cache_baseline  import generate_no_cache
    from step4_optimized_inference import (load_model_with_flash_attention,
                                           generate_with_cache)

    # Contexto RAG (reduzido para caber na GPU de desenvolvimento)
    CTX_TOKENS = 2048

    print("\n  Iniciando benchmark completo...\n")

    # Passo 3 — linha de base
    model_base, tokenizer_base, vram_load = load_qlora_model()
    ctx = generate_rag_context(tokenizer_base, target_tokens=CTX_TOKENS)
    m_no_cache = generate_no_cache(model_base, tokenizer_base, ctx["input_ids"])

    # Liberar modelo da memoria
    del model_base
    torch.cuda.empty_cache()

    # Passo 4 — otimizado
    model_opt, tokenizer_opt, _, attn_mode = load_model_with_flash_attention()
    ctx_opt = generate_rag_context(tokenizer_opt, target_tokens=CTX_TOKENS)
    m_cached = generate_with_cache(model_opt, tokenizer_opt, ctx_opt["input_ids"])

    # Relatorio final
    print_benchmark_report(
        vram_load_mb     = vram_load,
        token_count      = ctx["token_count"],
        metrics_no_cache = m_no_cache,
        metrics_cached   = m_cached,
        attn_mode        = attn_mode,
    )
