"""
main.py  —  Laboratorio 10: Pipeline Definitivo (RAG, QLoRA e Otimizacao de Inferencia na GPU)
Disciplina : Topicos em IA - 2026.1
Professor  : Prof. Dimmy Magalhaes
Instituicao: iCEV
Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor

Partes deste laboratorio foram geradas/complementadas com IA,
revisadas e validadas por Sean Victor Machado de Moraes.

Pipeline:
    Passo 1 -> Ingestao eficiente com QLoRA 4-bits
    Passo 2 -> Simulacao do RAG massivo (contexto medico)
    Passo 3 -> Geracao sem KV Cache (linha de base O(n^2))
    Passo 4 -> Geracao com KV Cache + FlashAttention-2
    Passo 5 -> Benchmark comparativo e relatorio de metricas
"""

import torch

from step1_qlora_loading       import load_qlora_model
from step2_rag_simulation      import generate_rag_context
from step3_no_cache_baseline   import generate_no_cache
from step4_optimized_inference import (load_model_with_flash_attention,
                                       generate_with_cache)
from step5_benchmark_report    import print_benchmark_report

print("=" * 60)
print("  Lab 10 - Pipeline Definitivo: RAG + QLoRA + Otimizacao")
print("  Aluno : Sean Victor Machado de Moraes")
print("  GitHub: SeanVictor")
print("=" * 60)

CTX_TOKENS = 2048   # Reduzir para 12000 em GPU com mais VRAM


# ════════════════════════════════════════════════════════════
# PASSO 1 — QLoRA 4-bits
# ════════════════════════════════════════════════════════════
model_base, tokenizer_base, vram_load = load_qlora_model()


# ════════════════════════════════════════════════════════════
# PASSO 2 — RAG Massivo
# ════════════════════════════════════════════════════════════
ctx = generate_rag_context(tokenizer_base, target_tokens=CTX_TOKENS)


# ════════════════════════════════════════════════════════════
# PASSO 3 — Geracao sem cache (linha de base)
# ════════════════════════════════════════════════════════════
m_no_cache = generate_no_cache(model_base, tokenizer_base, ctx["input_ids"])

del model_base
torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
# PASSO 4 — Geracao otimizada: KV Cache + FlashAttention
# ════════════════════════════════════════════════════════════
model_opt, tokenizer_opt, _, attn_mode = load_model_with_flash_attention()
ctx_opt = generate_rag_context(tokenizer_opt, target_tokens=CTX_TOKENS)
m_cached = generate_with_cache(model_opt, tokenizer_opt, ctx_opt["input_ids"])


# ════════════════════════════════════════════════════════════
# PASSO 5 — Relatorio de benchmark
# ════════════════════════════════════════════════════════════
print_benchmark_report(
    vram_load_mb     = vram_load,
    token_count      = ctx["token_count"],
    metrics_no_cache = m_no_cache,
    metrics_cached   = m_cached,
    attn_mode        = attn_mode,
)

print("  Pipeline concluido com sucesso!")
print("  Aluno: Sean Victor Machado de Moraes | GitHub: SeanVictor")
print("=" * 60)
