"""
Passo 4: A Engenharia de Otimizacao

Refatora o carregamento e a geracao com duas otimizacoes:

    1. KV Cache (use_cache=True):
       Armazena as matrizes K e V calculadas nos passos anteriores,
       evitando o recalculo a cada novo token gerado.

    2. FlashAttention-2 (attn_implementation="flash_attention_2"):
       Algoritmo Hardware-Aware que calcula o Scaled Dot-Product
       Attention diretamente na SRAM da GPU em blocos (tiling),
       evitando a materializacao da matriz O(n^2) na VRAM.

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Topicos em IA - 2026.1
Instituicao: iCEV
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_with_flash_attention(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
):
    """
    Carrega o modelo com QLoRA 4-bits e FlashAttention-2 ativado.

    FlashAttention-2 requer:
        - GPU Ampere ou superior (A100, RTX 3090+)
        - torch >= 2.0
        - flash-attn instalado

    Caso FlashAttention-2 nao esteja disponivel, cai para
    attn_implementation="sdpa" (Scaled Dot-Product Attention
    otimizado do PyTorch), que ja e superior ao attention padrao.

    Retorna:
        model     : modelo com FlashAttention ativo
        tokenizer : tokenizador
        vram_mb   : memoria VRAM ao carregar
    """
    print("=" * 60)
    print("  Passo 4 — Carregamento com FlashAttention-2")
    print("=" * 60)

    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Tenta FlashAttention-2, cai para SDPA se nao disponivel
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        attn_mode = "flash_attention_2"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        attn_mode = "sdpa (fallback)"

    torch.cuda.synchronize()
    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n  Atencao implementada : {attn_mode}")
    print(f"  VRAM ao carregar     : {vram_mb:.2f} MB")
    print("=" * 60)

    return model, tokenizer, vram_mb, attn_mode


def generate_with_cache(model, tokenizer, input_ids: torch.Tensor,
                        new_tokens: int = 100) -> dict:
    """
    Gera novos tokens com KV Cache ativado.

    Com use_cache=True, o modelo armazena os vetores K e V de cada
    camada calculados nos passos anteriores. A cada novo token,
    apenas o novo vetor e calculado e concatenado ao cache existente,
    reduzindo a complexidade efetiva de O(n^2) para O(n) por passo.

    Parametros:
        model      : modelo com FlashAttention
        tokenizer  : tokenizador
        input_ids  : tensor com o contexto tokenizado
        new_tokens : numero de tokens a gerar

    Retorna:
        dict com metricas de tempo e memoria
    """
    print("=" * 60)
    print("  Passo 4 — Geracao COM KV Cache + FlashAttention")
    print("=" * 60)
    print(f"\n  use_cache  = True")
    print(f"  Tokens     = {new_tokens}")
    print(f"  Contexto   = {input_ids.shape[1]:,} tokens")

    # Ativar KV Cache
    model.config.use_cache = True

    device = next(model.parameters()).device
    ids = input_ids.to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("\n  Gerando tokens...")
    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            ids,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    vram_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    tokens_per_sec = new_tokens / elapsed

    print(f"\n  Resultado COM cache + FlashAttention:")
    print(f"    Tempo total         : {elapsed:.2f} segundos")
    print(f"    Tokens por segundo  : {tokens_per_sec:.2f} tok/s")
    print(f"    Pico de VRAM        : {vram_peak_mb:.2f} MB")
    print("=" * 60)

    return {
        "mode":           "kv_cache_flash",
        "elapsed_s":      elapsed,
        "tokens_per_sec": tokens_per_sec,
        "vram_peak_mb":   vram_peak_mb,
        "output_ids":     output,
    }


if __name__ == "__main__":
    from step2_rag_simulation import generate_rag_context

    model, tokenizer, vram_load, attn_mode = load_model_with_flash_attention()
    ctx = generate_rag_context(tokenizer, target_tokens=2048)
    metrics = generate_with_cache(model, tokenizer, ctx["input_ids"])

    print(f"\n  Passo 4 OK")
    print(f"  Tempo com otimizacao : {metrics['elapsed_s']:.2f}s")
    print(f"  VRAM pico            : {metrics['vram_peak_mb']:.2f} MB")
