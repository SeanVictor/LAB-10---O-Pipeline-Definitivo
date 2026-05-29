"""
Passo 3: O Gargalo de Geracao — Sem KV Cache

Gera 100 tokens com use_cache=False, forcando o modelo a recalcular
Q, K, V a cada novo token gerado. Monitora tempo e pico de VRAM.

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Topicos em IA - 2026.1
Instituicao: iCEV
"""

import time
import torch


def generate_no_cache(model, tokenizer, input_ids: torch.Tensor,
                      new_tokens: int = 100) -> dict:
    """
    Gera novos tokens com KV Cache desativado.

    A cada novo token gerado, o modelo recalcula as matrizes Q, K e V
    para toda a sequencia de entrada, o que resulta em:
        - complexidade temporal O(n^2) por passo
        - crescimento quadratico do uso de VRAM

    Parametros:
        model      : modelo quantizado
        tokenizer  : tokenizador
        input_ids  : tensor com o contexto tokenizado
        new_tokens : numero de tokens a gerar

    Retorna:
        dict com metricas de tempo e memoria
    """
    print("=" * 60)
    print("  Passo 3 — Geracao SEM KV Cache (Linha de Base)")
    print("=" * 60)
    print(f"\n  use_cache  = False")
    print(f"  Tokens     = {new_tokens}")
    print(f"  Contexto   = {input_ids.shape[1]:,} tokens")

    # Desativar KV Cache
    model.config.use_cache = False

    device = next(model.parameters()).device
    ids = input_ids.to(device)

    # Zerar estatisticas de memoria
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("\n  Gerando tokens...")
    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            ids,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    vram_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    tokens_per_sec = new_tokens / elapsed

    print(f"\n  Resultado SEM cache:")
    print(f"    Tempo total         : {elapsed:.2f} segundos")
    print(f"    Tokens por segundo  : {tokens_per_sec:.2f} tok/s")
    print(f"    Pico de VRAM        : {vram_peak_mb:.2f} MB")
    print(f"\n  Observacao: O recalculo de Q, K, V a cada passo")
    print(f"  causa crescimento O(n^2) — o custo aumenta a cada token.")
    print("=" * 60)

    return {
        "mode":           "no_cache",
        "elapsed_s":      elapsed,
        "tokens_per_sec": tokens_per_sec,
        "vram_peak_mb":   vram_peak_mb,
        "output_ids":     output,
    }


if __name__ == "__main__":
    from step1_qlora_loading import load_qlora_model
    from step2_rag_simulation import generate_rag_context

    model, tokenizer, _ = load_qlora_model()
    ctx = generate_rag_context(tokenizer, target_tokens=2048)
    metrics = generate_no_cache(model, tokenizer, ctx["input_ids"])
    print(f"\n  Passo 3 OK")
    print(f"  Tempo sem cache  : {metrics['elapsed_s']:.2f}s")
    print(f"  VRAM pico        : {metrics['vram_peak_mb']:.2f} MB")
