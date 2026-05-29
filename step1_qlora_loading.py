"""
Passo 1: Ingestão Eficiente com QLoRA 4-bits

Carrega um modelo Llama compacto em 4-bits usando bitsandbytes.
Registra a memória VRAM utilizada ao carregar o modelo quantizado.

Aluno      : Sean Victor Machado de Moraes
GitHub     : https://github.com/SeanVictor
Disciplina : Tópicos em IA – 2026.1
Instituição: iCEV
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_qlora_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Carrega um modelo auto-regressivo em 4-bits com QLoRA.
    
    Configuração:
        - load_in_4bit=True      : quantização em 4-bits
        - bnb_4bit_compute_dtype : float16 para cálculos
    
    Retorna:
        model     : modelo quantizado
        tokenizer : tokenizador correspondente
        vram_mb   : memória VRAM ocupada em MB
    """
    print("=" * 60)
    print("  Passo 1 — Ingestão Eficiente com QLoRA 4-bits")
    print("=" * 60)
    
    print(f"\n  Carregando modelo: {model_name}")
    
    # Limpar cache antes
    torch.cuda.empty_cache()
    
    # Configuração BitsAndBytes para 4-bits
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Carregar modelo com quantização
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Registrar memória
    torch.cuda.synchronize()
    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Carregar tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\n  Modelo carregado com sucesso!")
    print(f"  Memória VRAM utilizada: {vram_mb:.2f} MB")
    print(f"  Tipo de quantização: 4-bits (QLoRA)")
    print(f"  Device: {model.device}")
    print("=" * 60)
    
    return model, tokenizer, vram_mb


if __name__ == "__main__":
    model, tokenizer, vram_mb = load_qlora_model()
    print(f"\n  Passo 1 OK — Modelo pronto para a próxima etapa")
    print(f"  Memória registrada: {vram_mb:.2f} MB")
