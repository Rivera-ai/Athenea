""" Implementation based on the Llama 3 architecture, from the repo: https://github.com/Rivera-ai/ZeusLLM/blob/main/llm.py """

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn


@dataclass
class ModelArgs:
    dim: int = 2560
    n_layers: int = 28
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    multiple_of: int = 256  # hacer que el tamaño de la capa oculta de SwiGLU sea múltiplo de una gran potencia de 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 5
    max_seq_len: int = 2048
    dropout: float = 0.1

def transfusion_config_to_model_args(config) -> ModelArgs:
    return ModelArgs(
        dim=config.d_model,
        n_layers=config.num_layers,
        n_heads=config.num_heads,
        n_kv_heads=config.n_kv_heads if hasattr(config, 'n_kv_heads') else None,
        vocab_size=config.lm_output_units,
        dropout=config.dropout_rate,
        max_seq_len=config.maxlen,
        # Otros parámetros específicos de tu implementación
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape

        
        # Proyecciones QKV
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Aplicar RoPE a Q y K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repetir K y V si es necesario
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Reordenar para atención
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if mask is not None:
            mask = mask[:, :, :seqlen, :seqlen]
        
        # Calcular scores
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)
        
        # Aplicar atención
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(2 * args.dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        # x: [batch_size, seq_len, dim]
        # freqs_cis: [seq_len, dim/2]
        # mask: [seq_len, seq_len] o [batch_size, 1, seq_len, seq_len]
        
        # Asegurar que la máscara tiene la forma correcta
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            
        # Attention y residual
        h = x + self.dropout(self.attention(self.attention_norm(x), freqs_cis, mask))
        
        # FFN y residual
        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))

        
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()

        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        # Precomputar las frecuencias para RoPE
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            params.dim // params.n_heads, 
            params.max_seq_len * 2,  # Multiplicar por 2 para tener margen 
            params.rope_theta
        ))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Obtener dimensiones actuales
        bsz, seqlen = h.size()[:2]
        #print(f"seqlen: {seqlen}")
        
        # Aplicar dropout
        h = self.dropout(h)
        
        # Ajustar freqs_cis al tamaño de la secuencia actual
        freqs_cis = self.freqs_cis[:seqlen]
        
        # Si la máscara es None, crear una máscara causal del tamaño correcto
        if mask is not None:
        # Asegurarse que la máscara tenga las dimensiones correctas para la secuencia actual
            mask = mask[:, :, :seqlen, :seqlen]
        else:
        # Crear máscara causal si no se proporciona
            mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Pasar por todas las capas
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        
        # Normalización final
        h = self.norm(h)
        
        return h