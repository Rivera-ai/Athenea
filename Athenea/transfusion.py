""" Repo based implementation: https://github.com/VachanVY/Transfusion.torch """

import typing as tp
import math

import torch
from torch import Tensor, nn, _assert

from .diffusion_utils import DiffusionUtils
from .llm import Transformer, transfusion_config_to_model_args


class TextOps(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.label_embed = nn.Embedding(
            num_embeddings=config.lm_output_units,
            embedding_dim=config.d_model,
        )
        nn.init.normal_(self.label_embed.weight, mean=0.0, std=0.02)
    
    def forward(self, labels:Tensor):
        return self.label_embed(labels)
    

class TimeEmbedding(nn.Module):
    def __init__(self, out_dim:int, hidden_dim:int):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.SELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.half_dim = hidden_dim // 2
        self.apply(self._init_weights)

    def _init_weights(self, module:nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def sinusoidal_embeddings(self, x:Tensor):
        """https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py#L87"""
        embeddings = torch.exp(-math.log(10000) * torch.arange(0, self.half_dim, device=x.device) / self.half_dim)
        embeddings = x[:, None] * embeddings[None]
        # in link implementation, concat(cos, sin, -1) is done
        embeddings = torch.concatenate([torch.cos(embeddings), torch.sin(embeddings)], -1)
        return embeddings

    def forward(self, x:Tensor): # (B,)
        x = self.sinusoidal_embeddings(x) # (B,) => (B, hidden_dim)
        x = self.mlp(x) # (B, out_dim)
        return x # (B, out_dim)
    

class PatchOps(nn.Module):
    def __init__(self, config:tp.Any, unbatched:bool=True):
        super().__init__()
        self.unbatched = unbatched
        self.p = config.patch_size

        # https://github.com/keras-team/keras/blob/master/keras/src/ops/image.py#L632
        out_dim:int = config.patch_size**2 * config.in_channels
        kernel = torch.eye(out_dim).reshape(
            self.p, self.p, config.in_channels, out_dim
        ).permute(3, 2, 0, 1)
        self.register_buffer("kernel", kernel)
        self.patch_proj = lambda x: nn.functional.conv2d(
            x, self.kernel, stride=(self.p, self.p), bias=None
        )

        self.H, self.W = config.H, config.W
        self.c = config.out_channels
        
        hidden_dim = int(config.d_model//2.5 if config.d_model//2.5 % 2 == 0 else config.d_model//2.5 + 1)
        self.time_emb = TimeEmbedding(out_dim, hidden_dim)

        self.linear = nn.Linear(out_dim, config.d_model)

    def patchify(self, images:Tensor):
        patches = self.patch_proj(images[None]) # (B=1, (P**2)*C, H//P, W//P)
        patches = patches.flatten(2).transpose(1, 2)
        return patches[0]  # (N, dim) # N = (H*W)/P**2 | dim = P**2)*C
    
    def unpatchify(self, x:Tensor): # (N = H*W//P**2, D = (P**2)*C)
        h, w = self.H//self.p, self.W//self.p # int(x.shape[1]**0.5) # h = H//P
        x = x.reshape(h, w, self.p, self.p, self.c) # (H//P, W//P, P, P, C)
        x = torch.einsum('hwpqc->chpwq', x) # (C, H//P, P, W//P, P)
        x = x.reshape(self.c, h*self.p, w*self.p) # (C, H, W)
        return x
    
    def forward(self, patches:Tensor, timesteps:Tensor): # (N, D) (B=1,)
        # We add an embedding of the timestep t to every patch vector before the linear layer)
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        return self.linear(patches[None] + time_emb)[0] # (B, N, d_model)


class Transfusion(nn.Module):
    def __init__(self, model:nn.Module, config:tp.Any):
        super().__init__()
        self.model = model
        self.embeddings = TextOps(config)
        self.patch_ops = PatchOps(config)
        self.lm_emb_linear = nn.Linear(config.d_model, config.lm_output_units)
        self.img_patch_linear = nn.Linear(config.d_model, (config.patch_size**2)*config.out_channels)

        self.register_buffer("BOI", config.BOI)
        self.register_buffer("IGNORE_TOKEN", config.IGNORE_TOKEN)
        self.register_buffer("EOI", config.EOI)
        self.register_buffer("EOS", config.EOS)
        self.register_buffer("causal_mask", torch.tril(torch.ones(1, 1, config.maxlen, config.maxlen), diagonal=0).bool())

        self.txt_maxlens = config.text_maxlen
    
    def _compute_mask(self, mask_params:list[tuple[int, int]]):
        mask = self.causal_mask.clone()
        for (from_, num_image_tokens) in mask_params:
            mask[:, :, from_:from_+num_image_tokens, from_:from_+num_image_tokens] = True
        return torch.where(mask, 0., -torch.inf)
    
    def forward_unbatched(
        self,
        modality_tokens:list[tp.Any],
        # [ (T1,), ( (N1, D), (,) ), (T2,), ( (N1, D), (,) )]
        #                   OR
        # [ (T3,), ( (N3, D), (,) ), (T4,),                 ]
        modality_strings:list[str]
        # ["text",    "image"      , "text",  "image"       ]
        #                   OR
        # ["text",    "image"      , "text",                ]
    ):
        _assert(len(modality_tokens) == len(modality_strings), "Length of modality_tokens and modality_strings should be same")
        inputs:list[Tensor] = []; mask_params:list[tuple[int, int]] = []; modality_lengths:list[int] = []

        # Prepare the inputs for the model
        for modality_str, modality_tok in zip(modality_strings, modality_tokens):
            if modality_str == 'text':
                inputs.append(self.embeddings(modality_tok)) # (seq,) => (seq, d_model)
                T = modality_tok.size(0)
                modality_lengths.append(T)
            elif modality_str == 'image':
                assert isinstance(modality_tok, (tuple, list)) and len(modality_tok) == 2
                N = modality_tok[0].size(0)
                mask_params.append((sum(modality_lengths), N))
                inputs.append(self.patch_ops(*modality_tok)) # (patches, timesteps) => (N, d_model)
                modality_lengths.append(N)
            else:
                raise ValueError(f"Unknown modality: {modality_str}")
            
        # Concatenate the embeddings of the different modalities
        # model input: (B=1, T, D) mask: (B=1, h=1, T, T)
        outputs:Tensor = self.model(torch.cat(inputs, dim=0)[None], self._compute_mask(mask_params)).squeeze(0) # (T, d_model)

        # Split the output back into the different modalities
        outputs = torch.split(outputs, modality_lengths, dim=0)

        # Apply the linear layers to the different modalities
        modality_token_emb:list[Tensor] = []
        for modality_str, modality_tok in zip(modality_strings, outputs):
            if modality_str == "text":
                modality_token_emb.append(self.lm_emb_linear(modality_tok[None]))
            elif modality_str == "image":
                modality_token_emb.append(self.img_patch_linear(modality_tok[None]))
            else:
                raise ValueError(f"Unknown modality: {modality_str}")
        return modality_token_emb
    # [ (T1,), (N1, D), (T2,), (N1, D) ] OR [ (T3,), (N3, D), (T4,) ]
    # ["text", "image", "text", "image"] OR ["text", "image", "text"]

    def forward(self, modality_tokens:list[list[tp.Any]], modality_strings:list[list[str]]):
        modality_token_emb = [
            self.forward_unbatched(modality_token, modality_string) for modality_token, modality_string in zip(modality_tokens, modality_strings)
        ]
        return modality_token_emb, modality_strings
    
    def lm_mode(self, modality_tokens:list[tp.Any], modality_strings:list[str], diff_utils:DiffusionUtils, autocast:torch.autocast):
        _assert((mod in ["text", "image"] for mod in modality_strings), f"Unknown Modality in {modality_strings}")
        text_length = sum([tok.size(0) for tok, _str in zip(modality_tokens, modality_strings) if _str == 'text'])
        for _ in range(self.txt_maxlens - text_length):
            with autocast:
                modality_token_emb, _ = self.forward_unbatched(modality_tokens, modality_strings) # list[Tensor]
            nxt_txt_tok:Tensor = modality_token_emb[-1][-1:].argmax(dim=-1) # (1,)
            if (nxt_txt_tok.squeeze() == self.EOS): break
            modality_tokens[-1] = torch.cat((modality_tokens[-1], nxt_txt_tok), dim=1)
            if (nxt_txt_tok.squeeze() == self.BOI):
                modality_tokens, modality_strings = self.diff_mode(modality_tokens, modality_strings, diff_utils, autocast)
                break
        return modality_tokens, modality_strings
    
    def diff_mode(self, modality_tokens:list[Tensor], modality_strings:list[str], diff_utils:DiffusionUtils, autocast:torch.autocast):
        _assert(modality_strings[-1] == "text", "Last modality should be text")
        _assert(modality_tokens[-1][:, -1].squeeze() == self.BOI, "Last token should be BOI")

        (modality_tokens, # (generated image tokens, timestep=0) gets appended
        modality_strings  # "image" gets appended
        ) = diff_utils.generate(
            model=self,
            modality_tokens=modality_tokens,
            modality_strings=modality_strings,
            autocast=autocast
        )
        modality_tokens.append(self.EOI.reshape((1,))); modality_strings.append("text")
        modality_tokens, modality_strings = self.lm_mode(modality_tokens, modality_strings, diff_utils, autocast)
        return modality_tokens, modality_strings
    
    @torch.no_grad()
    def generate(
        self,
        modality_tokens:list[tp.Any], # [ (T1,), (N1, D), (T2,), (N1, D) ]
        modality_strings:list[str],   # ["text", "image", "text", "image"]
        diff_utils:DiffusionUtils,
        autocast:torch.autocast
    ) -> tuple[list[tp.Any], list[str]]:
        _assert(self.training == False, "Should be in eval mode for generation")
        # if only image also fine, but add the <BOI> and <EOI> tokens (which are text modalities)
        _assert(modality_strings[0] == 'text' and modality_strings[-1] == 'text', "First and Last modality should be text")

        # LM mode ===recursive==> Diff mode ===recursive==> LM mode ...
        modality_tokens, modality_strings = self.lm_mode(modality_tokens, modality_strings, diff_utils, autocast)
        return modality_tokens, modality_strings
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_config = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_config)
        return optimizer


class CosineDecayWithWarmup:
    def __init__(
        self,
        warmup_steps:int,
        max_learning_rate:float,
        decay_steps:int,
        min_learning_rate:float
    ):
        self.warmup_steps = warmup_steps
        self.max_learning_rate = max_learning_rate
        self.decay_steps = decay_steps
        self.min_learning_rate = min_learning_rate

    def __call__(self, step):
        # linear warmup for warmup_steps steps
        if step < self.warmup_steps:
            return self.max_learning_rate * step / self.warmup_steps
        # if it > decay_steps, return min learning rate
        if step > self.decay_steps:
            return self.min_learning_rate
        # in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_learning_rate + coeff * (self.max_learning_rate - self.min_learning_rate)

"""
if __name__ == "__main__":
    # [ (T1,), ( (N1, D), (,) ), (T2,), ( (N1, D), (,) )]
    #                   OR
    # [ (T3,), ( (N3, D), (,) ), (T4,),                 ]
    from configs import MNIST_config as config
    from llm import Transformer, ModelArgs  # Tu nueva implementación de LLaMA 3

    # Crear ModelArgs mapeando los parámetros de MNIST_config
    model_args = ModelArgs(
        # Parámetros base del transformer
        dim=config.d_model,  # 348 para MNIST, 512 para FashionMNIST
        n_layers=config.num_layers,  # 7 para MNIST, 8 para FashionMNIST
        n_heads=config.num_heads,  # 6 para MNIST, 8 para FashionMNIST
        n_kv_heads=config.num_heads,  # Mismo número que heads
        
        # Vocabulario y dimensiones
        vocab_size=config.lm_output_units,  # Calculado dinámicamente basado en num_classes y tokens especiales
        max_seq_len=config.maxlen,  # 2*N + text_maxlen
        max_batch_size=config.batch_size,  # 64
        
        # Regularización
        dropout=config.dropout_rate,  # 0.0
        
        # Parámetros específicos de LLaMA 3 que no están en MNIST_config
        multiple_of=256,  # Valor por defecto de LLaMA 3
        ffn_dim_multiplier=None,  # Usar el valor por defecto
        norm_eps=1e-5,  # Valor estándar
        rope_theta=500000  # Valor por defecto de LLaMA 3
    )

    print("=== Configuración del Modelo ===")
    print(f"Dimension del modelo: {config.d_model}")
    print(f"Número de capas: {config.num_layers}")
    print(f"Número de heads: {config.num_heads}")
    print(f"Vocabulario: {config.lm_output_units}")
    print(f"Longitud máxima de secuencia: {config.maxlen}")

    # Imprimir estructura del ModelArgs
    print("\n=== Argumentos del Modelo ===")
    print(model_args)
    # Crear el modelo Transfusion
    model = Transfusion(
        model=Transformer(model_args),
        config=config
    )

    print("\n=== Estructura del Modelo ===")
    print(model)

    # Mostrar forma de los datos de entrada
    text_and_images = [
        [
            torch.randint(0, 10, (39,)), 
            (torch.randn(345, config.patch_size**2 * config.in_channels), torch.randint(0, config.num_timesteps, (1,))), 
            torch.randint(0, 10, (14,))
        ],
        [
            torch.randint(0, 10, (16,)), 
            (torch.randn(359, config.patch_size**2 * config.in_channels), torch.randint(0, config.num_timesteps, (1,))), 
            torch.randint(0, 10, (5,)), 
            (torch.randn(2, config.patch_size**2 * config.in_channels), torch.randint(0, config.num_timesteps, (1,))), 
            torch.randint(0, 10, (9,))
        ]
    ]

    print("\n=== Datos de Entrada ===")
    print("Secuencia 1:")
    print(f"Texto 1 shape: {text_and_images[0][0].shape}")
    print(f"Imagen 1 shape: {text_and_images[0][1][0].shape}")
    print(f"Texto 2 shape: {text_and_images[0][2].shape}")
    
    print("\nSecuencia 2:")
    print(f"Texto 1 shape: {text_and_images[1][0].shape}")
    print(f"Imagen 1 shape: {text_and_images[1][1][0].shape}")
    print(f"Texto 2 shape: {text_and_images[1][2].shape}")
    print(f"Imagen 2 shape: {text_and_images[1][3][0].shape}")
    print(f"Texto 3 shape: {text_and_images[1][4].shape}")

    # Intentar el forward pass y capturar cualquier error
    try:
        outputs = model(text_and_images, [["text", "image", "text"], ["text", "image", "text", "image", "text"]])
        print("\n=== Forward Pass Exitoso ===")
        print("Formas de salida:")
        if isinstance(outputs, tuple):
            modality_token_emb, modality_strings = outputs
            print("Modality Token Embeddings:")
            for i, emb in enumerate(modality_token_emb):
                print(f"Secuencia {i+1}:")
                for j, token_emb in enumerate(emb):
                    print(f"  Token {j+1} shape: {token_emb.shape}")
            print("\nModality Strings:")
            for i, strings in enumerate(modality_strings):
                print(f"Secuencia {i+1}: {strings}")
        else:
            print(f"Output type: {type(outputs)}")
            print(f"Output structure: {outputs}")
    except Exception as e:
        print("\n=== Error en Forward Pass ===")
        print(f"Error: {str(e)}")

    # Imprimir parámetros totales del modelo
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal de parámetros: {total_params:,}") """