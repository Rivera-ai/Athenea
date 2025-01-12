""" Repo based implementation: https://github.com/VachanVY/Transfusion.torch """

from dataclasses import dataclass
from typing import Optional, Literal, Any

import torch


@dataclass
class MNIST_config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_type = "bfloat16" if torch.cuda.is_available() else "float32"
    # device = torch.device("cpu") # uncomment for debugging

    # Diffusion Args
    var_range:tuple[float, float] = (1e-4, 2e-2)
    num_timesteps:int = 400

    # Vit Args
    patch_size:int = 2
    H:int = 28
    W:int = 28
    in_channels:int = 1
    out_channels:int = in_channels
    N:int = H*W//patch_size**2
    assert N*patch_size**2 == H*W

    # transformer Args
    d_model:int = 348
    num_heads:int = 6
    assert d_model % 2 == 0
    assert d_model % num_heads == 0
    num_layers:int = 7
    num_classes:int = 10
    dropout_rate:float = 0.0
    text_maxlen:int = 6
    maxlen:int = 2*N + text_maxlen

    # Training Args
    batch_size:int = 64
    num_steps:int = 15_000
    decay_steps:int = num_steps
    warmup_steps:int = 100
    max_lr:float = 3e-4
    min_lr:float = 0.0*max_lr
    no_decay:bool = True
    beta1:float = 0.9
    beta2:float = 0.99 # 0.95 in paper # for smaller datasets a bit higher is better
    clipnorm:float = 1e0
    weight_decay:float = 0.0 # 1e0 in paper
    
    patience:int = 10
    num_grad_accumalation_steps:int = 1
    return_best_train_states:bool = True
    log_interval:int = 25
    eval_freq:int = 400

    # Transfusion Args
    balancing_coeff:float = 5.0

    BOI:Optional[torch.Tensor] = torch.tensor(num_classes, dtype=torch.long) # 10
    IGNORE_TOKEN:Optional[torch.Tensor] = torch.tensor(num_classes+1, dtype=torch.long) # 11
    EOI:Optional[torch.Tensor] = torch.tensor(num_classes+2, dtype=torch.long) # 12
    EOS:Optional[torch.Tensor] = torch.tensor(num_classes+3, dtype=torch.long) # 13 

    lm_output_units:int = num_classes + int(BOI is not None) + int(IGNORE_TOKEN is not None) + int(EOI is not None) + int(EOS is not None)


@dataclass
class FashionMNIST_config(MNIST_config):
    d_model:int = 512
    num_heads:int = 8
    assert d_model % 2 == 0
    assert d_model % num_heads == 0
    num_layers:int = 8

    num_steps:int = 50_000
    ckpt_dir:str = "checkpoints/fashionmnist"

    eval_freq:int = 600
    log_interval:int = 1

@dataclass
class Flickr30kConfig:
    # Device y dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_type = "bfloat16" if torch.cuda.is_available() else "float32"

    # Paths
    data_dir: str = "data/flickr30k"
    image_dir: str = f"{data_dir}/images"
    csv_path: str = f"{data_dir}/flickr30k.csv"
    ckpt_dir: str = "checkpoints/flickr30k"
    tokenizer_path: str = "meta-llama/Llama-3.1-8B"

    # Diffusion Args
    var_range: tuple[float, float] = (1e-4, 2e-2)
    num_timesteps: int = 1000  # Aumentado para mejor calidad
    
    # Image Args
    patch_size: int = 16  # Patch más grande para imágenes de alta resolución
    H: int = 256
    W: int = 256
    in_channels: int = 3  # RGB
    out_channels: int = in_channels
    N: int = H*W//patch_size**2
    assert N*patch_size**2 == H*W

    # Transformer Args
    d_model: int = 1024  # Aumentado para manejar la complejidad
    num_heads: int = 16
    assert d_model % 2 == 0
    assert d_model % num_heads == 0
    num_layers: int = 12
    dropout_rate: float = 0.1  # Añadido dropout para regularización
    
    # Text Args
    max_seq_len: int = 512  # Longitud máxima para el tokenizer de Llama
    text_maxlen: int = max_seq_len
    maxlen: int = 2*N + text_maxlen  # Ajustado para patches + texto
    vocab_size: int = 32000  # Tamaño del vocabulario de Llama

    # Training Args
    batch_size: int = 32
    num_steps: int = 500_000  # Más pasos para dataset más grande
    decay_steps: int = 100_000
    warmup_steps: int = 2000  # Más warmup para estabilidad
    max_lr: float = 1e-4  # Learning rate más bajo para estabilidad
    min_lr: float = 1e-5
    no_decay: bool = False
    beta1: float = 0.9
    beta2: float = 0.95  # Valor del paper
    clipnorm: float = 1.0
    weight_decay: float = 0.1  # Añadido weight decay para regularización
    
    # Logging y Checkpointing
    patience: int = 15
    num_grad_accumulation_steps: int = 2  # Acumulación de gradientes para batch efectivo más grande
    return_best_train_states: bool = True
    log_interval: int = 100
    eval_freq: int = 1000
    save_every: int = 5000
    
    # Transfusion Args
    balancing_coeff: float = 5.0

    # Special Tokens - usando valores más altos que el vocab_size base
    BOI: Optional[torch.Tensor] = torch.tensor(vocab_size, dtype=torch.long)
    IGNORE_TOKEN: Optional[torch.Tensor] = torch.tensor(vocab_size+1, dtype=torch.long)
    EOI: Optional[torch.Tensor] = torch.tensor(vocab_size+2, dtype=torch.long)
    EOS: Optional[torch.Tensor] = torch.tensor(vocab_size+3, dtype=torch.long)

    # Output units incluye tokens especiales
    lm_output_units: int = vocab_size + int(BOI is not None) + int(IGNORE_TOKEN is not None) + \
                          int(EOI is not None) + int(EOS is not None)

    def __post_init__(self):
        """Verificaciones adicionales después de la inicialización"""
        # Verificar que N sea razonable para la atención
        assert self.N <= 256, f"Número de patches ({self.N}) demasiado alto. Considera aumentar patch_size"
        
        # Verificar que maxlen sea manejable
        assert self.maxlen <= 2048, f"Sequence length ({self.maxlen}) demasiado alta"
        
        # Crear directorios necesarios
        import os
        os.makedirs(self.ckpt_dir, exist_ok=True)

@dataclass
class Flickr30kConfigSmall(Flickr30kConfig):
    """Configuración más pequeña para pruebas o recursos limitados"""
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 8
    batch_size: int = 16
    num_steps: int = 100_000
    patch_size: int = 32  # Patches más grandes = menos secuencia

@dataclass
class Flickr30kConfigLarge(Flickr30kConfig):
    """Configuración más grande para mejor calidad"""
    d_model: int = 1536
    num_heads: int = 24
    num_layers: int = 16
    dropout_rate: float = 0.2
    num_steps: int = 1_000_000
    batch_size: int = 48
    num_grad_accumulation_steps: int = 4