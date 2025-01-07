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