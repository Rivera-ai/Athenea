""" Repo based implementation: https://github.com/VachanVY/Transfusion.torch 

And also based on the implementation of Llama 3 in the repo: https://github.com/Rivera-ai/ZeusLLM/blob/main/llm.py

"""

from .configs import MNIST_config, FashionMNIST_config
from .diffusion_utils import DiffusionUtils
from .transfusion import Transfusion, CosineDecayWithWarmup, PatchOps
from .llm import *
from .encoders import *
