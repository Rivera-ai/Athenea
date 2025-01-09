import torch
import torch.nn as nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=-1)

class TransfusionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        in_channels = config.in_channels
        dim_latent = config.d_model
        self.patch_size = config.patch_size
        
        # Calculamos el número de patches
        self.num_patches = (config.H // 16) * (config.W // 16)  # 16 por los 4 downsamplings
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            nn.Conv2d(256, dim_latent, 1),
        )
        
        self.rearrange = Rearrange('b d h w -> b (h w) d')
        self.normalize = Normalize()
        
        # Embedding temporal para diffusion
        self.time_mlp = nn.Sequential(
            nn.Linear(dim_latent, dim_latent * 4),
            nn.GELU(),
            nn.Linear(dim_latent * 4, dim_latent)
        )
    
    def get_timestep_embedding(self, timesteps, dim):
        """Crear embeddings posicionales para timesteps"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, timesteps=None):
        # Procesar imagen a través de la CNN
        x = self.features(x)  # [B, d_model, H//16, W//16]
        
        # Reorganizar a secuencia de patches
        x = self.rearrange(x)  # [B, (H//16)*(W//16), d_model]
        
        # Normalizar embeddings
        x = self.normalize(x)
        
        # Agregar información temporal si se proporciona
        if timesteps is not None:
            time_embed = self.time_mlp(self.get_timestep_embedding(timesteps, x.shape[-1]))
            x = x + time_embed.unsqueeze(1)
            
        return x

class TransfusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim_latent = config.d_model
        out_channels = config.in_channels
        
        # Calculamos las dimensiones espaciales correctas
        self.h = config.H // 16  # Por los 4 downsamplings
        self.w = config.W // 16
        
        # Corregimos el rearrange
        self.rearrange = Rearrange('b (h w) d -> b d h w', h=self.h, w=self.w)
        
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(dim_latent, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Reorganizar de secuencia a mapa de features 2D
        x = self.rearrange(x)
        # Decodificar
        x = self.decoder_net(x)
        return x

class TransfusionImageProcessor(nn.Module):
    """Procesador de imágenes completo para Transfusion"""
    def __init__(self, config):
        super().__init__()
        self.encoder = TransfusionEncoder(config)
        self.decoder = TransfusionDecoder(config)
        
    def encode(self, x, timesteps=None):
        """Codifica imágenes a embeddings"""
        return self.encoder(x, timesteps)
    
    def decode(self, x):
        """Decodifica embeddings a imágenes"""
        return self.decoder(x)
    
    def forward(self, x, timesteps=None):
        embeddings = self.encode(x, timesteps)
        reconstruction = self.decode(embeddings)
        return reconstruction, embeddings

def transfusion_loss(original, reconstruction, noise, predicted_noise, embeddings, lambda_recon=1.0):
    # Pérdida de reconstrucción
    recon_loss = F.mse_loss(reconstruction, original)
    
    # Pérdida de diffusion (solo si hay predicción de ruido)
    noise_loss = 0.0
    if predicted_noise is not None:
        noise_loss = F.mse_loss(predicted_noise, noise)
    
    # Pérdida de regularización de embeddings (opcional)
    embedding_loss = torch.mean(torch.abs(embeddings)) * 0.01
    
    return recon_loss * lambda_recon + noise_loss + embedding_loss

class DummyConfig:
    def __init__(self):
        self.in_channels = 3
        self.d_model = 256
        self.patch_size = 16
        self.H = 256
        self.W = 256

# Ejemplo de uso:
""" if __name__ == "__main__":
    class DummyConfig:
        def __init__(self):
            self.in_channels = 3
            self.d_model = 256
            self.patch_size = 16
            self.H = 256
            self.W = 256
    
    config = DummyConfig()
    processor = TransfusionImageProcessor(config)
    
    # Simular batch de imágenes
    dummy_images = torch.randn(4, 3, 256, 256)
    dummy_timesteps = torch.randint(0, 1000, (4,))
    
    # Procesar
    recon, embeddings = processor(dummy_images, dummy_timesteps)
    print(f"Embeddings shape: {embeddings.shape}")  # Debería ser [4, 256, 256]
    print(f"Reconstruction shape: {recon.shape}")   # Debería ser [4, 3, 256, 256]
    
    # Imprimir shapes intermedios para debug
    x = dummy_images
    print("\nShapes intermedios:")
    x = processor.encoder.features(x)
    print(f"Después de features: {x.shape}")
    x = processor.encoder.rearrange(x)
    print(f"Después de rearrange: {x.shape}")
    x = processor.encoder.normalize(x)
    print(f"Después de normalize: {x.shape}")

     Ejecución: 

        ⚡ main ~/Athenea/Athenea python encoders.py
        Embeddings shape: torch.Size([4, 256, 256])
        Reconstruction shape: torch.Size([4, 3, 256, 256])

        Shapes intermedios:
        Después de features: torch.Size([4, 256, 16, 16])
        Después de rearrange: torch.Size([4, 256, 256])
        Después de normalize: torch.Size([4, 256, 256])
    
    """