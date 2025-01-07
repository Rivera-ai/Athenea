""" This is a first test training file we are still working on more, and expanding the implementation, Encoder and Decoder chords for Flickr30k and multimodal training? """

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Athenea.transfusion import Transfusion, CosineDecayWithWarmup
from Athenea.configs import MNIST_config
from Athenea.llm import Transformer, transfusion_config_to_model_args
from Athenea.diffusion_utils import DiffusionUtils

def train_epoch(model, train_loader, optimizer, scheduler, diff_utils, config, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # Prepare text tokens (usando los labels como texto simple para prueba)
        text_tokens = labels.unsqueeze(-1)  # [B, 1]
        boi_tokens = torch.full((batch_size, 1), config.BOI.item(), device=device)
        eoi_tokens = torch.full((batch_size, 1), config.EOI.item(), device=device)
        
        # Random timesteps para difusión
        t = torch.randint(0, config.num_timesteps, (batch_size,), device=device)
        
        # Aplicar ruido a las imágenes
        noisy_images, noise = diff_utils.noisy_it(images, t)

        # Preparar tokens para Transfusion
        modality_tokens = [
            text_tokens,
            (noisy_images, t),
            torch.cat([boi_tokens, eoi_tokens], dim=1)
        ]
        modality_strings = ["text", "image", "text"]

        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model.forward_unbatched(modality_tokens, modality_strings)
        
        # Calcular pérdidas
        # Pérdida de difusión
        predicted_noise = outputs[1]
        diff_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
        # Pérdida de LM (solo en tokens de texto)
        lm_loss = torch.nn.functional.cross_entropy(
            outputs[0].view(-1, config.lm_output_units),
            text_tokens.view(-1)
        )
        
        # Combinar pérdidas
        loss = lm_loss + config.balancing_coeff * diff_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clipnorm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % config.log_interval == 0:
            print(f'Train Batch: {batch_idx} Loss: {loss.item():.6f}')
            
    return total_loss / len(train_loader)

def main():
    config = MNIST_config()
    device = config.device
    
    # Preparar dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Inicializar modelo
    model_args = transfusion_config_to_model_args(config)
    transformer = Transformer(model_args)
    model = Transfusion(transformer, config).to(device)
    
    # Inicializar utilidades de difusión
    diff_utils = DiffusionUtils(linear_schedule=True, config=config)
    
    # Optimizador y scheduler
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        betas=(config.beta1, config.beta2),
        device_type=device.type
    )
    
    scheduler = CosineDecayWithWarmup(
        warmup_steps=config.warmup_steps,
        max_learning_rate=config.max_lr,
        decay_steps=config.decay_steps,
        min_learning_rate=config.min_lr
    )
    
    # Entrenamiento
    print("Starting training...")
    for epoch in range(5):  # Entrenar por 5 épocas para prueba
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            diff_utils, config, device
        )
        print(f'Epoch: {epoch} Average Loss: {avg_loss:.6f}')

if __name__ == "__main__":
    main()