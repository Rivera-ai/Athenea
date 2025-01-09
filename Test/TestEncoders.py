import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
import os
from tqdm import tqdm
from pathlib import Path
from Athenea import transfusion_loss, DummyConfig, TransfusionImageProcessor

def create_flickr_dataloader(flickr_root, batch_size=32, num_workers=4):
    """Crear dataloaders para Flickr30k"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=flickr_root, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def sample_timesteps(batch_size, max_timesteps=1000):
    """Muestrear timesteps aleatorios para el entrenamiento"""
    return torch.randint(0, max_timesteps, (batch_size,))

def add_noise(images, noise_amount=0.1):
    """Añadir ruido gaussiano a las imágenes"""
    noise = torch.randn_like(images) * noise_amount
    noisy_images = images + noise
    return noisy_images, noise

def save_samples(model, original_images, epoch, output_dir):
    """Guardar muestras de reconstrucción"""
    model.eval()
    with torch.no_grad():
        reconstructions, _ = model(original_images)
        
        # Crear grid de imágenes
        samples = torch.cat([
            original_images[:8],  # Primeras 8 imágenes originales
            reconstructions[:8]   # Sus reconstrucciones
        ], dim=0)
        
        # Guardar grid
        save_image(
            samples,
            os.path.join(output_dir, f'samples_epoch_{epoch}.png'),
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )
    model.train()

def train(
    model,
    train_loader,
    num_epochs=100,
    device='cuda',
    lr=1e-4,
    save_every=5,
    output_dir='outputs'
):
    """Función principal de entrenamiento"""
    
    # Crear directorio para outputs si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Optimizador
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Preparar inputs
            noisy_images, noise = add_noise(images)
            timesteps = sample_timesteps(batch_size).to(device)
            
            # Forward pass
            reconstructions, embeddings = model(noisy_images, timesteps)
            
            # Calcular pérdida
            loss = transfusion_loss(
                original=images,
                reconstruction=reconstructions,
                noise=noise.to(device),
                predicted_noise=None,  # No tenemos predicción de ruido por ahora
                embeddings=embeddings
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Actualizar métricas
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Guardar muestras periódicamente
            if batch_idx == 0 and epoch % save_every == 0:
                save_samples(model, images, epoch, output_dir)
        
        # Log del epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}')
        
        # Guardar checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(
                checkpoint,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )

if __name__ == "__main__":
    # Configuración
    config = DummyConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear modelo
    model = TransfusionImageProcessor(config).to(device)
    
    # Crear dataloader
    flickr_root = "/content/flickr30k/flickr30k-images"  # Ajusta esto a tu path
    train_loader = create_flickr_dataloader(flickr_root)
    
    # Entrenar
    train(
        model=model,
        train_loader=train_loader,
        num_epochs=10,
        device=device,
        save_every=5,
        output_dir='outputs/transfusion_flickr'
    )

