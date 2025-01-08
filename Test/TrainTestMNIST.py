""" This is a first test training file we are still working on more, and expanding the implementation, Encoder and Decoder chords for Flickr30k and multimodal training? """

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as vutils
from pathlib import Path
import os
from Athenea import Transfusion, CosineDecayWithWarmup, MNIST_config, Transformer, transfusion_config_to_model_args, DiffusionUtils


def save_samples(model, images, diff_utils, device, epoch, batch_idx, sample_dir='samples'):
    """Guarda muestras originales, con ruido y reconstruidas"""
    model.eval()
    with torch.no_grad():
        # Crear directorio si no existe
        Path(sample_dir).mkdir(parents=True, exist_ok=True)
        
        # Seleccionar una imagen de muestra
        image = images[0].unsqueeze(0)  # Tomar primera imagen de la batch
        
        # Añadir ruido
        t = torch.tensor([diff_utils.num_timesteps-1], device=device)
        noisy_image, _ = diff_utils.noisy_it(image[0], t)
        
        # Generar secuencia de denoising
        denoised_image = noisy_image.clone()
        sequence = [image[0], noisy_image]
        
        # Proceso de denoising
        for i in range(diff_utils.num_timesteps-1, -1, -diff_utils.num_timesteps//10):
            t_i = torch.tensor([i], device=device)
            patches = model.patch_ops.patchify(denoised_image)
            
            with autocast(dtype=torch.float16):
                modality_token_emb = model.forward_unbatched(
                    [torch.tensor([0], device=device), (patches, t_i), torch.tensor([model.BOI, model.EOI], device=device)],
                    ["text", "image", "text"]
                )
            
            pred_noise = model.patch_ops.unpatchify(modality_token_emb[1].squeeze(0))
            denoised_image = diff_utils.one_step_ddpm(denoised_image.unsqueeze(0), pred_noise.unsqueeze(0), t_i)[0]
            
            if i % (diff_utils.num_timesteps//10) == 0:
                sequence.append(denoised_image)
        
        # Guardar grid de imágenes
        sequence = torch.stack(sequence)
        grid = vutils.make_grid(sequence, nrow=4, normalize=True)
        save_path = f"{sample_dir}/epoch_{epoch}_batch_{batch_idx}.png"
        vutils.save_image(grid, save_path)
        
        print(f"\nGuardada secuencia de imágenes en {save_path}")
        print("Secuencia: Original -> Ruido -> Pasos de denoising")
        
        # Visualizar en terminal (ASCII art simple)
        img_ascii = sequence.mean(dim=1).cpu().numpy()  # Convertir a escala de grises
        for idx, img in enumerate(img_ascii):
            print(f"\nImagen {idx}:")
            for row in img[::2]:  # Mostrar cada segunda fila para compactar
                print(''.join(['#' if pixel > 0 else ' ' for pixel in row[::2]]))
    
    model.train()

def prepare_batch_inputs(images, labels, config, device):
    # Preparar tokens para toda la batch
    batch_size = images.size(0)
    
    # Convertir etiquetas a tokens iniciales
    input_tokens = []
    modality_strings = []
    
    for i in range(batch_size):
        # Secuencia: [label, BOI, image, EOI]
        text_start = labels[i].unsqueeze(0)  # [1]
        text_end = torch.tensor([config.BOI, config.EOI], device=device)  # [2]
        
        input_tokens.append([
            text_start,
            None,  # Placeholder para la imagen
            text_end
        ])
        modality_strings.append(["text", "image", "text"])
    
    return input_tokens, modality_strings

def train_epoch(model, train_loader, optimizer, scheduler, scaler, diff_utils, config, device, epoch):
    model.train()
    total_loss = 0
    step = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # Preparar inputs para la batch
        batch_tokens, batch_strings = prepare_batch_inputs(images, labels, config, device)
        
        # Actualizar learning rate
        current_lr = scheduler(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        optimizer.zero_grad()
        
        # Acumular pérdidas para la batch
        batch_loss = 0
        
        for i in range(batch_size):
            t = torch.randint(0, config.num_timesteps, (1,), device=device)
            noisy_image, noise = diff_utils.noisy_it(images[i], t)
            patches = model.patch_ops.patchify(noisy_image)
            batch_tokens[i][1] = (patches, t)
            
            with autocast(dtype=torch.float16):
                modality_token_emb = model.forward_unbatched(batch_tokens[i], batch_strings[i])
                predicted_noise = model.patch_ops.unpatchify(modality_token_emb[1].squeeze(0))
                diff_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                
                text_start_loss = torch.nn.functional.cross_entropy(
                    modality_token_emb[0].squeeze(0),
                    batch_tokens[i][0]
                )
                
                text_end_loss = torch.nn.functional.cross_entropy(
                    modality_token_emb[2].squeeze(0),
                    batch_tokens[i][2]
                )
                
                loss = text_start_loss + text_end_loss + config.balancing_coeff * diff_loss
                loss = loss / batch_size
            
            scaler.scale(loss).backward()
            batch_loss += loss.item()
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clipnorm)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += batch_loss
        step += 1
        
        if batch_idx % config.log_interval == 0:
            print(f'Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {batch_loss:.6f}\t'
                  f'LR: {current_lr:.2e}')
            
            # Guardar y mostrar muestras
            save_samples(model, images, diff_utils, device, epoch, batch_idx)
    
    return total_loss / len(train_loader.dataset)

def main():
    # Configuración
    config = MNIST_config()
    device = config.device
    print(f"Using device: {device}")
    
    # Dataset y DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Modelo
    model_args = transfusion_config_to_model_args(config)
    transformer = Transformer(model_args)
    model = Transfusion(transformer, config).to(device)
    
    # Utilidades de difusión
    diff_utils = DiffusionUtils(linear_schedule=True, config=config)
    
    # Optimizador
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        betas=(config.beta1, config.beta2),
        device_type=device.type
    )
    
    # Scheduler
    scheduler = CosineDecayWithWarmup(
        warmup_steps=config.warmup_steps,
        max_learning_rate=config.max_lr,
        decay_steps=config.decay_steps,
        min_learning_rate=config.min_lr
    )
    
    # Gradient Scaler para mixed precision
    scaler = GradScaler()
    
    # Entrenamiento
    print("Starting training...")
    for epoch in range(config.num_steps):
        avg_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            diff_utils=diff_utils,
            config=config,
            device=device,
            epoch=epoch
        )
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
        
        

if __name__ == "__main__":
    main()