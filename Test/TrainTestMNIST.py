""" This is a first test training file we are still working on more, and expanding the implementation, Encoder and Decoder chords for Flickr30k and multimodal training? """

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
        print(f"Shape de toda la secuencia: {sequence.shape}")
        print("\nShapes individuales:")
        for idx, img in enumerate(sequence):
            print(f"Imagen {idx}: {img.shape}")

    
    model.train()

def prepare_batch_inputs(images, labels, config, device):
    batch_size = images.size(0)
    input_tokens = []
    modality_strings = []
    
    for i in range(batch_size):
        text_start = labels[i].unsqueeze(0)
        text_end = torch.tensor([config.BOI, config.EOI], device=device)
        
        input_tokens.append([
            text_start,
            None,
            text_end
        ])
        modality_strings.append(["text", "image", "text"])
    
    return input_tokens, modality_strings

def train_steps(model, train_loader, optimizer, scheduler, diff_utils, config, device, total_steps):
    model.train()
    step = 0
    running_loss = 0
    log_interval = 100  # Ajusta esto según necesites
    
    while step < total_steps:
        # Reiniciar el DataLoader cuando se agote
        for images, labels in train_loader:
            if step >= total_steps:
                break
                
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Preparar inputs para la batch
            batch_tokens, batch_strings = prepare_batch_inputs(images, labels, config, device)
            
            # Actualizar learning rate
            current_lr = scheduler(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                
            optimizer.zero_grad()
            batch_loss = 0
            
            for i in range(batch_size):
                t = torch.randint(0, config.num_timesteps, (1,), device=device)
                noisy_image, noise = diff_utils.noisy_it(images[i], t)
                patches = model.patch_ops.patchify(noisy_image)
                batch_tokens[i][1] = (patches, t)
                
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
                
                loss.backward()
                batch_loss += loss.item()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clipnorm)
            optimizer.step()
            
            running_loss += batch_loss
            
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval if step > 0 else running_loss
                print(f'Step: {step}/{total_steps} '
                      f'Loss: {avg_loss:.6f} '
                      f'LR: {current_lr:.2e}')
                running_loss = 0
                
                # Guardar y mostrar muestras
                save_samples(model, images, diff_utils, device, step, batch_idx=None)
            
            step += 1

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
    
    batch_size = 128
    total_steps = 50000  # Ajusta esto según necesites
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modelo y resto de la configuración igual que antes...
    model_args = transfusion_config_to_model_args(config)
    transformer = Transformer(model_args)
    model = Transfusion(transformer, config).to(device)
    
    diff_utils = DiffusionUtils(linear_schedule=True, config=config)
    
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
    
    # Entrenamiento por pasos
    print("Starting training...")
    train_steps(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        diff_utils=diff_utils,
        config=config,
        device=device,
        total_steps=total_steps
    )

if __name__ == "__main__":
    main()