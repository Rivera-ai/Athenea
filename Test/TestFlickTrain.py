import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import random
from torchvision import transforms
import ast
from Athenea import Transfusion, CosineDecayWithWarmup, Flickr30kConfig, Transformer, transfusion_config_to_model_args, DiffusionUtils

class Flickr30kDataset(Dataset):
    def __init__(
        self, 
        csv_file: str,
        image_dir: str,
        tokenizer,
        split: str = 'train',
        image_size: int = 256,
        max_length: int = 512
    ):
        """
        Args:
            csv_file: Path al archivo CSV de Flickr30k
            image_dir: Directorio que contiene las imágenes
            tokenizer: Tokenizador de Llama
            split: 'train', 'val' o 'test'
            image_size: Tamaño al que redimensionar las imágenes
            max_length: Longitud máxima para los tokens
        """
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Cargar y procesar CSV
        df = pd.read_csv(csv_file)
        # Filtrar por split
        self.data = df[df['split'] == split].reset_index(drop=True)
        
        # Convertir strings de listas a listas reales
        self.data['captions'] = self.data['raw'].apply(ast.literal_eval)
        
        # Transformaciones de imagen
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Loaded {len(self.data)} images for {split} split")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Cargar y procesar imagen
        img_path = self.image_dir / row['filename']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # En caso de error, retornar el siguiente item
            return self.__getitem__((idx + 1) % len(self))
            
        # Seleccionar una caption aleatoria de las 5 disponibles
        caption = random.choice(row['captions'])
        
        # Tokenizar caption
        encoded = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'caption': caption,  # Útil para debugging
            'image_id': row['img_id']
        }

def create_flickr_dataloaders(
    csv_path: str,
    image_dir: str,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 256,
    max_length: int = 512
):
    """
    Crea los dataloaders para train/val/test
    """
    datasets = {
        split: Flickr30kDataset(
            csv_file=csv_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            split=split,
            image_size=image_size,
            max_length=max_length
        )
        for split in ['train', 'val', 'test']
    }
    
    dataloaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
        for split, dataset in datasets.items()
    }
    
    return dataloaders

def save_samples(model, batch, diff_utils, device, epoch, batch_idx, sample_dir='flickr_samples'):
    """Guarda muestras originales, con ruido y reconstruidas"""
    model.eval()
    try:
        with torch.no_grad():
            Path(sample_dir).mkdir(parents=True, exist_ok=True)
            
            # Preparar imagen y caption
            image = batch['image'][0].to(device)  # [C, H, W]
            caption_ids = batch['input_ids'][0].to(device)
            
            # Guardar imagen original
            sequence = [image.cpu()]
            
            # Usar generate directamente (pero con un solo valor de retorno)
            with torch.autocast(device_type='cuda'):
                modality_token_emb = model.forward_unbatched(
                    [caption_ids, 
                     (model.patch_ops.patchify(torch.randn_like(image)), torch.tensor([0], device=device)),
                     torch.tensor([model.BOI, model.EOI], device=device)],
                    ["text", "image", "text"]
                )
                    
                # Obtener la imagen generada
                generated_image = model.patch_ops.unpatchify(modality_token_emb[1].squeeze(0))
                sequence.append(generated_image.cpu())
            
            # Guardar grid
            try:
                sequence = torch.stack(sequence)
                grid = vutils.make_grid(sequence, nrow=2, normalize=True)
                save_path = f"{sample_dir}/epoch_{epoch}_batch_{batch_idx}.png"
                vutils.save_image(grid, save_path)
            except Exception as e:
                print(f"Error al guardar grid: {str(e)}")
    
    except Exception as e:
        print(f"Error general en save_samples: {str(e)}")
    
    finally:
        model.train()

def prepare_batch_inputs(batch, config, device):
    batch_size = batch['image'].size(0)
    input_tokens = []
    modality_strings = []
    
    for i in range(batch_size):
        # Validar y clipear índices
        text_ids = batch['input_ids'][i].to(device)
        
        # Asegurarnos que los índices están dentro del rango válido
        if text_ids.max() >= config.lm_output_units:
            print(f"WARNING: Clipping token indices from {text_ids.max()} to {config.lm_output_units-1}")
            text_ids = torch.clamp(text_ids, 0, config.lm_output_units-1)
        
        text_end = torch.tensor([config.BOI, config.EOI], device=device)
        
        input_tokens.append([
            text_ids,
            None,  # Placeholder para la imagen
            text_end
        ])
        modality_strings.append(["text", "image", "text"])
    
    return input_tokens, modality_strings

def train_steps(model, train_loader, optimizer, scheduler, diff_utils, config, device, total_steps):
    model.train()
    step = 0
    running_loss = 0.0
    log_interval = 10
    
    pbar = tqdm(total=total_steps, desc='Training', position=0)
    current_lr = config.max_lr
    
    while step < total_steps:
        for batch in train_loader:
            try:
                if step >= total_steps:
                    break
                    
                images = batch['image'].to(device)
                batch_size = images.size(0)
                
                batch_tokens, batch_strings = prepare_batch_inputs(batch, config, device)
                
                # Actualizar learning rate
                current_lr = scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                    
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for i in range(batch_size):
                    # Proceso de diffusion con manejo de errores
                    t = torch.randint(0, config.num_timesteps, (1,), device=device)
                    noisy_image, noise = diff_utils.noisy_it(images[i], t)
                    patches = model.patch_ops.patchify(noisy_image)
                    batch_tokens[i][1] = (patches, t)
                    
                    # Forward pass
                    modality_token_emb = model.forward_unbatched(batch_tokens[i], batch_strings[i])
                    
                    # Pérdidas
                    predicted_noise = model.patch_ops.unpatchify(modality_token_emb[1].squeeze(0))
                    diff_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                    
                    text_start_loss = torch.nn.functional.cross_entropy(
                        modality_token_emb[0].squeeze(0),
                        batch_tokens[i][0],
                        ignore_index=config.IGNORE_TOKEN
                    )
                    
                    text_end_loss = torch.nn.functional.cross_entropy(
                        modality_token_emb[2].squeeze(0),
                        batch_tokens[i][2]
                    )
                    
                    loss = text_start_loss + text_end_loss + config.balancing_coeff * diff_loss
                    loss = loss / batch_size
                    
                    loss.backward()
                    batch_loss += loss.item()
                
                # Clip gradients y optimizar
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clipnorm)
                optimizer.step()
                
                running_loss += batch_loss
                
                if step % log_interval == 0 and step > 0:
                    avg_loss = running_loss / log_interval
                    pbar.set_description(f'Loss: {avg_loss:.4f} | LR: {current_lr:.2e}')
                    running_loss = 0.0
                    
                    try:
                        save_samples(model, batch, diff_utils, device, step, batch_idx=step)
                        print(f"Saved samples at step {step}")
                    except Exception as e:
                        print(f"Error guardando muestras: {str(e)}")
                
                pbar.update(1)
                step += 1
                
            except Exception as e:
                print(f"Error en el batch: {str(e)}")
                continue
    
    pbar.close()

def main():
    # Inicializar tokenizer primero
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        model_max_length=512,
        padding_side='left',
        truncation_side='left',
        bos_token='<s>',
        eos_token='</s>',
        unk_token='<unk>',
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configurar basándonos en el tokenizer
    config = Flickr30kConfig()
    config.vocab_size = len(tokenizer)  # Asegurarnos que coincide con el tokenizer
    device = config.device
    
    # Ajustar los tokens especiales para estar dentro del rango
    vocab_offset = config.vocab_size
    config.BOI = torch.tensor(vocab_offset, dtype=torch.long)
    config.IGNORE_TOKEN = torch.tensor(vocab_offset + 1, dtype=torch.long)
    config.EOI = torch.tensor(vocab_offset + 2, dtype=torch.long)
    config.EOS = torch.tensor(vocab_offset + 3, dtype=torch.long)
    
    # Actualizar lm_output_units para incluir los tokens especiales
    config.lm_output_units = vocab_offset + 4  # vocab_size + 4 tokens especiales
    
    # Configurar el resto de parámetros
    config.in_channels = 3
    config.H = config.W = 256
    config.N = (config.H * config.W) // (config.patch_size ** 2)
    
    # Validación de configuración
    print("\n=== Configuración ===")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Total output units: {config.lm_output_units}")
    print(f"Patch size: {config.patch_size}")
    print(f"Image size: {config.H}x{config.W}")
    print(f"Patches: {config.N}")
    
    # Validar dimensiones
    assert config.N * config.patch_size**2 == config.H * config.W, "Dimensiones incorrectas"
    assert config.lm_output_units > config.vocab_size, "Vocabulario no incluye tokens especiales"
    
    # Dataset y DataLoader
    dataloaders = create_flickr_dataloaders(csv_path='/teamspace/studios/this_studio/flickr30k/flickr_annotations_30k.csv', 
                                      image_dir='/teamspace/studios/this_studio/flickr30k/flickr30k-images', 
                                      tokenizer=tokenizer)

    train_loader = dataloaders['train']
    
    # Modelo y utilidades
    model_args = transfusion_config_to_model_args(config)
    transformer = Transformer(model_args)
    model = Transfusion(transformer, config).to(device)
    
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
    total_steps = 50000  # Ajustar según necesites
    
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