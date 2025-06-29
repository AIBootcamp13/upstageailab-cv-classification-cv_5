"""
Document Type Classification with WandB Integration
Based on the original baseline notebook
"""

import os
import time
from pathlib import Path

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Import our custom modules
from config import (
    WANDB_CONFIG, EXPERIMENT_CONFIG, DATA_CONFIG, 
    get_wandb_config, get_experiment_name, validate_config
)
from wandb_utils import (
    init_wandb, log_metrics, log_model_info, log_system_info,
    finish_run, create_run_name, log_confusion_matrix
)

class ImageDataset(Dataset):
    """Dataset class for image classification"""
    
    def __init__(self, csv_path, image_dir, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.is_train:
            image_name, target = row['image'], row['target']
        else:
            image_name, target = row['ID'], 0  # Test set doesn't have targets
            
        image_path = os.path.join(self.image_dir, image_name)
        image = np.array(Image.open(image_path))
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, target

def get_transforms(img_size, is_train=True):
    """Get data augmentation transforms"""
    
    if is_train:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return transform

def train_one_epoch(loader, model, optimizer, loss_fn, device, epoch):
    """Train for one epoch with WandB logging"""
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []
    
    pbar = tqdm(loader, desc=f'Training Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        preds_list.extend(predictions.detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Log batch metrics every 50 batches
        if batch_idx % 50 == 0:
            log_metrics({
                'batch_loss': loss.item(),
                'batch': epoch * len(loader) + batch_idx
            }, commit=False)
    
    # Calculate epoch metrics
    avg_loss = train_loss / len(loader)
    accuracy = accuracy_score(targets_list, preds_list)
    f1 = f1_score(targets_list, preds_list, average='macro')
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy,
        'train_f1': f1
    }

def validate_one_epoch(loader, model, loss_fn, device, epoch):
    """Validate for one epoch with WandB logging"""
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Validation Epoch {epoch}')
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            
            val_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            preds_list.extend(predictions.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
    
    # Calculate validation metrics
    avg_loss = val_loss / len(loader)
    accuracy = accuracy_score(targets_list, preds_list)
    f1 = f1_score(targets_list, preds_list, average='macro')
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_f1': f1,
        'predictions': preds_list,
        'targets': targets_list
    }

def train_model():
    """Main training function with WandB integration"""
    
    # Validate configuration
    if not validate_config():
        print("Configuration validation failed. Please fix the errors above.")
        return
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create run name
    run_name = create_run_name(
        model_name=EXPERIMENT_CONFIG['model_name'],
        experiment_type="baseline_wandb"
    )
    
    # Initialize WandB
    wandb_config = get_wandb_config()
    run = init_wandb(wandb_config, run_name=run_name)
    
    try:
        # Log system info
        log_system_info()
        
        # Data preparation
        print("Loading datasets...")
        
        # Read training data and create train/val split
        train_df = pd.read_csv(DATA_CONFIG['train_csv'])
        
        # Create train/validation split
        train_idx, val_idx = train_test_split(
            range(len(train_df)),
            test_size=DATA_CONFIG['val_split'],
            stratify=train_df['target'],
            random_state=DATA_CONFIG['random_seed']
        )
        
        # Create train and validation CSV files temporarily
        train_subset = train_df.iloc[train_idx].reset_index(drop=True)
        val_subset = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Get transforms
        img_size = EXPERIMENT_CONFIG['img_size']
        train_transform = get_transforms(img_size, is_train=True)
        val_transform = get_transforms(img_size, is_train=False)
        
        # Create datasets
        train_dataset = ImageDataset(
            csv_path=DATA_CONFIG['train_csv'],
            image_dir=DATA_CONFIG['train_dir'],
            transform=train_transform,
            is_train=True
        )
        
        # Filter datasets for train/val split
        train_dataset.df = train_subset
        
        val_dataset = ImageDataset(
            csv_path=DATA_CONFIG['train_csv'],
            image_dir=DATA_CONFIG['train_dir'],
            transform=val_transform,
            is_train=True
        )
        val_dataset.df = val_subset
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            shuffle=True,
            num_workers=EXPERIMENT_CONFIG['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            shuffle=False,
            num_workers=EXPERIMENT_CONFIG['num_workers'],
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Model setup
        print("Setting up model...")
        model = timm.create_model(
            EXPERIMENT_CONFIG['model_name'],
            pretrained=EXPERIMENT_CONFIG['pretrained'],
            num_classes=EXPERIMENT_CONFIG['num_classes']
        ).to(device)
        
        # Log model info
        log_model_info(model, input_shape=(3, img_size, img_size))
        
        # Loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(
            model.parameters(), 
            lr=EXPERIMENT_CONFIG['learning_rate'],
            weight_decay=EXPERIMENT_CONFIG['weight_decay']
        )
        
        # Scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=EXPERIMENT_CONFIG['epochs'],
            eta_min=EXPERIMENT_CONFIG['min_lr']
        )
        
        # Training loop
        best_val_f1 = 0.0
        best_epoch = 0
        
        print("Starting training...")
        for epoch in range(EXPERIMENT_CONFIG['epochs']):
            start_time = time.time()
            
            # Train
            train_metrics = train_one_epoch(
                train_loader, model, optimizer, loss_fn, device, epoch
            )
            
            # Validate
            val_metrics = validate_one_epoch(
                val_loader, model, loss_fn, device, epoch
            )
            
            # Step scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Combine metrics
            all_metrics = {
                **train_metrics,
                **val_metrics,
                'learning_rate': current_lr,
                'epoch': epoch,
                'epoch_time': epoch_time
            }
            
            # Log to WandB
            log_metrics(all_metrics)
            
            # Print progress
            print(f"Epoch {epoch+1}/{EXPERIMENT_CONFIG['epochs']}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_accuracy']:.4f}, "
                  f"Train F1: {train_metrics['train_f1']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                  f"Val F1: {val_metrics['val_f1']:.4f}")
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_metrics['val_f1'] > best_val_f1:
                best_val_f1 = val_metrics['val_f1']
                best_epoch = epoch
                
                # Save model
                os.makedirs(os.path.dirname(EXPERIMENT_CONFIG['best_model_path']), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'config': EXPERIMENT_CONFIG
                }, EXPERIMENT_CONFIG['best_model_path'])
                
                print(f"  New best model saved! Val F1: {best_val_f1:.4f}")
                
                # Log confusion matrix for best model
                log_confusion_matrix(
                    val_metrics['targets'], 
                    val_metrics['predictions']
                )
        
        print(f"\nTraining completed!")
        print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
        
        # Log final summary
        log_metrics({
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'total_epochs': EXPERIMENT_CONFIG['epochs']
        })
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        # Always finish WandB run
        finish_run()

if __name__ == "__main__":
    # Add model paths to experiment config
    from config import MODEL_PATHS
    EXPERIMENT_CONFIG.update({
        'best_model_path': MODEL_PATHS['best_model'],
        'latest_model_path': MODEL_PATHS['latest_model']
    })
    
    train_model()
