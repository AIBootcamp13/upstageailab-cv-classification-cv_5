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
from device_utils import setup_training_device, get_dataloader_config

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
            image_name, target = row['ID'], row['target']  # 'image' → 'ID'로 수정
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
    
    # Check if loader is empty
    if len(loader) == 0:
        print(f"Warning: Validation loader is empty for epoch {epoch}")
        return {
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'val_f1': 0.0,
            'predictions': None,
            'targets': None
        }
    
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
    
    # Additional safety check
    if len(preds_list) == 0 or len(targets_list) == 0:
        print(f"Warning: No validation data processed for epoch {epoch}")
        return {
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'val_f1': 0.0,
            'predictions': None,
            'targets': None
        }
    
    # Calculate validation metrics
    avg_loss = val_loss / len(loader)
    accuracy = accuracy_score(targets_list, preds_list)
    f1 = f1_score(targets_list, preds_list, average='macro')
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_f1': f1,
        'predictions': np.array(preds_list),
        'targets': np.array(targets_list)
    }

import logging
from datetime import datetime

# Configure logging
def setup_logging():
    """Setup file logging for training"""
    from config import LOGGING_CONFIG, LOGS_DIR
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOGS_DIR}/training_{timestamp}.log"
    
    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['log_level']),
        format=LOGGING_CONFIG['log_format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created: {log_file}")
    return logger, log_file

def print_configuration():
    """Print current configuration for logging"""
    print("\n" + "="*60)
    print("🚀 CV-Classification Training Configuration")
    print("="*60)
    print(f"📋 Model Settings:")
    print(f"  Model: {EXPERIMENT_CONFIG['model_name']}")
    print(f"  Number of classes: {EXPERIMENT_CONFIG['num_classes']}")
    print(f"  Pretrained: {EXPERIMENT_CONFIG['pretrained']}")
    print(f"\n📊 Training Settings:")
    print(f"  Image size: {EXPERIMENT_CONFIG['img_size']}")
    print(f"  Batch size: {EXPERIMENT_CONFIG['batch_size']}")
    print(f"  Learning rate: {EXPERIMENT_CONFIG['learning_rate']}")
    print(f"  Epochs: {EXPERIMENT_CONFIG['epochs']}")
    print(f"  Weight decay: {EXPERIMENT_CONFIG['weight_decay']}")
    print(f"\n🔧 Optimization Settings:")
    print(f"  Optimizer: {EXPERIMENT_CONFIG['optimizer']}")
    print(f"  Scheduler: {EXPERIMENT_CONFIG['scheduler']}")
    print(f"  Min LR: {EXPERIMENT_CONFIG['min_lr']}")
    print(f"\n📁 Data Settings:")
    print(f"  Validation split: {DATA_CONFIG['val_split']}")
    print(f"  Random seed: {DATA_CONFIG['random_seed']}")
    print("="*60 + "\n")

def predict_test_data(model, device, test_loader, output_file="submission.csv"):
    """Generate predictions on test data and save to CSV"""
    model.eval()
    predictions = []
    image_ids = []
    
    print("\n🔮 Generating predictions on test data...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Predicting")):
            images = images.to(device)
            outputs = model(images)
            batch_predictions = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(batch_predictions)
    
    # Create submission DataFrame
    test_df = pd.read_csv(DATA_CONFIG['test_csv'])
    submission_df = pd.DataFrame({
        'ID': test_df['ID'],
        'target': predictions
    })
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    submission_df.to_csv(output_file, index=False)
    
    print(f"📄 Predictions saved to: {output_file}")
    print(f"📊 Total predictions: {len(predictions)}")
    print(f"📈 Prediction distribution:")
    
    # Show prediction distribution
    unique, counts = np.unique(predictions, return_counts=True)
    for class_id, count in zip(unique, counts):
        percentage = count / len(predictions) * 100
        print(f"  Class {class_id}: {count} ({percentage:.1f}%)")
    
    return submission_df

def predict_test_data(model, device, test_loader, output_file="submission.csv"):
    """Generate predictions on test data and save to CSV"""
    model.eval()
    predictions = []
    
    print("\n🔮 Generating predictions on test data...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Predicting")):
            images = images.to(device)
            outputs = model(images)
            batch_predictions = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(batch_predictions)
    
    # Create submission DataFrame
    test_df = pd.read_csv(DATA_CONFIG['test_csv'])
    submission_df = pd.DataFrame({
        'ID': test_df['ID'],
        'target': predictions
    })
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    submission_df.to_csv(output_file, index=False)
    
    print(f"📄 Predictions saved to: {output_file}")
    print(f"📊 Total predictions: {len(predictions)}")
    
    return submission_df
def train_model():
    """Main training function with WandB integration"""
    
    # Setup logging
    logger, log_file = setup_logging()
    logger.info("Starting CV-Classification training")
    
    # Print configuration at start
    print_configuration()
    
    # Validate configuration
    if not validate_config():
        print("Configuration validation failed. Please fix the errors above.")
        return
    
    # Device setup with macOS optimization
    print("🔧 디바이스 설정 중...")
    device, device_type = setup_training_device()
    
    # Get optimized DataLoader configuration
    dataloader_config = get_dataloader_config(device_type)
    
    # Create run name
    run_name = create_run_name(
        model_name=EXPERIMENT_CONFIG['model_name'],
        experiment_type="baseline_wandb"
    )
    
    # Initialize WandB
    wandb_config = get_wandb_config()
    run = init_wandb(wandb_config, run_name=run_name)
    
    try:
        # Log system info including device information
        log_system_info()
        log_metrics({
            'device_type': device_type,
            'device_name': str(device),
            'dataloader_config': dataloader_config
        }, commit=False)
        
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
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            shuffle=True,
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            shuffle=False,
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory']
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
        total_start_time = time.time()
        
        print(f"🚀 Starting training for {EXPERIMENT_CONFIG['epochs']} epochs...")
        print(f"📊 Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
        print("="*60)
        
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
            
            # Print progress with better formatting
            progress_percent = (epoch + 1) / EXPERIMENT_CONFIG['epochs'] * 100
            print(f"\n📊 Epoch {epoch+1}/{EXPERIMENT_CONFIG['epochs']} ({progress_percent:.1f}%):")
            print(f"  🟢 Train - Loss: {train_metrics['train_loss']:.4f} | "
                  f"Acc: {train_metrics['train_accuracy']:.4f} | "
                  f"F1: {train_metrics['train_f1']:.4f}")
            print(f"  🟡 Valid - Loss: {val_metrics['val_loss']:.4f} | "
                  f"Acc: {val_metrics['val_accuracy']:.4f} | "
                  f"F1: {val_metrics['val_f1']:.4f}")
            print(f"  ⚙️  LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
            
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
                
                print(f"  🎆 NEW BEST MODEL! Val F1: {best_val_f1:.4f} (saved to {EXPERIMENT_CONFIG['best_model_path']})")
                print(f"  📥 Model checkpoint saved at epoch {epoch+1}")
                
                # Log confusion matrix for best model
                log_confusion_matrix(
                    val_metrics['targets'], 
                    val_metrics['predictions']
                )
        
        # Calculate total training time
        total_training_time = time.time() - total_start_time
        
        # Create test dataset and loader for predictions
        test_transform = get_transforms(img_size, is_train=False)
        test_dataset = ImageDataset(
            csv_path=DATA_CONFIG['test_csv'],
            image_dir=DATA_CONFIG['test_dir'],
            transform=test_transform,
            is_train=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            shuffle=False,
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory']
        )
        
        # Generate predictions on test data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = f"codes/submission_{timestamp}.csv"
        predict_test_data(model, device, test_loader, submission_file)
        
        # Log final results
        logger.info(f"Training completed in {total_training_time:.1f} seconds")
        logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
        logger.info(f"Best model saved to: {EXPERIMENT_CONFIG['best_model_path']}")
        logger.info(f"Predictions saved to: {submission_file}")
        
        # Print final summary
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED!")
        print("="*60)
        print(f"📊 Training Summary:")
        print(f"  Total epochs: {EXPERIMENT_CONFIG['epochs']}")
        print(f"  Total training time: {total_training_time/60:.1f} minutes ({total_training_time:.1f} seconds)")
        print(f"  Average time per epoch: {total_training_time/EXPERIMENT_CONFIG['epochs']:.1f} seconds")
        print(f"\n🏆 Best Results:")
        print(f"  Best validation F1: {best_val_f1:.4f}")
        print(f"  Best epoch: {best_epoch+1}/{EXPERIMENT_CONFIG['epochs']}")
        print(f"  Best model saved to: {EXPERIMENT_CONFIG['best_model_path']}")
        print(f"\n📄 Output Files:")
        print(f"  Submission file: {submission_file}")
        print(f"  Log file: {log_file}")
        print(f"  Model checkpoint: {EXPERIMENT_CONFIG['best_model_path']}")
        print(f"\n📊 Dataset Information:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Image size: {EXPERIMENT_CONFIG['img_size']}x{EXPERIMENT_CONFIG['img_size']}")
        print(f"  Batch size: {EXPERIMENT_CONFIG['batch_size']}")
        print("="*60)
        
        # Log final summary
        log_metrics({
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'total_epochs': EXPERIMENT_CONFIG['epochs'],
            'total_training_time': total_training_time,
            'avg_epoch_time': total_training_time / EXPERIMENT_CONFIG['epochs']
        })
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        # Always finish WandB run
        finish_run()

if __name__ == "__main__":
    import sys
    
    # Add model paths to experiment config
    from config import MODEL_PATHS
    EXPERIMENT_CONFIG.update({
        'best_model_path': MODEL_PATHS['best_model'],
        'latest_model_path': MODEL_PATHS['latest_model']
    })
    
    # Check for dry-run mode
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("🧪 DRY RUN 모드: 임포트 및 설정 테스트만 수행")
        print("="*60)
        
        # Test imports and basic setup
        try:
            print("1. 환경 검증 중...")
            if validate_config():
                print("   ✅ 설정 검증 성공")
            else:
                print("   ❌ 설정 검증 실패")
                sys.exit(1)
            
            print("\n2. 디바이스 설정 중...")
            device, device_type = setup_training_device()
            print(f"   ✅ 디바이스: {device} ({device_type})")
            
            print("\n3. WandB 연결 테스트 중...")
            wandb_config = get_wandb_config()
            print("   ✅ WandB 설정 로드 성공")
            
            print("\n4. 데이터 경로 확인 중...")
            train_csv_exists = os.path.exists(DATA_CONFIG['train_csv'])
            train_dir_exists = os.path.exists(DATA_CONFIG['train_dir'])
            print(f"   train.csv: {'✅' if train_csv_exists else '❌'}")
            print(f"   train/ 폴더: {'✅' if train_dir_exists else '❌'}")
            
            print("\n🎉 DRY RUN 완료: 모든 준비가 완료되었습니다!")
            print("실제 학습을 시작하려면 --dry-run 옵션 없이 실행하세요.")
            
        except Exception as e:
            print(f"\n❌ DRY RUN 실패: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("🚀 전체 학습 시작...")
        print(f"🕰️ 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        train_model()
