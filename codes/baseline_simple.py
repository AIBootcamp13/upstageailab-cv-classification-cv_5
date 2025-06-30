"""
Document Type Classification - Simple Baseline (matches official baseline)
Simplified version that matches the official baseline structure exactly
With macOS optimization support
"""

import os
import time
import random
import logging
from datetime import datetime

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Import configuration
from config import BASELINE_CONFIG, DATA_CONFIG, LOGS_DIR

def setup_logging():
    """Setup logging for baseline script"""
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOGS_DIR}/baseline_simple_{timestamp}.log"
    
    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Baseline simple training started - Log file: {log_file}")
    return logger, log_file

# Set seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

class ImageDataset(Dataset):
    """Simple dataset class matching baseline structure"""
    
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    """Training function matching baseline structure"""
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        preds = model(image)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return ret

def setup_device():
    """스마트 디바이스 설정 - 호환성과 최적화 균형"""
    
    # 호환성 모드 강제 사용
    if BASELINE_CONFIG.get('compatibility_mode', False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔒 호환성 모드: {device} (공식 베이스라인 100% 호환)")
        return device, {"pin_memory": True, "num_workers": 0}
    
    # macOS 최적화 모드
    if BASELINE_CONFIG.get('enable_macos_optimization', True):
        try:
            from device_utils import setup_training_device, get_dataloader_config
            device, device_type = setup_training_device()
            dataloader_config = get_dataloader_config(device_type)
            print(f"🚀 최적화 모드: {device} ({device_type})")
            return device, dataloader_config
        except (ImportError, AttributeError, RuntimeError) as e:
            print(f"⚠️  최적화 모듈 오류: {e}")
            print("🗞️  호환성 모드로 fallback")
    
    # 기본 방식 (원본 호환)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📟 기본 모드: {device}")
    return device, {"pin_memory": True, "num_workers": 0}

def main():
    """Main function matching baseline structure exactly"""
    
    # Setup logging
    logger, log_file = setup_logging()
    
    # Device setup with smart optimization
    device, dataloader_config = setup_device()
    logger.info(f"Device setup completed: {device}")
    
    # Configuration from baseline config
    img_size = BASELINE_CONFIG['img_size']
    batch_size = BASELINE_CONFIG['batch_size'] 
    learning_rate = BASELINE_CONFIG['learning_rate']
    epochs = BASELINE_CONFIG['epochs']
    model_name = BASELINE_CONFIG['model_name']
    
    config_info = f"Configuration: Model={model_name}, Image size={img_size}, Batch size={batch_size}, Learning rate={learning_rate}, Epochs={epochs}"
    print(config_info)
    logger.info(config_info)
    
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    
    # Data transforms (matching baseline exactly)
    trn_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    tst_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Datasets (using our data paths)
    trn_dataset = ImageDataset(
        DATA_CONFIG['train_csv'],
        DATA_CONFIG['train_dir'],
        transform=trn_transform
    )
    tst_dataset = ImageDataset(
        DATA_CONFIG['test_csv'],
        DATA_CONFIG['test_dir'],
        transform=tst_transform
    )
    print(f"Training samples: {len(trn_dataset)}")
    print(f"Test samples: {len(tst_dataset)}")
    logger.info(f"Dataset loaded - Training: {len(trn_dataset)}, Test: {len(tst_dataset)}")

    # DataLoaders with optimized settings
    trn_loader = DataLoader(
        trn_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config['pin_memory'],
        drop_last=False
    )
    tst_loader = DataLoader(
        tst_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config['pin_memory']
    )

    # Model setup (matching baseline exactly)
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model loaded: {model_name}")
    logger.info(f"Model loaded: {model_name} with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop (matching baseline structure)
    print("Starting training...")
    logger.info(f"Starting training for {epochs} epochs")
    training_start_time = time.time()
    for epoch in range(epochs):
        ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device)
        ret['epoch'] = epoch

        log = ""
        for k, v in ret.items():
            log += f"{k}: {v:.4f}\n"
        print(log)
        logger.info(f"Epoch {epoch+1}/{epochs} - {', '.join([f'{k}: {v:.4f}' for k, v in ret.items()])}")

    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Inference (matching baseline exactly)
    print("Starting inference...")
    logger.info("Starting inference on test data")
    inference_start_time = time.time()
    preds_list = []

    model.eval()
    for image, _ in tqdm(tst_loader):
        image = image.to(device)

        with torch.no_grad():
            preds = model(image)
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())

    # Save predictions (matching baseline format)
    pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
    pred_df['target'] = preds_list

    # Verify format matches sample submission
    sample_submission_df = pd.read_csv(DATA_CONFIG['test_csv'])
    assert (sample_submission_df['ID'] == pred_df['ID']).all()

    # Save results
    output_file = "pred_baseline.csv"
    pred_df.to_csv(output_file, index=False)
    
    inference_time = time.time() - inference_start_time
    total_time = time.time() - (training_start_time - training_time + inference_start_time)
    
    print(f"Predictions saved to: {output_file}")
    print("\nFirst 5 predictions:")
    print(pred_df.head())
    
    # Log final summary
    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Predictions saved to: {output_file}")
    logger.info(f"Total predictions: {len(preds_list)}")
    
    # Log prediction distribution
    unique, counts = np.unique(preds_list, return_counts=True)
    pred_dist = {f"class_{u}": c for u, c in zip(unique, counts)}
    logger.info(f"Prediction distribution: {pred_dist}")
    
    print(f"\n📊 Final Summary:")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Predictions: {len(preds_list)}")
    print(f"  Log file: {log_file}")

if __name__ == "__main__":
    main()
