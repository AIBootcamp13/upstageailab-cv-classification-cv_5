"""
Configuration module for CV Classification project with WandB integration
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# WandB Configuration
WANDB_CONFIG = {
    "api_key": os.getenv("WANDB_API_KEY"),
    "project": os.getenv("WANDB_PROJECT", "cv-classification"),
    "entity": os.getenv("WANDB_ENTITY"),
    "mode": os.getenv("WANDB_MODE", "online"),  # online, offline, disabled
    "tags": os.getenv("WANDB_TAGS", "cv,classification,document,upstage").split(","),
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    # Model settings
    "model_name": "resnet34",
    "num_classes": 17,
    "pretrained": True,
    
    # Training settings
    "img_size": 224,  # Increased from 32 for better performance
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 50,
    "num_workers": 4,
    
    # Optimizer settings
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "momentum": 0.9,
    
    # Scheduler settings
    "scheduler": "cosine",
    "min_lr": 1e-6,
    
    # Data augmentation
    "augmentation": {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation": 15,
        "brightness": 0.2,
        "contrast": 0.2,
    },
    
    # Early stopping
    "early_stopping": {
        "enabled": True,
        "patience": 10,
        "min_delta": 0.001,
    },
    
    # Checkpoint settings
    "save_best_only": True,
    "save_frequency": 5,  # Save every N epochs
}

# Data Configuration
DATA_CONFIG = {
    "train_csv": str(DATA_DIR / "train.csv"),
    "test_csv": str(DATA_DIR / "sample_submission.csv"),
    "train_dir": str(DATA_DIR / "train"),
    "test_dir": str(DATA_DIR / "test"),
    "submission_dir": str(DATA_DIR / "submissions"),
    
    # Data split
    "val_split": 0.2,
    "stratify": True,
    "random_seed": 42,
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_dir": str(LOGS_DIR),
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": str(LOGS_DIR / "training.log"),
}

# Model save paths
MODEL_PATHS = {
    "best_model": str(MODELS_DIR / "best_model.pth"),
    "latest_model": str(MODELS_DIR / "latest_model.pth"),
    "checkpoint_dir": str(MODELS_DIR / "checkpoints"),
}

def get_wandb_config() -> Dict[str, Any]:
    """Get WandB configuration for initialization"""
    return {
        "project": WANDB_CONFIG["project"],
        "entity": WANDB_CONFIG["entity"],
        "tags": WANDB_CONFIG["tags"],
        "config": EXPERIMENT_CONFIG,
        "mode": WANDB_CONFIG["mode"],
    }

def get_experiment_name(model_name: str = None, additional_info: str = None) -> str:
    """Generate experiment name for WandB runs"""
    model = model_name or EXPERIMENT_CONFIG["model_name"]
    img_size = EXPERIMENT_CONFIG["img_size"]
    batch_size = EXPERIMENT_CONFIG["batch_size"]
    lr = EXPERIMENT_CONFIG["learning_rate"]
    
    name = f"{model}_img{img_size}_bs{batch_size}_lr{lr}"
    
    if additional_info:
        name += f"_{additional_info}"
    
    return name

def validate_config() -> bool:
    """Validate configuration settings"""
    errors = []
    
    # Check if required directories exist
    for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        if not dir_path.exists():
            errors.append(f"Directory does not exist: {dir_path}")
    
    # Check WandB API key
    if not WANDB_CONFIG["api_key"] or WANDB_CONFIG["api_key"] == "your_wandb_api_key_here":
        errors.append("WandB API key not set. Please update .env file with your API key.")
    
    # Check data files
    train_csv = Path(DATA_CONFIG["train_csv"])
    if not train_csv.exists():
        errors.append(f"Training CSV not found: {train_csv}")
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

# Export main configurations
__all__ = [
    "WANDB_CONFIG",
    "EXPERIMENT_CONFIG", 
    "DATA_CONFIG",
    "LOGGING_CONFIG",
    "MODEL_PATHS",
    "get_wandb_config",
    "get_experiment_name",
    "validate_config",
]
