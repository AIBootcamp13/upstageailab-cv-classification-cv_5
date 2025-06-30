"""
WandB utility functions for run management and logging
"""

import wandb
import os
import re
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime
from pathlib import Path

# Handle wandb.Run type hint compatibility
if TYPE_CHECKING:
    try:
        from wandb.sdk.wandb_run import Run
    except ImportError:
        # Fallback for older wandb versions
        Run = Any
else:
    # Runtime - use Any to avoid import issues
    Run = Any

def init_wandb(config: Dict[str, Any], run_name: Optional[str] = None) -> "Run":
    """
    Initialize WandB run with configuration
    
    Args:
        config: WandB configuration dictionary
        run_name: Optional custom run name
        
    Returns:
        wandb.Run: Initialized WandB run
    """
    # Set WandB mode from environment or config
    wandb_mode = os.getenv("WANDB_MODE", config.get("mode", "online"))
    
    # Initialize run
    run = wandb.init(
        project=config["project"],
        entity=config.get("entity"),
        config=config.get("config", {}),
        tags=config.get("tags", []),
        name=run_name,
        mode=wandb_mode,
        reinit=True  # Allow multiple runs in same process
    )
    
    return run

def get_runs(project_name: str, entity: str = None, limit: int = 100) -> List["Run"]:
    """
    Get all runs from a WandB project
    
    Args:
        project_name: WandB project name
        entity: WandB entity (username/team)
        limit: Maximum number of runs to fetch
        
    Returns:
        List of WandB runs
    """
    try:
        api = wandb.Api()
        
        if entity:
            project_path = f"{entity}/{project_name}"
        else:
            project_path = project_name
            
        runs = api.runs(project_path, per_page=limit)
        return list(runs)
    
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return []

def get_latest_runs(project_name: str, entity: str = None, limit: int = 10) -> List["Run"]:
    """
    Get latest runs from a WandB project
    
    Args:
        project_name: WandB project name
        entity: WandB entity (username/team)
        limit: Number of latest runs to fetch
        
    Returns:
        List of latest WandB runs
    """
    runs = get_runs(project_name, entity, limit)
    
    # Sort by creation time (newest first)
    sorted_runs = sorted(runs, key=lambda x: x.created_at, reverse=True)
    return sorted_runs[:limit]

def auto_increment_run_suffix(base_name: str, project_name: str, entity: str = None) -> str:
    """
    Automatically increment run name suffix to avoid conflicts
    
    Args:
        base_name: Base name for the run
        project_name: WandB project name
        entity: WandB entity (username/team)
        
    Returns:
        Unique run name with incremented suffix
    """
    runs = get_runs(project_name, entity)
    
    if not runs:
        return f"{base_name}_001"
    
    # Extract existing run numbers
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)")
    existing_numbers = []
    
    for run in runs:
        if run.name:
            match = pattern.match(run.name)
            if match:
                existing_numbers.append(int(match.group(1)))
    
    # Find next available number
    if existing_numbers:
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1
    
    return f"{base_name}_{next_number:03d}"

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
    """
    Log metrics to WandB
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
        commit: Whether to commit the log
    """
    wandb.log(metrics, step=step, commit=commit)

def log_model_info(model, input_shape: tuple = None):
    """
    Log model architecture information to WandB
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape for parameter counting
    """
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "model_class_name": model.__class__.__name__,  # model_name → model_class_name으로 변경
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }
        
        # Log model info with allow_val_change
        wandb.config.update(model_info, allow_val_change=True)  # allow_val_change 추가
        
        # Log model summary if possible
        if input_shape:
            try:
                import torch
                dummy_input = torch.randn(1, *input_shape)
                wandb.watch(model, dummy_input, log="all", log_freq=100)
            except Exception as e:
                print(f"Could not log model graph: {e}")
                
    except Exception as e:
        print(f"Error logging model info: {e}")

def log_system_info():
    """Log system information to WandB"""
    try:
        import torch
        import platform
        
        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
            })
        
        wandb.config.update(system_info)
        
    except Exception as e:
        print(f"Error logging system info: {e}")

def save_model_artifact(model_path: str, name: str, type_: str = "model", 
                       metadata: Dict[str, Any] = None):
    """
    Save model as WandB artifact
    
    Args:
        model_path: Path to saved model
        name: Artifact name
        type_: Artifact type
        metadata: Optional metadata dictionary
    """
    try:
        artifact = wandb.Artifact(name=name, type=type_, metadata=metadata)
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
    except Exception as e:
        print(f"Error saving model artifact: {e}")

def finish_run():
    """Finish current WandB run"""
    try:
        wandb.finish()
    except Exception as e:
        print(f"Error finishing WandB run: {e}")

def create_run_name(model_name: str, experiment_type: str = None) -> str:
    """
    Create a descriptive run name with timestamp
    
    Args:
        model_name: Name of the model
        experiment_type: Type of experiment (e.g., 'baseline', 'augmented')
        
    Returns:
        Formatted run name
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    if experiment_type:
        return f"{model_name}_{experiment_type}_{timestamp}"
    else:
        return f"{model_name}_{timestamp}"

def log_confusion_matrix(y_true, y_pred, class_names: List[str] = None):
    """
    Log confusion matrix to WandB with comprehensive error handling
    
    Args:
        y_true: True labels (numpy array, list, or torch tensor)
        y_pred: Predicted labels (numpy array, list, or torch tensor)
        class_names: Optional class names
    """
    try:
        # Import required libraries
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert inputs to numpy arrays for consistent handling
        if hasattr(y_true, 'cpu'):  # PyTorch tensor
            y_true = y_true.cpu().numpy()
        elif hasattr(y_true, 'numpy'):  # Some other tensor type
            y_true = y_true.numpy()
        else:
            y_true = np.array(y_true)
            
        if hasattr(y_pred, 'cpu'):  # PyTorch tensor
            y_pred = y_pred.cpu().numpy()
        elif hasattr(y_pred, 'numpy'):  # Some other tensor type
            y_pred = y_pred.numpy()
        else:
            y_pred = np.array(y_pred)
        
        # Comprehensive validation
        if y_true is None or y_pred is None:
            return  # Silent return for None values
            
        if not hasattr(y_true, '__len__') or not hasattr(y_pred, '__len__'):
            return  # Silent return for non-array-like objects
            
        if len(y_true) == 0 or len(y_pred) == 0:
            return  # Silent return for empty arrays
            
        if len(y_true) != len(y_pred):
            return  # Silent return for length mismatch
        
        # Check for invalid values
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            return  # Silent return for NaN values
            
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        if class_names and len(class_names) == cm.shape[0]:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Log to WandB
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        
        # Close figure to prevent memory leaks
        plt.close()
        
    except ImportError:
        # Silent return for missing dependencies
        return
    except Exception:
        # Silent return for any other errors to prevent training interruption
        return

# Export functions
__all__ = [
    "init_wandb",
    "get_runs",
    "get_latest_runs", 
    "auto_increment_run_suffix",
    "log_metrics",
    "log_model_info",
    "log_system_info",
    "save_model_artifact",
    "finish_run",
    "create_run_name",
    "log_confusion_matrix",
]
