# CV-Classify 개발자 가이드

## 📋 개요

이 문서는 CV-Classify 시스템의 개발, 확장, 기여를 위한 종합 가이드입니다. 코드 구조, 개발 워크플로우, 확장 방법, 베스트 프랙티스를 포함합니다.

## 🏗️ 개발 환경 설정

### 개발용 설치

```bash
# 1. 저장소 클론
git clone <repository-url>
cd cv-classify

# 2. 개발 모드 설정
chmod +x setup.sh
./setup.sh

# 3. 개발 의존성 설치 (개발용 패키지 포함)
pip install -r requirements.txt
pip install black flake8 pytest mypy jupyter

# 4. 환경 변수 설정
cp .env.template .env
# .env 파일을 편집하여 WandB API 키 등 설정

# 5. 개발 환경 확인
python3 codes/device_utils.py  # 디바이스 테스트
./menu.sh                      # 메뉴 시스템 테스트
```

### IDE 설정

#### VS Code 설정

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".git": true,
        "wandb": true,
        "logs": true
    }
}
```

`.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train Baseline",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/codes/baseline_simple.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/codes"
        },
        {
            "name": "Train with WandB",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/codes/train_with_wandb.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/codes"
        }
    ]
}
```

---

## 🏛️ 코드 아키텍처

### 핵심 설계 원칙

#### 1. 모듈성 (Modularity)
```python
# 각 모듈은 단일 책임을 가짐
config.py       → 설정 관리
device_utils.py → 하드웨어 추상화
wandb_utils.py  → 실험 추적
```

#### 2. 의존성 주입 (Dependency Injection)
```python
# 하드코딩된 의존성 대신 주입 받음
def train_model(config, device_manager, experiment_tracker):
    # 설정과 의존성을 외부에서 주입
    pass
```

#### 3. 플랫폼 추상화 (Platform Abstraction)
```python
# 플랫폼별 차이를 추상화
def get_optimal_device():
    # 내부적으로 플랫폼을 감지하고 최적 설정 반환
    pass
```

### 레이어별 역할

#### 1. 설정 레이어 (Configuration Layer)
```python
# config.py
PROJECT_ROOT = Path(__file__).parent.parent

# 환경별 설정 분리
WANDB_CONFIG = {...}
EXPERIMENT_CONFIG = {...}  # 실제 실험용
BASELINE_CONFIG = {...}    # 빠른 테스트용
```

#### 2. 추상화 레이어 (Abstraction Layer)
```python
# device_utils.py
class DeviceManager:
    def get_optimal_device(self):
        # 플랫폼별 최적 디바이스 감지
    
    def get_dataloader_config(self):
        # 디바이스에 맞는 DataLoader 설정
```

#### 3. 유틸리티 레이어 (Utility Layer)
```python
# wandb_utils.py
class ExperimentTracker:
    def init_experiment(self, config):
        # 실험 초기화
    
    def log_metrics(self, metrics):
        # 메트릭 로깅
```

#### 4. 실행 레이어 (Execution Layer)
```python
# train_with_wandb.py
def main():
    # 1. 설정 로드
    # 2. 디바이스 설정
    # 3. 데이터 준비
    # 4. 모델 학습
    # 5. 결과 저장
```

---

## 🔧 개발 워크플로우

### Git 브랜치 전략

```bash
main            # 안정된 배포 버전
├── develop     # 개발 통합 브랜치
├── feature/*   # 새로운 기능 개발
├── bugfix/*    # 버그 수정
└── hotfix/*    # 긴급 수정
```

#### 브랜치 명명 규칙

```bash
# 기능 개발
feature/add-efficientnet-support
feature/improve-ocr-integration
feature/cross-platform-optimization

# 버그 수정
bugfix/fix-mps-memory-leak
bugfix/wandb-connection-error

# 핫픽스
hotfix/critical-cuda-compatibility
```

### 커밋 메시지 컨벤션

```bash
# 형식: <type>(<scope>): <description>

feat(device): add Apple Silicon MPS support
fix(wandb): resolve API key validation error
docs(readme): update installation instructions
refactor(config): simplify configuration structure
test(device): add cross-platform device tests
perf(dataloader): optimize num_workers for each platform
```

### 개발 프로세스

#### 1. 새 기능 개발

```bash
# 1. 새 브랜치 생성
git checkout -b feature/add-new-model

# 2. 기능 구현
# - 코드 작성
# - 테스트 추가
# - 문서 업데이트

# 3. 테스트 실행
python -m pytest tests/
./scripts/test_baseline.sh

# 4. 코드 포맷팅
black codes/
flake8 codes/

# 5. 커밋 및 푸시
git add .
git commit -m "feat(model): add EfficientNet support"
git push origin feature/add-new-model

# 6. Pull Request 생성
```

#### 2. 버그 수정

```bash
# 1. 버그 재현
python codes/train_with_wandb.py  # 오류 확인

# 2. 테스트 케이스 작성
# tests/test_bug_reproduction.py

# 3. 수정 구현
# codes/wandb_utils.py 수정

# 4. 테스트 확인
python -m pytest tests/test_bug_reproduction.py

# 5. 회귀 테스트
./menu.sh  # 전체 시스템 테스트
```

---

## 🧪 테스트 전략

### 테스트 구조

```
tests/
├── unit/                    # 단위 테스트
│   ├── test_config.py      # 설정 모듈 테스트
│   ├── test_device_utils.py # 디바이스 유틸 테스트
│   └── test_wandb_utils.py  # WandB 유틸 테스트
├── integration/             # 통합 테스트
│   ├── test_training_pipeline.py
│   └── test_cross_platform.py
├── e2e/                     # 엔드투엔드 테스트
│   ├── test_baseline_execution.py
│   └── test_full_training.py
└── fixtures/                # 테스트 데이터
    ├── sample_images/
    └── mock_configs/
```

### 단위 테스트 예제

```python
# tests/unit/test_device_utils.py
import pytest
import torch
from unittest.mock import patch, MagicMock

from codes.device_utils import get_optimal_device, get_dataloader_config


class TestDeviceUtils:
    """디바이스 유틸리티 테스트"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_cuda_device_selection(self, mock_mps, mock_cuda):
        """CUDA 디바이스 선택 테스트"""
        mock_cuda.return_value = True
        mock_mps.return_value = False
        
        device, device_type = get_optimal_device()
        
        assert device.type == 'cuda'
        assert device_type == 'CUDA'
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_mps_device_selection(self, mock_mps, mock_cuda):
        """MPS 디바이스 선택 테스트"""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        
        device, device_type = get_optimal_device()
        
        assert device.type == 'mps'
        assert device_type == 'MPS'
    
    def test_dataloader_config_cuda(self):
        """CUDA DataLoader 설정 테스트"""
        config = get_dataloader_config('CUDA')
        
        assert config['pin_memory'] == True
        assert config['num_workers'] >= 4
    
    def test_dataloader_config_mps(self):
        """MPS DataLoader 설정 테스트"""
        config = get_dataloader_config('MPS')
        
        assert config['pin_memory'] == False
        assert config['num_workers'] == 0
```

### 통합 테스트 예제

```python
# tests/integration/test_training_pipeline.py
import pytest
import tempfile
import os
from pathlib import Path

from codes.baseline_simple import main as baseline_main
from codes.config import BASELINE_CONFIG


class TestTrainingPipeline:
    """학습 파이프라인 통합 테스트"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """임시 데이터 디렉토리 생성"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 테스트용 더미 데이터 생성
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            
            # train.csv 생성
            train_csv = data_dir / "train.csv"
            train_csv.write_text("ID,target\ntest1.jpg,0\ntest2.jpg,1\n")
            
            # test.csv 생성  
            test_csv = data_dir / "sample_submission.csv"
            test_csv.write_text("ID,target\ntest3.jpg,0\n")
            
            # 이미지 폴더 생성
            (data_dir / "train").mkdir()
            (data_dir / "test").mkdir()
            
            yield data_dir
    
    def test_baseline_execution(self, temp_data_dir, monkeypatch):
        """베이스라인 실행 테스트"""
        # 설정을 테스트 데이터로 변경
        monkeypatch.setattr('codes.config.DATA_DIR', temp_data_dir)
        monkeypatch.setattr('codes.config.BASELINE_CONFIG.epochs', 1)
        
        # 베이스라인 실행 (오류 없이 완료되어야 함)
        try:
            baseline_main()
            assert True  # 예외 없이 완료됨
        except Exception as e:
            pytest.fail(f"베이스라인 실행 실패: {e}")
```

### 크로스 플랫폼 테스트

```python
# tests/integration/test_cross_platform.py
import pytest
import platform
import subprocess

from codes.device_utils import setup_training_device
from codes.platform_utils import detect_platform


class TestCrossPlatform:
    """크로스 플랫폼 호환성 테스트"""
    
    def test_platform_detection(self):
        """플랫폼 감지 테스트"""
        detected = detect_platform()
        current = platform.system().lower()
        
        if current == 'darwin':
            assert detected == 'macos'
        elif current == 'linux':
            assert detected in ['ubuntu', 'centos', 'linux']
        elif current in ['windows', 'cygwin']:
            assert detected == 'windows'
    
    def test_device_setup_cross_platform(self):
        """크로스 플랫폼 디바이스 설정 테스트"""
        device, device_type = setup_training_device()
        
        # 모든 플랫폼에서 유효한 디바이스 반환해야 함
        assert device_type in ['CUDA', 'MPS', 'CPU']
        assert hasattr(device, 'type')
    
    @pytest.mark.skipif(
        not subprocess.run(['which', 'bash'], capture_output=True).returncode == 0,
        reason="Bash shell not available"
    )
    def test_shell_script_execution(self):
        """셸 스크립트 실행 테스트"""
        result = subprocess.run(
            ['bash', 'scripts/platform_utils.sh'],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "시스템 정보" in result.stdout
```

### 테스트 실행

```bash
# 전체 테스트 실행
python -m pytest tests/

# 특정 테스트 그룹 실행
python -m pytest tests/unit/
python -m pytest tests/integration/

# 커버리지 포함 실행
python -m pytest tests/ --cov=codes/ --cov-report=html

# 마커별 실행
python -m pytest -m "not slow"  # 느린 테스트 제외
python -m pytest -m "gpu"       # GPU 테스트만
```

---

## 🔌 확장 가이드

### 새로운 모델 추가

#### 1. 설정 업데이트

```python
# codes/config.py
SUPPORTED_MODELS = {
    'resnet34': {
        'family': 'resnet',
        'input_size': 224,
        'memory_multiplier': 1.0
    },
    'efficientnet_b0': {
        'family': 'efficientnet', 
        'input_size': 224,
        'memory_multiplier': 0.8
    },
    'vit_base_patch16_224': {  # 새 모델 추가
        'family': 'transformer',
        'input_size': 224,
        'memory_multiplier': 1.5
    }
}
```

#### 2. 모델 로더 확장

```python
# codes/model_factory.py (새 파일 생성)
import timm
from typing import Dict, Any

class ModelFactory:
    """모델 생성 팩토리"""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int, **kwargs) -> torch.nn.Module:
        """모델 생성"""
        
        if model_name.startswith('vit_'):
            # Vision Transformer 특별 처리
            model = timm.create_model(
                model_name,
                pretrained=kwargs.get('pretrained', True),
                num_classes=num_classes,
                drop_rate=kwargs.get('drop_rate', 0.1)
            )
        else:
            # 기본 모델 생성
            model = timm.create_model(
                model_name,
                pretrained=kwargs.get('pretrained', True),
                num_classes=num_classes
            )
        
        return model
    
    @staticmethod
    def get_model_config(model_name: str) -> Dict[str, Any]:
        """모델별 권장 설정 반환"""
        from config import SUPPORTED_MODELS
        return SUPPORTED_MODELS.get(model_name, {})
```

#### 3. 기존 코드 업데이트

```python
# codes/train_with_wandb.py
from model_factory import ModelFactory

# 기존: model = timm.create_model(...)
# 변경:
model = ModelFactory.create_model(
    model_name=EXPERIMENT_CONFIG['model_name'],
    num_classes=EXPERIMENT_CONFIG['num_classes'],
    pretrained=EXPERIMENT_CONFIG['pretrained']
)
```

### 새로운 데이터 증강 추가

#### 1. 증강 라이브러리 확장

```python
# codes/augmentation.py (새 파일 생성)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Any

class AugmentationFactory:
    """데이터 증강 팩토리"""
    
    @staticmethod
    def get_transform(
        img_size: int,
        is_train: bool = True,
        augmentation_level: str = 'medium'
    ) -> A.Compose:
        """증강 수준별 변환 생성"""
        
        base_transforms = [
            A.Resize(height=img_size, width=img_size),
        ]
        
        if is_train:
            if augmentation_level == 'light':
                train_transforms = [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ]
            elif augmentation_level == 'medium':
                train_transforms = [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.3),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.2),
                ]
            elif augmentation_level == 'heavy':
                train_transforms = [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=25, p=0.5),
                    A.RandomBrightnessContrast(p=0.4),
                    A.GaussNoise(p=0.3),
                    A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3),
                    A.CoarseDropout(p=0.3),
                ]
            else:
                train_transforms = []
            
            base_transforms.extend(train_transforms)
        
        # 공통 후처리
        base_transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        return A.Compose(base_transforms)
```

#### 2. 설정에 증강 옵션 추가

```python
# codes/config.py
EXPERIMENT_CONFIG.update({
    'augmentation_level': 'medium',  # light, medium, heavy
    'custom_augmentations': {
        'mixup': False,
        'cutmix': False,
        'autoaugment': False
    }
})
```

### 새로운 손실 함수 추가

```python
# codes/losses.py (새 파일 생성)
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced datasets"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        
        smooth_targets = targets_one_hot * (1 - self.smoothing) + \
                        self.smoothing / num_classes
        
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()

class LossFactory:
    """손실 함수 팩토리"""
    
    @staticmethod
    def create_loss(loss_type: str, **kwargs):
        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            return FocalLoss(**kwargs)
        elif loss_type == 'label_smoothing':
            return LabelSmoothingCrossEntropy(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
```

### 새로운 스케줄러 추가

```python
# codes/schedulers.py (새 파일 생성)
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine Annealing with Warm Restarts"""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * 
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class SchedulerFactory:
    """스케줄러 팩토리"""
    
    @staticmethod
    def create_scheduler(scheduler_type: str, optimizer, **kwargs):
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **kwargs
            )
        elif scheduler_type == 'cosine_restart':
            return CosineAnnealingWarmRestarts(
                optimizer, **kwargs
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, **kwargs
            )
        elif scheduler_type == 'reduce_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **kwargs
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
```

---

## 🎯 베스트 프랙티스

### 코딩 스타일

#### 1. Python 코딩 표준

```python
# 1. Import 순서
import os                    # 표준 라이브러리
import sys
from pathlib import Path

import torch                 # 서드파티 라이브러리
import numpy as np
import pandas as pd

from config import CONFIG    # 로컬 모듈
from device_utils import setup_device

# 2. 함수 문서화
def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    한 에포크 학습 실행
    
    Args:
        loader: 학습 데이터 로더
        model: PyTorch 모델
        optimizer: 옵티마이저
        device: 학습 디바이스
    
    Returns:
        학습 메트릭 딕셔너리 (loss, accuracy, f1)
    
    Raises:
        RuntimeError: CUDA out of memory 등 실행 오류
    """
    model.train()
    # ... 구현
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy,
        'train_f1': f1_score
    }

# 3. 클래스 설계
class ExperimentManager:
    """실험 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._device = None
        self._model = None
    
    @property
    def device(self) -> torch.device:
        """지연 초기화된 디바이스"""
        if self._device is None:
            self._device = self._setup_device()
        return self._device
    
    def _setup_device(self) -> torch.device:
        """private 메서드는 _ 접두사"""
        # 디바이스 설정 로직
        pass
```

#### 2. 오류 처리

```python
# 구체적인 예외 처리
def load_model(model_path: str) -> nn.Module:
    """모델 로드 with 적절한 예외 처리"""
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    except RuntimeError as e:
        raise RuntimeError(f"모델 로드 실패: {e}")
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        raise ValueError("잘못된 체크포인트 형식입니다")
    except RuntimeError as e:
        raise ValueError(f"모델 상태 로드 실패: {e}")
    
    return model

# 리소스 정리
def train_with_cleanup():
    """리소스 정리를 보장하는 학습 함수"""
    
    device = None
    model = None
    
    try:
        device, _ = setup_training_device()
        model = create_model().to(device)
        
        # 학습 실행
        train_model(model, device)
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise
    
    finally:
        # 메모리 정리
        if model is not None:
            del model
        
        if device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device is not None and device.type == 'mps':
            torch.mps.empty_cache()
        
        import gc
        gc.collect()
```

#### 3. 로깅

```python
import logging
from datetime import datetime

# 로거 설정
def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """구조화된 로거 설정"""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # 파일 핸들러
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"logs/{name}_{timestamp}.log")
    file_handler.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 사용 예제
logger = setup_logger('training')

def train_epoch():
    logger.info("에포크 학습 시작")
    
    try:
        # 학습 로직
        metrics = train_one_epoch(...)
        logger.info(f"에포크 완료: {metrics}")
        
    except Exception as e:
        logger.error(f"에포크 학습 실패: {e}", exc_info=True)
        raise
```

### 성능 최적화

#### 1. 메모리 관리

```python
def optimize_memory_usage():
    """메모리 사용량 최적화"""
    
    # 1. 그래디언트 체크포인팅 (메모리 vs 계산 트레이드오프)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # 2. 혼합 정밀도 학습
    from torch.cuda.amp import autocast, GradScaler
    
    scaler = GradScaler()
    
    for batch in dataloader:
        with autocast():
            outputs = model(batch)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # 3. 메모리 모니터링
    def log_memory_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU 메모리: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

#### 2. 데이터 로딩 최적화

```python
class OptimizedDataset(Dataset):
    """메모리 효율적인 데이터셋"""
    
    def __init__(self, csv_path: str, image_dir: str, cache_images: bool = False):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.cache_images = cache_images
        self._image_cache = {} if cache_images else None
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row['ID']
        
        # 이미지 캐싱 (메모리가 충분한 경우)
        if self.cache_images and image_path in self._image_cache:
            image = self._image_cache[image_path]
        else:
            image = self._load_image(image_path)
            if self.cache_images:
                self._image_cache[image_path] = image
        
        return image, row['target']
    
    def _load_image(self, path: Path):
        """효율적인 이미지 로딩"""
        try:
            # PIL보다 빠른 cv2 사용 고려
            import cv2
            image = cv2.imread(str(path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception:
            # fallback to PIL
            from PIL import Image
            return np.array(Image.open(path))
```

### 크로스 플랫폼 호환성

```python
def ensure_cross_platform_compatibility():
    """크로스 플랫폼 호환성 보장"""
    
    # 1. 경로 처리
    from pathlib import Path
    
    # 잘못된 방법
    # path = "data/train/image.jpg"  # Windows에서 문제
    
    # 올바른 방법
    path = Path("data") / "train" / "image.jpg"
    
    # 2. 환경 변수 처리
    import os
    
    def get_cache_dir():
        if os.name == 'nt':  # Windows
            return Path.home() / "AppData" / "Local" / "cv-classify"
        else:  # Unix-like
            return Path.home() / ".cache" / "cv-classify"
    
    # 3. 멀티프로세싱 호환성
    if __name__ == '__main__':
        # Windows에서 멀티프로세싱 문제 방지
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
```

---

## 📝 문서화 가이드

### 코드 문서화

#### 1. 모듈 문서화

```python
"""
CV-Classify Device Utilities

이 모듈은 크로스 플랫폼 디바이스 감지 및 최적화 기능을 제공합니다.

주요 기능:
- 자동 디바이스 감지 (CUDA, MPS, CPU)
- 플랫폼별 최적화 설정
- DataLoader 설정 자동 조정

예제:
    >>> from device_utils import setup_training_device
    >>> device, device_type = setup_training_device()
    >>> print(f"사용 디바이스: {device}")

작성자: CV-Classify Team
버전: 1.0.0
"""

import torch
from typing import Tuple, Dict, Any
```

#### 2. 함수 문서화

```python
def get_dataloader_config(device_type: str) -> Dict[str, Any]:
    """
    디바이스 타입에 따른 최적화된 DataLoader 설정 반환
    
    각 플랫폼의 특성에 맞는 DataLoader 설정을 제공하여
    최적의 학습 성능을 달성할 수 있도록 합니다.
    
    Args:
        device_type (str): 디바이스 타입
            - "CUDA": NVIDIA GPU
            - "MPS": Apple Silicon GPU  
            - "CPU": CPU 전용
    
    Returns:
        Dict[str, Any]: DataLoader 설정 딕셔너리
            - pin_memory (bool): 메모리 고정 사용 여부
            - num_workers (int): 병렬 워커 수
            - persistent_workers (bool): 워커 재사용 여부
    
    Raises:
        ValueError: 지원하지 않는 device_type인 경우
    
    예제:
        >>> config = get_dataloader_config("CUDA")
        >>> dataloader = DataLoader(dataset, **config)
    
    참고:
        - MPS는 멀티프로세싱 이슈로 num_workers=0 권장
        - CUDA는 pin_memory=True로 GPU 전송 최적화
    """
    # 구현...
```

### README 업데이트

```markdown
# CV-Classify

## 새 기능 추가 시 업데이트 필요 섹션

### 🆕 새로운 기능 (v1.1.0)
- ✅ EfficientNet 모델 지원 추가
- ✅ Focal Loss 손실 함수 추가  
- ✅ AutoAugment 데이터 증강 지원
- ✅ 성능 프로파일링 도구 추가

### 📊 지원 모델
| 모델 | 입력 크기 | 메모리 요구량 | 성능 |
|------|-----------|---------------|------|
| ResNet34 | 224x224 | 4GB | ⭐⭐⭐⭐ |
| EfficientNet-B0 | 224x224 | 3GB | ⭐⭐⭐⭐⭐ |
| Vision Transformer | 224x224 | 6GB | ⭐⭐⭐⭐⭐ |

### 🔧 사용법 예제
```bash
# 새로운 모델로 학습
python codes/train_with_wandb.py --model efficientnet_b0

# 커스텀 증강 레벨 사용
python codes/train_with_wandb.py --augmentation heavy
```
```

---

## 🔄 CI/CD 파이프라인

### GitHub Actions 설정

`.github/workflows/ci.yml`:
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: [3.8, 3.9, '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest black flake8
    
    - name: Run linting
      run: |
        black --check codes/
        flake8 codes/
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=codes/
    
    - name: Test cross-platform execution
      run: |
        python codes/device_utils.py  # 디바이스 테스트
        # bash 스크립트는 Windows에서 제외
        if [ "$RUNNER_OS" != "Windows" ]; then
          bash scripts/platform_utils.sh
        fi
      shell: bash

  integration-test:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run integration tests
      run: |
        python -m pytest tests/integration/ -v
    
    - name: Test baseline execution
      run: |
        # 실제 베이스라인 실행 테스트 (더미 데이터)
        python codes/baseline_simple.py --dry-run

  security-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security checks
      run: |
        pip install safety bandit
        safety check -r requirements.txt
        bandit -r codes/ -f json
```

### 자동 릴리스 설정

`.github/workflows/release.yml`:
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
        body: |
          ## 변경사항
          - 새로운 기능 추가
          - 버그 수정
          - 성능 개선
          
          ## 설치 방법
          ```bash
          git clone <repo-url>
          cd cv-classify
          git checkout ${{ github.ref }}
          ./setup.sh
          ```
```

---

## 🎯 컨트리뷰션 가이드

### 기여 프로세스

1. **이슈 생성**: 새 기능이나 버그 리포트
2. **포크 생성**: 개인 저장소로 포크
3. **브랜치 생성**: 기능별 브랜치 생성
4. **개발 진행**: 코드 작성 + 테스트 추가
5. **Pull Request**: 코드 리뷰 요청
6. **코드 리뷰**: 피드백 반영
7. **병합**: main 브랜치에 병합

### Pull Request 템플릿

```markdown
## 변경사항 요약
<!-- 이 PR에서 변경된 내용을 간략히 설명해주세요 -->

## 변경 타입
- [ ] 버그 수정
- [ ] 새 기능
- [ ] 성능 개선
- [ ] 리팩토링
- [ ] 문서 업데이트
- [ ] 테스트 추가

## 테스트 결과
- [ ] 단위 테스트 통과
- [ ] 통합 테스트 통과
- [ ] 크로스 플랫폼 테스트 완료
- [ ] 수동 테스트 완료

## 체크리스트
- [ ] 코드가 프로젝트 스타일 가이드를 따름
- [ ] 적절한 테스트가 추가됨
- [ ] 문서가 업데이트됨
- [ ] 브레이킹 체인지가 있다면 마이그레이션 가이드 포함

## 스크린샷/로그 (해당하는 경우)
<!-- 변경사항을 보여주는 스크린샷이나 로그를 첨부해주세요 -->

## 관련 이슈
Closes #이슈번호
```

### 코드 리뷰 가이드라인

#### 리뷰어를 위한 체크리스트

```markdown
## 코드 리뷰 체크리스트

### 기능성
- [ ] 코드가 의도한 대로 작동하는가?
- [ ] 엣지 케이스가 적절히 처리되는가?
- [ ] 에러 핸들링이 적절한가?

### 성능
- [ ] 성능상 문제가 없는가?
- [ ] 메모리 누수 가능성은 없는가?
- [ ] 불필요한 계산이 없는가?

### 보안
- [ ] 보안 취약점이 없는가?
- [ ] 입력 검증이 적절한가?
- [ ] 민감한 정보가 노출되지 않는가?

### 코드 품질
- [ ] 코드가 읽기 쉬운가?
- [ ] 네이밍이 명확한가?
- [ ] 중복 코드가 없는가?
- [ ] 주석이 적절한가?

### 테스트
- [ ] 테스트 커버리지가 충분한가?
- [ ] 테스트가 의미 있는가?
- [ ] 모든 테스트가 통과하는가?

### 문서화
- [ ] API 문서가 업데이트되었는가?
- [ ] README가 업데이트되었는가?
- [ ] 변경사항이 적절히 문서화되었는가?
```

### 이슈 템플릿

#### 버그 리포트

```markdown
---
name: 버그 리포트
about: 버그를 발견했을 때 사용하세요
labels: bug
---

## 버그 설명
버그에 대한 명확하고 간결한 설명

## 재현 방법
1. '...' 로 이동
2. '...' 클릭
3. '...' 까지 스크롤
4. 오류 확인

## 예상 동작
어떤 일이 일어날 것으로 예상했는지 설명

## 실제 동작
실제로 무엇이 일어났는지 설명

## 스크린샷
해당하는 경우 스크린샷 첨부

## 환경 정보
- OS: [예: macOS, Ubuntu]
- Python 버전: [예: 3.9]
- GPU: [예: NVIDIA RTX 3080, Apple M1]
- 기타 관련 정보

## 추가 정보
버그에 대한 기타 추가 정보
```

#### 기능 요청

```markdown
---
name: 기능 요청
about: 새로운 기능을 제안하고 싶을 때 사용하세요
labels: enhancement
---

## 기능 설명
원하는 기능에 대한 명확하고 간결한 설명

## 문제점
이 기능이 해결하고자 하는 문제는 무엇인가요?

## 제안하는 해결책
어떤 방식으로 해결하고 싶은지 설명

## 대안
고려해본 다른 대안들이 있다면 설명

## 추가 정보
기능 요청에 대한 기타 추가 정보, 스크린샷, 참고 자료 등
```

---

## 🔚 마무리

이 개발자 가이드는 CV-Classify 시스템의 개발, 확장, 기여를 위한 종합적인 지침을 제공합니다.

**핵심 원칙**:
- **모듈성**: 독립적이고 재사용 가능한 컴포넌트
- **확장성**: 새로운 기능을 쉽게 추가할 수 있는 구조
- **품질**: 테스트와 코드 리뷰를 통한 품질 보장
- **문서화**: 명확하고 최신의 문서 유지

**개발 워크플로우**:
1. 이슈 생성 → 브랜치 생성 → 개발 → 테스트 → PR → 리뷰 → 병합

**확장 가이드**:
- 새로운 모델, 손실 함수, 스케줄러 추가 방법
- 팩토리 패턴을 통한 확장성 확보
- 크로스 플랫폼 호환성 유지

이 가이드를 따라 CV-Classify 시스템을 효과적으로 개발하고 발전시킬 수 있습니다.