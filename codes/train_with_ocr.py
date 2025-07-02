"""
OCR 통합 문서 분류 모델 - 실제 적용 버전
기존 train_with_wandb.py를 기반으로 OCR 기능 추가
"""

import os
import time
import pickle
import sys
from pathlib import Path
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 기존 모듈 임포트
from config import (
    WANDB_CONFIG, EXPERIMENT_CONFIG, DATA_CONFIG, 
    get_wandb_config, get_experiment_name, validate_config
)
from wandb_utils import (
    init_wandb, log_metrics, log_model_info, log_system_info,
    finish_run, create_run_name, log_confusion_matrix
)
from device_utils import setup_training_device, get_dataloader_config

# OCR 관련 임포트
try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_BACKEND = 'easyocr'
except ImportError:
    try:
        import pytesseract
        OCR_AVAILABLE = True
        OCR_BACKEND = 'tesseract'
    except ImportError:
        OCR_AVAILABLE = False
        OCR_BACKEND = None

class OCRExtractor:
    """이미지에서 텍스트 추출"""
    
    def __init__(self, backend='easyocr', cache_dir='data/ocr_cache'):
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if backend == 'easyocr' and OCR_AVAILABLE:
            print("🔤 EasyOCR 초기화 중...")
            self.reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
        elif backend == 'tesseract' and OCR_AVAILABLE:
            print("🔤 Tesseract 초기화 중...")
            self.tesseract_config = '--psm 6 -l kor+eng'
    
    def extract_text(self, image_path):
        """이미지에서 텍스트 추출 (캐싱 지원)"""
        image_name = Path(image_path).name
        cache_file = self.cache_dir / f"{image_name}.txt"
        
        # 캐시된 결과가 있으면 사용
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        # OCR 실행
        text = self._extract_text_from_image(image_path)
        
        # 결과 캐싱
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return text
    
    def _extract_text_from_image(self, image_path):
        """실제 OCR 실행"""
        try:
            if self.backend == 'easyocr':
                results = self.reader.readtext(image_path)
                # 신뢰도 30% 이상인 텍스트만 사용
                texts = [result[1] for result in results if result[2] > 0.3]
                return ' '.join(texts)
            
            elif self.backend == 'tesseract':
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image, config=self.tesseract_config)
                return text.strip()
                
        except Exception as e:
            print(f"OCR 실패 {image_path}: {e}")
            return ""
        
        return ""

class EnhancedImageDataset(Dataset):
    """OCR 텍스트가 포함된 향상된 데이터셋"""
    
    def __init__(self, csv_path, image_dir, transform=None, is_train=True, use_ocr=False, text_features=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        self.use_ocr = use_ocr
        self.text_features = text_features
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.is_train:
            image_name, target = row['ID'], row['target']
        else:
            image_name, target = row['ID'], 0
            
        image_path = os.path.join(self.image_dir, image_name)
        image = np.array(Image.open(image_path))
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        # OCR 텍스트 특징이 있으면 추가
        if self.use_ocr and self.text_features is not None:
            text_feature = torch.FloatTensor(self.text_features[idx])
            return image, text_feature, target
        else:
            return image, target

class MultiModalModel(nn.Module):
    """이미지 + 텍스트 융합 모델"""
    
    def __init__(self, model_name='resnet34', num_classes=17, text_feature_dim=1000, use_ocr=False):
        super().__init__()
        
        self.use_ocr = use_ocr
        
        # 이미지 인코더
        self.image_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes if not use_ocr else 0  # OCR 사용시 특징만 추출
        )
        
        if use_ocr:
            # 멀티모달 설정
            image_feature_dim = self.image_model.num_features
            
            # 텍스트 처리기
            self.text_processor = nn.Sequential(
                nn.Linear(text_feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # 융합 분류기
            fusion_dim = image_feature_dim + 256
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, images, text_features=None):
        if self.use_ocr and text_features is not None:
            # 멀티모달 처리
            image_features = self.image_model(images)
            text_features = self.text_processor(text_features)
            
            # 특징 융합
            fused_features = torch.cat([image_features, text_features], dim=1)
            return self.classifier(fused_features)
        else:
            # 이미지만 사용
            return self.image_model(images)

def extract_and_cache_ocr_features(csv_path, image_dir, save_path=None, backend='easyocr'):
    """전체 데이터셋에서 OCR 특징 추출 및 캐싱"""
    
    if save_path and Path(save_path).exists():
        print(f"📁 캐시된 OCR 특징 로드: {save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"🔤 OCR 특징 추출 시작 ({backend})...")
    
    # 현재 작업 디렉토리를 기준으로 절대 경로 생성
    current_dir = Path.cwd()
    if current_dir.name == 'codes':
        # codes 디렉토리에서 실행되는 경우
        project_root = current_dir.parent
    else:
        # 프로젝트 루트에서 실행되는 경우
        project_root = current_dir
    
    cache_dir = project_root / 'data' / 'ocr_cache'
    ocr_extractor = OCRExtractor(backend=backend, cache_dir=str(cache_dir))
    df = pd.read_csv(csv_path)
    
    texts = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="OCR 처리"):
        image_path = os.path.join(image_dir, row['ID'])
        text = ocr_extractor.extract_text(image_path)
        texts.append(text if text.strip() else "빈문서")
    
    # TF-IDF 특징화
    print("📊 TF-IDF 특징 추출 중...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words=None
    )
    
    text_features = vectorizer.fit_transform(texts).toarray()
    
    # 특징 저장
    feature_data = {
        'text_features': text_features,
        'vectorizer': vectorizer,
        'texts': texts
    }
    
    if save_path:
        save_path = project_root / save_path if not Path(save_path).is_absolute() else Path(save_path)
        print(f"💾 OCR 특징 저장: {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(feature_data, f)
    
    print(f"✅ OCR 특징 추출 완료. 형태: {text_features.shape}")
    return feature_data

def train_one_epoch_multimodal(loader, model, optimizer, loss_fn, device, epoch, use_ocr=False):
    """멀티모달 모델을 위한 훈련 함수"""
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []
    
    pbar = tqdm(loader, desc=f'Training Epoch {epoch}')
    for batch_idx, batch_data in enumerate(pbar):
        
        if use_ocr:
            images, text_features, targets = batch_data
            text_features = text_features.to(device)
        else:
            images, targets = batch_data
            text_features = None
        
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if use_ocr:
            outputs = model(images, text_features)
        else:
            outputs = model(images)
            
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        preds_list.extend(predictions.detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        if batch_idx % 50 == 0:
            log_metrics({
                'batch_loss': loss.item(),
                'batch': epoch * len(loader) + batch_idx
            }, commit=False)
    
    avg_loss = train_loss / len(loader)
    accuracy = accuracy_score(targets_list, preds_list)
    f1 = f1_score(targets_list, preds_list, average='macro')
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy,
        'train_f1': f1
    }

def train_with_ocr():
    """OCR 통합 훈련 함수"""
    
    # OCR 사용 여부 확인
    use_ocr = OCR_AVAILABLE and EXPERIMENT_CONFIG.get('use_ocr', True)
    
    if use_ocr:
        print(f"🔤 OCR 모드 활성화 ({OCR_BACKEND})")
    else:
        print("🖼️  이미지 전용 모드 (OCR 비활성화)")
    
    # 기존 설정 로드
    print_configuration()
    
    if not validate_config():
        print("Configuration validation failed.")
        return
    
    device, device_type = setup_training_device()
    dataloader_config = get_dataloader_config(device_type)
    
    # OCR 특징 추출 (use_ocr=True인 경우에만)
    train_text_features = None
    test_text_features = None
    
    if use_ocr:
        # 훈련 데이터 OCR 처리
        train_ocr_path = "data/ocr_cache/train_ocr_features.pkl"
        train_feature_data = extract_and_cache_ocr_features(
            DATA_CONFIG['train_csv'], 
            DATA_CONFIG['train_dir'], 
            train_ocr_path, 
            backend=OCR_BACKEND
        )
        train_text_features = train_feature_data['text_features']
        
        # 테스트 데이터 OCR 처리
        test_ocr_path = "data/ocr_cache/test_ocr_features.pkl"
        test_feature_data = extract_and_cache_ocr_features(
            DATA_CONFIG['test_csv'], 
            DATA_CONFIG['test_dir'], 
            test_ocr_path, 
            backend=OCR_BACKEND
        )
        test_text_features = test_feature_data['text_features']
    
    # WandB 초기화
    run_name = create_run_name(
        model_name=EXPERIMENT_CONFIG['model_name'],
        experiment_type="ocr_multimodal" if use_ocr else "image_only"
    )
    
    wandb_config = get_wandb_config()
    wandb_config['use_ocr'] = use_ocr
    wandb_config['ocr_backend'] = OCR_BACKEND if use_ocr else None
    
    run = init_wandb(wandb_config, run_name=run_name)
    
    try:
        # 데이터 준비
        train_df = pd.read_csv(DATA_CONFIG['train_csv'])
        train_idx, val_idx = train_test_split(
            range(len(train_df)),
            test_size=DATA_CONFIG['val_split'],
            stratify=train_df['target'],
            random_state=DATA_CONFIG['random_seed']
        )
        
        train_subset = train_df.iloc[train_idx].reset_index(drop=True)
        val_subset = train_df.iloc[val_idx].reset_index(drop=True)
        
        # 이미지 변환
        img_size = EXPERIMENT_CONFIG['img_size']
        train_transform = get_transforms(img_size, is_train=True)
        val_transform = get_transforms(img_size, is_train=False)
        
        # 데이터셋 생성
        train_dataset = EnhancedImageDataset(
            DATA_CONFIG['train_csv'],
            DATA_CONFIG['train_dir'],
            transform=train_transform,
            is_train=True,
            use_ocr=use_ocr,
            text_features=train_text_features[train_idx] if use_ocr else None
        )
        train_dataset.df = train_subset
        
        val_dataset = EnhancedImageDataset(
            DATA_CONFIG['train_csv'],
            DATA_CONFIG['train_dir'],
            transform=val_transform,
            is_train=True,
            use_ocr=use_ocr,
            text_features=train_text_features[val_idx] if use_ocr else None
        )
        val_dataset.df = val_subset
        
        # 데이터 로더 생성 시 디바이스 타입에 따른 최적화 설정 사용
        train_loader = DataLoader(
            train_dataset,
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            shuffle=True,
            **dataloader_config  # pin_memory와 num_workers 설정 사용
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=EXPERIMENT_CONFIG['batch_size'],
            shuffle=False,
            **dataloader_config  # pin_memory와 num_workers 설정 사용
        )
        
        # 모델 생성 (텍스트 특징 차원 동적 설정)
        text_dim = train_text_features.shape[1] if use_ocr and train_text_features is not None else 1000
        print(f"🔤 텍스트 특징 차원: {text_dim}")
        
        model = MultiModalModel(
            model_name=EXPERIMENT_CONFIG['model_name'],
            num_classes=EXPERIMENT_CONFIG['num_classes'],
            text_feature_dim=text_dim,
            use_ocr=use_ocr
        ).to(device)
        
        # 훈련 설정
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(
            model.parameters(), 
            lr=EXPERIMENT_CONFIG['learning_rate'],
            weight_decay=EXPERIMENT_CONFIG['weight_decay']
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=EXPERIMENT_CONFIG['epochs'],
            eta_min=EXPERIMENT_CONFIG['min_lr']
        )
        
        # 훈련 루프
        best_val_f1 = 0.0
        best_epoch = 0
        total_start_time = time.time()
        
        print(f"🚀 OCR 통합 훈련 시작 ({EXPERIMENT_CONFIG['epochs']} epochs)...")
        
        for epoch in range(EXPERIMENT_CONFIG['epochs']):
            start_time = time.time()
            
            # 훈련
            train_metrics = train_one_epoch_multimodal(
                train_loader, model, optimizer, loss_fn, device, epoch, use_ocr
            )
            
            # 검증 (기존 함수 재사용, OCR 지원 추가 필요)
            val_metrics = validate_one_epoch_multimodal(
                val_loader, model, loss_fn, device, epoch, use_ocr
            )
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time
            
            # 메트릭 로깅
            all_metrics = {
                **train_metrics,
                **val_metrics,
                'learning_rate': current_lr,
                'epoch': epoch,
                'epoch_time': epoch_time
            }
            
            log_metrics(all_metrics)
            
            # 진행 상황 출력
            print(f"\n📊 Epoch {epoch+1}/{EXPERIMENT_CONFIG['epochs']}:")
            print(f"  🟢 Train - Loss: {train_metrics['train_loss']:.4f} | "
                  f"F1: {train_metrics['train_f1']:.4f}")
            print(f"  🟡 Valid - Loss: {val_metrics['val_loss']:.4f} | "
                  f"F1: {val_metrics['val_f1']:.4f}")
            
            # 최고 모델 저장
            if val_metrics['val_f1'] > best_val_f1:
                best_val_f1 = val_metrics['val_f1']
                best_epoch = epoch
                
                model_save_path = EXPERIMENT_CONFIG['best_model_path']
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'config': EXPERIMENT_CONFIG,
                    'use_ocr': use_ocr,
                    'ocr_backend': OCR_BACKEND,
                    'text_feature_dim': text_dim  # 텍스트 차원 정보 저장
                }, model_save_path)
                
                print(f"  🎆 NEW BEST! Val F1: {best_val_f1:.4f}")
        
        # 테스트 예측 (훈련 완료 후)
        if use_ocr:
            print(f"\n🔮 OCR 통합 모델로 테스트 예측 시작...")
            print(f"📊 훈련 시 텍스트 차원: {text_dim}")
            print(f"📊 테스트 시 텍스트 차원: {test_text_features.shape[1] if test_text_features is not None else 'None'}")
            
            # 차원 일치 확인
            if test_text_features is not None and test_text_features.shape[1] != text_dim:
                print(f"⚠️  차원 불일치 감지! 훈련:{text_dim} vs 테스트:{test_text_features.shape[1]}")
                print(f"🔄 테스트 특징을 훈련 차원에 맞춰 조정합니다...")
                
                # 차원 조정 (패딩 또는 자르기)
                if test_text_features.shape[1] < text_dim:
                    # 패딩으로 차원 늘리기
                    padding = text_dim - test_text_features.shape[1]
                    test_text_features = np.pad(test_text_features, ((0, 0), (0, padding)), mode='constant')
                    print(f"✅ 패딩으로 {text_dim}차원으로 조정 완료")
                else:
                    # 자르기로 차원 줄이기
                    test_text_features = test_text_features[:, :text_dim]
                    print(f"✅ 자르기로 {text_dim}차원으로 조정 완료")
            
            predict_with_ocr(model, device, test_text_features)
        else:
            print("\n🖼️  이미지 전용 모델 - OCR 예측 생략")
        
        print(f"\n🎉 OCR 통합 훈련 완료! 최고 F1: {best_val_f1:.4f}")
        
    except Exception as e:
        print(f"훈련 중 오류: {e}")
        raise
    finally:
        finish_run()

def get_transforms(img_size, is_train=True):
    """기존 transform 함수 (변경 없음)"""
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

def validate_one_epoch_multimodal(loader, model, loss_fn, device, epoch, use_ocr=False):
    """멀티모달 검증 함수"""
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []
    
    if len(loader) == 0:
        return {
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'val_f1': 0.0,
            'predictions': None,
            'targets': None
        }
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Validation Epoch {epoch}')
        for batch_data in pbar:
            
            if use_ocr:
                images, text_features, targets = batch_data
                text_features = text_features.to(device)
            else:
                images, targets = batch_data
                text_features = None
            
            images = images.to(device)
            targets = targets.to(device)
            
            if use_ocr:
                outputs = model(images, text_features)
            else:
                outputs = model(images)
                
            loss = loss_fn(outputs, targets)
            
            val_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            preds_list.extend(predictions.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
    
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

def print_configuration():
    """기존 설정 출력 함수 (변경 없음)"""
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
    print(f"\n🔤 OCR Settings:")
    print(f"  OCR Available: {OCR_AVAILABLE}")
    print(f"  OCR Backend: {OCR_BACKEND}")
    print(f"  Use OCR: {EXPERIMENT_CONFIG.get('use_ocr', True) and OCR_AVAILABLE}")
    print("="*60 + "\n")

def predict_with_ocr(model, device, test_text_features):
    """고급 OCR 통합 모델을 사용한 테스트 예측"""
    # 테스트 데이터셋 준비
    img_size = EXPERIMENT_CONFIG['img_size']
    test_transform = get_transforms(img_size, is_train=False)
    
    # test.csv 경로 확인 및 수정
    test_csv_path = DATA_CONFIG.get('test_csv', 'data/sample_submission.csv')
    test_dir_path = DATA_CONFIG.get('test_dir', 'data/test')
    
    print(f"📊 테스트 데이터: {test_csv_path}")
    print(f"📁 테스트 폴더: {test_dir_path}")
    print(f"🔤 텍스트 특징 상태: {test_text_features.shape if test_text_features is not None else 'None'}")
    
    test_dataset = EnhancedImageDataset(
        test_csv_path,
        test_dir_path,
        transform=test_transform,
        is_train=False,
        use_ocr=True,
        text_features=test_text_features
    )
    
    dataloader_config = get_dataloader_config(device.type)  # 디바이스 타입 전달
    test_loader = DataLoader(
        test_dataset,
        batch_size=EXPERIMENT_CONFIG['batch_size'],
        shuffle=False,
        **dataloader_config  # pin_memory와 num_workers 설정 사용
    )
    
    # 예측 수행
    model.eval()
    predictions = []
    
    print("\n🔮 OCR 통합 모델로 테스트 데이터 예측 중...")
    with torch.no_grad():
        for batch_idx, (images, text_features, _) in enumerate(tqdm(test_loader, desc="OCR 예측")):
            images = images.to(device)
            text_features = text_features.to(device)
            
            outputs = model(images, text_features)
            batch_predictions = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(batch_predictions)
    
    # 제출 파일 생성
    test_csv_path = DATA_CONFIG.get('test_csv', 'data/sample_submission.csv')
    test_df = pd.read_csv(test_csv_path)
    submission_df = pd.DataFrame({
        'ID': test_df['ID'],
        'target': predictions
    })
    
    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"codes/ocr_submission_{timestamp}.csv"
    os.makedirs(os.path.dirname(submission_file), exist_ok=True)
    submission_df.to_csv(submission_file, index=False)
    
    print(f"📄 OCR 예측 결과 저장: {submission_file}")
    print(f"📊 총 예측 수: {len(predictions)}")
    
    # 예측 분포 표시
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"📈 OCR 모델 예측 분포:")
    for class_id, count in zip(unique, counts):
        percentage = count / len(predictions) * 100
        print(f"  Class {class_id}: {count} ({percentage:.1f}%)")
    
    return submission_df

if __name__ == "__main__":
    
    # 모델 경로 설정
    from config import MODEL_PATHS
    EXPERIMENT_CONFIG.update({
        'best_model_path': MODEL_PATHS['best_model'],
        'latest_model_path': MODEL_PATHS['latest_model'],
        'use_ocr': True  # OCR 사용 여부
    })
    
    # DRY RUN 모드
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("🧪 OCR DRY RUN 모드")
        print("="*60)
        
        try:
            print("1. OCR 라이브러리 확인...")
            if OCR_AVAILABLE:
                print(f"   ✅ OCR 백엔드: {OCR_BACKEND}")
            else:
                print("   ⚠️  OCR 라이브러리 없음 (이미지 전용 모드)")
            
            print("\n2. 환경 검증...")
            if validate_config():
                print("   ✅ 설정 검증 성공")
            
            print("\n3. 디바이스 설정...")
            device, device_type = setup_training_device()
            print(f"   ✅ 디바이스: {device}")
            
            print("\n🎉 DRY RUN 완료! OCR 통합 훈련 준비 완료")
            
        except Exception as e:
            print(f"\n❌ DRY RUN 실패: {e}")
            sys.exit(1)
    else:
        train_with_ocr()
