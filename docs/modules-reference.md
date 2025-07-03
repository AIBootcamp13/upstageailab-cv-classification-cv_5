# CV-Classify 모듈 참조 문서

## 📋 개요

CV-Classify 시스템의 주요 Python 모듈들에 대한 상세한 API 참조 문서입니다. 각 모듈의 기능, 클래스, 함수, 사용 예제를 포함합니다.

## 📂 모듈 구조

```
codes/
├── config.py           # 설정 관리
├── device_utils.py     # 디바이스 최적화
├── wandb_utils.py      # 실험 추적
├── baseline_simple.py  # 간단한 베이스라인
├── train_with_wandb.py # 고급 실험 실행
└── train_with_ocr.py   # OCR 통합 모델
```

---

## 🔧 config.py

**용도**: 전체 시스템의 설정을 중앙 집중식으로 관리하는 모듈

### 주요 설정 객체

#### `WANDB_CONFIG`
WandB 실험 추적 관련 설정

```python
WANDB_CONFIG = {
    "api_key": os.getenv("WANDB_API_KEY"),
    "project": os.getenv("WANDB_PROJECT", "cv-classification"),
    "entity": os.getenv("WANDB_ENTITY"),
    "mode": os.getenv("WANDB_MODE", "online"),
    "tags": ["cv", "classification", "document", "upstage"]
}
```

**사용 예제**:
```python
from config import WANDB_CONFIG
import wandb

# WandB 초기화
wandb.init(
    project=WANDB_CONFIG["project"],
    tags=WANDB_CONFIG["tags"]
)
```

#### `EXPERIMENT_CONFIG`
고급 실험용 설정 (실제 경진대회용)

```python
EXPERIMENT_CONFIG = {
    # 모델 설정
    "model_name": "resnet34",
    "num_classes": 17,
    "pretrained": True,
    
    # 학습 설정
    "img_size": 224,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 50,
    "num_workers": 4,
    
    # 데이터 증강
    "augmentation": {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation": 15,
        "brightness": 0.2,
        "contrast": 0.2,
    },
    
    # 조기 종료
    "early_stopping": {
        "enabled": True,
        "patience": 10,
        "min_delta": 0.001,
    }
}
```

#### `BASELINE_CONFIG`
간단한 베이스라인용 설정 (빠른 테스트용)

```python
BASELINE_CONFIG = {
    "model_name": "resnet34",
    "num_classes": 17,
    "pretrained": True,
    "img_size": 32,        # 빠른 처리를 위한 작은 이미지
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 1,           # 빠른 테스트
    "num_workers": 0,
    "enable_macos_optimization": True,
    "compatibility_mode": False
}
```

#### `DATA_CONFIG`
데이터 경로 및 분할 설정

```python
DATA_CONFIG = {
    "train_csv": str(DATA_DIR / "train.csv"),
    "test_csv": str(DATA_DIR / "sample_submission.csv"),
    "train_dir": str(DATA_DIR / "train"),
    "test_dir": str(DATA_DIR / "test"),
    "submission_dir": str(DATA_DIR / "submissions"),
    "val_split": 0.2,
    "stratify": True,
    "random_seed": 42
}
```

### 주요 함수

#### `get_wandb_config() -> Dict[str, Any]`
WandB 초기화를 위한 설정 반환

**반환값**: WandB 초기화에 필요한 설정 딕셔너리

**사용 예제**:
```python
from config import get_wandb_config
from wandb_utils import init_wandb

config = get_wandb_config()
run = init_wandb(config, run_name="my_experiment")
```

#### `get_experiment_name(model_name: str = None, additional_info: str = None) -> str`
실험 이름 자동 생성

**매개변수**:
- `model_name`: 모델 이름 (선택사항)
- `additional_info`: 추가 정보 (선택사항)

**반환값**: 생성된 실험 이름

**사용 예제**:
```python
from config import get_experiment_name

name = get_experiment_name("resnet34", "augmented")
print(name)  # "resnet34_img224_bs32_lr0.001_augmented"
```

#### `validate_config() -> bool`
설정 유효성 검증

**반환값**: 설정이 유효하면 True, 아니면 False

**사용 예제**:
```python
from config import validate_config

if validate_config():
    print("설정이 유효합니다")
else:
    print("설정에 오류가 있습니다")
```

---

## 🚀 device_utils.py

**용도**: 크로스 플랫폼 디바이스 최적화 및 하드웨어 감지

### 주요 함수

#### `get_optimal_device() -> Tuple[torch.device, str]`
최적 디바이스 자동 감지

**반환값**: (디바이스 객체, 디바이스 타입 문자열)

**지원 디바이스**:
- **CUDA**: NVIDIA GPU (Linux/Windows)
- **MPS**: Apple Silicon GPU (macOS)
- **CPU**: CPU fallback (모든 플랫폼)

**사용 예제**:
```python
from device_utils import get_optimal_device

device, device_type = get_optimal_device()
print(f"사용 디바이스: {device} ({device_type})")

# 모델을 디바이스로 이동
model = model.to(device)
```

#### `setup_training_device() -> Tuple[torch.device, str]`
학습용 디바이스 설정 및 최적화

**반환값**: (디바이스 객체, 디바이스 타입 문자열)

**최적화 기능**:
- **MPS**: Apple Silicon 최적화, 메모리 정리
- **CUDA**: cuDNN 벤치마킹, 성능 최적화
- **CPU**: 멀티스레딩 최적화

**사용 예제**:
```python
from device_utils import setup_training_device

device, device_type = setup_training_device()
# 자동으로 최적화 설정이 적용됨
```

#### `get_dataloader_config(device_type: str) -> Dict[str, Any]`
디바이스 타입에 따른 DataLoader 최적화 설정

**매개변수**:
- `device_type`: "MPS", "CUDA", "CPU" 중 하나

**반환값**: DataLoader 설정 딕셔너리

**최적화 전략**:

```python
# MPS (Apple Silicon)
{
    "pin_memory": False,  # MPS는 pin_memory 불필요
    "num_workers": 0      # 멀티프로세싱 이슈 방지
}

# CUDA (NVIDIA GPU)
{
    "pin_memory": True,   # GPU 메모리 최적화
    "num_workers": 4-8    # 병렬 데이터 로딩
}

# CPU
{
    "pin_memory": False,
    "num_workers": 2-4    # 적당한 병렬 처리
}
```

**사용 예제**:
```python
from device_utils import setup_training_device, get_dataloader_config
from torch.utils.data import DataLoader

device, device_type = setup_training_device()
dataloader_config = get_dataloader_config(device_type)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    **dataloader_config  # 최적화된 설정 적용
)
```

#### `check_gpu_memory(device: torch.device)`
GPU 메모리 정보 확인

**매개변수**:
- `device`: 확인할 디바이스

**기능**:
- **CUDA**: GPU 메모리 사용량 표시
- **MPS**: 통합 메모리 정보 표시
- **CPU**: 시스템 메모리 정보 표시

#### `test_device() -> Tuple[torch.device, str]`
디바이스 테스트 및 검증

**반환값**: (테스트된 디바이스, 디바이스 타입)

**테스트 내용**:
- 텐서 연산 테스트
- 메모리 사용량 확인
- DataLoader 설정 권장사항 출력

---

## 📊 wandb_utils.py

**용도**: WandB 실험 추적 및 관리 유틸리티

### 주요 함수

#### `init_wandb(config: Dict[str, Any], run_name: str = None) -> wandb.Run`
WandB 런 초기화

**매개변수**:
- `config`: WandB 설정 딕셔너리
- `run_name`: 런 이름 (선택사항)

**반환값**: 초기화된 WandB 런 객체

**사용 예제**:
```python
from wandb_utils import init_wandb
from config import get_wandb_config

config = get_wandb_config()
run = init_wandb(config, run_name="baseline_test")
```

#### `log_metrics(metrics: Dict[str, Any], step: int = None, commit: bool = True)`
메트릭 로깅

**매개변수**:
- `metrics`: 로깅할 메트릭 딕셔너리
- `step`: 스텝 번호 (선택사항)
- `commit`: 즉시 커밋 여부

**사용 예제**:
```python
from wandb_utils import log_metrics

# 학습 메트릭 로깅
log_metrics({
    'train_loss': 0.5,
    'train_accuracy': 0.85,
    'val_loss': 0.6,
    'val_accuracy': 0.82,
    'epoch': 10
})

# 배치별 메트릭 (커밋하지 않고)
log_metrics({
    'batch_loss': 0.45
}, commit=False)
```

#### `log_model_info(model, input_shape: tuple = None)`
모델 정보 자동 로깅

**매개변수**:
- `model`: PyTorch 모델
- `input_shape`: 입력 텐서 모양 (선택사항)

**로깅 정보**:
- 모델 클래스 이름
- 총 파라미터 수
- 학습 가능한 파라미터 수
- 모델 그래프 (input_shape 제공 시)

**사용 예제**:
```python
from wandb_utils import log_model_info
import timm

model = timm.create_model('resnet34', num_classes=17)
log_model_info(model, input_shape=(3, 224, 224))
```

#### `log_system_info()`
시스템 정보 로깅

**로깅 정보**:
- Python 버전
- 플랫폼 정보
- PyTorch 버전
- CUDA 정보 (사용 가능한 경우)
- GPU 정보

**사용 예제**:
```python
from wandb_utils import log_system_info

# 실험 시작 시 시스템 정보 로깅
log_system_info()
```

#### `log_confusion_matrix(y_true, y_pred, class_names: List[str] = None)`
혼동 행렬 시각화 및 로깅

**매개변수**:
- `y_true`: 실제 라벨
- `y_pred`: 예측 라벨
- `class_names`: 클래스 이름 목록 (선택사항)

**사용 예제**:
```python
from wandb_utils import log_confusion_matrix

# 검증 결과 혼동 행렬 로깅
log_confusion_matrix(
    y_true=val_targets,
    y_pred=val_predictions,
    class_names=[f"Class_{i}" for i in range(17)]
)
```

#### `create_run_name(model_name: str, experiment_type: str = None) -> str`
타임스탬프 포함 런 이름 생성

**매개변수**:
- `model_name`: 모델 이름
- `experiment_type`: 실험 타입 (선택사항)

**반환값**: 생성된 런 이름

**사용 예제**:
```python
from wandb_utils import create_run_name

run_name = create_run_name("resnet34", "baseline")
print(run_name)  # "resnet34_baseline_1202_1430"
```

#### `save_model_artifact(model_path: str, name: str, type_: str = "model", metadata: Dict = None)`
모델을 WandB 아티팩트로 저장

**매개변수**:
- `model_path`: 저장된 모델 파일 경로
- `name`: 아티팩트 이름
- `type_`: 아티팩트 타입
- `metadata`: 메타데이터 (선택사항)

**사용 예제**:
```python
from wandb_utils import save_model_artifact

save_model_artifact(
    model_path="models/best_model.pth",
    name="best_resnet34",
    metadata={"val_f1": 0.85, "epoch": 25}
)
```

#### `finish_run()`
현재 WandB 런 종료

**사용 예제**:
```python
from wandb_utils import finish_run

try:
    # 학습 코드
    pass
finally:
    finish_run()  # 항상 런 종료
```

---

## 🎯 baseline_simple.py

**용도**: 빠른 환경 검증을 위한 간단한 베이스라인 실행

### 주요 클래스

#### `ImageDataset(Dataset)`
간단한 이미지 데이터셋 클래스

**매개변수**:
- `csv`: CSV 파일 경로
- `path`: 이미지 폴더 경로
- `transform`: 데이터 변환 (선택사항)

**사용 예제**:
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(32, 32),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

dataset = ImageDataset(
    csv="data/train.csv",
    path="data/train",
    transform=transform
)
```

### 주요 함수

#### `train_one_epoch(loader, model, optimizer, loss_fn, device) -> Dict[str, float]`
한 에포크 학습 실행

**매개변수**:
- `loader`: 데이터 로더
- `model`: PyTorch 모델
- `optimizer`: 옵티마이저
- `loss_fn`: 손실 함수
- `device`: 디바이스

**반환값**: 학습 메트릭 딕셔너리

#### `setup_device() -> Tuple[torch.device, Dict[str, Any]]`
호환성 우선 디바이스 설정

**반환값**: (디바이스, DataLoader 설정)

**특징**:
- 호환성 모드 지원
- macOS 최적화 모드 지원
- Fallback 메커니즘

#### `main()`
메인 실행 함수

**실행 과정**:
1. 로깅 설정
2. 디바이스 설정
3. 데이터 로딩
4. 모델 학습 (1 epoch)
5. 테스트 데이터 예측
6. 결과 저장

---

## 🚀 train_with_wandb.py

**용도**: WandB 통합 고급 실험 실행

### 주요 클래스

#### `ImageDataset(Dataset)`
고급 이미지 데이터셋 클래스

**특징**:
- 학습/테스트 모드 지원
- 검증 세트 분할 지원
- 고급 데이터 증강

### 주요 함수

#### `get_transforms(img_size: int, is_train: bool = True) -> A.Compose`
데이터 증강 변환 생성

**매개변수**:
- `img_size`: 이미지 크기
- `is_train`: 학습용 여부

**학습용 증강**:
- 수평 뒤집기
- 회전 (±15도)
- 밝기/대비 조정
- 정규화

**검증/테스트용**:
- 크기 조정만
- 정규화

**사용 예제**:
```python
from train_with_wandb import get_transforms

train_transform = get_transforms(224, is_train=True)
val_transform = get_transforms(224, is_train=False)
```

#### `train_one_epoch(loader, model, optimizer, loss_fn, device, epoch) -> Dict[str, float]`
WandB 로깅이 포함된 학습 함수

**매개변수**:
- `loader`: 학습 데이터 로더
- `model`: PyTorch 모델
- `optimizer`: 옵티마이저
- `loss_fn`: 손실 함수
- `device`: 디바이스
- `epoch`: 현재 에포크

**반환값**: 학습 메트릭 딕셔너리

**특징**:
- 실시간 진행률 표시
- 배치별 메트릭 로깅 (50배치마다)
- 에포크별 정확도/F1 스코어 계산

**사용 예제**:
```python
from train_with_wandb import train_one_epoch

metrics = train_one_epoch(
    loader=train_loader,
    model=model,
    optimizer=optimizer,
    loss_fn=criterion,
    device=device,
    epoch=current_epoch
)

print(f"Train Loss: {metrics['train_loss']:.4f}")
print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
```

#### `validate_one_epoch(loader, model, loss_fn, device, epoch) -> Dict[str, Any]`
검증 실행 함수

**매개변수**:
- `loader`: 검증 데이터 로더
- `model`: PyTorch 모델
- `loss_fn`: 손실 함수
- `device`: 디바이스
- `epoch`: 현재 에포크

**반환값**: 검증 메트릭 및 예측 결과

```python
{
    'val_loss': float,
    'val_accuracy': float,
    'val_f1': float,
    'predictions': np.array,
    'targets': np.array
}
```

**특징**:
- 안전한 검증 (빈 로더 처리)
- 혼동 행렬용 예측/타겟 반환
- 그래디언트 계산 비활성화

#### `predict_test_data(model, device, test_loader, output_file) -> pd.DataFrame`
테스트 데이터 예측 및 저장

**매개변수**:
- `model`: 학습된 모델
- `device`: 디바이스
- `test_loader`: 테스트 데이터 로더
- `output_file`: 출력 파일 경로

**반환값**: 제출용 DataFrame

**기능**:
- 테스트 데이터 예측
- 제출 형식으로 저장
- 예측 분포 출력

**사용 예제**:
```python
from train_with_wandb import predict_test_data

submission_df = predict_test_data(
    model=best_model,
    device=device,
    test_loader=test_loader,
    output_file="submission_20241202.csv"
)
```

#### `train_model()`
메인 학습 함수

**실행 과정**:
1. 로깅 설정
2. 설정 검증
3. 디바이스 설정
4. WandB 초기화
5. 데이터 준비 (train/val 분할)
6. 모델 설정
7. 학습 루프 실행
8. 최적 모델 저장
9. 테스트 예측 생성

**특징**:
- 완전 자동화된 실험 파이프라인
- 검증 기반 최적 모델 저장
- 상세한 로깅 및 진행률 표시
- 에러 처리 및 리소스 정리

---

## 🔤 train_with_ocr.py

**용도**: OCR 통합 멀티모달 문서 분류

### 주요 특징

**멀티모달 접근**:
- 이미지 특징 추출 (CNN)
- 텍스트 특징 추출 (OCR → NLP)
- 특징 융합 및 분류

**지원 OCR 엔진**:
- **EasyOCR**: 다국어 지원, GPU 가속
- **Tesseract**: 전통적인 OCR 엔진

### 주요 클래스

#### `MultimodalDataset(Dataset)`
이미지 + 텍스트 데이터셋

**특징**:
- OCR 텍스트 자동 추출
- 이미지 + 텍스트 동시 처리
- 캐싱 메커니즘 (선택사항)

#### `MultimodalModel(nn.Module)`
멀티모달 분류 모델

**아키텍처**:
```python
# 이미지 인코더
image_encoder = timm.create_model('resnet34', pretrained=True)

# 텍스트 인코더  
text_encoder = nn.LSTM(embedding_dim, hidden_dim)

# 융합 레이어
fusion_layer = nn.Linear(image_dim + text_dim, hidden_dim)

# 분류기
classifier = nn.Linear(hidden_dim, num_classes)
```

### 주요 함수

#### `extract_text_with_ocr(image_path: str, ocr_engine: str = "easyocr") -> str`
이미지에서 텍스트 추출

**매개변수**:
- `image_path`: 이미지 파일 경로
- `ocr_engine`: "easyocr" 또는 "tesseract"

**반환값**: 추출된 텍스트

**사용 예제**:
```python
from train_with_ocr import extract_text_with_ocr

text = extract_text_with_ocr("document.jpg", "easyocr")
print(f"추출된 텍스트: {text}")
```

#### `preprocess_text(text: str) -> List[str]`
텍스트 전처리

**전처리 과정**:
- 특수문자 제거
- 소문자 변환
- 토큰화
- 불용어 제거

#### `train_multimodal_model()`
멀티모달 모델 학습

**특징**:
- 이미지 + 텍스트 동시 학습
- 가중치 조정 가능한 손실 함수
- OCR 품질에 따른 적응적 학습

---

## 🛠️ 사용 패턴 및 모범 사례

### 1. 설정 관리 패턴

```python
# 1. 설정 로드 및 검증
from config import validate_config, EXPERIMENT_CONFIG, DATA_CONFIG

if not validate_config():
    raise ValueError("설정 오류")

# 2. 디바이스 설정
from device_utils import setup_training_device, get_dataloader_config

device, device_type = setup_training_device()
dataloader_config = get_dataloader_config(device_type)

# 3. WandB 초기화
from wandb_utils import init_wandb, create_run_name
from config import get_wandb_config

run_name = create_run_name("resnet34", "experiment")
wandb_config = get_wandb_config()
run = init_wandb(wandb_config, run_name)
```

### 2. 학습 루프 패턴

```python
from wandb_utils import log_metrics, log_model_info, finish_run

try:
    # 모델 정보 로깅
    log_model_info(model, input_shape=(3, 224, 224))
    
    for epoch in range(epochs):
        # 학습
        train_metrics = train_one_epoch(...)
        
        # 검증
        val_metrics = validate_one_epoch(...)
        
        # 메트릭 로깅
        log_metrics({
            **train_metrics,
            **val_metrics,
            'epoch': epoch,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        # 최적 모델 저장
        if val_metrics['val_f1'] > best_f1:
            torch.save(model.state_dict(), 'best_model.pth')
            
finally:
    finish_run()
```

### 3. 크로스 플랫폼 DataLoader 패턴

```python
from device_utils import setup_training_device, get_dataloader_config

# 디바이스별 최적화 설정 자동 적용
device, device_type = setup_training_device()
dataloader_config = get_dataloader_config(device_type)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    **dataloader_config  # 플랫폼에 최적화된 설정
)
```

### 4. 에러 처리 패턴

```python
import logging
from wandb_utils import finish_run

logger = logging.getLogger(__name__)

try:
    # 학습 코드
    train_model()
    
except Exception as e:
    logger.error(f"학습 중 오류 발생: {e}")
    raise
    
finally:
    # 리소스 정리
    finish_run()
    logger.info("실험 종료")
```

### 5. OCR 통합 패턴

```python
from train_with_ocr import extract_text_with_ocr, MultimodalDataset

# OCR 설정 확인
try:
    import easyocr
    ocr_engine = "easyocr"
except ImportError:
    try:
        import pytesseract
        ocr_engine = "tesseract"
    except ImportError:
        raise ImportError("OCR 라이브러리가 설치되지 않았습니다")

# 멀티모달 데이터셋 생성
dataset = MultimodalDataset(
    csv_path="data/train.csv",
    image_dir="data/train",
    ocr_engine=ocr_engine,
    cache_text=True  # 텍스트 캐싱으로 성능 향상
)
```

---

## 🔧 확장 가이드

### 새로운 모델 추가

1. **config.py 수정**:
```python
EXPERIMENT_CONFIG["model_name"] = "efficientnet_b0"
```

2. **모델 로드 부분 수정**:
```python
# train_with_wandb.py
model = timm.create_model(
    EXPERIMENT_CONFIG['model_name'],
    pretrained=True,
    num_classes=17
)
```

### 새로운 메트릭 추가

1. **계산 함수 정의**:
```python
def calculate_custom_metric(y_true, y_pred):
    # 사용자 정의 메트릭 계산
    return metric_value
```

2. **로깅 추가**:
```python
custom_metric = calculate_custom_metric(targets, predictions)
log_metrics({'custom_metric': custom_metric})
```

### 새로운 데이터 증강 추가

```python
def get_advanced_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),
        # 새로운 증강 기법 추가
        A.Normalize(),
        ToTensorV2()
    ])
```

이 모듈 참조 문서는 CV-Classify 시스템의 모든 주요 컴포넌트에 대한 완전한 API 참조를 제공합니다. 각 함수와 클래스의 용도, 매개변수, 반환값, 사용 예제를 포함하여 개발자가 시스템을 효과적으로 활용할 수 있도록 돕습니다.