# Task) WandB 설정 추가

**Issue ID:** #6  
**Status:** Open  
**Author:** skier-song9  
**Created:** About 2 days ago  
**Comments:** 0  
**Labels:** 개발, 설정  
**Assignees:** skier-song9  

---

## 📊 Description

**WandB API 설정과 코드 추가**

Weights & Biases를 프로젝트에 통합하여 실험 추적 및 모델 성능 모니터링을 구현합니다.

---

## ✅ Tasks

### 1. API Key 및 환경 설정
- [ ] **WandB API key용 `.env`, `.env.template` 파일 생성**
- [ ] **`.gitignore`에 `.env` 파일 추가**

### 2. Run 관리 함수 구현
- [ ] **최근 run 이름을 불러오는 함수 추가:**
  - `get_runs()` - 모든 run 목록 조회
  - `get_latest_runs()` - 최신 run 조회
  - `auto_increment_run_suffix()` - run 이름 자동 증가

### 3. 설정 시스템 구축
- [ ] **확장성 있는 config 딕셔너리 설계**

### 4. WandB 초기화
- [ ] **wandb init 구현**

### 5. 로깅 시스템 통합
- [ ] **train 코드에 wandb log 추가**
- [ ] **validation 코드에 wandb log 추가**

---

## 🛠️ 구현 가이드

### 1. 환경 설정 파일 생성

**.env.template**
```bash
# WandB API Key
WANDB_API_KEY=your_wandb_api_key_here

# WandB Project Settings
WANDB_PROJECT=cv-classification
WANDB_ENTITY=your_username_or_team
```

**.env** (실제 사용)
```bash
# WandB API Key
WANDB_API_KEY=실제_API_키_입력

# WandB Project Settings
WANDB_PROJECT=cv-classification
WANDB_ENTITY=실제_사용자명
```

### 2. .gitignore 업데이트
```gitignore
# Environment variables
.env

# WandB files
wandb/
```

### 3. 예상 코드 구조

**config.py**
```python
import os
from dotenv import load_dotenv

load_dotenv()

WANDB_CONFIG = {
    \"api_key\": os.getenv(\"WANDB_API_KEY\"),
    \"project\": os.getenv(\"WANDB_PROJECT\", \"cv-classification\"),
    \"entity\": os.getenv(\"WANDB_ENTITY\"),
    \"tags\": [\"cv\", \"classification\", \"document\"],
}

EXPERIMENT_CONFIG = {
    \"model_name\": \"resnet34\",
    \"img_size\": 224,
    \"batch_size\": 32,
    \"learning_rate\": 1e-3,
    \"epochs\": 50,
    \"num_classes\": 17
}
```

**wandb_utils.py**
```python
import wandb
from typing import List, Dict, Optional

def get_runs() -> List[wandb.Run]:
    \"\"\"모든 run 목록 조회\"\"\"
    pass

def get_latest_runs() -> Optional[wandb.Run]:
    \"\"\"최신 run 조회\"\"\"
    pass

def auto_increment_run_suffix() -> str:
    \"\"\"run 이름 자동 증가\"\"\"
    pass
```

### 4. 학습 코드 통합 예시

**train.py**
```python
import wandb

# WandB 초기화
wandb.init(
    project=WANDB_CONFIG[\"project\"],
    entity=WANDB_CONFIG[\"entity\"],
    config=EXPERIMENT_CONFIG,
    tags=WANDB_CONFIG[\"tags\"]
)

# 학습 루프에서 로깅
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = validate(...)
    
    # WandB 로깅
    wandb.log({
        \"epoch\": epoch,
        \"train_loss\": train_loss,
        \"train_accuracy\": train_acc,
        \"val_loss\": val_loss,
        \"val_accuracy\": val_acc
    })
```

---

## 📈 로깅할 메트릭

### Training Metrics
- Loss (train/validation)
- Accuracy (train/validation)
- F1-Score (train/validation)
- Learning Rate
- Epoch

### Model Metrics
- Model Architecture
- Parameter Count
- FLOPs

### System Metrics
- GPU Usage
- Memory Usage
- Training Time

---

## 🔗 References

- **WandB 정리 노션:** [WandB 가이드](https://skier-song9.notion.site/WanDB-1d2c8d3f60f580dbb91fff477a108770?source=copy_link)
- **WandB 공식 문서:** [https://docs.wandb.ai/](https://docs.wandb.ai/)

---

## 📋 완료 체크리스트

설정 완료 후 다음 사항들을 확인하세요:

- [ ] WandB 계정 생성 및 API key 발급
- [ ] `.env` 파일에 API key 저장
- [ ] Git에 `.env` 파일이 추가되지 않았는지 확인
- [ ] WandB 프로젝트가 웹에서 정상 생성되는지 확인
- [ ] 학습 실행 시 메트릭이 WandB에 기록되는지 확인

---

**원본 이슈:** [GitHub에서 보기](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/issues/6)

---

## 💡 추가 팁

### WandB 설치
```bash
# WandB 설치
pip install wandb

# 또는 uv로
uv add wandb
```

### API Key 설정 확인
```bash
# WandB 로그인 (한 번만 실행)
wandb login

# 또는 환경변수로 설정
export WANDB_API_KEY=your_api_key
```

### 오프라인 모드 (인터넷 연결이 없을 때)
```python
# 오프라인 모드로 실행
os.environ[\"WANDB_MODE\"] = \"offline\"
```
