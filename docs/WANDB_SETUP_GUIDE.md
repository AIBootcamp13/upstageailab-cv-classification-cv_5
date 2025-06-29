# WandB 설정 완료 가이드

## ✅ 완료된 설정들

다음 파일들이 자동으로 생성되었습니다:

### 📁 환경 설정 파일
- ✅ `.env.template` - 환경 변수 템플릿
- ✅ `.env` - 실제 환경 변수 파일 (API 키 입력 필요)
- ✅ `.gitignore` - WandB 파일 제외 규칙 추가

### 🐍 Python 모듈
- ✅ `codes/config.py` - 프로젝트 설정 관리
- ✅ `codes/wandb_utils.py` - WandB 유틸리티 함수
- ✅ `codes/train_with_wandb.py` - WandB 통합 학습 코드

### 📦 의존성
- ✅ `pyproject.toml` - wandb, seaborn 의존성 추가

---

## 🔄 다음에 해야 할 작업들

### 1️⃣ WandB 계정 생성 및 API 키 발급
```bash
# 1. https://wandb.ai 방문하여 계정 생성
# 2. Settings > API Keys에서 API 키 발급
# 3. .env 파일에 실제 API 키 입력
```

### 2️⃣ 환경 변수 설정
```bash
# .env 파일을 편집하여 실제 정보 입력
nano .env

# 다음 항목들을 실제 값으로 변경:
WANDB_API_KEY=실제_발급받은_API_키
WANDB_ENTITY=실제_사용자명_또는_팀명
```

### 3️⃣ 의존성 설치
```bash
# UV로 새로운 의존성 설치
uv sync

# 또는 pip로
pip install wandb python-dotenv seaborn
```

### 4️⃣ WandB 로그인
```bash
# 터미널에서 WandB 로그인
wandb login

# API 키 입력하라고 나오면 .env에 입력한 키 사용
```

### 5️⃣ 설정 확인
```bash
# Python에서 설정 확인
python -c "from codes.config import validate_config; validate_config()"
```

---

## 🚀 사용 방법

### 기본 학습 실행
```bash
# WandB 통합 학습 실행
cd codes
python train_with_wandb.py
```

### 설정 커스터마이징
```python
# codes/config.py에서 다음 항목들 수정 가능:
EXPERIMENT_CONFIG = {
    "model_name": "resnet34",  # 다른 모델로 변경 가능
    "img_size": 224,           # 이미지 크기
    "batch_size": 32,          # 배치 크기
    "learning_rate": 1e-3,     # 학습률
    "epochs": 50,              # 에포크 수
    # ... 기타 설정들
}
```

---

## 📊 WandB 대시보드에서 확인할 수 있는 것들

### 학습 지표
- 🔥 Loss (Train/Validation)
- 📈 Accuracy (Train/Validation) 
- 🎯 F1-Score (Train/Validation)
- ⚡ Learning Rate 변화
- ⏱️ Epoch 시간

### 모델 정보
- 🏗️ 모델 아키텍처
- 🔢 파라미터 수
- 💾 시스템 정보

### 시각화
- 📊 Confusion Matrix
- 📈 메트릭 그래프
- 🔄 실시간 학습 진행상황

---

## 🛠️ 문제 해결

### API 키 오류
```bash
# API 키가 올바르지 않은 경우
wandb login --relogin
```

### 오프라인 모드
```bash
# 인터넷 연결이 없는 경우
export WANDB_MODE=offline
python train_with_wandb.py
```

### 로그 확인
```bash
# 학습 로그 확인
tail -f logs/training.log
```

---

## 📋 체크리스트

완료 후 다음 항목들을 확인하세요:

- [ ] WandB 계정 생성 완료
- [ ] API 키 발급 및 .env 파일 업데이트 완료
- [ ] `wandb login` 성공
- [ ] `uv sync` 또는 의존성 설치 완료
- [ ] 설정 검증 (`validate_config()`) 통과
- [ ] 첫 번째 실험 실행 성공
- [ ] WandB 웹 대시보드에서 결과 확인

---

## 🎉 완료!

모든 설정이 완료되면 WandB 대시보드에서 실시간으로 학습 진행상황을 모니터링하고, 팀원들과 실험 결과를 공유할 수 있습니다!

**WandB 프로젝트 URL**: `https://wandb.ai/[your-entity]/cv-classification`
