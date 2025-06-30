# 베이스라인 실행 가이드

## 개요

이 프로젝트는 두 가지 베이스라인 버전을 제공합니다:

1. **간단한 베이스라인** (`baseline_simple.py`) - 공식 베이스라인과 동일한 구조
2. **고급 베이스라인** (`train_with_wandb.py`) - WandB 통합 및 고급 기능

## 1. 간단한 베이스라인 실행

공식 베이스라인과 정확히 동일한 구조로 빠른 테스트에 적합합니다.

### 실행 방법
```bash
# 프로젝트 루트 디렉토리에서
cd codes
python baseline_simple.py
```

### 특징
- **이미지 크기**: 32x32 (공식 베이스라인과 동일)
- **에포크**: 1 (빠른 테스트용)
- **모델**: ResNet34
- **출력**: `pred_baseline.csv`

### 설정 변경
`config.py`의 `BASELINE_CONFIG`에서 설정 변경 가능:
```python
BASELINE_CONFIG = {
    \"img_size\": 32,      # 이미지 크기
    \"epochs\": 1,         # 에포크 수
    \"batch_size\": 32,    # 배치 크기
    # ...
}
```

## 2. 고급 베이스라인 실행 (WandB 통합)

완전한 기능을 갖춘 훈련 시스템으로 실제 실험에 적합합니다.

### 사전 준비
1. **WandB 설정**:
   ```bash
   # .env 파일 생성 (.env.template 참고)
   cp .env.template .env
   # .env 파일에 WandB API 키 입력
   ```

2. **환경 확인**:
   ```bash
   python -c \"from codes.config import validate_config; validate_config()\"
   ```

### 실행 방법
```bash
# 프로젝트 루트 디렉토리에서
cd codes
python train_with_wandb.py
```

### 특징
- **이미지 크기**: 224x224 (더 좋은 성능)
- **에포크**: 50 (완전한 훈련)
- **검증 세트**: 20% 자동 분할
- **WandB 로깅**: 실시간 모니터링
- **모델 저장**: 최고 성능 모델 자동 저장
- **스케줄러**: CosineAnnealingLR

### 설정 변경
`config.py`의 `EXPERIMENT_CONFIG`에서 설정 변경 가능:
```python
EXPERIMENT_CONFIG = {
    \"img_size\": 224,     # 더 큰 이미지 크기
    \"epochs\": 50,        # 충분한 훈련
    \"batch_size\": 32,
    # ...
}
```

## 3. 환경 설정

### 의존성 설치
```bash
# UV 사용 (권장)
uv sync

# 또는 pip 사용
pip install -r requirements.txt
```

### 데이터 확인
```bash
# 데이터 구조 확인
ls -la data/
# 다음이 있어야 함:
# - train.csv (1570개 레이블)
# - sample_submission.csv (3140개 제출 양식)
# - train/ (1570개 학습 이미지)
# - test/ (3140개 테스트 이미지)
# - meta.csv (17개 클래스 정보)
```

## 4. 백그라운드 실행

### nohup 사용
```bash
# 간단한 베이스라인
nohup python codes/baseline_simple.py > logs/baseline_simple.log 2>&1 &

# 고급 베이스라인
nohup python codes/train_with_wandb.py > logs/train_wandb.log 2>&1 &
```

### 실행 상태 확인
```bash
# 프로세스 확인
ps aux | grep python

# 로그 확인
tail -f logs/baseline_simple.log
tail -f logs/train_wandb.log
```

## 5. 결과 확인

### 예측 파일
- **간단한 베이스라인**: `pred_baseline.csv`
- **고급 베이스라인**: WandB 아티팩트로 저장

### 성능 모니터링
- **간단한 베이스라인**: 콘솔 출력
- **고급 베이스라인**: WandB 대시보드 (https://wandb.ai)

## 6. 문제 해결

### 일반적인 오류

1. **CUDA 메모리 부족**:
   ```python
   # config.py에서 배치 크기 줄이기
   \"batch_size\": 16,  # 32에서 16으로
   ```

2. **WandB 연결 오류**:
   ```bash
   # API 키 확인
   cat .env | grep WANDB_API_KEY
   # 또는 오프라인 모드
   export WANDB_MODE=offline
   ```

3. **데이터 경로 오류**:
   ```bash
   # 데이터 구조 확인
   python -c \"from codes.config import validate_config; validate_config()\"
   ```

### 로그 확인
```bash
# 에러 로그 확인
grep -i error logs/*.log

# WandB 상태 확인
grep -i wandb logs/*.log
```

## 7. 성능 비교

| 구분 | 간단한 베이스라인 | 고급 베이스라인 |
|------|------------------|----------------|
| 실행 시간 | ~5초 | ~30분 |
| 이미지 크기 | 32x32 | 224x224 |
| 에포크 | 1 | 50 |
| 검증 | 없음 | 20% 분할 |
| 로깅 | 콘솔 | WandB |
| 예상 성능 | ~0.16 F1 | ~0.6+ F1 |

## 8. 다음 단계

1. **간단한 베이스라인**으로 환경 테스트
2. **고급 베이스라인**으로 실제 실험
3. **하이퍼파라미터 튜닝**: config.py 수정
4. **모델 변경**: `model_name` 설정 변경
5. **앙상블**: 여러 모델 결과 조합
