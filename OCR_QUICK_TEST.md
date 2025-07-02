# 🛠️ OCR 모델 실행 간단 테스트

## 빠른 테스트 방법

### 1. DRY RUN으로 환경 확인
```bash
cd codes
python train_with_ocr.py --dry-run
```

### 2. 정상 실행 (짧은 테스트)
```bash
cd codes
python train_with_ocr.py
```

### 3. 메뉴에서 실행
```bash
./menu.sh
# 8번 선택 (OCR DRY RUN)
# 또는 9번 선택 (OCR 포그라운드 실행)
```

## 🐛 일반적인 문제 해결

### ImportError 발생 시:
```bash
pip install easyocr transformers scikit-learn
```

### 메모리 부족 시:
config.py에서 배치 크기 줄이기:
```python
'batch_size': 8,  # 기본값에서 줄임
```

### OCR 처리 시간이 오래 걸릴 때:
- 첫 실행 시 OCR 캐시 생성으로 시간이 오래 걸립니다
- 두 번째 실행부터는 캐시를 사용하여 빨라집니다
