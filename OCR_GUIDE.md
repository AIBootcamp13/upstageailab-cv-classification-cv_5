# 🔤 OCR 통합 가이드

CV-Classify 프로젝트에 OCR(Optical Character Recognition) 기능이 추가되었습니다!

## 📋 현재 상태

✅ **기존 이미지 분류**: F1 스코어 0.9547 달성  
🆕 **OCR 통합 기능**: 이미지 + 텍스트 멀티모달 분류

## 🚀 OCR 통합 실행 가이드

### 1단계: OCR 환경 설정
```bash
# 메뉴에서 7번 선택 또는
./setup_ocr.sh
```

### 2단계: OCR 기능 테스트
```bash
# 메뉴에서 8번 선택 또는
cd codes && python train_with_ocr.py --dry-run
```

### 3단계: OCR 통합 훈련
```bash
# 포그라운드 실행 (메뉴 9번)
cd codes && python train_with_ocr.py

# 또는 백그라운드 실행 (메뉴 10번)
nohup python train_with_ocr.py > ../logs/ocr_training.log 2>&1 &
```

## 🔤 OCR 기능 특징

### 📊 멀티모달 접근법
- **이미지 특징**: ResNet34로 이미지 특징 추출
- **텍스트 특징**: OCR로 추출한 텍스트를 TF-IDF로 벡터화
- **융합 분류**: 이미지 + 텍스트 특징을 결합하여 최종 분류

### 🔧 OCR 백엔드 지원
1. **EasyOCR** (추천)
   - GPU 가속 지원
   - 한국어/영어 동시 인식
   - 설정이 간단
   
2. **Tesseract** (대안)
   - 빠른 처리 속도
   - macOS/Linux 네이티브 지원
   - 더 가벼운 메모리 사용

### 💾 캐싱 시스템
- OCR 결과를 `data/ocr_cache/`에 자동 저장
- 재실행 시 이전 결과 재사용으로 시간 단축
- 수동 삭제로 OCR 재처리 가능

## 📈 성능 비교

| 방법 | F1 스코어 | 장점 | 단점 |
|------|-----------|------|------|
| 이미지만 | 0.9547 | 빠른 처리 | 텍스트 정보 무시 |
| OCR 통합 | **TBD** | 텍스트 정보 활용 | 처리 시간 증가 |

## 🛠️ 기술 스택

- **OCR**: EasyOCR / Tesseract
- **텍스트 처리**: TF-IDF, scikit-learn
- **모델 융합**: PyTorch 멀티모달 아키텍처
- **캐싱**: Pickle 기반 결과 저장

## 📝 사용 예시

```python
# 기본 이미지 분류 (기존)
python train_with_wandb.py

# OCR 통합 분류 (신규)
python train_with_ocr.py

# DRY RUN으로 환경 확인
python train_with_ocr.py --dry-run
```

## 🔍 문서 타입 예시

프로젝트는 다음 17가지 문서 타입을 분류합니다:

- 계좌번호 (account_number)
- 진단서 (diagnosis)  
- 운전면허증 (driver_license)
- 의료비 영수증 (medical_bill_receipts)
- 주민등록증 (national_id_card)
- 여권 (passport)
- 약국 영수증 (pharmaceutical_receipt)
- 처방전 (prescription)
- 이력서 (resume)
- 차량등록증 (vehicle_registration_certificate)
- 번호판 (vehicle_registration_plate)
- 등...

이런 문서들은 **텍스트 내용이 분류에 매우 중요**하므로 OCR 통합이 성능 향상에 도움이 될 것으로 예상됩니다.

## 🎯 기대 효과

1. **정확도 향상**: 이미지 + 텍스트 정보로 더 정확한 분류
2. **강건성**: 이미지 품질이 낮아도 텍스트로 보완
3. **해석 가능성**: 어떤 텍스트가 분류에 영향을 주는지 분석 가능

## ⚠️ 주의사항

- OCR 처리 시간이 추가로 소요됩니다 (첫 실행 시)
- GPU 메모리 사용량이 증가할 수 있습니다
- 한국어 텍스트가 포함된 문서에서 특히 효과적입니다

## 🆘 문제 해결

### OCR 라이브러리 설치 실패
```bash
# macOS에서 Tesseract 설치
brew install tesseract tesseract-lang

# Ubuntu에서 Tesseract 설치  
sudo apt-get install tesseract-ocr tesseract-ocr-kor

# EasyOCR 설치 (대안)
pip install easyocr
```

### 메모리 부족
- 배치 크기를 줄이세요 (config.py에서 batch_size 조정)
- 이미지 크기를 줄이세요 (img_size 조정)

### OCR 정확도 개선
- EasyOCR 사용 (Tesseract보다 정확)
- 이미지 전처리 추가 (노이즈 제거, 대비 조정)

---

**TIP**: 메뉴 시스템(`./menu.sh`)을 사용하면 모든 OCR 기능을 쉽게 실행할 수 있습니다!
