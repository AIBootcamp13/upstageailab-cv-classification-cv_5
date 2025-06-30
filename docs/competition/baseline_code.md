# 베이스라인 코드

**출처:** [aistages](https://stages.ai/competitions/356/data/baseline)  
**클립 날짜:** 2025-06-30T12:43:40+09:00

## 데이터 다운로드 링크

https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000356/data/code.tar.gz

## 베이스라인 코드 다운로드

다운로드 탭을 참고, wget 명령어를 통해 데이터셋을 본인의 작업 환경에 다운로드 합니다.

```bash
# 코드 다운로드
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000356/data/code.tar.gz
```

## 베이스라인 코드 설명

### 코드 파일 구조

```
├─code/
│ ├──baseline_code.ipynb # 베이스라인 코드가 작성되어 있습니다.
│ ├──requirements.txt # 학습을 위해 필요 라이브러리가 작성되어 있습니다.
```

### 베이스라인 코드 설명

**기본 정보**
- **사용 모델** : ResNet34
- **데이터셋 전처리** : Resize & Normalize
- **이미지 사이즈** : 32
- **Epochs** : 1
- **Seed** : 42
- **소요 학습 시간** : 약 5초 (RTX 3090 기준)
- **Public score** : 0.1659

**코드 내 목차** (상세 내용은 노트북 파일을 참고해주세요!)
- Prepare Environments
- Import Library & Define Functions
- Hyper-parameters
- Load Data
- Train Model
- Inference & Save File
