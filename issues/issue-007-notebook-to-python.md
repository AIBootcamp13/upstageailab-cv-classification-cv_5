# (모델 학습) .ipynb 파일을 .py 파일로 바꿔서 학습시키기

**Issue ID:** #7  
**Status:** Open  
**Author:** skier-song9  
**Created:** About 2 days ago  
**Comments:** 0  
**Labels:** 문서, 필수  
**Assignees:** skier-song9  

---

## 🙌 Intro

딥러닝 모델 학습은 시간이 오래 걸리기 때문에 jupyter notebook으로 학습을 돌리면 컴퓨터를 끌 수 없습니다.

AIStages 서버는 24시간 운영되지만, **로컬 PC가 AIStages와 연결이 끊기면 jupyter notebook에서의 학습도 종료됩니다.**

따라서 오랜 시간이 걸리는 학습을 진행할 때는 아래의 과정을 거쳐 **`.py` 파일로 변환해 AIStages 서버 백그라운드에서 실행**해야 합니다.

---

## 📗 .ipynb 파일 준비

### 사전 테스트 필수!
- 중간에 오류가 발생하지 않는지, **`epoch=1`** 등으로 학습을 짧게 하여 처음부터 끝까지 코드를 실행해봅니다
- **학습 완료 후, 모델 저장 및 파일 저장까지 잘 되는지 확인**

### 체크리스트
- [ ] 전체 코드가 에러 없이 실행되는지 확인
- [ ] 모델 저장 경로가 올바른지 확인
- [ ] 결과 파일(CSV 등) 저장이 정상인지 확인
- [ ] 데이터 경로가 서버 환경에 맞는지 확인

---

## 📕 .py 파일로 변환

### 1단계: 터미널에서 .ipynb 파일 위치로 이동
```bash
cd ~/upstageailab-cv-classification-cv_5/codes
```

### 2단계: ipynb를 py 파일로 변환
```bash
# 기본 변환 명령어
jupyter nbconvert --to script [파일이름.ipynb]

# 예시
jupyter nbconvert --to script \"[Image Classification] 베이스라인 코드 해설.ipynb\"
```

### 3단계: 변환된 .py 파일 검토
- 불필요한 주석이나 마법 명령어(`%`, `!`) 제거
- 파일 경로가 올바른지 확인
- import 구문이 정상인지 확인

---

## ⚙️ Background에서 Python 파일 실행하기

> **Reference:** [nohup 명령어 정리 노션](https://www.notion.so/skier-song9/Linux-251ed38c07b14f37b6617c96de075456?source=copy_link#11fc8d3f60f58086a17ddff5698038cd)

### 1단계: nohup 명령어 설치 (필요시)
```bash
# nohup 명령어가 없다면 설치
apt update && apt install -y coreutils
```

### 2단계: 실행할 Python 파일 위치로 이동
```bash
# 파일이 codes/practice/baseline.py 라면
cd ~/upstageailab-cv-classification-cv_5/codes/practice
```

### 3단계: Python 파일을 백그라운드에서 실행 & 로그 파일 설정
```bash
# log 디렉토리: upstageailab-cv-classification-cv_5/logs
nohup python baseline.py > ../../logs/[log파일명지정.log] 2>&1 &

# 예시
nohup python baseline.py > ../../logs/baseline_training.log 2>&1 &
```

### 4단계: 프로세스 확인
- 3번 명령어를 실행하면 **숫자가 터미널에 출력**되는데, 이는 Python 파일에 대한 **PID(Process ID)**입니다.
- 터미널에 `ps aux | grep [PID]`를 입력했는데 `Done`이 나오면 성공적으로 코드가 수행된 것입니다.

---

## 📊 프로세스 모니터링

### 실행 중인 프로세스 확인
```bash
# 특정 PID로 프로세스 확인
ps aux | grep [PID]

# Python 프로세스 전체 확인
ps aux | grep python

# 백그라운드 작업 목록 확인
jobs
```

### 로그 파일 실시간 확인
```bash
# 로그 파일 실시간 모니터링
tail -f ../../logs/baseline_training.log

# 로그 파일 전체 확인
cat ../../logs/baseline_training.log

# 로그 파일 마지막 n줄 확인
tail -n 50 ../../logs/baseline_training.log
```

### 프로세스 종료 (필요시)
```bash
# PID로 프로세스 종료
kill [PID]

# 강제 종료
kill -9 [PID]
```

---

## 📁 파일 구조 예시

```
upstageailab-cv-classification-cv_5/
├── codes/
│   ├── [Image Classification] 베이스라인 코드 해설.ipynb
│   ├── [Image Classification] 베이스라인 코드 해설.py  # 변환된 파일
│   └── practice/
│       └── baseline.py  # 실제 실행할 파일
├── logs/
│   ├── baseline_training.log     # 학습 로그
│   ├── model_experiment_1.log    # 실험 로그
│   └── wandb_sync.log           # WandB 동기화 로그
└── models/
    ├── best_model.pth
    └── checkpoint_epoch_50.pth
```

---

## 💡 추가 팁

### 1. 환경 변수 설정
```bash
# GPU 사용 설정
export CUDA_VISIBLE_DEVICES=0

# WandB 오프라인 모드 (인터넷 연결 불안정시)
export WANDB_MODE=offline
```

### 2. 메모리 모니터링
```bash
# GPU 메모리 사용량 확인
nvidia-smi

# 시스템 메모리 확인
free -h

# 디스크 사용량 확인
df -h
```

### 3. 안전한 학습 실행을 위한 체크리스트
- [ ] 데이터 경로가 올바른가?
- [ ] 모델 저장 경로가 존재하는가?
- [ ] 충분한 디스크 공간이 있는가?
- [ ] GPU 메모리가 충분한가?
- [ ] 로그 디렉토리가 존재하는가?

---

## 🚨 주의사항

1. **로그 파일 크기**: 장시간 학습 시 로그 파일이 매우 커질 수 있으니 주기적으로 확인
2. **디스크 공간**: 모델 체크포인트와 로그로 인한 디스크 공간 부족 주의
3. **프로세스 관리**: 불필요한 프로세스는 적절히 종료하여 리소스 절약
4. **네트워크 연결**: WandB 등 외부 서비스 사용시 네트워크 상태 확인

---

**원본 이슈:** [GitHub에서 보기](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/issues/7)

---

## 📋 실습 예제

### 베이스라인 코드 변환 및 실행 전체 과정

```bash
# 1. 노트북 위치로 이동
cd ~/upstageailab-cv-classification-cv_5/codes

# 2. ipynb를 py로 변환
jupyter nbconvert --to script \"[Image Classification] 베이스라인 코드 해설.ipynb\"

# 3. practice 폴더로 복사
cp \"[Image Classification] 베이스라인 코드 해설.py\" practice/baseline.py

# 4. practice 폴더로 이동
cd practice

# 5. 백그라운드에서 실행
nohup python baseline.py > ../../logs/baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 6. PID 확인 및 저장
echo $! > ../../logs/baseline.pid

# 7. 로그 실시간 확인
tail -f ../../logs/baseline_$(date +%Y%m%d_%H%M%S).log
```
