# 📋 Issue #7 실행 가이드 - Notebook to Python

## ✅ 구현 완료된 도구들

다음 도구들이 자동으로 생성되었습니다:

### 🛠️ 생성된 스크립트들

| 파일 | 용도 | 설명 |
|------|------|------|
| `scripts/run_training.sh` | 범용 백그라운드 실행 | Linux/macOS 모두 지원 |
| `scripts/macos_launcher.sh` | macOS 최적화 실행 | Screen/Tmux 세션 지원 |
| `scripts/monitor.py` | 실시간 모니터링 | 프로세스/시스템 리소스 확인 |

### 🎯 핵심 기능들

- ✅ **백그라운드 실행**: 터미널 종료해도 학습 계속
- ✅ **실시간 모니터링**: CPU, 메모리, GPU 사용량 확인
- ✅ **로그 관리**: 자동 로그 파일 생성 및 관리
- ✅ **프로세스 제어**: 시작/중지/상태확인
- ✅ **macOS 최적화**: Screen/Tmux 세션 지원

---

# 🚀 사용자가 해야 할 단계별 가이드

## 1️⃣ **스크립트 실행 권한 설정**

```bash
# 실행 권한 부여
chmod +x scripts/run_training.sh
chmod +x scripts/macos_launcher.sh
```

## 2️⃣ **방법 선택**

### 🍎 **macOS 사용자 (권장)**
```bash
# 대화형 방법 선택 (가장 쉬움)
./scripts/macos_launcher.sh start

# 또는 직접 Screen 세션으로 시작
./scripts/macos_launcher.sh screen
```

### 🐧 **Linux/범용 방법**
```bash
# 백그라운드 학습 시작
./scripts/run_training.sh start

# 환경 확인
./scripts/run_training.sh check
```

## 3️⃣ **실시간 모니터링**

```bash
# 연속 모니터링 (5초마다 업데이트)
python3 scripts/monitor.py monitor

# 한번만 상태 확인
python3 scripts/monitor.py status

# 로그 확인
python3 scripts/monitor.py logs --lines 100
```

## 4️⃣ **프로세스 관리**

### 상태 확인
```bash
# 범용 방법
./scripts/run_training.sh status

# macOS 방법
./scripts/macos_launcher.sh list
```

### 학습 중지
```bash
# 범용 방법
./scripts/run_training.sh stop

# 또는 직접 kill
kill [PID번호]
```

## 5️⃣ **로그 확인**

```bash
# 실시간 로그 확인
./scripts/run_training.sh follow

# 최근 로그 확인
./scripts/run_training.sh logs 50
```

---

# 💡 상황별 사용법

## 🔥 **빠른 시작 (macOS)**

```bash
# 1. 권한 설정
chmod +x scripts/*.sh

# 2. 대화형 실행
./scripts/macos_launcher.sh start

# 3. 모니터링
python3 scripts/monitor.py monitor
```

## ⚡ **백그라운드 실행 (모든 OS)**

```bash
# 1. 백그라운드 시작
./scripts/run_training.sh start

# 2. 상태 확인
./scripts/run_training.sh status

# 3. 로그 확인
./scripts/run_training.sh follow
```

## 🖥️ **Screen 세션 사용 (고급)**

```bash
# Screen이 없다면 설치
brew install screen  # macOS
# 또는 apt install screen  # Linux

# Screen 세션으로 시작
./scripts/macos_launcher.sh screen

# 세션 접속
screen -r [세션이름]

# 세션에서 나가기: Ctrl+A, D
# 세션 종료: Ctrl+A, K
```

---

# 🔍 문제 해결

## ❌ **권한 오류**
```bash
# 해결: 실행 권한 부여
chmod +x scripts/run_training.sh
chmod +x scripts/macos_launcher.sh
```

## ❌ **Python 모듈 오류**
```bash
# 해결: 의존성 설치
python3 -m pip install psutil wandb python-dotenv seaborn
```

## ❌ **데이터 없음 오류**
```bash
# 해결: 더미 데이터 생성 또는 실제 데이터 배치
# (이전에 안내한 더미 데이터 생성 방법 사용)
```

## ❌ **포트/GPU 오류**
```bash
# GPU 확인
nvidia-smi

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
```

---

# 📊 모니터링 예시

## 실시간 모니터링 화면
```
🚀 CV-Classify Training Monitor
📅 2025-06-29 15:30:45

📊 Training Process Status
  PID: 12345
  Status: running
  CPU: 85.2%
  Memory: 2048.5 MB (12.3%)
  Runtime: 02:15:30

💻 System Resources
  CPU: 45.2% (8 cores)
  Memory: 8.2GB / 16.0GB (51.2%)
  GPU: Tesla V100 - 78% usage

📝 Recent Log (last 5 lines)
  Epoch 15/50 - Train Loss: 0.234
  Val Accuracy: 87.5%
  Best model saved!
```

---

# 🎯 Issue #7 체크리스트

## ✅ **완료된 항목들**

- [x] **백그라운드 실행 도구 구현**
- [x] **프로세스 모니터링 시스템**
- [x] **로그 관리 자동화**
- [x] **macOS 환경 최적화**
- [x] **실시간 상태 확인**
- [x] **Screen/Tmux 세션 지원**
- [x] **GPU 모니터링**
- [x] **자동 환경 확인**

## 🔄 **사용자 실행 단계**

1. [ ] 스크립트 실행 권한 설정
2. [ ] 학습 실행 방법 선택
3. [ ] 백그라운드 학습 시작
4. [ ] 모니터링 및 로그 확인

---

# 🎉 결론

**Issue #7의 모든 요구사항이 구현되었고, 원본 문서보다 훨씬 더 강력한 기능들이 추가되었습니다!**

- 📱 **사용자 친화적**: 대화형 인터페이스
- 🔧 **macOS 최적화**: Screen/Tmux 세션 지원  
- 📊 **실시간 모니터링**: GPU, 메모리, CPU 확인
- 🛡️ **안정성**: 자동 환경 확인 및 에러 처리
- 🎮 **편의성**: 한 명령어로 모든 작업 수행

이제 **실행 권한 설정 후 바로 사용**하실 수 있습니다! 🚀
