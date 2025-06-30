# 환경 설정 가이드

이 프로젝트는 macOS와 Ubuntu에서 모두 실행할 수 있도록 설계되었습니다.

## 자동 설치 (권장)

```bash
# 자동 환경 설정
chmod +x setup.sh
./setup.sh

# 가상환경 활성화
source .venv/bin/activate

# 프로젝트 실행
./menu.sh
```

## 수동 설치

### macOS
```bash
# 가상환경 생성
python3.11 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements-macos.txt
```

### Ubuntu (GPU 서버)
```bash
# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements-ubuntu-gpu.txt
```

### Ubuntu (CPU 전용)
```bash
# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements-ubuntu-cpu.txt
```

## Requirements 파일 설명

- `requirements-macos.txt`: macOS용 (MPS 지원)
- `requirements-ubuntu-gpu.txt`: Ubuntu GPU 서버용 (CUDA 12.1)
- `requirements-ubuntu-cpu.txt`: Ubuntu CPU 전용
- `requirements.txt`: 기본 (macOS 호환)

## 환경 확인

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print('CUDA GPU 사용 가능')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon) 사용 가능')
else:
    print('CPU 모드')
"
```

## 문제 해결

### Python 3.11 설치

**macOS:**
```bash
brew install python@3.11
```

**Ubuntu:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### CUDA 설정 (Ubuntu GPU 서버)

NVIDIA 드라이버와 CUDA 12.1이 설치되어 있는지 확인:
```bash
nvidia-smi
nvcc --version
```
