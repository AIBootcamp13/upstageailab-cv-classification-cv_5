# Ubuntu 환경 설치 가이드

## 🚀 Ubuntu에서 CV-Classify 실행하기

Ubuntu 환경에서 CV-Classify 프로젝트를 설정하고 실행하는 방법을 안내합니다.

## 📋 시스템 요구사항

- Ubuntu 18.04 LTS 이상 (20.04/22.04 LTS 권장)
- Python 3.7 이상 (3.8+ 권장)
- 최소 8GB RAM (16GB 권장)
- GPU: NVIDIA GPU (선택사항, CUDA 지원)

## 🔧 1단계: 시스템 업데이트 및 기본 패키지 설치

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 기본 개발 도구 설치
sudo apt install -y build-essential git curl wget vim

# Python 관련 패키지 설치
sudo apt install -y python3 python3-pip python3-dev python3-venv

# 유틸리티 도구
sudo apt install -y screen tmux htop tree

# 이미지 처리 라이브러리 (OpenCV 의존성)
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

## 🐍 2단계: 프로젝트 설정

```bash
# 프로젝트 클론
git clone <your-repository-url>
cd cv-classify

# 실행 권한 부여
chmod +x setup.sh
chmod +x menu.sh

# 자동 설정 실행 (권장)
./setup.sh
```

### 수동 설정 (선택사항)

```bash
# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# pip 업그레이드
pip install --upgrade pip
```

## 📦 3단계: 패키지 설치

### CPU 전용 설치
```bash
# requirements.txt에서 CPU 버전 주석 해제 후
pip install -r requirements.txt
```

### GPU 지원 설치 (NVIDIA GPU)
```bash
# CUDA 버전 확인
nvidia-smi

# 기본 CUDA 12.1 버전 (requirements.txt 기본값)
pip install -r requirements.txt

# 다른 CUDA 버전이 필요한 경우
# CUDA 11.8 예시:
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

## 🎮 4단계: NVIDIA GPU 설정 (선택사항)

### NVIDIA 드라이버 설치
```bash
# 권장 드라이버 확인
ubuntu-drivers devices

# 자동 설치 (권장)
sudo ubuntu-drivers autoinstall

# 재부팅
sudo reboot

# 설치 확인
nvidia-smi
```

### CUDA 툴킷 설치 (필요시)
```bash
# CUDA 12.1 설치 (Ubuntu 20.04 기준)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1

# 환경변수 설정
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## ✅ 5단계: 환경 검증

```bash
# Python 환경 확인
python3 --version
pip --version

# 가상환경 활성화 확인
source .venv/bin/activate

# GPU 확인 (GPU 설치한 경우)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# 디바이스 테스트
cd codes
python3 -c "from device_utils import test_device; test_device()"
```

## 🚀 6단계: 프로젝트 실행

### 메뉴 시스템 사용 (권장)
```bash
# 메뉴 실행
./menu.sh
```

### 직접 실행
```bash
# 간단한 베이스라인 실행 (30초)
cd codes
python3 baseline_simple.py

# WandB 통합 훈련 (DRY RUN)
python3 train_with_wandb.py --dry-run

# 실제 훈련
python3 train_with_wandb.py
```

## 🔧 문제 해결

### 공통 문제들

#### 1. Permission Denied 오류
```bash
# 실행 권한 설정
chmod +x setup.sh menu.sh
sudo chown -R $USER:$USER .
```

#### 2. Python 명령어 오류
```bash
# python3가 기본이 아닌 경우
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
```

#### 3. 패키지 설치 오류
```bash
# 개발 헤더 설치
sudo apt install -y python3-dev libssl-dev libffi-dev

# 이미지 처리 라이브러리
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev

# OpenCV 의존성
sudo apt install -y libopencv-dev python3-opencv
```

#### 4. 메모리 부족
```bash
# 스왑 파일 생성 (4GB)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구 설정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### GPU 관련 문제들

#### 1. CUDA 버전 불일치
```bash
# CUDA 버전 확인
nvcc --version
nvidia-smi

# PyTorch 재설치 (CUDA 12.1)
pip uninstall torch torchvision torchaudio
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

#### 2. NVIDIA 드라이버 충돌
```bash
# 기존 드라이버 제거
sudo apt purge nvidia*
sudo apt autoremove

# 재설치
sudo ubuntu-drivers autoinstall
sudo reboot
```

#### 3. CUDA 라이브러리 경로 문제
```bash
# 라이브러리 경로 확인 및 추가
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# .bashrc에 영구 추가
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

## 📊 성능 최적화

### DataLoader 최적화
- CPU 전용: `num_workers=4-8` (코어 수에 따라)
- GPU 사용: `num_workers=8-16`, `pin_memory=True`

### 메모리 최적화
```bash
# 메모리 사용량 모니터링
watch -n 1 'free -h && nvidia-smi'

# 배치 크기 조정 (메모리 부족 시)
# codes/config.py에서 batch_size 감소: 32 → 16 → 8
```

## 🎯 Ubuntu 전용 최적화

### 1. CPU 성능 최적화
```bash
# CPU 거버너 설정 (성능 모드)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. 디스크 I/O 최적화
```bash
# SSD인 경우 TRIM 활성화
sudo systemctl enable fstrim.timer
sudo systemctl start fstrim.timer
```

### 3. 네트워크 최적화 (WandB 사용 시)
```bash
# DNS 성능 개선
echo 'nameserver 8.8.8.8' | sudo tee -a /etc/resolv.conf
echo 'nameserver 8.8.4.4' | sudo tee -a /etc/resolv.conf
```

## 📝 환경 설정

### 환경변수 설정
```bash
# .bashrc에 추가 (선택사항)
echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc  # 특정 GPU 사용
source ~/.bashrc
```

### 자동 시작 스크립트
```bash
#!/bin/bash
# start_training.sh
cd /path/to/cv-classify
source .venv/bin/activate
cd codes
python3 train_with_wandb.py
```

### 백그라운드 실행
```bash
# Screen 사용 (권장)
screen -S cv_training
source .venv/bin/activate
cd codes
python3 train_with_wandb.py
# Ctrl+A, D로 detach

# 다시 접속
screen -r cv_training

# Tmux 사용
tmux new-session -d -s cv_training
tmux send-keys -t cv_training 'source .venv/bin/activate' Enter
tmux send-keys -t cv_training 'cd codes && python3 train_with_wandb.py' Enter
```

## 🔍 모니터링 및 로그

```bash
# 실시간 로그 확인
tail -f logs/training_*.log

# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 시스템 리소스 모니터링
htop

# 디스크 사용량 확인
df -h

# 프로세스 확인
ps aux | grep python
```

## 🚀 빠른 시작 요약

Ubuntu에서 가장 빠르게 시작하는 방법:

```bash
# 1. 기본 패키지 설치
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git screen

# 2. 프로젝트 클론 및 설정
git clone <your-repository-url>
cd cv-classify
chmod +x setup.sh menu.sh

# 3. 자동 설정 실행
./setup.sh

# 4. 메뉴 실행
./menu.sh
```

이 가이드를 따라하면 Ubuntu 환경에서 CV-Classify 프로젝트를 성공적으로 실행할 수 있습니다. 문제가 발생하면 위의 문제 해결 섹션을 참고하세요.
