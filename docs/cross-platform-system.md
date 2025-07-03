# CV-Classify 크로스 플랫폼 시스템 가이드

## 📋 개요

CV-Classify는 다양한 운영체제와 하드웨어 환경에서 일관된 성능을 제공하도록 설계된 크로스 플랫폼 시스템입니다. 이 문서는 각 플랫폼별 특징, 최적화 전략, 설치 및 설정 방법을 상세히 설명합니다.

## 🖥️ 지원 플랫폼

### 완전 지원 플랫폼

| 플랫폼 | 버전 | GPU 지원 | 패키지 관리자 | 상태 |
|--------|------|----------|---------------|------|
| **macOS** | 10.15+ | Apple Silicon MPS | Homebrew | ✅ 완전 지원 |
| **Ubuntu** | 20.04/22.04 LTS | NVIDIA CUDA | APT | ✅ 완전 지원 |
| **CentOS** | 7/8 | NVIDIA CUDA | YUM/DNF | ✅ 완전 지원 |
| **Windows WSL2** | Ubuntu 20.04+ | NVIDIA CUDA | APT | ✅ 완전 지원 |

### 부분 지원 플랫폼

| 플랫폼 | 제한사항 | 권장 대안 |
|--------|----------|-----------|
| **Windows 네이티브** | 셸 스크립트 미지원 | WSL2 사용 권장 |
| **기타 Linux 배포판** | 패키지 관리자 차이 | 수동 설치 필요 |

---

## 🔧 플랫폼별 아키텍처

### 1. macOS (Apple Silicon + Intel)

#### 시스템 아키텍처

```
macOS 환경
├── Homebrew 패키지 관리
├── Python 3.7+ (Homebrew)
├── Apple Silicon GPU (MPS)
│   ├── 통합 메모리 시스템
│   ├── Metal Performance Shaders
│   └── 자동 메모리 관리
└── Intel x86_64 호환성
```

#### 하드웨어 최적화

**Apple Silicon (M1/M2/M3/M4)**:
```python
# MPS 최적화 설정
device = torch.device('mps')
dataloader_config = {
    "pin_memory": False,    # 통합 메모리로 불필요
    "num_workers": 0,       # 멀티프로세싱 이슈 방지
    "persistent_workers": False
}

# 메모리 최적화
torch.mps.empty_cache()  # 메모리 정리
```

**Intel Mac**:
```python
# CPU 최적화 설정  
device = torch.device('cpu')
torch.set_num_threads(8)  # CPU 코어 활용
dataloader_config = {
    "pin_memory": False,
    "num_workers": 4
}
```

#### 특별 고려사항

**장점**:
- 통합 메모리로 대용량 모델 처리 가능
- 저전력 고성능
- 안정적인 개발 환경

**제한사항**:
- MPS는 일부 PyTorch 연산 미지원
- 멀티프로세싱 DataLoader 이슈
- CUDA 전용 라이브러리 호환성 문제

### 2. Ubuntu Linux

#### 시스템 아키텍처

```
Ubuntu 환경
├── APT 패키지 관리
├── Python 3.7+ (시스템/PPA)
├── NVIDIA GPU (CUDA)
│   ├── CUDA Toolkit
│   ├── cuDNN 라이브러리
│   └── GPU 메모리 관리
└── 고성능 컴퓨팅 최적화
```

#### 하드웨어 최적화

**NVIDIA GPU**:
```python
# CUDA 최적화 설정
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

dataloader_config = {
    "pin_memory": True,     # GPU 메모리 전송 최적화
    "num_workers": 8,       # 병렬 데이터 로딩
    "persistent_workers": True,
    "prefetch_factor": 2
}

# 혼합 정밀도 학습
scaler = torch.cuda.amp.GradScaler()
```

**CPU 전용**:
```python
# CPU 최적화 설정
device = torch.device('cpu')
torch.set_num_threads(min(os.cpu_count(), 16))

dataloader_config = {
    "pin_memory": False,
    "num_workers": min(os.cpu_count() // 2, 8)
}
```

#### 특별 고려사항

**장점**:
- 최고 성능의 CUDA 지원
- 대부분의 딥러닝 라이브러리 완전 호환
- 서버 환경 최적화

**제한사항**:
- NVIDIA 드라이버 의존성
- CUDA 설치 복잡성

### 3. CentOS/RHEL

#### 시스템 아키텍처

```
CentOS 환경
├── YUM/DNF 패키지 관리
├── Python 3.7+ (EPEL/SCL)
├── NVIDIA GPU (CUDA)
│   ├── 엔터프라이즈 지원
│   └── 장기 안정성
└── 서버 환경 최적화
```

#### 특별 설정

**Python 설치**:
```bash
# CentOS 7
sudo yum install epel-release
sudo yum install python3 python3-pip

# CentOS 8
sudo dnf install python3 python3-pip
```

**CUDA 설치**:
```bash
# NVIDIA 리포지토리 추가
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

sudo dnf install cuda
```

### 4. Windows WSL2

#### 시스템 아키텍처

```
Windows WSL2 환경
├── Ubuntu 20.04+ 서브시스템
├── Windows GPU 공유
├── NVIDIA GPU (CUDA on WSL)
│   ├── Windows NVIDIA 드라이버
│   ├── WSL2 CUDA 지원
│   └── 하이브리드 메모리 관리
└── 파일 시스템 연동
```

#### 설정 요구사항

**WSL2 CUDA 설정**:
1. Windows NVIDIA 드라이버 (471.41+)
2. WSL2 커널 업데이트
3. Ubuntu WSL에서 CUDA Toolkit 설치

```bash
# WSL2에서 CUDA 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda
```

---

## ⚙️ 자동 플랫폼 감지 시스템

### platform_utils.sh의 감지 로직

```bash
detect_platform() {
    case "$(uname -s)" in
        Darwin)
            echo "macos"
            ;;
        Linux)
            if [ -f /etc/lsb-release ] || [ -f /etc/debian_version ]; then
                echo "ubuntu"
            elif [ -f /etc/redhat-release ] || [ -f /etc/centos-release ]; then
                echo "centos"
            else
                echo "linux"
            fi
            ;;
        CYGWIN*|MINGW*|MSYS*)
            echo "windows"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}
```

### device_utils.py의 디바이스 감지

```python
def get_optimal_device():
    """크로스 플랫폼 최적화된 디바이스 감지"""
    
    # 1. CUDA 확인 (최우선)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        return device, "CUDA"
    
    # 2. MPS 확인 (macOS Apple Silicon)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        return device, "MPS"
    
    # 3. CPU fallback
    else:
        device = torch.device('cpu')
        return device, "CPU"
```

---

## 📦 플랫폼별 설치 가이드

### 1. macOS 설치

#### 자동 설치 (권장)

```bash
# 리포지토리 클론
git clone <repository-url>
cd cv-classify

# 자동 설정 실행
chmod +x setup_macos.sh
./setup_macos.sh

# 또는 통합 설정
chmod +x setup.sh
./setup.sh
```

#### 수동 설치

**1단계: Homebrew 설치**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2단계: 필수 패키지 설치**
```bash
brew install python3 git screen tmux
```

**3단계: Python 의존성 설치**
```bash
pip3 install -r requirements-macos.txt
```

### 2. Ubuntu 설치

#### 자동 설치 (권장)

```bash
# 리포지토리 클론
git clone <repository-url>
cd cv-classify

# 자동 설정 실행
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh

# 또는 통합 설정
chmod +x setup.sh
./setup.sh
```

#### 수동 설치

**1단계: 시스템 패키지 업데이트**
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

**2단계: 필수 패키지 설치**
```bash
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    git curl wget screen tmux \
    build-essential
```

**3단계: NVIDIA GPU 설정 (선택사항)**
```bash
# NVIDIA 드라이버 설치
sudo apt-get install nvidia-driver-470

# CUDA 설치
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**4단계: Python 의존성 설치**
```bash
# GPU 버전
pip3 install -r requirements-ubuntu-gpu.txt

# CPU 버전
pip3 install -r requirements-ubuntu-cpu.txt
```

### 3. CentOS 설치

#### 자동 설치

```bash
# 리포지토리 클론
git clone <repository-url>
cd cv-classify

# 자동 설정 실행 (CentOS는 통합 스크립트 사용)
chmod +x setup.sh
./setup.sh
```

#### 수동 설치

**CentOS 7**:
```bash
# EPEL 저장소 추가
sudo yum install epel-release

# 필수 패키지 설치
sudo yum install python3 python3-pip git screen tmux

# Python 의존성 설치
pip3 install -r requirements.txt
```

**CentOS 8**:
```bash
# 필수 패키지 설치
sudo dnf install python3 python3-pip git screen tmux

# Python 의존성 설치
pip3 install -r requirements.txt
```

### 4. Windows WSL2 설치

#### 1단계: WSL2 설정

```powershell
# Windows PowerShell (관리자 권한)
wsl --install -d Ubuntu-20.04
```

#### 2단계: Ubuntu 설정

WSL2 Ubuntu 터미널에서:
```bash
# 시스템 업데이트
sudo apt-get update && sudo apt-get upgrade -y

# CV-Classify 설치 (Ubuntu와 동일)
git clone <repository-url>
cd cv-classify
chmod +x setup.sh
./setup.sh
```

#### 3단계: CUDA on WSL 설정 (선택사항)

```bash
# WSL2 CUDA 설치
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda
```

---

## 🔄 플랫폼 간 호환성 관리

### 파일 경로 처리

```python
# config.py에서 크로스 플랫폼 경로 처리
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# 자동으로 플랫폼별 구분자 사용 (/ vs \)
```

### 스크립트 호환성

```bash
# menu.sh에서 크로스 플랫폼 명령어 처리
clear_screen() {
    if command -v clear &> /dev/null; then
        clear
    else
        printf '\033[2J\033[H'  # ANSI 이스케이프 시퀀스
    fi
}
```

### Python 명령어 통일

```bash
# platform_utils.sh에서 Python 명령어 자동 감지
detect_python() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        echo "python3"  # 기본값
    fi
}
```

---

## 🚀 성능 최적화 전략

### 플랫폼별 최적화 매트릭스

| 플랫폼 | 디바이스 | pin_memory | num_workers | 특별 설정 |
|--------|----------|------------|-------------|-----------|
| **macOS Apple Silicon** | MPS | False | 0 | 통합 메모리 |
| **macOS Intel** | CPU | False | 4 | 멀티스레딩 |
| **Ubuntu CUDA** | CUDA | True | 8 | cuDNN 벤치마킹 |
| **Ubuntu CPU** | CPU | False | 4-8 | OpenMP 최적화 |
| **WSL2 CUDA** | CUDA | True | 6 | 하이브리드 메모리 |

### 자동 성능 조정

```python
def get_optimal_batch_size(device_type, model_name):
    """플랫폼별 최적 배치 크기 자동 결정"""
    
    base_batch_sizes = {
        "MPS": 32,      # Apple Silicon 통합 메모리
        "CUDA": 64,     # NVIDIA GPU 전용 메모리
        "CPU": 16       # CPU 제한된 메모리
    }
    
    # 모델별 조정
    model_multipliers = {
        "resnet34": 1.0,
        "resnet50": 0.8,
        "efficientnet_b0": 1.2,
        "efficientnet_b4": 0.6
    }
    
    base_size = base_batch_sizes.get(device_type, 16)
    multiplier = model_multipliers.get(model_name, 1.0)
    
    return int(base_size * multiplier)
```

### 메모리 사용량 모니터링

```python
def monitor_memory_usage(device):
    """플랫폼별 메모리 사용량 모니터링"""
    
    if device.type == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        cached = torch.cuda.memory_reserved(0)
        
        return {
            'total': total / 1e9,
            'allocated': allocated / 1e9,
            'cached': cached / 1e9,
            'free': (total - allocated) / 1e9
        }
    
    elif device.type == 'mps':
        # Apple Silicon 통합 메모리
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total': mem.total / 1e9,
            'available': mem.available / 1e9,
            'used': mem.used / 1e9,
            'percent': mem.percent
        }
    
    else:  # CPU
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total': mem.total / 1e9,
            'available': mem.available / 1e9,
            'used': mem.used / 1e9,
            'percent': mem.percent
        }
```

---

## 🛠️ 트러블슈팅 가이드

### 공통 문제 해결

#### 1. Python 명령어 인식 오류

**증상**:
```
python: command not found
python3: command not found
```

**해결책**:
```bash
# macOS
brew install python3

# Ubuntu
sudo apt-get install python3 python3-pip

# CentOS
sudo yum install python3 python3-pip
```

#### 2. 권한 오류

**증상**:
```
Permission denied: ./setup.sh
```

**해결책**:
```bash
chmod +x setup.sh
chmod +x menu.sh
chmod +x scripts/*.sh
```

#### 3. 패키지 관리자 오류

**증상**:
```
brew: command not found (macOS)
apt-get: command not found (Ubuntu)
```

**해결책**:
```bash
# macOS - Homebrew 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Ubuntu - APT 업데이트
sudo apt-get update

# CentOS - EPEL 저장소 추가
sudo yum install epel-release
```

### 플랫폼별 특수 문제

#### macOS 특수 문제

**1. MPS 디바이스 오류**

**증상**:
```
RuntimeError: MPS backend out of memory
```

**해결책**:
```python
# 메모리 정리
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# 배치 크기 줄이기
BATCH_SIZE = 16  # 기본 32에서 16으로 감소
```

**2. Rosetta 호환성 문제**

**증상**:
```
ImportError: cannot import name '_C' from 'torch'
```

**해결책**:
```bash
# Apple Silicon 네이티브 Python 사용
arch -arm64 brew install python3
arch -arm64 pip3 install torch torchvision
```

**3. Xcode Command Line Tools 누락**

**증상**:
```
xcrun: error: invalid active developer path
```

**해결책**:
```bash
xcode-select --install
```

#### Ubuntu 특수 문제

**1. CUDA 버전 불일치**

**증상**:
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**해결책**:
```bash
# CUDA 버전 확인
nvcc --version
nvidia-smi

# PyTorch 재설치 (CUDA 버전에 맞게)
pip3 uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**2. NVIDIA 드라이버 충돌**

**증상**:
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

**해결책**:
```bash
# 드라이버 재설치
sudo apt-get purge nvidia-*
sudo apt-get autoremove
sudo apt-get install nvidia-driver-470
sudo reboot
```

**3. CUDNN 라이브러리 누락**

**증상**:
```
UserWarning: cuDNN is not available
```

**해결책**:
```bash
# cuDNN 설치
sudo apt-get install libcudnn8 libcudnn8-dev
```

#### Windows WSL2 특수 문제

**1. WSL2 CUDA 지원 문제**

**증상**:
```
torch.cuda.is_available() returns False
```

**해결책**:
```bash
# WSL2 CUDA 지원 확인
ls /usr/lib/wsl/lib/

# CUDA 환경 변수 설정
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**2. 파일 시스템 권한 문제**

**증상**:
```
PermissionError: [Errno 13] Permission denied
```

**해결책**:
```bash
# WSL2에서 Windows 파일 시스템 사용 시
# Linux 파일 시스템으로 프로젝트 이동
cp -r /mnt/c/cv-classify ~/cv-classify
cd ~/cv-classify
```

#### CentOS 특수 문제

**1. Python 3.7+ 설치 문제**

**증상**:
```
No package python3.8 available
```

**해결책**:
```bash
# CentOS 7 - Software Collections 사용
sudo yum install centos-release-scl
sudo yum install rh-python38
scl enable rh-python38 bash

# CentOS 8 - AppStream 사용
sudo dnf install python38
```

**2. 컴파일러 도구 누락**

**증상**:
```
error: Microsoft Visual C++ 14.0 is required
```

**해결책**:
```bash
# 개발 도구 설치
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

---

## 📊 성능 벤치마크

### 플랫폼별 성능 비교

#### 베이스라인 테스트 (ResNet34, 1 epoch)

| 플랫폼 | 하드웨어 | 배치 크기 | 시간 (초) | 메모리 (GB) | 성능 점수 |
|--------|----------|-----------|-----------|------------|-----------|
| **macOS M1 Pro** | 8코어 CPU + 14코어 GPU | 32 | 45 | 8.5 | ⭐⭐⭐⭐⭐ |
| **Ubuntu RTX 3080** | CUDA | 64 | 25 | 6.2 | ⭐⭐⭐⭐⭐ |
| **Ubuntu RTX 2060** | CUDA | 32 | 35 | 4.1 | ⭐⭐⭐⭐ |
| **macOS Intel i7** | 8코어 CPU | 16 | 180 | 12.3 | ⭐⭐⭐ |
| **Ubuntu CPU (16코어)** | CPU 전용 | 16 | 220 | 8.7 | ⭐⭐⭐ |
| **WSL2 RTX 3070** | CUDA on WSL | 32 | 30 | 5.8 | ⭐⭐⭐⭐ |

#### 고급 실험 테스트 (ResNet34, 50 epochs)

| 플랫폼 | 예상 시간 | 최적 배치 크기 | 권장 설정 |
|--------|-----------|----------------|-----------|
| **Apple Silicon** | 45분 | 32 | MPS + 통합 메모리 |
| **NVIDIA RTX 30xx** | 25분 | 64 | CUDA + 혼합 정밀도 |
| **NVIDIA RTX 20xx** | 35분 | 32 | CUDA + 표준 정밀도 |
| **고성능 CPU** | 3-4시간 | 16 | 멀티스레딩 + 캐싱 |

### 자동 성능 테스트

```python
def run_performance_test():
    """플랫폼별 성능 테스트 실행"""
    
    from device_utils import setup_training_device
    import time
    import torch
    
    device, device_type = setup_training_device()
    
    # 테스트 설정
    batch_size = 32
    img_size = 224
    num_classes = 17
    
    # 모델 생성
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, num_classes)
    ).to(device)
    
    # 더미 데이터
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    dummy_target = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # 성능 측정
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # 워밍업
    for _ in range(5):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # 실제 측정
    start_time = time.time()
    for _ in range(100):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    end_time = time.time()
    
    # 결과 출력
    total_time = end_time - start_time
    samples_per_second = (100 * batch_size) / total_time
    
    print(f"플랫폼: {device_type}")
    print(f"디바이스: {device}")
    print(f"총 시간: {total_time:.2f}초")
    print(f"처리량: {samples_per_second:.1f} samples/second")
    
    return {
        'device_type': device_type,
        'total_time': total_time,
        'throughput': samples_per_second
    }
```

---

## 🔄 배포 및 이식성

### 환경 독립적 실행

#### Docker 지원 (계획 중)

```dockerfile
# Dockerfile.cuda (NVIDIA GPU 지원)
FROM nvidia/cuda:11.8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git

COPY requirements-ubuntu-gpu.txt .
RUN pip3 install -r requirements-ubuntu-gpu.txt

COPY . /cv-classify
WORKDIR /cv-classify

CMD ["python3", "codes/train_with_wandb.py"]
```

```dockerfile
# Dockerfile.cpu (CPU 전용)
FROM python:3.9-slim

RUN apt-get update && apt-get install -y git

COPY requirements-ubuntu-cpu.txt .
RUN pip install -r requirements-ubuntu-cpu.txt

COPY . /cv-classify
WORKDIR /cv-classify

CMD ["python3", "codes/train_with_wandb.py"]
```

#### 가상 환경 관리

```bash
# Python venv 사용
python3 -m venv cv-classify-env
source cv-classify-env/bin/activate  # Linux/macOS
# cv-classify-env\Scripts\activate  # Windows

pip install -r requirements.txt
```

```bash
# Conda 환경 사용
conda create -n cv-classify python=3.9
conda activate cv-classify
pip install -r requirements.txt
```

### 설정 포터빌리티

#### 환경 변수 통합 관리

```bash
# .env.template 파일
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=cv-classification
WANDB_ENTITY=your_wandb_entity
WANDB_MODE=online

# 플랫폼별 오버라이드
MACOS_OPTIMIZATION=true
CUDA_OPTIMIZATION=true
CPU_THREADS=auto
```

#### 플랫폼별 설정 프로파일

```python
# config.py에서 플랫폼별 설정 프로파일
PLATFORM_PROFILES = {
    "macos_m1": {
        "device_preference": ["mps", "cpu"],
        "dataloader_workers": 0,
        "pin_memory": False,
        "batch_size_multiplier": 1.0
    },
    "ubuntu_cuda": {
        "device_preference": ["cuda", "cpu"],
        "dataloader_workers": 8,
        "pin_memory": True,
        "batch_size_multiplier": 1.5
    },
    "cpu_only": {
        "device_preference": ["cpu"],
        "dataloader_workers": 4,
        "pin_memory": False,
        "batch_size_multiplier": 0.5
    }
}
```

---

## 📈 모니터링 및 디버깅

### 플랫폼별 시스템 모니터링

```python
def get_system_metrics():
    """플랫폼별 시스템 메트릭 수집"""
    
    import psutil
    import platform
    
    metrics = {
        'platform': platform.system(),
        'architecture': platform.machine(),
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory': {
            'total': psutil.virtual_memory().total / 1e9,
            'available': psutil.virtual_memory().available / 1e9,
            'percent': psutil.virtual_memory().percent
        }
    }
    
    # GPU 정보 추가
    if torch.cuda.is_available():
        metrics['gpu'] = {
            'name': torch.cuda.get_device_name(0),
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'memory_allocated': torch.cuda.memory_allocated(0) / 1e9
        }
    elif torch.backends.mps.is_available():
        metrics['gpu'] = {
            'name': 'Apple Silicon',
            'type': 'MPS'
        }
    
    return metrics
```

### 로그 통합 관리

```python
def setup_cross_platform_logging():
    """크로스 플랫폼 로깅 설정"""
    
    import logging
    import os
    from datetime import datetime
    
    # 플랫폼별 로그 디렉토리
    if os.name == 'nt':  # Windows
        log_dir = os.path.expanduser('~\\AppData\\Local\\cv-classify\\logs')
    else:  # Unix-like
        log_dir = os.path.expanduser('~/.cv-classify/logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 타임스탬프 포함 로그 파일
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cv_classify_{timestamp}.log")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
```

---

## 🎯 최적화 권장사항

### 플랫폼별 베스트 프랙티스

#### macOS 최적화

```python
# macOS 특화 설정
if platform.system() == 'Darwin':
    # MPS 사용 시 배치 크기 조정
    if torch.backends.mps.is_available():
        BATCH_SIZE = min(BATCH_SIZE, 32)
        NUM_WORKERS = 0  # 멀티프로세싱 문제 방지
    
    # 메모리 관리
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
```

#### Ubuntu 최적화

```python
# Ubuntu 특화 설정
if platform.system() == 'Linux':
    # CUDA 사용 시 성능 최적화
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 혼합 정밀도 학습
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        
        # 배치 크기 증가
        BATCH_SIZE = min(BATCH_SIZE * 2, 128)
```

#### 범용 CPU 최적화

```python
# CPU 전용 최적화
if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
    import multiprocessing
    
    # CPU 스레드 수 최적화
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(min(num_cores, 16))
    
    # 배치 크기 조정
    BATCH_SIZE = min(BATCH_SIZE, 16)
    
    # 데이터 로딩 최적화
    NUM_WORKERS = min(num_cores // 2, 8)
```

### 메모리 효율성

```python
def optimize_memory_usage():
    """메모리 사용량 최적화"""
    
    # 그래디언트 체크포인팅
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # 혼합 정밀도 학습 (CUDA)
    if torch.cuda.is_available():
        model = model.half()  # FP16 사용
    
    # 메모리 정리
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
```

---

## 🔚 결론

CV-Classify의 크로스 플랫폼 시스템은 다양한 하드웨어와 운영체제에서 일관된 성능과 사용자 경험을 제공합니다. 

**주요 장점**:
- **자동 최적화**: 플랫폼별 하드웨어에 맞는 자동 설정
- **일관된 인터페이스**: 모든 플랫폼에서 동일한 사용법
- **성능 최적화**: 각 플랫폼의 특성을 살린 최적화
- **문제 해결**: 플랫폼별 트러블슈팅 가이드 제공

**지원 우선순위**:
1. **Tier 1**: macOS Apple Silicon, Ubuntu CUDA
2. **Tier 2**: macOS Intel, Windows WSL2
3. **Tier 3**: CentOS, 기타 Linux 배포판

이 가이드를 통해 어떤 환경에서든 CV-Classify를 효과적으로 활용할 수 있습니다.