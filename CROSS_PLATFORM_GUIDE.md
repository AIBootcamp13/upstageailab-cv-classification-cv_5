# 🌍 크로스 플랫폼 지원 가이드

CV-Classify 프로젝트는 **macOS**와 **Ubuntu** 환경을 모두 지원합니다.

## 🍎 macOS 환경

### 지원 버전
- **macOS 12.0+** (Apple Silicon M1/M2/M3 권장)
- **Intel Mac** 지원 (CUDA 불가, CPU 모드)

### 자동 최적화
- **MPS GPU 가속**: Apple Silicon에서 자동 활성화
- **통합 메모리**: 68GB+ 대용량 메모리 활용
- **Homebrew 통합**: 자동 패키지 관리

### 실행 방법
```bash
# 환경 설정 (최초 1회)
./setup.sh

# OCR 환경 설정 (선택사항)
./setup_ocr.sh

# 프로젝트 실행
./menu.sh
```

## 🐧 Ubuntu 환경

### 지원 버전
- **Ubuntu 20.04 LTS** ✅ 완전 지원
- **Ubuntu 22.04 LTS** ✅ 완전 지원  
- **Ubuntu 24.04 LTS** ✅ 완전 지원

### GPU 지원
- **NVIDIA GPU + CUDA**: 자동 GPU 가속
- **CPU 전용**: CPU 최적화 모드

### 실행 방법
```bash
# 전체 환경 설정 (최초 1회) - 권장
./setup_ubuntu.sh

# 또는 수동 설정
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh

# 프로젝트 실행
./run_ubuntu.sh

# 또는 수동 실행
source .venv/bin/activate
./menu.sh
```

## 📊 플랫폼별 성능 비교

| 항목 | macOS (M3 Max) | Ubuntu (NVIDIA RTX) | Ubuntu (CPU) |
|------|----------------|---------------------|--------------|
| **GPU 가속** | MPS | CUDA | 없음 |
| **메모리** | 통합 메모리 68GB | VRAM + RAM | RAM만 |
| **성능** | 🚀 매우 빠름 | 🚀 매우 빠름 | ⚡ 보통 |
| **전력 효율** | 🍃 매우 좋음 | ⚡ 보통 | 🍃 좋음 |

## 🔧 주요 차이점

### macOS 특화 기능
- **MPS 최적화**: `pin_memory=False`, `num_workers=0`
- **Homebrew 통합**: 자동 패키지 설치
- **Apple Silicon 최적화**: 네이티브 성능

### Ubuntu 특화 기능  
- **CUDA 지원**: NVIDIA GPU 가속
- **APT 패키지 관리**: 시스템 레벨 패키지 설치
- **멀티 GPU 지원**: 여러 GPU 활용 가능

## 🐳 Docker 지원 (개발 중)

Ubuntu 환경에서 Docker를 사용한 실행도 지원 예정:

```bash
# Docker 이미지 빌드
docker build -t cv-classify .

# GPU 지원 실행
docker run --gpus all -it cv-classify

# CPU 전용 실행  
docker run -it cv-classify
```

## 📋 환경별 Requirements

### macOS
```bash
# 기본 (MPS 지원)
pip install -r requirements.txt

# 또는 macOS 전용
pip install -r requirements-macos.txt
```

### Ubuntu GPU
```bash
# CUDA 12.1 지원
pip install -r requirements-ubuntu-gpu.txt
```

### Ubuntu CPU
```bash
# CPU 전용 최적화
pip install -r requirements-ubuntu-cpu.txt
```

## 🔍 환경별 체크리스트

### macOS 체크리스트
- [ ] macOS 12.0+ 확인
- [ ] Homebrew 설치
- [ ] Python 3.8+ 설치
- [ ] Xcode Command Line Tools
- [ ] MPS 지원 확인

### Ubuntu 체크리스트
- [ ] Ubuntu 20.04+ 확인
- [ ] Python 3.8+ 설치
- [ ] NVIDIA 드라이버 (GPU 사용 시)
- [ ] CUDA Toolkit (GPU 사용 시)
- [ ] 개발 도구 패키지

## 🚀 빠른 시작 가이드

### macOS 사용자
```bash
# 1. 프로젝트 클론
git clone <repository-url>
cd cv-classify

# 2. 환경 설정
./setup.sh

# 3. 실행
./menu.sh
# 3번 선택 → Y (OCR 사용)
```

### Ubuntu 사용자
```bash
# 1. 프로젝트 클론  
git clone <repository-url>
cd cv-classify

# 2. 환경 설정
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh

# 3. 실행
./run_ubuntu.sh
# 3번 선택 → Y (OCR 사용)
```

## 🐛 플랫폼별 문제해결

### macOS 문제해결
- **MPS 오류**: macOS 12.0+ 및 Apple Silicon 필요
- **Homebrew 없음**: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- **권한 오류**: `chmod +x *.sh`

### Ubuntu 문제해결
- **CUDA 오류**: NVIDIA 드라이버 및 CUDA 재설치
- **패키지 오류**: `sudo apt update && sudo apt upgrade`
- **권한 오류**: `sudo` 없이 실행, 필요시에만 sudo 사용

## 📞 지원

각 플랫폼별 상세한 설치 가이드와 문제해결은:
- **macOS**: `docs/MACOS_SETUP.md`
- **Ubuntu**: `docs/UBUNTU_SETUP.md`
- **일반**: GitHub Issues 탭

---

**참고**: 두 플랫폼 모두에서 동일한 F1 스코어 0.9547+ 성능을 보장합니다! 🎯
