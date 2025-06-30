#!/bin/bash

# 크로스 플랫폼 Python 환경 설정 스크립트
set -e

echo "🚀 Python 환경 자동 설정 시작..."

# 플랫폼 감지
detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

# GPU 지원 확인 (Linux만)
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "gpu"
        else
            echo "cpu"
        fi
    else
        echo "cpu"
    fi
}

# Python 명령어 찾기
find_python() {
    local platform=$1
    
    if [[ "$platform" == "macos" ]]; then
        # macOS에서 Python 3.11 찾기
        if [[ -f "/opt/homebrew/bin/python3.11" ]]; then
            echo "/opt/homebrew/bin/python3.11"
        elif [[ -f "/usr/local/bin/python3.11" ]]; then
            echo "/usr/local/bin/python3.11"
        elif command -v python3.11 &> /dev/null; then
            echo "python3.11"
        elif command -v python3 &> /dev/null; then
            echo "python3"
        else
            echo ""
        fi
    else
        # Linux에서 Python 찾기
        if command -v python3.11 &> /dev/null; then
            echo "python3.11"
        elif command -v python3.10 &> /dev/null; then
            echo "python3.10"
        elif command -v python3 &> /dev/null; then
            echo "python3"
        else
            echo ""
        fi
    fi
}

# 환경 정보 출력
PLATFORM=$(detect_platform)
echo "🖥️  플랫폼: $PLATFORM"

if [[ "$PLATFORM" == "linux" ]]; then
    GPU_SUPPORT=$(detect_gpu)
    echo "🎮 GPU 지원: $GPU_SUPPORT"
fi

# Python 찾기
PYTHON_CMD=$(find_python $PLATFORM)
if [[ -z "$PYTHON_CMD" ]]; then
    echo "❌ Python을 찾을 수 없습니다."
    echo "Python 3.10+ 설치가 필요합니다."
    exit 1
fi

echo "🐍 Python: $($PYTHON_CMD --version) ($PYTHON_CMD)"

# Requirements 파일 선택
if [[ "$PLATFORM" == "macos" ]]; then
    REQUIREMENTS_FILE="requirements-macos.txt"
elif [[ "$PLATFORM" == "linux" && "$GPU_SUPPORT" == "gpu" ]]; then
    REQUIREMENTS_FILE="requirements-ubuntu-gpu.txt"
elif [[ "$PLATFORM" == "linux" && "$GPU_SUPPORT" == "cpu" ]]; then
    REQUIREMENTS_FILE="requirements-ubuntu-cpu.txt"
else
    REQUIREMENTS_FILE="requirements.txt"
fi

echo "📋 사용할 Requirements: $REQUIREMENTS_FILE"

# Requirements 파일 확인
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "❌ $REQUIREMENTS_FILE 파일을 찾을 수 없습니다."
    echo "기본 requirements.txt를 사용합니다."
    REQUIREMENTS_FILE="requirements.txt"
fi

# 기존 가상환경 제거
echo "🗑️  기존 가상환경 제거..."
if [[ -d ".venv" ]]; then
    rm -rf .venv
fi

# 새 가상환경 생성
echo "📦 새 가상환경 생성..."
$PYTHON_CMD -m venv .venv

# 가상환경 활성화
echo "🔄 가상환경 활성화..."
source .venv/bin/activate

# pip 업그레이드
echo "⬆️  pip 업그레이드..."
pip install --upgrade pip

# 패키지 설치
echo "📚 패키지 설치 중 ($REQUIREMENTS_FILE)..."
pip install -r $REQUIREMENTS_FILE

# 설치 확인
echo "✅ 설치 확인..."
python -c "
import sys
import torch
import torchvision
import timm
import numpy
import pandas

print('🎉 모든 패키지가 성공적으로 설치되었습니다!')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'TIMM: {timm.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'Pandas: {pandas.__version__}')

# 가속 장치 확인
if torch.cuda.is_available():
    print(f'🚀 CUDA GPU 가속 사용 가능! (디바이스: {torch.cuda.get_device_name()})')
    print(f'   CUDA 버전: {torch.version.cuda}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('🚀 MPS (Apple Silicon GPU) 가속 사용 가능!')
else:
    print('💻 CPU 모드로 실행됩니다.')
"

echo ""
echo "✅ 환경 설정 완료!"
echo ""
echo "📝 다음 명령어로 가상환경을 활성화하세요:"
echo "   source .venv/bin/activate"
echo ""
echo "🚀 그 다음 메뉴를 실행하세요:"
echo "   ./menu.sh"
echo ""
echo "📋 사용된 설정:"
echo "   플랫폼: $PLATFORM"
if [[ "$PLATFORM" == "linux" ]]; then
    echo "   GPU 지원: $GPU_SUPPORT"
fi
echo "   Requirements: $REQUIREMENTS_FILE"
echo "   Python: $PYTHON_CMD"
