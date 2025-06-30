#!/bin/bash

# macOS Python 환경 설정 스크립트
set -e

echo "🚀 macOS Python 환경 설정 시작..."

# 현재 디렉토리 확인
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt 파일을 찾을 수 없습니다."
    exit 1
fi

# 기존 가상환경 제거
echo "🗑️  기존 가상환경 제거..."
if [ -d ".venv" ]; then
    rm -rf .venv
fi

# Python 3.11 확인
PYTHON_CMD="/opt/homebrew/bin/python3.11"
if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ Python 3.11이 설치되지 않았습니다."
    echo "다음 명령어로 설치하세요: brew install python@3.11"
    exit 1
fi

echo "🐍 Python 버전: $($PYTHON_CMD --version)"

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
echo "📚 패키지 설치 중..."
pip install -r requirements.txt

# 설치 확인
echo "✅ 설치 확인..."
python -c "
import torch
import torchvision
import timm
import numpy
import pandas
import cv2
import albumentations
print('🎉 모든 패키지가 성공적으로 설치되었습니다!')
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'TIMM: {timm.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'Pandas: {pandas.__version__}')

# MPS 지원 확인 (Apple Silicon)
if torch.backends.mps.is_available():
    print('🚀 MPS (Apple Silicon GPU) 가속 사용 가능!')
else:
    print('💻 CPU 모드로 실행됩니다.')
"

echo ""
echo "✅ 환경 설정 완료!"
echo "다음 명령어로 가상환경을 활성화하세요:"
echo "source .venv/bin/activate"
echo ""
echo "그 다음 메뉴를 실행하세요:"
echo "./menu.sh"
