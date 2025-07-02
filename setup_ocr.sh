#!/bin/bash

# OCR 설정 스크립트 for cv-classify
echo "🔧 OCR 환경 설정 시작..."

# 현재 디렉토리 확인
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Python 환경 활성화 (가상환경 사용 중이라면)
if [ -d ".venv" ]; then
    echo "📦 가상환경 활성화 중..."
    source .venv/bin/activate
fi

# OCR 라이브러리 설치
echo "📥 OCR 라이브러리 설치 중..."

# EasyOCR (추천) - GPU 가속, 한국어 지원
echo "1. EasyOCR 설치 중..."
pip install easyocr

# Tesseract (대안) - 더 빠름
echo "2. Tesseract 설치 중..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        brew install tesseract tesseract-lang
    else
        echo "⚠️  Homebrew가 설치되지 않았습니다. EasyOCR을 사용하세요."
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu/Debian
    echo "🐧 Ubuntu/Debian 환경 감지"
    
    # 패키지 목록 업데이트
    sudo apt-get update
    
    # Tesseract 및 한국어 언어팩 설치
    echo "📦 Tesseract OCR 설치 중..."
    sudo apt-get install -y tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng
    
    # 개발 도구 설치 (OpenCV 등을 위해)
    echo "🔧 개발 도구 설치 중..."
    sudo apt-get install -y build-essential cmake pkg-config
    sudo apt-get install -y libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev
    sudo apt-get install -y libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev
    sudo apt-get install -y gfortran openexr libatlas-base-dev python3-dev python3-numpy
    sudo apt-get install -y libtbb2 libtbb-dev libdc1394-22-dev
    
    echo "✅ Ubuntu OCR 환경 설정 완료"
fi

pip install pytesseract

# 추가 텍스트 처리 라이브러리
echo "3. 텍스트 처리 라이브러리 설치 중..."
pip install transformers scikit-learn

# 설치 확인
echo "✅ 설치 확인 중..."
python -c "
try:
    import easyocr
    print('✅ EasyOCR 설치 완료')
except ImportError:
    print('❌ EasyOCR 설치 실패')

try:
    import pytesseract
    print('✅ Tesseract 설치 완료')
except ImportError:
    print('❌ Tesseract 설치 실패')

try:
    from transformers import AutoTokenizer
    print('✅ Transformers 설치 완료')
except ImportError:
    print('❌ Transformers 설치 실패')

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    print('✅ Scikit-learn 설치 완료')
except ImportError:
    print('❌ Scikit-learn 설치 실패')
"

echo "🎉 OCR 환경 설정 완료!"
echo "이제 codes/train_with_ocr.py를 실행할 수 있습니다."
