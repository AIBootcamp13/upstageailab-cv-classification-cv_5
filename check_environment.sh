#!/bin/bash

# 환경 검증 스크립트

echo "🔍 CV-Classify 환경 검증"
echo "========================================"

# Python 및 가상환경 확인
echo "🐍 Python 환경:"
echo "  Python 경로: $(which python)"
echo "  Python 버전: $(python --version)"
echo "  가상환경: ${VIRTUAL_ENV:-'활성화되지 않음'}"
echo

# 필수 패키지 확인
echo "📦 필수 패키지 확인:"
python -c "
import sys
packages = {
    'torch': None,
    'torchvision': None, 
    'torchaudio': None,
    'timm': None,
    'albumentations': None,
    'numpy': None,
    'pandas': None,
    'opencv-python': 'cv2'
}

for package, import_name in packages.items():
    try:
        if import_name:
            exec(f'import {import_name}')
            module = sys.modules[import_name]
        else:
            exec(f'import {package}')
            module = sys.modules[package]
        
        version = getattr(module, '__version__', 'Unknown')
        print(f'  ✅ {package}: {version}')
    except ImportError:
        print(f'  ❌ {package}: 설치되지 않음')
"

echo
echo "🎮 GPU/가속 장치 확인:"
python -c "
import torch
print(f'  PyTorch 버전: {torch.__version__}')

if torch.cuda.is_available():
    print(f'  ✅ CUDA 사용 가능 - {torch.cuda.get_device_name()}')
    print(f'     CUDA 버전: {torch.version.cuda}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('  ✅ MPS (Apple Silicon) 사용 가능')
else:
    print('  💻 CPU 모드')
"

echo
echo "📁 프로젝트 구조 확인:"
echo "  ✅ 현재 위치: $(pwd)"

files_to_check=(
    "data/train.csv:훈련 라벨 데이터"
    "data/train/:훈련 이미지 폴더"
    "data/test/:테스트 이미지 폴더"
    "data/sample_submission.csv:제출 포맷 파일"
    "codes/baseline_simple.py:간단한 베이스라인"
    "codes/train_with_wandb.py:고급 베이스라인"
    "requirements.txt:패키지 요구사항"
)

for item in "${files_to_check[@]}"; do
    file="${item%:*}"
    desc="${item#*:}"
    if [ -e "$file" ]; then
        echo "  ✅ $desc ($file)"
    else
        echo "  ❌ $desc ($file)"
    fi
done

echo
echo "📊 데이터 통계:"
if [ -f "data/train.csv" ]; then
    train_count=$(wc -l < data/train.csv)
    echo "  📈 훈련 데이터: $((train_count - 1)) 개 (헤더 제외)"
fi

if [ -d "data/train" ]; then
    train_images=$(ls data/train/*.jpg 2>/dev/null | wc -l)
    echo "  🖼️  훈련 이미지: $train_images 개"
fi

if [ -d "data/test" ]; then
    test_images=$(ls data/test/*.jpg 2>/dev/null | wc -l)
    echo "  🖼️  테스트 이미지: $test_images 개"
fi

if [ -f "data/sample_submission.csv" ]; then
    submission_count=$(wc -l < data/sample_submission.csv)
    echo "  📝 제출 포맷: $((submission_count - 1)) 개 예측 필요 (헤더 제외)"
fi

echo
echo "📋 요약:"
if [ -n "${VIRTUAL_ENV}" ] && python -c "import torch, timm" 2>/dev/null; then
    echo "  🎉 환경이 올바르게 설정되었습니다!"
    echo "  🚀 ./menu.sh로 학습을 시작할 수 있습니다."
    echo ""
    echo "  📌 참고: test.csv가 없는 것은 정상입니다."
    echo "     테스트 데이터에는 정답 라벨이 제공되지 않습니다."
else
    echo "  ⚠️  환경 설정이 완료되지 않았습니다."
    echo "  🔧 ./setup.sh를 실행하여 환경을 설정해주세요."
fi
