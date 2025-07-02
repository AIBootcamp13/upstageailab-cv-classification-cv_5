#!/bin/bash

# Ubuntu 환경 체크 및 설정 스크립트
# CUDA, OCR, Python 환경을 자동으로 확인하고 설정

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_header() {
    echo -e "${BLUE}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}${BOLD}║                   🐧 Ubuntu 환경 설정 도구                   ║${NC}"
    echo -e "${BLUE}${BOLD}║                  CV-Classify 프로젝트용                     ║${NC}"
    echo -e "${BLUE}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
}

check_ubuntu_version() {
    echo -e "${CYAN}🔍 Ubuntu 버전 확인 중...${NC}"
    
    if [ -f /etc/lsb-release ]; then
        source /etc/lsb-release
        echo -e "${GREEN}✅ Ubuntu ${DISTRIB_RELEASE} ${DISTRIB_CODENAME} 감지${NC}"
        
        # Ubuntu 버전별 호환성 체크
        case "$DISTRIB_RELEASE" in
            "20.04"|"22.04"|"24.04")
                echo -e "${GREEN}✅ 지원되는 Ubuntu 버전입니다${NC}"
                ;;
            *)
                echo -e "${YELLOW}⚠️  테스트되지 않은 Ubuntu 버전입니다. 호환성 문제가 있을 수 있습니다.${NC}"
                ;;
        esac
    else
        echo -e "${RED}❌ Ubuntu가 아니거나 버전을 확인할 수 없습니다${NC}"
        exit 1
    fi
    echo
}

check_python_environment() {
    echo -e "${CYAN}🐍 Python 환경 확인 중...${NC}"
    
    # Python 버전 확인
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        echo -e "${GREEN}✅ Python ${PYTHON_VERSION} 설치됨${NC}"
        
        # Python 3.8+ 확인
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            echo -e "${GREEN}✅ Python 버전 요구사항 충족 (3.8+)${NC}"
        else
            echo -e "${RED}❌ Python 3.8 이상이 필요합니다${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Python3가 설치되지 않았습니다${NC}"
        echo -e "${YELLOW}설치 명령어: sudo apt update && sudo apt install python3 python3-pip${NC}"
        exit 1
    fi
    
    # pip 확인
    if command -v pip3 &> /dev/null; then
        echo -e "${GREEN}✅ pip3 설치됨${NC}"
    else
        echo -e "${YELLOW}⚠️  pip3가 설치되지 않았습니다${NC}"
        echo -e "${CYAN}설치 중: sudo apt install python3-pip${NC}"
        sudo apt update && sudo apt install python3-pip -y
    fi
    
    # 가상환경 도구 확인
    if python3 -m venv --help &> /dev/null; then
        echo -e "${GREEN}✅ Python venv 사용 가능${NC}"
    else
        echo -e "${YELLOW}⚠️  python3-venv가 설치되지 않았습니다${NC}"
        echo -e "${CYAN}설치 중: sudo apt install python3-venv${NC}"
        sudo apt update && sudo apt install python3-venv -y
    fi
    echo
}

check_gpu_environment() {
    echo -e "${CYAN}🎮 GPU 환경 확인 중...${NC}"
    
    # NVIDIA GPU 확인
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ NVIDIA 드라이버 설치됨${NC}"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | while read line; do
            echo -e "${CYAN}  GPU: $line${NC}"
        done
        
        # CUDA 확인
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
            echo -e "${GREEN}✅ CUDA ${CUDA_VERSION} 설치됨${NC}"
            
            # PyTorch CUDA 호환성 확인
            case "$CUDA_VERSION" in
                "V12.1"|"V12.0")
                    echo -e "${GREEN}✅ PyTorch 2.5.0 호환 CUDA 버전${NC}"
                    ;;
                "V11.8"|"V11.7")
                    echo -e "${YELLOW}⚠️  이전 CUDA 버전입니다. requirements-ubuntu-gpu.txt 수정이 필요할 수 있습니다${NC}"
                    ;;
                *)
                    echo -e "${YELLOW}⚠️  호환성이 확인되지 않은 CUDA 버전입니다${NC}"
                    ;;
            esac
        else
            echo -e "${YELLOW}⚠️  CUDA toolkit이 설치되지 않았습니다${NC}"
            echo -e "${CYAN}CUDA 설치 가이드: https://developer.nvidia.com/cuda-downloads${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  NVIDIA GPU가 감지되지 않거나 드라이버가 설치되지 않았습니다${NC}"
        echo -e "${CYAN}CPU 모드로 실행됩니다${NC}"
    fi
    echo
}

check_ocr_environment() {
    echo -e "${CYAN}🔤 OCR 환경 확인 중...${NC}"
    
    # Tesseract 확인
    if command -v tesseract &> /dev/null; then
        TESSERACT_VERSION=$(tesseract --version | head -1 | awk '{print $2}')
        echo -e "${GREEN}✅ Tesseract ${TESSERACT_VERSION} 설치됨${NC}"
        
        # 한국어 언어팩 확인
        if tesseract --list-langs | grep -q kor; then
            echo -e "${GREEN}✅ 한국어 언어팩 설치됨${NC}"
        else
            echo -e "${YELLOW}⚠️  한국어 언어팩이 설치되지 않았습니다${NC}"
            echo -e "${CYAN}설치 중: sudo apt install tesseract-ocr-kor${NC}"
            sudo apt update && sudo apt install tesseract-ocr-kor -y
        fi
    else
        echo -e "${YELLOW}⚠️  Tesseract가 설치되지 않았습니다${NC}"
        echo -e "${CYAN}설치 중: sudo apt install tesseract-ocr tesseract-ocr-kor${NC}"
        sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-kor -y
    fi
    
    # 개발 도구 확인 (OpenCV 컴파일용)
    echo -e "${CYAN}개발 도구 확인 중...${NC}"
    MISSING_PACKAGES=""
    
    for pkg in build-essential cmake pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
        fi
    done
    
    if [ -n "$MISSING_PACKAGES" ]; then
        echo -e "${YELLOW}⚠️  누락된 개발 도구가 있습니다:$MISSING_PACKAGES${NC}"
        echo -e "${CYAN}설치 중...${NC}"
        sudo apt update && sudo apt install $MISSING_PACKAGES -y
    else
        echo -e "${GREEN}✅ 필요한 개발 도구가 모두 설치되어 있습니다${NC}"
    fi
    echo
}

setup_python_environment() {
    echo -e "${CYAN}🔧 Python 환경 설정 중...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # 가상환경 생성 (없는 경우에만)
    if [ ! -d ".venv" ]; then
        echo -e "${CYAN}가상환경 생성 중...${NC}"
        python3 -m venv .venv
        echo -e "${GREEN}✅ 가상환경 생성 완료${NC}"
    else
        echo -e "${GREEN}✅ 가상환경이 이미 존재합니다${NC}"
    fi
    
    # 가상환경 활성화
    source .venv/bin/activate
    
    # pip 업그레이드
    echo -e "${CYAN}pip 업그레이드 중...${NC}"
    python -m pip install --upgrade pip
    
    # GPU 환경에 따른 requirements 파일 선택
    if command -v nvidia-smi &> /dev/null && command -v nvcc &> /dev/null; then
        REQUIREMENTS_FILE="requirements-ubuntu-gpu.txt"
        echo -e "${GREEN}🎮 GPU 환경 감지: ${REQUIREMENTS_FILE} 사용${NC}"
    else
        REQUIREMENTS_FILE="requirements-ubuntu-cpu.txt"
        echo -e "${CYAN}🖥️  CPU 환경: ${REQUIREMENTS_FILE} 사용${NC}"
    fi
    
    # 패키지 설치
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo -e "${CYAN}패키지 설치 중... (시간이 오래 걸릴 수 있습니다)${NC}"
        pip install -r "$REQUIREMENTS_FILE"
        echo -e "${GREEN}✅ 패키지 설치 완료${NC}"
    else
        echo -e "${RED}❌ $REQUIREMENTS_FILE 파일을 찾을 수 없습니다${NC}"
        exit 1
    fi
    
    # OCR 라이브러리 설치
    echo -e "${CYAN}OCR 라이브러리 설치 중...${NC}"
    pip install easyocr pytesseract
    echo -e "${GREEN}✅ OCR 라이브러리 설치 완료${NC}"
    
    echo
}

run_environment_test() {
    echo -e "${CYAN}🧪 환경 테스트 실행 중...${NC}"
    
    cd "$PROJECT_ROOT/codes"
    
    # 가상환경 활성화
    source ../.venv/bin/activate
    
    # 디바이스 테스트
    echo -e "${CYAN}디바이스 테스트:${NC}"
    python -c "
import torch
import sys

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')

# CUDA 테스트
if torch.cuda.is_available():
    print(f'✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}')
    print(f'CUDA 버전: {torch.version.cuda}')
else:
    print('⚠️  CUDA 사용 불가 - CPU 모드')

# 간단한 텐서 연산 테스트
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_tensor = torch.randn(100, 100).to(device)
result = torch.matmul(test_tensor, test_tensor)
print(f'✅ 텐서 연산 테스트 성공 ({device})')
"
    
    # OCR 테스트
    echo -e "${CYAN}OCR 테스트:${NC}"
    python -c "
try:
    import easyocr
    print('✅ EasyOCR 임포트 성공')
except ImportError as e:
    print(f'❌ EasyOCR 임포트 실패: {e}')

try:
    import pytesseract
    print('✅ Tesseract 임포트 성공')
except ImportError as e:
    print(f'❌ Tesseract 임포트 실패: {e}')
"
    echo
}

create_ubuntu_launcher() {
    echo -e "${CYAN}🚀 Ubuntu 실행 스크립트 생성 중...${NC}"
    
    cat > "$PROJECT_ROOT/run_ubuntu.sh" << 'EOF'
#!/bin/bash

# Ubuntu 환경용 CV-Classify 실행 스크립트
# 가상환경 자동 활성화 및 메뉴 실행

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 가상환경 활성화
if [ -d ".venv" ]; then
    echo "🐍 Python 가상환경 활성화 중..."
    source .venv/bin/activate
else
    echo "❌ 가상환경이 없습니다. setup_ubuntu.sh를 먼저 실행하세요."
    exit 1
fi

# 메뉴 실행
echo "🚀 CV-Classify 메뉴 시작..."
./menu.sh
EOF
    
    chmod +x "$PROJECT_ROOT/run_ubuntu.sh"
    echo -e "${GREEN}✅ Ubuntu 실행 스크립트 생성: run_ubuntu.sh${NC}"
    echo
}

print_summary() {
    echo -e "${GREEN}${BOLD}🎉 Ubuntu 환경 설정 완료!${NC}"
    echo
    echo -e "${YELLOW}📋 다음 단계:${NC}"
    echo -e "${CYAN}  1. 프로젝트 실행: ${BOLD}./run_ubuntu.sh${NC}"
    echo -e "${CYAN}  2. 또는 수동 실행:${NC}"
    echo -e "${CYAN}     source .venv/bin/activate${NC}"
    echo -e "${CYAN}     ./menu.sh${NC}"
    echo
    echo -e "${YELLOW}💡 팁:${NC}"
    echo -e "${CYAN}  • GPU 사용: CUDA가 설치된 경우 자동으로 GPU 가속됩니다${NC}"
    echo -e "${CYAN}  • OCR 기능: EasyOCR과 Tesseract 모두 사용 가능합니다${NC}"
    echo -e "${CYAN}  • 로그 확인: logs/ 폴더에서 실행 로그를 확인할 수 있습니다${NC}"
    echo
}

main() {
    print_header
    
    # 루트 권한 확인
    if [ "$EUID" -eq 0 ]; then
        echo -e "${RED}❌ 루트 권한으로 실행하지 마세요. 일반 사용자로 실행하세요.${NC}"
        exit 1
    fi
    
    echo -e "${BOLD}Ubuntu 환경에서 CV-Classify 프로젝트 설정을 시작합니다.${NC}"
    echo -e "${YELLOW}이 스크립트는 다음 작업을 수행합니다:${NC}"
    echo -e "${CYAN}  • Ubuntu 환경 호환성 확인${NC}"
    echo -e "${CYAN}  • Python 및 CUDA 환경 검증${NC}"
    echo -e "${CYAN}  • OCR 라이브러리 설치${NC}"
    echo -e "${CYAN}  • 필요한 Python 패키지 설치${NC}"
    echo -e "${CYAN}  • 환경 테스트 실행${NC}"
    echo
    
    read -p "계속하시겠습니까? [Y/n]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        echo "설정을 취소했습니다."
        exit 0
    fi
    
    check_ubuntu_version
    check_python_environment
    check_gpu_environment
    check_ocr_environment
    setup_python_environment
    run_environment_test
    create_ubuntu_launcher
    print_summary
}

main "$@"
