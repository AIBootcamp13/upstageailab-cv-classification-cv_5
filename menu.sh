#!/bin/bash

# =============================================================================
# CV-Classify 메인 메뉴 스크립트
# 크로스 플랫폼 지원: macOS, Linux (Ubuntu/CentOS), Windows (WSL)
# =============================================================================

# 플랫폼 감지
detect_platform() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)
            if [ -f /etc/lsb-release ] || [ -f /etc/debian_version ]; then
                echo "ubuntu"
            elif [ -f /etc/redhat-release ] || [ -f /etc/centos-release ]; then
                echo "centos"
            else
                echo "linux"
            fi ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

# Python 명령어 감지
detect_python() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        echo "python3"  # 기본값
    fi
}

# 크로스 플랫폼 파일 수정 시간
get_file_mtime() {
    local file="$1"
    local platform=$(detect_platform)
    
    if [ ! -f "$file" ]; then
        echo "파일 없음"
        return 1
    fi
    
    case "$platform" in
        macos)
            stat -f "%Sm" "$file" 2>/dev/null || echo "알 수 없음"
            ;;
        *)
            stat --format="%y" "$file" 2>/dev/null | cut -d'.' -f1 || echo "알 수 없음"
            ;;
    esac
}

# 크로스 플랫폼 clear
clear_screen() {
    if command -v clear &> /dev/null; then
        clear
    else
        printf '\033[2J\033[H'
    fi
}

# 색상 정의 (터미널 지원 확인)
if [ -t 1 ] && [ "${TERM:-}" != "dumb" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    PURPLE='\033[0;35m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' PURPLE='' BOLD='' NC=''
fi

# 프로젝트 경로
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 시스템 변수
PLATFORM=$(detect_platform)
PYTHON_CMD=$(detect_python)

# 헤더 출력
print_header() {
    clear_screen
    echo -e "${BLUE}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}${BOLD}║                   🚀 CV-Classify 메인 메뉴                   ║${NC}"
    echo -e "${BLUE}${BOLD}║                      통합 실행 인터페이스                     ║${NC}"
    echo -e "${BLUE}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${CYAN}💻 시스템: $PLATFORM | 🐍 Python: $PYTHON_CMD${NC}"
    echo
}

# 메인 메뉴 출력
show_main_menu() {
    echo -e "${BOLD}📋 실행 옵션을 선택하세요:${NC}"
    echo
    echo -e "${GREEN}${BOLD}🚀 베이스라인 실행${NC}"
    echo -e "${CYAN}  1)${NC} 간단한 베이스라인 테스트 ${YELLOW}(30초, 환경 검증)${NC}"
    echo -e "${CYAN}  2)${NC} 간단한 베이스라인 백그라운드 실행 ${YELLOW}(30초)${NC}"
    echo -e "${CYAN}  3)${NC} 고급 베이스라인 포그라운드 실행 ${YELLOW}(이미지/OCR 선택)${NC}"
    echo -e "${CYAN}  4)${NC} 고급 베이스라인 백그라운드 실행 ${YELLOW}(이미지/OCR 선택)${NC}"
    echo -e "${CYAN}  5)${NC} 고급 베이스라인 DRY RUN ${YELLOW}(환경 검증만)${NC}"
    echo -e "${CYAN}  6)${NC} 베이스라인 메뉴 (상세 옵션) ${YELLOW}(대화형)${NC}"
    echo
    echo -e "${GREEN}${BOLD}🔤 OCR 전용 메뉴 (고급)${NC}"
    echo -e "${CYAN}  7)${NC} OCR 환경 설정 ${YELLOW}(EasyOCR/Tesseract 설치)${NC}"
    echo -e "${CYAN}  8)${NC} OCR 모델 DRY RUN ${YELLOW}(환경 검증 전용)${NC}"
    echo -e "${CYAN}  9)${NC} OCR 모델 포그라운드 실행 ${YELLOW}(OCR 전용)${NC}"
    echo -e "${CYAN} 10)${NC} OCR 모델 백그라운드 실행 ${YELLOW}(OCR 전용)${NC}"
    echo
    echo -e "${GREEN}${BOLD}📊 모니터링 & 관리${NC}"
    echo -e "${CYAN} 11)${NC} 실행 상태 확인"
    echo -e "${CYAN} 12)${NC} 실시간 로그 보기"
    echo -e "${CYAN} 13)${NC} 실시간 모니터링 (그래프)"
    echo -e "${CYAN} 14)${NC} 실행 중인 프로세스 중지"
    echo
    echo -e "${GREEN}${BOLD}🔧 환경 & 도구${NC}"
    echo -e "${CYAN} 15)${NC} 환경 설정 확인"
    echo -e "${CYAN} 16)${NC} 노트북 → Python 변환"
    
    # 플랫폼별 메뉴 조정
    if [[ "$PLATFORM" == "macos" ]]; then
        echo -e "${CYAN} 17)${NC} 고급 실행 메뉴 ${YELLOW}(Screen/Tmux)${NC}"
        echo -e "${CYAN} 18)${NC} GPU/MPS 디바이스 테스트 ${YELLOW}(성능 확인)${NC}"
    else
        echo -e "${CYAN} 17)${NC} 고급 실행 메뉴 ${YELLOW}(Screen/Tmux)${NC}"
        echo -e "${CYAN} 18)${NC} GPU/CUDA 디바이스 테스트 ${YELLOW}(성능 확인)${NC}"
    fi
    echo
    echo -e "${GREEN}${BOLD}📋 정보${NC}"
    echo -e "${CYAN} 19)${NC} 프로젝트 상태 요약"
    echo -e "${CYAN} 20)${NC} 사용 가능한 파일 목록"
    echo -e "${CYAN} 21)${NC} 도움말 보기"
    echo
    echo -e "${RED}${BOLD}  0)${NC} 종료"
    echo
    echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
}

# 프로젝트 상태 요약
show_project_status() {
    echo -e "${BOLD}📋 프로젝트 상태 요약${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    
    # 시스템 정보
    echo -e "${YELLOW}💻 시스템 정보:${NC}"
    echo "  플랫폼: $PLATFORM"
    echo "  Python: $PYTHON_CMD"
    if command -v "$PYTHON_CMD" &> /dev/null; then
        local python_version=$($PYTHON_CMD --version 2>&1)
        echo "  버전: $python_version"
    fi
    echo
    
    # 데이터 확인
    echo -e "${YELLOW}📊 데이터 상태:${NC}"
    if [ -f "$PROJECT_ROOT/data/train.csv" ] && [ -d "$PROJECT_ROOT/data/train" ]; then
        echo "  ✅ 훈련 데이터 준비 완료"
        echo "     - train.csv: $(wc -l < "$PROJECT_ROOT/data/train.csv" 2>/dev/null || echo "?") 행"
        echo "     - train/ 폴더: $(ls "$PROJECT_ROOT/data/train" 2>/dev/null | wc -l || echo "?") 개 이미지"
    else
        echo "  ❌ 훈련 데이터 누락"
    fi
    
    if [ -f "$PROJECT_ROOT/data/test.csv" ] && [ -d "$PROJECT_ROOT/data/test" ]; then
        echo "  ✅ 테스트 데이터 준비 완료"
    else
        echo "  ❌ 테스트 데이터 누락"
    fi
    echo
    
    # 코드 파일 확인
    echo -e "${YELLOW}📄 코드 파일 상태:${NC}"
    if [ -f "$PROJECT_ROOT/codes/baseline_code.ipynb" ]; then
        echo "  ✅ 공식 베이스라인 노트북 (baseline_code.ipynb)"
    else
        echo "  ❌ 공식 베이스라인 노트북 누락"
    fi
    
    if [ -f "$PROJECT_ROOT/codes/baseline_simple.py" ]; then
        echo "  ✅ 간단한 베이스라인 스크립트 (baseline_simple.py)"
    else
        echo "  ⚠️  간단한 베이스라인 스크립트 누락 (변환 필요)"
    fi
    
    if [ -f "$PROJECT_ROOT/codes/train_with_wandb.py" ]; then
        echo "  ✅ 고급 베이스라인 스크립트 (train_with_wandb.py)"
    else
        echo "  ❌ 고급 베이스라인 스크립트 누락"
    fi
    echo
    
    # 실행 상태 확인
    echo -e "${YELLOW}🏃 실행 상태:${NC}"
    local pid_file="$PROJECT_ROOT/logs/training.pid"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "  🟢 학습 실행 중 (PID: $pid)"
        else
            echo "  🔴 학습 중지됨"
            rm -f "$pid_file"
        fi
    else
        echo "  ⚪ 대기 중 (실행 중인 학습 없음)"
    fi
    echo
    
    # 최근 결과 확인 (크로스 플랫폼)
    echo -e "${YELLOW}📈 최근 결과:${NC}"
    if [ -f "$PROJECT_ROOT/codes/pred_baseline.csv" ]; then
        echo "  ✅ 최근 예측 파일: codes/pred_baseline.csv"
        echo "     생성 시간: $(get_file_mtime "$PROJECT_ROOT/codes/pred_baseline.csv")"
    else
        echo "  ⚪ 예측 파일 없음 (아직 실행하지 않음)"
    fi
    
    # 로그 파일 확인
    local latest_log=$(ls -t "$PROJECT_ROOT/logs/"*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "  📝 최근 로그: $(basename "$latest_log")"
        echo "     생성 시간: $(get_file_mtime "$latest_log")"
    else
        echo "  ⚪ 로그 파일 없음"
    fi
}

# 사용 가능한 파일 목록
show_available_files() {
    echo -e "${BOLD}📁 사용 가능한 파일 목록${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    
    echo -e "${YELLOW}📓 노트북 파일:${NC}"
    if ls "$PROJECT_ROOT/codes/"*.ipynb &> /dev/null; then
        ls "$PROJECT_ROOT/codes/"*.ipynb 2>/dev/null | while read file; do
            echo "  📓 $(basename "$file")"
        done
    else
        echo "  ⚪ 노트북 파일 없음"
    fi
    echo
    
    echo -e "${YELLOW}🐍 Python 스크립트:${NC}"
    if ls "$PROJECT_ROOT/codes/"*.py &> /dev/null; then
        ls "$PROJECT_ROOT/codes/"*.py 2>/dev/null | while read file; do
            echo "  🐍 $(basename "$file")"
        done
    else
        echo "  ⚪ Python 파일 없음"
    fi
    echo
    
    echo -e "${YELLOW}🔧 실행 스크립트:${NC}"
    if ls "$PROJECT_ROOT/scripts/"*.sh &> /dev/null; then
        ls "$PROJECT_ROOT/scripts/"*.sh 2>/dev/null | while read file; do
            echo "  🔧 $(basename "$file")"
        done
    else
        echo "  ⚪ 스크립트 파일 없음"
    fi
    echo
    
    echo -e "${YELLOW}📊 데이터 파일:${NC}"
    if [ -d "$PROJECT_ROOT/data" ]; then
        if ls "$PROJECT_ROOT/data/"*.csv &> /dev/null; then
            ls "$PROJECT_ROOT/data/"*.csv 2>/dev/null | while read file; do
                echo "  📊 $(basename "$file")"
            done
        else
            echo "  ⚪ CSV 파일 없음"
        fi
        
        echo "  📁 폴더:"
        ls -la "$PROJECT_ROOT/data/" 2>/dev/null | grep "^d" | awk '{print "     📁 " $9}' | grep -v "^\.$\|^\.\.$" || echo "     📁 없음"
    else
        echo "  ❌ data/ 폴더가 없습니다"
    fi
}

# 도움말 (플랫폼별 안내)
show_help() {
    echo -e "${BOLD}📖 CV-Classify 사용 가이드 (${PLATFORM})${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo -e "${YELLOW}🚀 베이스라인 선택 가이드:${NC}"
    echo "  1️⃣  간단한 베이스라인 (baseline_simple.py)"
    echo "      ├─ 용도: 환경 테스트, 빠른 검증"
    echo "      ├─ 시간: ~30초"
    echo "      ├─ 특징: 32x32 이미지, 1 epoch"
    echo "      └─ 호환: 공식 베이스라인 100% 동일"
    echo
    echo "  2️⃣  고급 베이스라인 (train_with_wandb.py)"
    echo "      ├─ 용도: 실제 실험, 성능 최적화"
    echo "      ├─ 시간: ~30분"
    echo "      ├─ 특징: 224x224 이미지, 50 epochs"
    echo "      └─ 추가: WandB 통합, 검증 세트 분할"
    echo
    echo -e "${YELLOW}📋 실행 단계:${NC}"
    echo "  1. 환경 테스트: 메뉴 1번 → 간단한 베이스라인 테스트"
    echo "  2. 고급 테스트: 메뉴 3번 → 고급 베이스라인 테스트"
    echo "  3. 실제 실험: 메뉴 4번 → 고급 베이스라인 백그라운드 실행"
    echo "  4. 모니터링: 메뉴 8번 → 실시간 모니터링"
    echo "  5. 결과 확인: codes/pred_baseline.csv 파일"
    echo
    echo -e "${YELLOW}🔧 플랫폼별 트러블슈팅:${NC}"
    case "$PLATFORM" in
        macos)
            echo "  • Homebrew 없음: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "  • MPS 디바이스 오류: macOS 12.0+ 및 Apple Silicon 필요"
            echo "  • Screen 없음: brew install screen"
            ;;
        ubuntu|linux)
            echo "  • 패키지 없음: sudo apt-get update && sudo apt-get install screen tmux"
            echo "  • CUDA 없음: NVIDIA 드라이버 및 CUDA 설치 필요"
            echo "  • Python3 없음: sudo apt-get install python3 python3-pip"
            ;;
        centos)
            echo "  • 패키지 없음: sudo yum install screen tmux python3"
            echo "  • 또는: sudo dnf install screen tmux python3"
            ;;
        *)
            echo "  • 일반적인 문제: 패키지 관리자로 필요한 도구 설치"
            ;;
    esac
    echo "  • 파일 없음 오류: 메뉴 11번으로 노트북 변환"
    echo "  • 환경 문제: 메뉴 10번으로 환경 확인"
    echo "  • 실행 중단: 메뉴 9번으로 프로세스 중지"
    echo "  • 상태 확인: 메뉴 14번으로 프로젝트 상태 확인"
    echo
    echo -e "${YELLOW}📞 지원:${NC}"
    echo "  • GitHub Issues: 프로젝트 리포지토리 Issues 탭"
    echo "  • 로그 확인: logs/ 폴더의 최신 로그 파일"
    echo "  • 실시간 도움: 메뉴 8번 모니터링 사용"
    echo "  • Python 명령어: $PYTHON_CMD (자동 감지됨)"
}

# 사용자 입력 대기
wait_for_input() {
    echo
    echo -e "${BOLD}계속하려면 Enter를 누르세요...${NC}"
    read -r
}

# 스크립트 실행 함수 (크로스 플랫폼)
execute_script() {
    local script_path="$1"
    local script_name="$2"
    
    echo -e "${GREEN}🚀 $script_name 실행 중...${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    
    if [ -f "$script_path" ]; then
        exec "$script_path" "${@:3}"
    else
        echo -e "${RED}❌ 스크립트를 찾을 수 없습니다: $script_path${NC}"
        echo -e "${YELLOW}🔍 디버그 정보:${NC}"
        echo "  PROJECT_ROOT: $PROJECT_ROOT"
        echo "  현재 디렉토리: $(pwd)"
        echo "  스크립트 경로: $script_path"
        echo "  플랫폼: $PLATFORM"
        echo "  Python: $PYTHON_CMD"
        echo
        echo -e "${YELLOW}📁 scripts 폴더 내용:${NC}"
        ls -la "$PROJECT_ROOT/scripts/" 2>/dev/null || echo "  scripts 폴더를 찾을 수 없습니다"
        wait_for_input
    fi
}

# 메인 로직
main() {
    while true; do
        print_header
        show_main_menu
        
        echo -ne "${BOLD}선택하세요 (0-21): ${NC}"
        read -r choice
        echo
        
        case $choice in
            1)
                execute_script "$PROJECT_ROOT/scripts/test_baseline.sh" "간단한 베이스라인 테스트"
                ;;
            2)
                execute_script "$PROJECT_ROOT/scripts/run_training.sh" "간단한 베이스라인 백그라운드 실행" "start" "baseline_simple.py"
                ;;
            3)
                # 고급 베이스라인 포그라운드 실행 (크로스 플랫폼)
                echo -e "${GREEN}🚀 고급 베이스라인 포그라운드 실행 중...${NC}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo
                echo -e "${YELLOW}🤔 OCR(광학 문자 인식)을 사용하시겠습니까?${NC}"
                echo -e "${CYAN}ℹ️  OCR은 문서 이미지에서 텍스트를 추출하여 분류 성능을 향상시킵니다.${NC}"
                echo -e "${CYAN}  - 이미지만: 빠른 처리, 기존 방식${NC}"
                echo -e "${CYAN}  - OCR 통합: 텍스트 + 이미지, 더 정확한 분류 기대${NC}"
                echo
                echo -ne "${BOLD}OCR을 사용하시겠습니까? [y/N]: ${NC}"
                read -r use_ocr_choice
                echo
                
                if [[ "$use_ocr_choice" =~ ^[Yy]$ ]]; then
                    echo -e "${GREEN}🔤 OCR 통합 모델로 실행합니다...${NC}"
                    echo
                    # OCR 환경 채크
                    cd "$PROJECT_ROOT/codes" || exit 1
                    if $PYTHON_CMD -c "import easyocr" 2>/dev/null || $PYTHON_CMD -c "import pytesseract" 2>/dev/null; then
                        echo -e "${GREEN}✅ OCR 환경이 설정되어 있습니다.${NC}"
                        $PYTHON_CMD train_with_ocr.py
                    else
                        echo -e "${YELLOW}⚠️  OCR 라이브러리가 설치되지 않았습니다.${NC}"
                        echo -e "${CYAN}🔧 OCR 환경을 설정하시겠습니까? [Y/n]: ${NC}"
                        read -r setup_ocr
                        if [[ "$setup_ocr" =~ ^[Nn]$ ]]; then
                            echo -e "${BLUE}🚀 이미지 전용 모델로 실행합니다...${NC}"
                            $PYTHON_CMD train_with_wandb.py
                        else
                            echo -e "${GREEN}🔧 OCR 환경 설정 시작...${NC}"
                            cd "$PROJECT_ROOT" || exit 1
                            chmod +x setup_ocr.sh
                            ./setup_ocr.sh
                            echo
                            echo -e "${GREEN}🔤 OCR 설정 완료! OCR 모델 실행 중...${NC}"
                            cd codes || exit 1
                            $PYTHON_CMD train_with_ocr.py
                        fi
                    fi
                else
                    echo -e "${BLUE}🚀 이미지 전용 모델로 실행합니다...${NC}"
                    cd "$PROJECT_ROOT/codes" || exit 1
                    $PYTHON_CMD train_with_wandb.py
                fi
                wait_for_input
                ;;
            4)
                # 고급 베이스라인 백그라운드 실행
                echo -e "${GREEN}🚀 고급 베이스라인 백그라운드 실행 시작...${NC}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo
                echo -e "${YELLOW}🤔 백그라운드에서 OCR을 사용하시겠습니까?${NC}"
                echo -e "${CYAN}ℹ️  OCR은 처리 시간이 더 오래 걸리지만 더 정확한 분류를 기대할 수 있습니다.${NC}"
                echo
                echo -ne "${BOLD}OCR을 사용하시겠습니까? [y/N]: ${NC}"
                read -r use_ocr_bg
                echo
                
                if [[ "$use_ocr_bg" =~ ^[Yy]$ ]]; then
                    # OCR 환경 채크
                    cd "$PROJECT_ROOT/codes" || exit 1
                    if $PYTHON_CMD -c "import easyocr" 2>/dev/null || $PYTHON_CMD -c "import pytesseract" 2>/dev/null; then
                        echo -e "${GREEN}🔤 OCR 모델 백그라운드 실행 시작...${NC}"
                        nohup $PYTHON_CMD train_with_ocr.py > "../logs/ocr_training_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
                        echo $! > "../logs/ocr_training.pid"
                        echo -e "${GREEN}✅ OCR 모델 백그라운드 실행 시작됨${NC}"
                        echo -e "${CYAN}📄 로그: logs/ocr_training_$(date +%Y%m%d_%H%M%S).log${NC}"
                    else
                        echo -e "${YELLOW}⚠️  OCR 라이브러리가 설치되지 않았습니다.${NC}"
                        echo -e "${CYAN}🔧 먼저 메뉴 7번으로 OCR 환경을 설정해주세요.${NC}"
                        echo -e "${BLUE}🚀 이미지 전용 모델로 백그라운드 실행합니다...${NC}"
                        execute_script "$PROJECT_ROOT/scripts/run_training.sh" "고급 베이스라인 백그라운드 실행" "start" "train_with_wandb.py"
                        return
                    fi
                else
                    echo -e "${BLUE}🚀 이미지 전용 모델로 백그라운드 실행합니다...${NC}"
                    execute_script "$PROJECT_ROOT/scripts/run_training.sh" "고급 베이스라인 백그라운드 실행" "start" "train_with_wandb.py"
                    return
                fi
                echo -e "${CYAN}🔍 상태 확인: 메뉴 11번${NC}"
                wait_for_input
                ;;
            5)
                # 고급 베이스라인 DRY RUN (환경 검증)
                echo -e "${GREEN}🧪 고급 베이스라인 DRY RUN (환경 검증)${NC}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo
                cd "$PROJECT_ROOT/codes" || exit 1
                $PYTHON_CMD train_with_wandb.py --dry-run
                echo
                echo -e "${YELLOW}💡 DRY RUN이 성공했다면, 메뉴 3번으로 포그라운드 실행 또는 4번으로 백그라운드 실행하세요.${NC}"
                wait_for_input
                ;;
            6)
                execute_script "$PROJECT_ROOT/scripts/run_baseline.sh" "베이스라인 상세 메뉴"
                ;;
            7)
                # OCR 환경 설정
                echo -e "${GREEN}🔧 OCR 환경 설정 시작...${NC}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo
                chmod +x "$PROJECT_ROOT/setup_ocr.sh"
                "$PROJECT_ROOT/setup_ocr.sh"
                wait_for_input
                ;;
            8)
                # OCR 통합 모델 DRY RUN
                echo -e "${GREEN}🧪 OCR 통합 모델 DRY RUN (환경 검증)${NC}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo
                cd "$PROJECT_ROOT/codes" || exit 1
                $PYTHON_CMD train_with_ocr.py --dry-run
                echo
                echo -e "${YELLOW}💡 DRY RUN이 성공했다면, 메뉴 9번으로 포그라운드 실행하세요.${NC}"
                wait_for_input
                ;;
            9)
                # OCR 통합 모델 포그라운드 실행
                echo -e "${GREEN}🔤 OCR 통합 모델 포그라운드 실행 중...${NC}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo
                cd "$PROJECT_ROOT/codes" || exit 1
                $PYTHON_CMD train_with_ocr.py
                wait_for_input
                ;;
            10)
                # OCR 통합 모델 백그라운드 실행
                echo -e "${GREEN}🚀 OCR 통합 모델 백그라운드 실행 시작...${NC}"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo
                cd "$PROJECT_ROOT/codes" || exit 1
                nohup $PYTHON_CMD train_with_ocr.py > "../logs/ocr_training_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
                echo $! > "../logs/ocr_training.pid"
                echo -e "${GREEN}✅ OCR 통합 모델 백그라운드 실행 시작됨${NC}"
                echo -e "${CYAN}📄 로그: logs/ocr_training_$(date +%Y%m%d_%H%M%S).log${NC}"
                echo -e "${CYAN}🔍 상태 확인: 메뉴 11번${NC}"
                wait_for_input
                ;;
            11)
                execute_script "$PROJECT_ROOT/scripts/run_training.sh" "실행 상태 확인" "status"
                ;;
            12)
                execute_script "$PROJECT_ROOT/scripts/run_training.sh" "실시간 로그 보기" "follow"
                ;;
            13)
                execute_script "$PYTHON_CMD" "실시간 모니터링" "$PROJECT_ROOT/scripts/monitor.py" "monitor"
                ;;
            14)
                execute_script "$PROJECT_ROOT/scripts/run_training.sh" "프로세스 중지" "stop"
                ;;
            15)
                execute_script "$PROJECT_ROOT/scripts/run_training.sh" "환경 설정 확인" "check"
                ;;
            16)
                execute_script "$PROJECT_ROOT/scripts/convert_notebook.sh" "노트북 → Python 변환"
                ;;
            17)
                execute_script "$PROJECT_ROOT/scripts/advanced_launcher.sh" "고급 실행 메뉴"
                ;;
            18)
                cd "$PROJECT_ROOT/codes" && $PYTHON_CMD -c "from device_utils import test_device; test_device()"
                wait_for_input
                ;;
            19)
                show_project_status
                wait_for_input
                ;;
            20)
                show_available_files
                wait_for_input
                ;;
            21)
                show_help
                wait_for_input
                ;;
            0)
                echo -e "${GREEN}👋 CV-Classify를 이용해 주셔서 감사합니다!${NC}"
                echo -e "${CYAN}행운을 빕니다! 🍀${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}❌ 잘못된 선택입니다. 0-21 범위의 숫자를 입력해주세요.${NC}"
                wait_for_input
                ;;
        esac
    done
}

# 스크립트 실행
main "$@"