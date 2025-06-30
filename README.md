# CV-Classify 프로젝트

## 🚀 빠른 시작 가이드

### 1. Git Clone
```bash
git clone <repository-url>
cd cv-classify
```

### 2. 초기 설정 (필수)
```bash
chmod +x setup.sh
./setup.sh
```

### 3. 실행
```bash
./menu.sh
```

---

## 💻 지원 플랫폼
- ✅ **macOS** (Intel/Apple Silicon) - Homebrew 기반
- ✅ **Ubuntu** 20.04/22.04 LTS - APT 기반  
- ✅ **CentOS** 7/8 - YUM/DNF 기반
- ✅ **Windows WSL2** - Linux 호환

---

## 📋 시스템 요구사항

### 필수
- Python 3.7+
- Git
- curl, wget

### 선택사항 (권장)
- screen 또는 tmux (백그라운드 실행용)
- NVIDIA GPU + CUDA (GPU 가속용)

---

## 🛠️ 주요 기능

### 베이스라인 실행
1. **간단한 베이스라인**: 30초, 환경 검증용
2. **고급 베이스라인**: 30분, WandB 통합

### 모니터링 & 관리
- 실시간 로그 확인
- 그래프 모니터링
- 프로세스 관리

### 크로스 플랫폼 지원
- 자동 플랫폼 감지
- Python 명령어 자동 선택
- 패키지 관리자 자동 감지

---

## 📁 주요 파일 구조

```
cv-classify/
├── menu.sh                    # 메인 메뉴 (크로스 플랫폼)
├── setup.sh                   # 초기 설정 스크립트
├── scripts/
│   ├── advanced_launcher.sh   # 고급 실행 (Screen/Tmux)
│   ├── platform_utils.sh      # 플랫폼 유틸리티
│   └── *.sh                    # 기타 스크립트들
├── codes/                      # Python 코드
├── data/                       # 데이터
└── logs/                       # 로그 파일
```

---

## 🔧 트러블슈팅

### 권한 오류
```bash
chmod +x setup.sh
./setup.sh
```

### 플랫폼별 패키지 설치

#### macOS
```bash
# Homebrew 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 패키지 설치
brew install python3 screen tmux
```

#### Ubuntu
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip screen tmux curl wget git
```

#### CentOS
```bash
sudo yum install python3 python3-pip screen tmux curl wget git
# 또는
sudo dnf install python3 python3-pip screen tmux curl wget git
```

---

## 📞 지원

- **GitHub Issues**: 문제 신고 및 제안
- **메뉴 16번**: 도움말 보기
- **메뉴 14번**: 프로젝트 상태 확인

---

## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

---

## 1. Competition Info

### Overview
- _Write competition information_

### Timeline
- ex) January 10, 2024 - Start Date
- ex) February 10, 2024 - Final submission deadline

---

## 2. Data Description

### Dataset Overview
- _Explain using data_

### EDA
- _Describe your EDA process and step-by-step conclusion_

### Data Processing
- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

---

## 3. Modeling

### Model Description
- _Write model information and why your select this model_

### Modeling Process
- _Write model train and test process with capture_

---

## 4. Result

### Leader Board
- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation
- _Insert your presentation file(pdf) link_

---

## etc

### Meeting Log
- _Insert your meeting log link like Notion or Google Docs_

### Reference
- _Insert related reference_
