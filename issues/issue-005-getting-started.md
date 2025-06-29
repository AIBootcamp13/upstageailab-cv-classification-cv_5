# Getting Started ⭐

**Issue ID:** #5  
**Status:** Open  
**Author:** skier-song9  
**Created:** About 2 days ago  
**Comments:** 1  
**Labels:** 문서, 필수  
**Assignees:** skier-song9  

---

## 🔻 AIStages GitHub 설정

### Git 설치 및 기본 설정
```bash
# Git 설치
apt update
apt install -y git

# Git 설정
git config --global credential.helper store
git config --global user.name "깃헙 사용자 이름 입력"
git config --global user.email "깃헙 사용자 이메일 입력"
git config --global core.pager "cat"

# Vim 설치 및 에디터 설정
apt install -y vim
git config --global core.editor "vim"
```

---

## 🔻 기타 라이브러리 설치

### 한국어 폰트 설치
```bash
# 한국어 폰트 설치
apt-get update
apt-get install -y fonts-nanum*

# 폰트 파일 위치 확인 (폰트들이 잘 나오면 설치 완료)
ls /usr/share/fonts/truetype/nanum/Nanum*
```

### UV 설치를 위한 준비
```bash
# uv 설치를 위한 curl 설치
apt-get install -y curl
```

---

## 🔻 Fork & Clone

> **참고:** [GitHub Flow 팀플 방법 노션](https://skier-song9.notion.site/github-flow-1dec8d3f60f580948bb2c9a112266c46?source=copy_link)

### 1단계: Fork 생성
1. 팀 레포 [AIBootcamp13/upstageailab-cv-classification-cv_5](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5)에서 **Fork**
2. 개인 계정에 forked repository 생성

### 2단계: 홈 디렉토리로 이동
```bash
# 터미널에서 현재 위치를 ${HOME}으로 이동
cd ~/
```
> **참고:** AIStages 서버 컨테이너의 `${HOME}` 위치는 `/data/ephemeral/home/`

### 3단계: Repository Clone
```bash
# forked repository를 clone
git clone https://github.com/[각자의 깃헙 계정]/upstageailab-cv-classification-cv_5.git
```

### 4단계: Upstream Remote 설정
```bash
# 팀 레포를 upstream이라는 이름의 remote 저장소로 지정
# fetch & merge할 때 사용
git remote add upstream https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5.git

# upstream으로의 push를 방지 (안전장치)
git remote set-url --push upstream no-push
```

---

## 📋 체크리스트

프로젝트 시작 전 다음 항목들을 확인하세요:

- [ ] Git 설치 및 설정 완료
- [ ] 한국어 폰트 설치 완료
- [ ] curl 설치 완료
- [ ] 팀 레포 Fork 완료
- [ ] 개인 계정에서 Clone 완료
- [ ] Upstream remote 설정 완료
- [ ] UV 설치 준비 완료

---

## 🔗 관련 링크

- **팀 레포:** [AIBootcamp13/upstageailab-cv-classification-cv_5](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5)
- **GitHub Flow 가이드:** [노션 페이지](https://skier-song9.notion.site/github-flow-1dec8d3f60f580948bb2c9a112266c46?source=copy_link)

---

**원본 이슈:** [GitHub에서 보기](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/issues/5)

---

## 💡 추가 팁

### Git 설정 확인
```bash
# 현재 Git 설정 확인
git config --list

# 특정 설정 확인
git config user.name
git config user.email
```

### Remote 확인
```bash
# 등록된 remote 저장소 확인
git remote -v

# 예상 결과:
# origin    https://github.com/[개인계정]/upstageailab-cv-classification-cv_5.git (fetch)
# origin    https://github.com/[개인계정]/upstageailab-cv-classification-cv_5.git (push)
# upstream  https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5.git (fetch)
# upstream  https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5.git (push)
```
