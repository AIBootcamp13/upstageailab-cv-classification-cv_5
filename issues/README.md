# Issues 관리 가이드

이 폴더는 프로젝트의 GitHub Issues를 로컬에서 관리하기 위한 공간입니다.

## 📁 폴더 구조

```
issues/
├── README.md                     # 이 파일
├── issue-001-template-guide.md   # Issue & PR 템플릿 가이드
└── templates/                    # 템플릿 파일들
    ├── bug-report.md
    ├── feature-request.md
    └── pull-request.md
```

## 📋 파일 명명 규칙

```
issue-{번호}-{간단한-제목}.md
```

**예시:**
- `issue-001-template-guide.md`
- `issue-002-data-loading-bug.md`
- `issue-003-model-performance-improvement.md`

## 🔄 이슈 동기화

### GitHub에서 로컬로 가져오기
```bash
# 특정 이슈 가져오기
gh issue view {이슈번호} --repo AIBootcamp13/upstageailab-cv-classification-cv_5 > issues/issue-{번호}-{제목}.md

# 모든 열린 이슈 목록 보기
gh issue list --repo AIBootcamp13/upstageailab-cv-classification-cv_5
```

### 새 이슈 생성
```bash
# GitHub에 새 이슈 생성
gh issue create --repo AIBootcamp13/upstageailab-cv-classification-cv_5 --title "제목" --body "내용"
```

## 📊 이슈 상태 관리

- **📕 버그 리포트**: `bug` 라벨
- **📗 기능 요청**: `enhancement` 라벨  
- **📘 문서화**: `documentation` 라벨
- **📙 질문**: `question` 라벨

## 🔗 유용한 명령어

```bash
# 이슈 목록 보기
gh issue list --repo AIBootcamp13/upstageailab-cv-classification-cv_5

# 이슈 상세 보기
gh issue view {번호} --repo AIBootcamp13/upstageailab-cv-classification-cv_5

# 이슈 닫기
gh issue close {번호} --repo AIBootcamp13/upstageailab-cv-classification-cv_5

# 이슈에 댓글 달기
gh issue comment {번호} --repo AIBootcamp13/upstageailab-cv-classification-cv_5 --body "댓글 내용"
```

## 📌 팀 협업 가이드

1. **이슈 생성 전**: 중복 이슈가 없는지 확인
2. **템플릿 사용**: 제공된 템플릿을 따라 작성
3. **라벨 지정**: 적절한 라벨로 분류
4. **담당자 지정**: Assignee 설정
5. **연관 작업**: PR과 이슈 연결 (`close #번호`)

---

**참고:** 이 폴더의 파일들은 GitHub Issues의 로컬 백업 및 오프라인 참조용입니다.
