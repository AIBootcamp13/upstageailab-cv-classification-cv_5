# 데이터 개요

**출처:** [aistages](https://stages.ai/competitions/356/data/overview)  
**클립 날짜:** 2025-06-30T12:41:51+09:00

## 데이터 다운로드 링크

https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000356/data/data.tar.gz

## 데이터셋 다운로드

다운로드 탭을 참고, wget 명령어를 통해 데이터셋을 본인의 작업 환경에 다운로드 합니다.

```bash
# 데이터 다운로드
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000356/data/data.tar.gz
```

터미널에서 아래 명령어를 활용하여 파일을 압축 해제합니다.

```bash
tar -zxvf data.tar.gz
```

---

## 학습 데이터셋 정보

주어진 학습 데이터에 대한 정보는 다음과 같습니다.

- **train [폴더]**
	- 1570장의 이미지가 저장되어 있습니다.

- **train.csv [파일]**
	
	![train.csv 예시](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/832b4982-bd93-4480-936f-3c93a1aee98b.png)
	
	- 1570개의 행으로 이루어져 있습니다. `train/` 폴더에 존재하는 1570개의 이미지에 대한 정답 클래스를 제공합니다.
	- `ID` 학습 샘플의 파일명
	- `target` 학습 샘플의 정답 클래스 번호

- **meta.csv [파일]**
	
	![meta.csv 예시](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/d4b872ca-b669-4166-b146-5ce12af01deb.png)
	
	- 17개의 행으로 이루어져 있습니다.
	- `target` 17개의 클래스 번호입니다.
	- `class_name` 클래스 번호에 대응하는 클래스 이름입니다.

---

## 평가 데이터셋 정보

평가 데이터에 대한 정보는 다음과 같습니다.

- **test [폴더]**
	- 3140장의 이미지가 저장되어 있습니다.

- **sample_submission.csv [파일]**
	- 3140개의 행으로 이루어져 있습니다.
	- `ID` 평가 샘플의 파일명이 저장되어 있습니다.
	- `target` 예측 결과가 입력될 컬럼입니다. 값이 전부 0으로 저장되어 있습니다.

![sample_submission.csv 예시](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/86c6b7ed-f8a4-4909-a614-a8d3bdfc94a7.png)

그 밖에 평가 데이터는 학습 데이터와 달리 랜덤하게 Rotation 및 Flip 등이 되었고 훼손된 이미지들이 존재합니다.
