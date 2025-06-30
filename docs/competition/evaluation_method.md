# 평가방법

**출처:** [aistages](https://stages.ai/competitions/356/overview/evaluation)  
**클립 날짜:** 2025-06-30T12:37:22+09:00

### 평가지표

- **평가지표** : Macro F1

F1 score는 Precision과 Recall의 조화 평균을 의미합니다. 클래스마다 개수가 불균형할 때 모델의 성능을 더욱 정확하게 평가할 수 있습니다. 수식은 다음과 같습니다.

![F1 Score 수식](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/4d81bedc-a500-4910-8334-dc96995fa1e1.png)

![Precision 수식](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/edc12558-6c45-4610-89b5-9b933b1f4b39.png)

![Recall 수식](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/6fee07b1-3b0a-4a12-8913-0ed2119e5ec8.png)

**[참고자료]** https://www.linkedin.com/pulse/understanding-confusion-matrix-tanvi-mittal/

Macro F1 score는 multi classification을 위한 평가 지표로 클래스 별로 계산된 F1 score를 단순 평균한 지표입니다.

![Macro F1 Score 수식](https://aistages-api-public-prod.s3.amazonaws.com/app/Files/01555d7c-ad8a-4ce3-9692-33d2be0eaaf6.png)

- **Public 평가** : 전체 Test 데이터 중 랜덤 샘플링 된 50%
- **Private 평가** : 전체 Test 데이터 중 나머지 50%
