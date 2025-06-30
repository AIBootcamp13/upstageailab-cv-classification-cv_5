# 대회 팁 및 가이드

**출처:** [aistages](https://stages.ai/competitions/356/board/community/post/2915)  
**작성자:** 크리스(운영진)  
**작성일:** 2025.04.21 14:39  
**클립 날짜:** 2025-06-30T12:45:32+09:00

## 1. 안내되는 기초 코드부터 살펴보는건 어떨까요?

대회에서 안내하는 기초 코드를 최대한 먼저 흡수하도록 노력하는 것이 좋습니다. 안내되는 코드가 최고의 점수를 내는 코드는 아닐 수 있습니다. 하지만 경험 있는 사람이 작성한 코드를 읽고 그 위에서 시작해 보는건 그 이상으로 의미가 있습니다. 길지는 않은 시간이기 때문에 시간을 아끼는 것도 중요합니다.

혹은 더 좋은 기초 코드를 발견한 뒤 기존에 작업하던 템플릿과도 어떤 차이가 있는지 비교해 보는 것만으로도 공부가 많이 됩니다. 다양한 기초 템플릿을 공부해보세요!

**참고 자료:**
- https://github.com/bentrevett/pytorch-image-classification
- https://www.kaggle.com/code/androbomb/using-cnn-to-classify-images-w-pytorch

## 2. Metric의 중요성을 간과하지 마세요!

머신러닝 모델링은 결국 주어진 문제를 해결하기 위한 것입니다. 하지만 여기서 어떤 모델을 사용하느냐가 문제를 해결하는데 핵심일까요? 어떤 모델은 당연히 중요합니다. 다만 어떻게 모델 성능을 측정하는지에 대한 이해가 선행되지 않으면 결과적으로 모래 위에 성을 쌓는 것과 같은 결과로 대회를 마무리하게 될 수도 있습니다. (점수가 중요한게 아닙니다!) 

처음에는 최대한 작은 모델로 고정해서 학습하면서 여러가지 실험을 진행해 보신 뒤에 좋은 모델로 테스트 해보는 방향을 권장드립니다.

아래 캐글 글을 참고해서 직접 confusion matrix를 시각화 해보는 것도 metric을 이해하는데 있어서도, 모델링을 하는데 있어서도 반드시 도움이 됩니다.

**참고 자료:**
- https://arize.com/blog-course/f1-score/
- https://www.kaggle.com/code/valentynsichkar/confusion-matrix-for-image-classification
- https://www.analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/#F1_Score

## 3. 주피터 노트북만을 활용해서 협업을 하는게 정답은 아닐 수 있습니다.

기초 베이스라인 코드는 주피터 노트북으로 통일해서 제공하고 있습니다. 노트북은 브라우저 기반에서 돌아가기 때문에 시각화를 잘하는데 강점이 있습니다. 다만, 협업하기 위해서는 불편함이 당연히 존재합니다. 현업에서 큰 프로젝트를 진행하고 관리할 때 노트북만을 활용하는 경우는 거의 없기 때문에 스크립트(py) 기반의 협업에도 지금부터 익숙해 지시면 더욱 좋습니다.

주피터 노트북 사용에 대한 이점과 단점에 대한 내용은 아래 글을 참고해 주세요! 노트북을 한 번 스크립트로 변경해서 팀원들과 협업을 해보는건 어떨까요? (깃헙까지 같이 활용하면 베스트!)

**참고 자료:**
- https://www.quora.com/What-are-the-pros-and-cons-of-using-Jupyter-Notebooks-for-development-in-Python-as-opposed-to-using-an-IDE
- https://pieriantraining.com/convert-jupyter-notebooks-to-py-a-beginners-guide/

## 4. 앙상블 기법은 반드시 적용해야 하는걸까요?

모델의 점수를 극한까지 끌어올리기 위해 많이 쓰는 앙상블 기법에 대해서는 알고 있을 필요가 있습니다. 조금의 점수를 올리고자 하는 것이 목표라면 앙상블 기법을 적극적으로 활용할 수도 있습니다. 하지만 앙상블이 항상 정답일까요? 몇 가지 단점이 있습니다. 이를 알고 시도하는 것과 모르고 시도하는 것에는 큰 차이가 있다고 생각합니다.

앙상블은 정확도를 올려줌과 동시에 오버피팅과 언더피팅의 리스크를 줄여줍니다. 이것은 다양한 종류의 하위 데이터셋과 피처를 활용해 추론을 한 결과입니다.

하지만 추론 결과에 대해 설명을 제공하기가 다소 복잡하고 어렵기도 합니다. 또한 연산을 앙상블 하는 모델의 개수 만큼 추가로 더 하기 때문에 computational 비용 측면에서도 단점이 있습니다. 이와 같은 단점은 현업에서 도입을 망설이게 하는 큰 요인 중 하나입니다.

앙상블 코드는 직접 작성을 해보거나 공개된 코드를 적용해 보는 쪽으로 권장드려봅니다.

**참고 자료:**
- https://medium.com/@alexppppp/how-to-train-an-ensemble-of-convolutional-neural-networks-for-image-classification-8fc69b087d3
- https://www.linkedin.com/advice/1/what-pros-cons-using-ensemble-methods-ml-skills-algorithms

## 5. 대회 참여는 처음부터 혼자만의 아이디어와 생각으로 진행하는게 좋을까요?

개인 차이가 있을 수 있지만 오랜 시간 동안 붙잡고 있었음에도 진전이 없다면 다른 사람의 도움을 얻어보는 것도 중요합니다.

도움은 어디서 얻을 수 있을까요? 동료 혹은 멘토를 통해 얻을 수도 있겠지만 모든 자료는 인터넷에 공개되어 있습니다. 예를 들어 보면 이미지 분류와 같은 태스크에서는 어떤 시도들을 해볼 수 있을까? 하는 의문에서 검색을 하다보면 아래와 같은 글을 찾을 수 있습니다. '문서' 데이터셋은 흔치 않지만 '이미지 분류' 자체는 유사한 내용으로 대회가 진행된 사례가 매우 많습니다.

**참고 자료:**
- https://www.kaggle.com/competitions/mayo-clinic-strip-ai/discussion/335726
