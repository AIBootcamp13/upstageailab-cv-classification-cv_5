# autoDocstring - 함수 설명 주석

**Issue ID:** #9  
**Status:** Open  
**Author:** skier-song9  
**Created:** About 1 hour ago  
**Comments:** 0  
**Labels:** 문서, 설정  
**Assignees:** skier-song9  

---

## 📊 Description

**autoDocstring - 함수 설명 주석 자동 생성 도구**

VSCode, Cursor AI 등 코드 에디터에서 함수 docstring을 자동으로 생성해주는 확장 프로그램 설정 및 사용법 안내입니다.

---

## 🖥️ Installation

### 1. 확장 프로그램 설치
1. **VSCode, Cursor AI 등 코드 에디터에서 extensions 검색**
2. **"autoDocstring" 검색**
3. **설치**

---

## ⚙️ Setting

### 1. 설정 변경
![설정 이미지](https://github.com/user-attachments/assets/3cad9196-6efd-4266-8ec2-e96e7a64abef)

1. **설치가 완료되면 톱니바퀴를 클릭**
2. **"Settings" 클릭하여 설정창으로 이동**
3. **Docstring Format을 "one-line-sphinx"로 변경**

### 2. 권장 설정 옵션

**settings.json에 추가할 설정:**
```json
{
    \"autoDocstring.docstringFormat\": \"one-line-sphinx\",
    \"autoDocstring.generateDocstringOnEnter\": true,
    \"autoDocstring.includeExtendedSummary\": true,
    \"autoDocstring.includeName\": false,
    \"autoDocstring.startOnNewLine\": false
}
```

---

## 👤 Usage

### 1. 기본 사용법
1. **함수 작성 후 큰 따옴표(`\"\"\"`) 3번 작성**
2. **autoDocstring 옵션이 나타나면 Enter를 눌러 자동완성**

![사용법 이미지](https://github.com/user-attachments/assets/1d349719-0de4-492d-853c-f0df70724d98)

### 2. 사용 예시

**Before (함수만 작성된 상태):**
```python
def train_model(model, dataloader, optimizer, epochs):
    # 함수 구현 내용
    pass
```

**After (autoDocstring 적용 후):**
```python
def train_model(model, dataloader, optimizer, epochs):
    \"\"\"모델을 훈련합니다.

    :param model: 훈련할 모델
    :param dataloader: 데이터로더
    :param optimizer: 옵티마이저
    :param epochs: 훈련 에포크 수
    :return: 훈련된 모델
    \"\"\"
    # 함수 구현 내용
    pass
```

---

## 📝 Docstring 포맷 종류

### 1. one-line-sphinx (권장)
```python
def function(param1, param2):
    \"\"\"함수 설명 :param param1: 설명 :param param2: 설명 :return: 반환값 설명\"\"\"
    pass
```

### 2. sphinx
```python
def function(param1, param2):
    \"\"\"함수 설명
    
    :param param1: 설명
    :param param2: 설명
    :return: 반환값 설명
    \"\"\"
    pass
```

### 3. google
```python
def function(param1, param2):
    \"\"\"함수 설명
    
    Args:
        param1: 설명
        param2: 설명
        
    Returns:
        반환값 설명
    \"\"\"
    pass
```

### 4. numpy
```python
def function(param1, param2):
    \"\"\"함수 설명
    
    Parameters
    ----------
    param1 : type
        설명
    param2 : type
        설명
        
    Returns
    -------
    type
        반환값 설명
    \"\"\"
    pass
```

---

## 🛠️ 프로젝트 적용 가이드

### 1. 주요 함수들에 docstring 추가
```python
# config.py
def load_config():
    \"\"\"설정 파일을 로드합니다. :return: 설정 딕셔너리\"\"\"
    pass

# train_with_wandb.py
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    \"\"\"한 에포크 동안 모델을 훈련합니다. :param model: 훈련할 모델 :param dataloader: 훈련 데이터로더 :param optimizer: 옵티마이저 :param criterion: 손실 함수 :param device: 디바이스 :return: 평균 손실값과 정확도\"\"\"
    pass

# wandb_utils.py
def get_latest_runs():
    \"\"\"최신 WandB run을 조회합니다. :return: 최신 run 객체\"\"\"
    pass
```

### 2. 클래스에도 docstring 추가
```python
class CustomDataset:
    \"\"\"이미지 분류를 위한 커스텀 데이터셋 클래스\"\"\"
    
    def __init__(self, data_path, transform=None):
        \"\"\"데이터셋을 초기화합니다. :param data_path: 데이터 경로 :param transform: 데이터 변환 함수\"\"\"
        pass
```

---

## 📋 적용 체크리스트

설정 완료 후 다음 사항들을 확인하세요:

- [ ] autoDocstring 확장 프로그램 설치 완료
- [ ] Docstring Format을 "one-line-sphinx"로 설정
- [ ] `\"\"\"` 입력 시 자동완성 옵션이 나타나는지 확인
- [ ] 기존 함수들에 docstring 추가
- [ ] 새로 작성하는 함수에 docstring 습관화

---

## 🔗 References

- **autoDocstring GitHub:** [https://github.com/NilsJPWerner/autoDocstring](https://github.com/NilsJPWerner/autoDocstring)
- **VSCode Marketplace:** [autoDocstring 확장](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
- **Sphinx Documentation:** [https://www.sphinx-doc.org/](https://www.sphinx-doc.org/)

---

## 💡 추가 팁

### 1. 키보드 단축키 설정
```json
{
    \"key\": \"ctrl+shift+2\",
    \"command\": \"autoDocstring.generateDocstring\",
    \"when\": \"editorTextFocus\"
}
```

### 2. 타입 힌트와 함께 사용
```python
def process_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    \"\"\"이미지를 처리합니다. :param image: 입력 이미지 배열 :param size: 리사이즈할 크기 (width, height) :return: 처리된 이미지 배열\"\"\"
    pass
```

### 3. 복잡한 함수의 예시
```python
def train_with_validation(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    epochs: int,
    device: str = \"cuda\"
) -> dict:
    \"\"\"검증과 함께 모델을 훈련합니다. :param model: 훈련할 PyTorch 모델 :param train_loader: 훈련 데이터로더 :param val_loader: 검증 데이터로더 :param optimizer: 옵티마이저 :param criterion: 손실 함수 :param epochs: 훈련 에포크 수 :param device: 디바이스 ('cuda' 또는 'cpu') :return: 훈련 기록이 담긴 딕셔너리\"\"\"
    pass
```

---

**원본 이슈:** [GitHub에서 보기](https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5/issues/9)
