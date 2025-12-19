# Med-TENT: Source-Free Domain Adaptation을 활용한 전자 의료 기록(EHR) 텍스트 기반 질병 예측 모델의 병원 간 데이터 공유 한계 극복 및 성능 개선
**Med-TENT**는 **전자 의료 기록(EHR)**을 활용하여 다른 병원으로부터 공유받은 질병 예측 모델을 **원본 학습 데이터 없이(source-free)** 보유하고 있는 데이터만으로 모델의 예측 성능을 유지시킬 수 있도록 제안한 **도메인 적응(Domain Adaptation)** 기법이다. <br>
<br>
본 연구에서는 패혈증 예측을 위해 텍스트 기반 전자의료기록 데이터로 모델을 학습하고, Med-TENT 적용 전후 성능 변화를 분석하였다. 그 결과, Med-TENT 적용 전 대비 모델의 평균 예측 AUROC가 평균 3%, AUPRC는 평균 4% 향상되었다.

<br>

## 연구 배경
의료 AI 모델을 서로 다른 병원끼리 공유하는 시나리오에는 다음과 같은 문제점이 존재한다.

### 데이터 이질성에 따른 병원 간 Domain Shift
병원마다 데이터 측정 기기 및 파라미터, 수집한 환자 집단 등의 차이로 데이터 분포의 차이, Domain Shift가 발생한다. 이는 한 병원에서 빌드한 모델을 다른 병원에서 적용하는 경우에도 나타난다.

### 제한적인 의료 데이터 공유
HIPAA, GDPR, PIPA 등 각국의 개인정보보호 규제에 의해 개인의 건강 정보 및 읠 데이터는 직접적인 공유가 어렵다. <br>
<br>
위의 요인들은 **공유된 의료 AI 모델의 예측 성능 저하** 문제로 이어진다.

<br>

## 실험 결과
### AUROC
| Method | Source Test | Target Test |
|-----|:-----:|:-----:|
|Baseline|0.82|0.51|
|**Med-TENT(ours)**|0.82|**0.54 (최대 0.64)**|

### AUPRC
| Method | Source Test | Target Test |
|-----|:-----:|:-----:|
|Baseline|0.43|0.19|
|**Med-TENT(ours)**|0.43|**0.23 (최대 0.30)**|

<br>
Med-TENT 적용 후, 기존 baseline 대비 AUROC는 평균 3%, 최대 10% 향상하였다. AUPRC는 baseline 대비 평균 4%, 최대 11%의 성능 향상을 보였다.

<br>

## Reproducing Med-TENT
각 단계는 해당하는 ipynb 파일을 호환되는 환경(e.g., Google Colab, Jupyter Notebook)에서 전체 실행하여 재생성 가능하다.

### Preprocessing
```
open Preprocessing/01_AddYearOffset.ipynb
run > run all(ctrl+F9)
...
Repeat for all ipynb files seqentially.
```

### Pretraining
```
open Pretraining/pretraining.ipynb
run > run all(ctrl+F9)
```

### Finetuning
```
open Fine-Tuning/fine-tuning.ipynb
run > run all(ctrl+F9)
```

<br>

## Testing Med-TENT
Med-TENT 적용 실험을 통해 Target Domain(병원 ID 175) 대상 예측 성능을 출력한다.
```
open test/test.ipynb
run > run all(ctrl+F9)
```

<br>

## 코드 체계
```
project/
│
├── v1  # 초기 버전
│
├── v2  # 현재 버전
│   ├── data
│   |
│   ├── Preprocessing
│   |
│   ├── Pretraining
│   |
│   ├── Fine-Tuning
│   |
│   ├── medtent.py  # Med-TENT 알고리즘이 정의된 소스코드
│   ├── Med-TENT.ipynb  # Med-TENT 적용 실험 코드
│   └── ...
│
└── README.md

```
자세한 소스코드 설명은 각 directory 내에서 확인 가능하다.

<br>
 
## 사용 소프트웨어 및 패키지
### Baseline Model
[Med-BERT](github.com/ZhiGroup/Med-BERT) <br>

### Dataset
[eICU Collaborative Research Database](eicu-crd.mit.edu)

<br>

## 포스터
[medtent_poster](medtent_poster.png)