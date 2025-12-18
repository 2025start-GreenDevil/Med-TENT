# Fine-Tuning
Baseline model의 fine tuning을 진행하여 패혈증 예측을 실행한다.
<br>
**실험 결과**
| Method | Source Test | Target Test |
|-----|-----|-----|
|Baseline|0.82|0.51|

실험 결과, Target domain에서의 예측 AUROOC는 0.51로, Source domain에서의 예측 대비 30.1% 낮은 것을 확인할 수 있다. 이를 Med-TENT 적용 전의 실험 기준 성능으로 둔다.

## fine-tuning.ipynb
실제 finetuning을 진행하는 notebook. 모델이 패혈증을 예측할 수 있도록 학습한다.

## converted_pytorch_model.bin
Finetuning이 완료된 모델을 PyTorch로 변환하여 저장한 binary file. <br>
기존 모델은 TensrFlow library로 구성된 반면, 추후 적용할 Med-TENT는 PyTorch 라이브러리로 구성되어 있어 서로 호환되지 않는다. 이를 해결하기 위해 기존 모델을 PyTorch로 변환하여 저장한다.

## 