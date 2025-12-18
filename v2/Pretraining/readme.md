# Pretraining
실험의 Baseline model인 [Med-BERT](https://github.com/ZhiGroup/Med-BERT)의 pretraining 단계이다. <br>
BERT 모델이 데이터를 통해 진단 시퀀스를 학습하는 단계이다.

## config.json
Pretraining의 output file로, BERT 모델 구조와 하이퍼파라미터가 정의된 설정 파일

## create_BERTpretrain_EHRfeatures.py
Training에 필요한 feature를 추출하는 스크립트

## modeling.py
BERT 아키텍처를 정의하는 모델 클래스와 레이어 구현 스크립트

## optimization.py
학습에 사용되는 옵티마이저와 학습 스케줄러를 정의한 스크립트

## preprocess_pretrain_data.py
데이터를 모델에 입력하기 위한 전처리 함수를 정의한 스크립트

## pretraining.ipynb
실제 model pretraining을 진행하는 notebook

## run_EHRpretraining.py
pretraining 함수를 정의한 스크립트

## run_EHRpretraining_utils.py
데이터 로딩, 전처리, 손실 계산 등의 보조 함수들이 포함된 스크립트