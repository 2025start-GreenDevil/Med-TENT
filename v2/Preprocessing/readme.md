# Preprocessing
데이터 전처리를 진행하는 소스코드이다.

## AddYearOffset.ipynb
입원의 total offset을 계산하여 입원 년도(admit year) 행을 추가한다.

## SelectRandomStay.ipynb
한 환자에 대해 여러 개의 입원 기록이 존재하는 경우, 입원 기록의 순서를 추정한다. 추정 불가능한 경우, 임의의 한 입원 기록을 선택한다.

## AddRandomdate.ipynb
baseline model의 입력 형식을 맞추기 위해 입/퇴원 날짜 행을 추가한다.

## UnifyICDCodes.ipynb
혼용된 두 버전의 진단코드 ICD-9 code와 ICD-10 code를 ICD-9 기준으로 통일한다. 단, ICD-10만 존재하는 경우 그 진단 코드를 유지한다.

## code_processing.ipynb
예측 대상 질병인 패혈증(Sepsis)의 진단 코드 '038.9'를 기준으로 라벨링한다. <br>
이때, 추후 모델이 해당 코드 존재 여부로 질병을 예측하지 않도록 라벨링 후 해당 코드를 코드 시퀀스에서 삭제한다.