# Rough Sketch-Simplification using Deep Learning

이 저장소에는 [Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup](https://esslab.jp/~ess/publications/SimoSerraSIGGRAPH2016.pdf) 논문의 코드가 포함되어 있습니다. 이 코드는 사용자 지정 데이터셋에서 테스트하고 훈련되었으며 PyTorch를 기반으로 합니다.



## 개요

해당 논문은 일련의 합성 연산자를 학습하여 스케치 드로잉을 단순화하는 새로운 기술을 제시합니다. 모든 차원의 이미지를 네트워크에 입력할 수 있으며, 네트워크는 입력 이미지와 동일한 차원의 이미지를 출력합니다.

![model](images/model.png)

이 아키텍처는 인코더와 디코더로 구성되어 있습니다. 첫 번째 부분은 인코더로 작용하여 이미지를 공간적으로 압축하고, 두 번째 부분은 이미지에서 핵심적인 선을 처리하고 추출하며, 세 번째 부분은 디코더로 작용하여 더 단순한 표현을 입력과 동일한 해상도의 회색조 이미지로 변환합니다. 이 모든 것은 합성곱을 사용하여 수행됩니다. 다운 및 업 샘플링 아키텍처는 간단한 필터 뱅크와 유사할 수 있습니다. 그러나 해상도가 낮은 곳에서 채널 수가 훨씬 더 크다는 점에 유의하는 것이 중요합니다. 예를 들어, 크기가 1/8일 때 1024입니다. 이를 통해 깨끗한 선으로 이어지는 정보가 저해상도 부분을 통해 전달됩니다. 네트워크는 인코더-디코더 아키텍처에 의해 전달할 정보를 선택하도록 훈련됩니다. 스트라이드가 1인 경우 패딩을 사용하여 커널 크기를 보상하고 출력이 입력과 동일한 크기가 되도록 보장합니다. 풀링 레이어는 이전 레이어의 해상도를 줄이기 위해 스트라이드를 늘린 합성곱 레이어로 대체됩니다.



## 내용

- Rough Sketch Simplification using FCNN in PyTorch

  - [개요](https://chat.openai.com/c/905726a9-018d-4e5c-b4ed-f71a564d0256#개요)

  - [내용](https://chat.openai.com/c/905726a9-018d-4e5c-b4ed-f71a564d0256#내용)

  - [1. 설정 지침 및 종속성](https://chat.openai.com/c/905726a9-018d-4e5c-b4ed-f71a564d0256#1-설정-지침-및-종속성)

  - [2. 데이터셋](https://chat.openai.com/c/905726a9-018d-4e5c-b4ed-f71a564d0256#2-데이터셋)

  - [3. 모델 훈련](https://chat.openai.com/c/905726a9-018d-4e5c-b4ed-f71a564d0256#3-모델-훈련)

  - [5. 모델 아키텍처](https://chat.openai.com/c/905726a9-018d-4e5c-b4ed-f71a564d0256#5-모델-아키텍처)

  - [6. 관측](https://github.com/kimbap918/RoughSketchSimplification-using-FCNN/edit/master/README.md#6-%EA%B4%80%EC%B8%A1)

    - [훈련](https://chat.openai.com/c/905726a9-018d-4e5c-b4ed-f71a564d0256#훈련)
    - [예측](https://chat.openai.com/c/905726a9-018d-4e5c-b4ed-f71a564d0256#예측)

  - [7. 저장소 개요](https://chat.openai.com/c/905726a9-018d-4e5c-b4ed-f71a564d0256#7-저장소-개요)

  

## 1. 설정 지침 및 종속성

로컬 머신에 저장소를 복제하세요.

```
git clone https://github.com/ishanrai05/rough-sketch-simplification-using-FCNN
```

Python3를 사용하여 가상 환경을 시작하세요.

```
virtualenv env
```

의존성을 설치하세요.

```
pip install -r requirements.txt
```

Google Colab 노트북을 사용할 수도 있습니다. 이 경우에는 저장소에 제공된 노트북을 업로드하면 됩니다.



## 2. 데이터셋

저자들은 논문에 대한 데이터셋을 제공하지 않았습니다. 따라서 직접 데이터셋을 만들었습니다. 데이터셋은 드라이브에 업로드되어 있으며, 링크는 [여기](https://drive.google.com/open?id=14NQTqITAiw8o-JgdnumQ-K0asLRwJy7q)에서 찾을 수 있습니다. 자유롭게 사용하세요.



## 3. 모델 훈련

모델을 훈련하려면 다음을 실행하세요.

```
python main.py --train=True
```

선택적 매개변수:

| argument     | default | desciption                          |
| ------------ | ------- | ----------------------------------- |
| -h, --help   |         | 도움말 메시지 표시 및 종료          |
| --use_cuda   | False   | 훈련할 장치. 기본값은 CPU입니다.    |
| --samples    | False   | 샘플 이미지 보기                    |
| --num_epochs | 10      | 훈련할 epoch 수                     |
| --train      | True    | 모델 훈련                           |
| --root       | '.'     | 입력 및 대상 이미지의 루트 디렉토리 |



## 5. 모델 아키텍처

![archi](images/archi.png)  



## 6. 관측

구글 Colab에서 Nvidia Tesla K80 GPU를 사용하여 150 epoch에 대해 약 63분이 걸립니다.

### 훈련

| Epoch | Prediction                |
| ----- | ------------------------- |
| 2     | ![epoch2](pred/2.png)     |
| 60    | ![epoch40](pred/60.png)   |
| 100   | ![epoch80](pred/100.png)  |
| 140   | ![epoch120](pred/140.png) |



### 예측

![pred1](pred/pred1.png)
![pred2](pred/pred2.png)
![pred3](pred/pred3.png)



## 7. 저장소 개요

이 저장소에는 다음 파일과 폴더가 포함되어 있습니다.

1. **notebook**: 코드용 주피터 노트북이 들어 있습니다.
2. **images**: 이미지가 들어 있습니다.
3. **pred**: 예측 이미지가 들어 있습니다.
4. `constants.py`: 훈련 중 이미지 너비와 크기.
5. `CustomDataset.py`: 데이터셋 생성용 코드입니다.
6. `model.py`: 논문에 설명된 모델 코드입니다.
7. `predict.py`: 모델을 사용하여 이미지를 간단하게 하는 함수입니다.
8. `read_data.py`: 이미지를 읽는 코드입니다.
9. `visualize.py`: 시각화용 코드입니다.
10. `utils.py`: 도우미 함수가 들어 있습니다.
11. `train.py`: 모델을 처음부터 훈련하는 함수입니다.
12. `main.py`: 모델을 실행하는 메인 코드가 들어 있습니다.
13. `requirements.txt`: 가상 환경에서 쉽게 설정할 수 있는 종속성이 나열되어 있습니다.
