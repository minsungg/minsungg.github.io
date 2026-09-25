---
layout: page
title: fMRI 기반 인지 조건 분류 및 이미지 캡션 생성
description: fMRI beta map을 활용한 CNN 기반 encoder 설계 및 실험
img:
importance: 1
category: Research
---

# fMRI 기반 인지 조건 분류 및 이미지 캡션 생성
## 1. 프로젝트 개요
fMRI beta map을 활용하여 뇌 활동 데이터를 표현하는 딥러닝 encoder를 설계하고, 학습된 표현을 서로 다른 두 가지 과제에 적용한 프로젝트입니다.
- **HCP 데이터셋:** fMRI beta map을 활용한 19개 인지 조건 분류
- **NSD 데이터셋:** fMRI beta map과 사전 학습된 언어 모델을 연결한 이미지 캡션 생성
먼저 HCP 데이터셋을 사용하여 인지 조건 분류 모델을 학습했습니다. 이후 classification head를 제거하고, encoder가 추출한 feature를 bridge network를 통해 GPT-2의 입력 임베딩 공간으로 변환했습니다.
이를 바탕으로 fMRI beta map에서 이미지 캡션을 생성하는 모델을 구성하고, encoder 구조와 feature 차원에 따른 학습 및 생성 결과의 변화를 실험했습니다.
---
# 2. fMRI 데이터 전처리 및 입력 표현
## 2.1 전처리 과정
HCP 및 NSD의 fMRI 데이터에 대해 다음과 같은 전처리 과정을 수행했습니다.
1. NIfTI 형식의 4D fMRI 데이터 로딩
2. Slice-timing correction
3. Rigid-body 기반 head motion correction
4. Framewise displacement 계산
5. Spatial smoothing
6. First-level GLM을 통한 beta map 추출
7. Brain mask를 활용한 유효 voxel 추출
8. 3차원 beta map을 1차원 벡터로 변환
HCP 데이터에서는 TR 0.72초를 사용했으며, spatial smoothing에는 6.0mm FWHM을 적용했습니다.
First-level GLM을 사용하여 자극 조건별 beta map을 추출하고, 이를 딥러닝 encoder의 입력으로 사용했습니다.


## 2.2 입력 차원 통일
### 문제
Beta map은 1차원 벡터 형태로 변환되었지만, 데이터에 따라 입력 벡터의 길이가 달라질 수 있었습니다.
또한 원본 beta map을 그대로 사용하면 입력 차원이 매우 커져 encoder의 입력 크기를 일관되게 유지하기 어려웠습니다.
### 해결 방법
`nn.AdaptiveAvgPool1d`를 사용하여 beta map의 입력 차원을 **100,000차원으로 통일**했습니다.
이를 통해 서로 다른 길이의 입력 벡터를 동일한 encoder 구조에 전달할 수 있도록 구성했습니다.
다만 이 과정은 실제 뇌의 3차원 공간 구조를 보존하는 방법은 아닙니다. 1차원으로 변환된 beta map의 순서를 기준으로 구간별 평균을 계산하기 때문에, voxel 간의 실제 공간적 인접성을 보장하지 못합니다.
따라서 본 프로젝트에서는 다음과 같은 가정을 두었습니다.
> 1차원으로 변환된 beta map에서도 인접한 값 사이에 학습 가능한 패턴이 존재할 수 있다고 가정하고, 1D CNN을 통해 해당 패턴을 추출했습니다. 이후 Adaptive Average Pooling을 사용하여 입력 크기를 통일했습니다.
---
# 3. 전처리 과정에서의 한계 및 회고
전처리 과정에서는 로컬 환경의 메모리 제약으로 인해 일부 표준적인 fMRI 전처리 과정을 수행하지 못했습니다.
특히 **coregistration과 spatial normalization은 메모리 할당 문제로 생략**했습니다. 대신 MNI152 brain mask를 functional image의 공간에 맞게 resampling한 후, 해당 mask를 사용하여 beta map을 추출했습니다.
또한 전체 데이터를 한 번에 메모리에 로드하기 어려워 HCP 및 NSD 데이터 처리 과정에서 로드할 파일 수와 세션 수를 제한했습니다. 이로 인해 주어진 raw 데이터를 충분히 활용하지 못했으며, 모델링 과정에서는 이미 전처리된 데이터를 사용하는 방식으로 진행했습니다.
이 과정에서 다음과 같은 한계를 확인했습니다.
- 메모리 제약으로 인한 Coregistration 및 spatial normalization 생략
- 그로 인한 Subject 및 session 간 voxel 위치 대응의 불확실성
이 경험을 통해 모델 구조뿐 아니라 **데이터 로딩 방식, 중간 데이터 관리, 메모리 사용량을 고려한 파이프라인 설계가 중요하다**는 점을 배웠습니다.
이후 Python의 오브젝트 라이프 사이클과 generator class를 학습하며 대규모 데이터를 처리할 때 불필요한 객체 유지와 중간 결과 저장을 줄이는 메모리 관리 방법을 공부했습니다. 다만 이러한 학습은 본 프로젝트의 전처리 문제를 완전히 해결한 결과라기보다는, 향후 대규모 데이터 처리 파이프라인을 개선하기 위한 방향을 탐색한 과정이었습니다.
---
# 4. HCP 인지 조건 분류 Encoder 설계
## 4.1 MLP 기반 초기 모델
처음에는 beta map을 입력으로 받는 MLP 기반 모델을 구성했습니다.
그러나 학습 과정에서 loss가 충분히 감소하지 않았으며, Q3에서 training loss가 약 1.0 이하로 감소하지 못하는 문제가 발생했습니다.
이를 통해 입력의 모든 값을 독립적으로 처리하는 MLP 구조만으로는 beta map에 존재할 수 있는 국소적 패턴을 효과적으로 학습하기 어려울 수 있다고 판단했습니다.
## 4.2 1D CNN 도입
Beta map은 1차원 벡터로 표현되어 있었지만, 인접한 입력 값 사이에 일정한 패턴이 존재할 가능성을 고려했습니다.
이에 따라 1D CNN을 도입하여 입력의 국소적 패턴을 추출하도록 encoder를 변경했습니다.
CNN을 선택한 이유는 다음과 같습니다.
- 인접한 입력 값 사이의 패턴 학습
- Convolution을 통한 효율적인 feature 추출
- RNN보다 간단한 구조
- Transformer보다 상대적으로 낮은 계산 비용
- 제한된 데이터 환경에서 모델 복잡도 조절
초기 CNN encoder는 3개의 convolution layer와 2개의 fully connected layer로 구성했으며, 최종 feature 차원은 256으로 설정했습니다.




---
# 5. NSD 전이 과정에서의 Overfitting 문제
HCP 인지 조건 분류를 위해 학습한 encoder를 NSD 이미지 캡션 생성 과제에 적용하는 과정에서 overfitting 문제가 관찰되었습니다.
초기 encoder는 256차원의 feature를 출력했으며, 이를 bridge network를 통해 GPT-2의 입력 차원인 768로 변환했습니다.
그러나 초기 모델에서는 fMRI feature와 언어 표현 사이의 일반적인 관계를 학습하기보다 학습 데이터의 caption 패턴을 암기하는 양상이 나타났습니다.
주요 문제는 다음과 같았습니다.
- 학습 데이터에 대한 과적합
- 입력 feature와 GPT-2 임베딩 공간 사이의 변환 과정에서 발생하는 병목 가능성
- 학습 데이터와 유사한 문장의 반복적인 생성
- fMRI 입력과 관련성이 낮은 caption 생성
특히 256차원의 feature를 통해 여러 개의 768차원 brain token을 생성해야 했기 때문에, encoder가 추출한 표현이 bridge network에서 지나치게 압축된 형태로 사용될 가능성을 고려했습니다.
---
# 6. Encoder 구조 변경 및 일반화 성능 개선
## 6.1 Feature 차원 확장
초기 encoder의 feature 차원은 256이었지만, 이후 이를 **768차원으로 확장**했습니다.
이는 GPT-2의 임베딩 차원과 동일한 크기의 표현 공간을 encoder 단계에서 확보하여, bridge network가 수행해야 하는 차원 확장 부담을 줄이기 위한 변경이었습니다.
기존 구조에서는 256차원의 feature를 기반으로 여러 개의 768차원 brain token을 생성해야 했습니다. 반면 feature 차원을 768로 확장함으로써 encoder가 더 높은 차원의 표현을 학습할 수 있도록 구성했습니다.
이 변경 이후 GELU, Dropout, convolution layer 증가 등의 구조 변경과 함께 초기 모델에서 나타났던 반복적 caption 생성 및 과적합 양상이 완화되는 것을 관찰했습니다.
따라서 feature 차원 확장은 다음과 같은 목적을 가졌습니다.
- 저차원 feature bottleneck 완화
- fMRI 표현의 정보 손실 가능성 감소
- Bridge network의 차원 변환 부담 완화
- downstream caption generation task에 대한 표현력 및 일반화 가능성 개선
단, 이러한 개선이 feature 차원 확장만으로 발생했다고 단정하기는 어렵습니다. GELU, Dropout, convolution block 증가 및 brain token 개수 조정이 함께 적용되었기 때문에, 성능 변화는 여러 구조적 변경의 결과로 해석해야 합니다.
## 6.2 Convolution Block 증가
기존 3개의 convolution layer를 4개로 변경했습니다.
이를 통해 입력 beta map에서 보다 다양한 수준의 feature를 추출하고, encoder의 표현 능력을 확장하고자 했습니다.
## 6.3 활성화 함수 변경
기존 ReLU 활성화 함수를 GELU로 변경했습니다.
GELU를 적용하여 활성화 함수의 비선형성을 변경하고, feature 학습 과정에서의 표현 방식과 일반화 성능 변화를 확인했습니다.
다만 GELU가 항상 ReLU보다 우수한 일반화 성능을 보장하는 것은 아니며, 본 프로젝트에서는 다른 구조 변경과 함께 적용한 실험적 요소로 해석했습니다.
## 6.4 Dropout 및 Weight Decay 적용
과적합을 완화하기 위해 convolution block과 fully connected layer에 Dropout을 적용했습니다.
- Convolution block Dropout: 0.1
- Fully connected layer Dropout: 0.3
- Adam optimizer의 Weight Decay 적용
Dropout과 Weight Decay를 통해 모델이 학습 데이터의 특정 패턴을 과도하게 암기하는 것을 줄이고, 새로운 입력에 대한 표현의 일반화 가능성을 높이고자 했습니다.
---
# 7. 최종 Encoder 구조
최종 encoder는 다음과 같이 구성했습니다.
- 4개의 `Conv1d` layer
- Batch Normalization
- GELU activation
- Dropout
- Max Pooling
- Adaptive Average Pooling
- Fully Connected Layer
- 768차원 feature vector
Encoder의 출력은 HCP 데이터셋에서 19개 인지 조건 분류에 사용했으며, 이후 classification head를 제거하고 NSD 이미지 캡션 생성 모델의 입력 feature로 재사용했습니다.
---
# 8. HCP 인지 조건 분류 실험
HCP beta map을 사용하여 19개 인지 조건을 분류하는 모델을 학습했습니다.
## 실험 과정
1. HCP beta map 데이터 로딩
2. 19개 인지 조건에 해당하는 데이터 구성
3. Adaptive Average Pooling을 통한 입력 차원 통일
4. 학습·검증·테스트 데이터 분할
5. CNN encoder와 classification head 공동 학습
6. Cross-entropy loss를 사용한 분류 학습
7. Macro-F1 및 confusion matrix를 활용한 평가
8. t-SNE를 통한 feature 분포 시각화
9. 학습된 encoder weight 저장
10. NSD 캡션 생성 과제에 encoder 재사용
Encoder와 classification head를 함께 학습한 후, downstream task에서 재사용하기 위해 classification head를 제거하고 encoder의 feature extractor 부분을 활용했습니다.
---
# 9. fMRI Encoder와 GPT-2 연결
HCP 분류를 통해 학습한 encoder에서 classification head를 제거했습니다.
이후 encoder가 출력하는 768차원 feature를 bridge network에 입력하여 GPT-2의 입력 임베딩 차원에 맞는 brain token으로 변환했습니다.
```plain text
fMRI beta map
      ↓
fMRI Encoder
      ↓
768차원 feature
      ↓
Bridge Network
      ↓
3개의 brain token
      ↓
GPT-2
      ↓
생성된 이미지 캡션
```
Bridge network는 encoder의 feature를 GPT-2의 입력 임베딩 공간으로 변환하도록 구성했습니다.
GPT-2의 파라미터는 고정하고, fMRI encoder와 bridge network를 학습하는 방식을 적용했습니다.
HCP 분류에서 학습한 feature가 NSD 캡션 생성에 최적화된 표현이라고 보기는 어려웠기 때문에, NSD 과제에서는 encoder도 함께 fine-tuning했습니다.
---
# 10. Caption 생성 과정에서의 문제와 실험
## 문제
초기 실험에서는 brain token 개수가 증가할 때 training loss가 감소하는 양상이 나타났습니다.
그러나 loss 감소가 생성 결과의 개선으로 직접 이어지지는 않았습니다. 평가 과정에서는 학습 데이터와 유사한 문장을 반복하거나, fMRI 입력과 관련성이 낮은 caption을 생성하는 문제가 관찰되었습니다.
이는 모델이 fMRI feature와 caption 사이의 의미적 대응 관계를 충분히 학습하지 못하고, caption의 통계적 패턴을 학습했을 가능성을 보여주었습니다.
## 구조 변경
이 문제를 개선하기 위해 다음과 같은 구조 변경을 수행했습니다.
- Brain token 개수를 3개로 조정
- Encoder feature 차원을 256차원에서 768차원으로 확장
- GELU 활성화 함수 적용
- Convolution block 및 fully connected layer에 Dropout 적용
- Weight Decay 적용
- GPT-2 파라미터 고정
- Encoder와 bridge network 공동 fine-tuning
특히 256차원에서 768차원으로 feature 차원을 확장하여 encoder 내부에서 보다 높은 차원의 표현을 학습할 수 있도록 했습니다.
그 결과 초기 모델에서 나타났던 반복적인 caption 생성 및 과적합 양상이 완화되었으며, 이전 설정에 비해 생성 결과의 일반화 양상이 개선되는 것을 관찰했습니다.
다만 제한된 데이터와 학습 자원으로 인해 이러한 개선을 정량적으로 충분히 검증하지는 못했습니다.
---
# 11. 평가 및 한계

모델은 다음과 같은 방법으로 평가했습니다.
- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- METEOR
- ROUGE-L
- CIDEr
- 생성된 caption 샘플 확인
초기 모델에서는 학습 데이터에 대한 과적합과 반복적인 caption 생성 문제가 관찰되었습니다.
이후 feature 차원을 768로 확장하고, GELU 및 Dropout을 적용한 구조에서는 초기 설정에 비해 생성 결과가 개선되는 양상이 나타났습니다.
그러나 다음과 같은 한계가 존재합니다.
- 제한된 학습 데이터 및 학습 시간
- 전처리 단계에서 coregistration 및 spatial normalization 생략
- 1차원 beta map 표현에 따른 공간 정보 손실
- 제한된 데이터로 인한 모델 일반화 성능 검증 부족
- Caption 생성 결과와 fMRI feature 간 의미적 대응 관계에 대한 정량적 검증 부족
따라서 본 프로젝트의 결과를 **fMRI에서 언어를 성공적으로 복원했다**고 표현하는 것은 적절하지 않습니다.
보다 정확한 결론은 다음과 같습니다.
> fMRI beta map에서 feature를 추출하고, 이를 사전 학습된 언어 모델과 연결하여 이미지 caption을 생성하는 전체 파이프라인을 구현했습니다. 초기 MLP 기반 구조에서 1D CNN으로 변경하고, convolution block 증가, GELU 및 Dropout 적용, feature 차원 확장 등의 실험을 수행했습니다. 특히 encoder feature 차원을 256차원에서 768차원으로 확장하여 feature bottleneck을 완화하고, 초기 모델에서 나타난 반복적인 caption 생성 및 과적합 양상이 개선되는 것을 관찰했습니다.
	그러나 전처리 과정에서의 공간 정렬 제한, 제한된 학습 자원 및 데이터 규모로 인해 fMRI 표현과 언어 표현 사이의 안정적인 대응 관계를 충분히 검증하지는 못했습니다. 향후에는 공간 정규화가 적용된 데이터와 더 큰 학습 규모를 활용하고, caption 생성 성능과 fMRI feature의 의미적 대응 관계를 정량적으로 평가할 필요가 있습니다.

## \[Sample 1\]<br>• Image/fMRI ID : Batch 0, Item 0<br>• Reference: several red ripe apples in two large wooden boxes<br>• Decoded Pred: A large bowl of rice and a large bowl of noodles.<br>• Metrics : BLEU-4: 0.0200 \| METEOR: 0.2381 \| ROUGE-L: 0.1667
## \[Sample 2\]<br>• Image/fMRI ID : Batch 0, Item 1<br>• Reference: A woman sitting at a table with plates of food.<br>• Decoded Pred: A woman is sitting on a bench with her hands on her hips.<br>• Metrics : BLEU-4: 0.0209 \| METEOR: 0.2174 \| ROUGE-L: 0.1429
## \[Sample 3\]<br>• Image/fMRI ID : Batch 0, Item 2<br>• Reference: Room with wood covered wall and floor with two beds.<br>• Decoded Pred: A large room with a bed and a desk.<br>• Metrics : BLEU-4: 0.0279 \| METEOR: 0.2632 \| ROUGE-L: 0.2000
## \[Sample 4\]<br>• Image/fMRI ID : Batch 0, Item 3<br>• Reference: a green field that has a man with a kite<br>• Decoded Pred: A man standing on a tree.<br>• Metrics : BLEU-4: 0.0324 \| METEOR: 0.3125 \| ROUGE-L: 0.2857
## \[Sample 5\]<br>• Image/fMRI ID : Batch 0, Item 4<br>• Reference: Black and white photograph of many people in a train station.<br>• Decoded Pred: A man sitting on a chair with his hands on his knees.<br>• Metrics : BLEU-4: 0.0223 \| METEOR: 0.2273 \| ROUGE-L: 0.1538
## \[Sample 6\]<br>• Image/fMRI ID : Batch 0, Item 5<br>• Reference: A seagull rests on a concrete wall, with the ocean and a lighthouse in the background.<br>• Decoded Pred: A man is standing on a tree in the middle of the road.<br>• Metrics : BLEU-4: 0.0209 \| METEOR: 0.2174 \| ROUGE-L: 0.1429
## \[Sample 7\]<br>• Image/fMRI ID : Batch 0, Item 6<br>• Reference: A zebra is standing in a field and looking right at the camera.<br>• Decoded Pred: A man is standing on a tree in the middle of the forest.<br>• Metrics : BLEU-4: 0.0209 \| METEOR: 0.2174 \| ROUGE-L: 0.1429
## \[Sample 8\]<br>• Image/fMRI ID : Batch 0, Item 7<br>• Reference: A beige and white cow with horns in the vegetation.<br>• Decoded Pred: A woman sitting on a bench with her legs spread out on the floor.<br>• Metrics : BLEU-4: 0.0197 \| METEOR: 0.2083 \| ROUGE-L: 0.1333
## \[Sample 9\]<br>• Image/fMRI ID : Batch 0, Item 8<br>• Reference: Two women sit on a sidewalk near a street with a train passing by.<br>• Decoded Pred: A man standing on a tree in the middle of the forest.<br>• Metrics : BLEU-4: 0.0223 \| METEOR: 0.2273 \| ROUGE-L: 0.1538
## \[Sample 10\]<br>• Image/fMRI ID : Batch 0, Item 9<br>• Reference: A large vanity mirror mounted above a sink in a bathroom.<br>• Decoded Pred: A large kitchen with a large kitchen sink.<br>• Metrics : BLEU-4: 0.0306 \| METEOR: 0.2778 \| ROUGE-L: 0.2222
BLEU-1: 0.1622<br>BLEU-2: 0.0623<br>BLEU-3: 0.0389<br>BLEU-4: 0.0255<br>METEOR: 0.2511<br>ROUGE-L: 0.1886
