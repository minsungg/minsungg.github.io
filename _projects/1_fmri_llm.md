---
layout: page
title: fMRI 기반 인지 조건 분류 및 이미지 캡션 생성
description: fMRI beta map을 활용한 CNN 기반 encoder 설계 및 실험
img: /assets/img/fmri.jpg
importance: 1
category: Main
---

![NSD와 fMRI 데이터 예시](/assets/img/fmri.jpg)

## 프로젝트 개요

**수업 프로젝트**로, 제공된 HCP·NSD fMRI 데이터셋과 실험 환경을 활용하여 fMRI beta map을 입력으로 받는 encoder를 설계하고, 이를 인지 조건 분류와 이미지 캡션 생성에 적용했습니다.

- **HCP:** fMRI beta map을 이용한 19개 인지 조건 분류
- **NSD:** fMRI feature와 사전 학습 언어 모델을 연결한 이미지 캡션 생성

제가 주로 수행한 작업은 **fMRI 입력 전처리, 1D CNN encoder 설계 및 구조 변경, GPT-2와의 연결, 실험 결과 분석**입니다.

---

## 1. 데이터 전처리 및 입력 구성

HCP와 NSD의 fMRI 데이터에 대해 다음 전처리를 수행했습니다.

1. NIfTI 4D 데이터 로딩
2. Slice-timing correction
3. Rigid-body motion correction
4. Framewise displacement 계산
5. Spatial smoothing
6. First-level GLM을 통한 beta map 추출
7. Brain mask를 이용한 voxel 추출
8. 1차원 벡터 변환

HCP에서는 TR 0.72초, spatial smoothing 6.0mm FWHM을 사용했습니다.

Beta map의 입력 길이를 통일하기 위해 AdaptiveAvgPool1d를 사용하여 **100,000차원**으로 변환했습니다. 이 과정에서는 beta map을 1차원으로 펼치기 때문에 실제 voxel의 3차원 공간적 인접성이 보존되지 않는다는 한계가 있습니다.

또한 로컬 환경의 메모리 제약으로 **coregistration과 spatial normalization을 생략**하고, MNI152 brain mask를 functional image 공간에 맞춰 resampling하여 사용했습니다.

---

## 2. HCP 인지 조건 분류

초기에는 MLP 기반 encoder를 구성했으나 학습이 충분히 진행되지 않았습니다. 이에 beta map의 인접한 입력 값에서 패턴을 추출하기 위해 **1D CNN encoder**를 적용했습니다.

초기 CNN은 3개의 convolution layer와 2개의 fully connected layer, 256차원 feature로 구성했습니다.

이후 실험에서 다음과 같이 구조를 변경했습니다.

- Convolution block: 3개 → 4개
- Feature dimension: 256 → **768**
- ReLU → **GELU**
- Dropout 및 Weight Decay 적용
- Max Pooling 및 Adaptive Average Pooling 적용

최종 encoder는 768차원 feature를 출력하도록 구성했고, HCP의 19개 인지 조건 분류에 사용했습니다. Macro-F1, confusion matrix, t-SNE 등을 이용해 결과를 분석했습니다.

---

## 3. NSD 이미지 캡션 생성

HCP 분류에서 학습한 encoder의 classification head를 제거하고, encoder feature를 NSD 캡션 생성에 재사용했습니다.

```
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
이미지 caption
```

GPT-2는 고정하고 encoder와 bridge network를 fine-tuning했습니다.

초기 모델에서는 학습 데이터에 대한 과적합과 반복적인 caption 생성이 관찰되었습니다. 이를 완화하기 위해 feature dimension 확장, convolution block 증가, GELU, Dropout, Weight Decay 등의 구조 변경을 실험했습니다.

구조 변경 이후 초기 설정에 비해 반복적인 caption 생성이 완화되는 양상을 관찰했지만, 제한된 데이터와 학습 자원으로 인해 개선 효과를 충분히 정량적으로 검증하지는 못했습니다.

---

## 4. 결과 및 한계

최종 caption generation 결과는 다음과 같습니다.

| Metric | Score |
|---|---:|
| BLEU-1 | 0.1622 |
| BLEU-2 | 0.0623 |
| BLEU-3 | 0.0389 |
| BLEU-4 | 0.0255 |
| METEOR | 0.2511 |
| ROUGE-L | 0.1886 |

본 프로젝트에서는 fMRI beta map에서 feature를 추출하고 이를 GPT-2와 연결하는 전체 파이프라인을 구현했습니다. 다만 다음과 같은 한계가 있습니다.

- Coregistration 및 spatial normalization 생략
- 1차원 입력 표현에 따른 공간 정보 손실
- 제한된 데이터 및 학습 자원
- fMRI feature와 caption의 의미적 대응 관계에 대한 정량적 검증 부족

따라서 본 결과를 fMRI에서 언어를 성공적으로 복원한 것으로 해석하기보다는, **fMRI representation을 언어 모델과 연결하는 방법을 설계하고 실험한 프로젝트**로 보는 것이 적절합니다.
