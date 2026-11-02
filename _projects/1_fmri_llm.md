---
layout: page
title: fMRI Representation Learning & Brain-to-Text Decoding
description: HCP task classification으로 학습한 1D CNN encoder를 GPT-2에 연결해 NSD 이미지 캡션 생성으로 확장한 프로젝트
img: https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/t1image0.png
importance: 1
category: Main
---

![fMRI 평균 BOLD 영상](https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/t1image0.png)

## 프로젝트 개요

fMRI 뇌 활동 data로 인지 조건을 분류하고, 피험자가 본 이미지의 내용을 문장으로 생성하는 **개인 수업 프로젝트**입니다. HCP betamap으로 1D CNN encoder를 학습한 뒤, 이를 GPT-2에 연결해 NSD 캡션 생성으로 확장했습니다.

- **기간:** 2026.04–2026.06
- **담당 작업:** HCP·NSD 전처리, encoder 설계·학습, Bridge와 GPT-2 연결, 결과 분석
- **주요 결과:** HCP 19개 task condition 분류에서 **test Macro-F1 0.8667**, NSD betamap 기반 캡션 생성 및 ablation

[프로젝트 GitHub](https://github.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding) · [실험 과정](https://github.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/blob/main/Docs/experiment_log.md)

## 1. 서로 다른 입력을 같은 encoder로 처리하기

HCP와 NSD의 betamap은 1차원 벡터로 사용하지만 길이가 달랐습니다. 같은 encoder에 넣기 위해 **AdaptiveAvgPool1d로 입력 길이를 100,000으로 통일**했습니다.

PCA도 고려했지만, 입력 차원에 비해 sample이 적어 제한된 data에서 추정한 주성분에 의존하기보다 별도 학습이 필요 없는 pooling을 선택했습니다. 벡터 순서에 따라 인접 구간을 평균 내면서 입력 크기를 조절하는 방식입니다.

별도로 raw BOLD 영상에 slice-timing correction, motion correction, spatial smoothing을 적용하고 GLM으로 betamap을 추출했습니다. 메모리 제약으로 coregistration과 spatial normalization은 생략했습니다. **모델 학습에는 수업에서 제공한 betamap을 사용하고, 직접 추출한 data는 ablation에 활용했습니다.**

## 2. MLP에서 CNN으로 encoder 변경

초기 MLP는 training loss가 1.0 아래로 내려가지 않았습니다. 인접한 입력 값의 패턴을 학습하는 **1D CNN**으로 변경했으며, 긴 입력을 순차 처리하는 RNN의 실행 시간과 적은 data에서 Transformer를 학습하는 부담을 고려했습니다.

최종 encoder는 convolution block 4개와 FC layer 2개로 구성했습니다. GELU, dropout, weight decay를 적용하고, 768차원 feature vector를 출력하도록 설계했습니다. Classification head는 분리해 HCP 학습 이후 제거할 수 있도록 했습니다.

![HCP classification loss curve](https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/encoder-loss.png)

**1,520개 sample을 8:1:1로 분할한 결과, test Macro-F1 0.8667을 기록했습니다.** Sample 단위 무작위 분할 기준이며, 피험자별로 분리한 평가는 아닙니다.

![HCP 19개 task condition confusion matrix](https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/encoder-cm.png)

[Confusion matrix와 분류 결과](https://github.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/blob/main/Results/Classification/Results.md)

## 3. 표현의 폭을 조정해 GPT-2에 연결

```text
NSD betamap → HCP pretrained encoder → feature vector (768)
           → Linear Bridge → brain-prefix 3개 → frozen GPT-2 → 캡션
```

Bridge는 encoder 출력을 GPT-2 입력 임베딩 공간의 벡터 3개로 변환합니다. **GPT-2는 고정하고 encoder와 Bridge를 함께 학습**해, HCP 분류에서 학습한 표현을 NSD 캡션 생성에 맞게 조정했습니다. 학습 시에는 prefix 뒤에 정답 캡션을 연결하고, 추론 시에는 prefix만 입력합니다.

초기에는 256차원 feature vector에서 여러 prefix를 만들었습니다. Prefix를 늘리면 training loss는 낮아졌지만, 실제 생성에서는 자극과 관계없는 문장과 반복 표현이 나타났습니다.

이에 출력 prefix 수뿐 아니라 **encoder와 Bridge 사이의 중간 표현을 256에서 768차원으로 확대**했습니다. 당시 실험에서는 이 변경이 생성 결과 개선의 주요 전환점이었습니다. Prefix 수는 추가 증가에 따른 개선과 학습 시간을 고려해 최종 3개로 설정했습니다.

![NSD caption decoding loss curve](https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/btt-loss.png)

## 4. 생성 결과와 ablation

아래 reference는 COCO 이미지 설명의 한국어 요약이며, 생성 문장은 실제 출력입니다. 피험자의 발화를 복원하는 과제는 아닙니다.

| Reference 요약 | Generated caption |
|---|---|
| 침대 두 개가 있는 방 | A large room with a bed and a desk. |
| 음식이 놓인 테이블에 앉은 여성 | A woman is sitting on a bench with her hands on her hips. |
| 들판에 서 있는 얼룩말 | A man is standing on a tree in the middle of the forest. |

방·침대처럼 겹치는 개념이 나타난 사례도 있었지만, 대상 불일치와 반복도 남았습니다. Zero·random 입력과 직접 전처리한 NSD 입력도 비교했으며, random 입력에서는 서로 같은 문장이 생성됐습니다.

[캡션 생성 예시와 ablation](https://github.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/blob/main/Results/Caption_examples/example.md)

## 회고

- fMRI data 전처리 중 메모리 제약으로 일부 과정을 생략했습니다. 하지만 나중에 돌아보니 중간 배열의 보관 방식과 처리 순서를 바꾸면 메모리를 더 아낄 수 있었을 것 같습니다. 다음에는 같은 이유로 필요한 과정을 생략하지 않도록 Python의 메모리 관리 방식인 reference counting과 GC 등을 공부하고 있습니다.
- 모델에 여러 방법론을 적용하는 것만으로 문제가 해결되지는 않았습니다. 모델의 어느 부분에 문제가 있는지 추정하고, 그에 맞는 해결 방법을 찾아 여러 파라미터를 바꿔보는 시행착오를 통해 모델을 개선하는 과정을 배웠습니다.
