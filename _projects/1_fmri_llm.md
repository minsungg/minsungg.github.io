---
layout: page
title: fMRI Representation Learning & Brain-to-Text Decoding
description: HCP task classification으로 학습한 1D CNN encoder를 GPT-2에 연결해 NSD 이미지 캡션 생성으로 확장한 프로젝트
img: https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/t1image0.png
importance: 1
category: Main
---

**HCP betamap으로 인지 조건을 분류하는 encoder를 학습하고, 이를 GPT-2에 연결해 NSD 이미지 캡션 생성으로 확장했습니다.**

| 항목 | 내용 |
|---|---|
| 활동·기간 | 개인 수업 프로젝트 · 2026.04–2026.06 |
| 팀·역할 | 1명 · 전처리, 모델 설계·학습, 결과 분석 |
| 데이터 | HCP·NSD fMRI 및 COCO 캡션 |

![fMRI 평균 BOLD 영상](https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/t1image0.png)

[프로젝트 GitHub](https://github.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding) · [실험 과정](https://github.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/blob/main/Docs/experiment_log.md)

## 문제 — 서로 다른 fMRI 입력을 분류와 캡션 생성으로 연결하기

HCP의 task condition을 분류하는 encoder를 학습하고, 피험자가 본 이미지의 내용을 설명하는 NSD 캡션 생성에 재사용하고자 했습니다. HCP와 NSD의 betamap은 길이가 달랐고, 분류용 feature를 GPT-2 입력으로 변환하는 과정도 필요했습니다.

## 접근 — 입력 길이 통일과 encoder–GPT-2 연결

- **전처리:** raw BOLD 영상에 slice-timing correction, motion correction, spatial smoothing을 적용하고 GLM으로 betamap을 추출했습니다. 메모리 제약으로 coregistration과 spatial normalization은 생략했습니다. 모델 학습에는 수업에서 제공한 betamap을 사용하고, 직접 추출한 data는 ablation에 활용했습니다.
- **입력 길이 통일:** `AdaptiveAvgPool1d`로 길이를 100,000으로 맞췄습니다. PCA도 고려했지만 입력 차원에 비해 sample이 적어, 별도 학습 없이 인접 구간을 평균 내는 pooling을 선택했습니다.
- **Encoder 설계:** convolution block 4개와 FC layer 2개로 768차원 feature vector를 출력하도록 구성했습니다. Classification head는 분리해 HCP 학습 이후 제거할 수 있도록 했습니다.
- **GPT-2 연결:** 단일 linear layer인 Bridge로 feature를 각 768차원의 brain-prefix 3개로 변환했습니다. GPT-2는 고정하고 encoder와 Bridge를 함께 학습했습니다. 학습 시에는 prefix 뒤에 정답 캡션을 연결하고, 추론 시에는 prefix만 입력했습니다.

```text
NSD betamap → HCP pretrained encoder → feature vector (768)
           → Linear Bridge → brain-prefix 3개 → frozen GPT-2 → 캡션
```

## 시행착오 — encoder 구조와 중간 표현의 폭 조정

- **MLP에서 CNN으로 변경:** 초기 MLP는 training loss가 1.0 아래로 내려가지 않았습니다. 인접한 입력 값의 패턴을 학습하는 1D CNN으로 변경했습니다. 긴 입력을 순차 처리하는 RNN의 실행 시간과 적은 data에서 Transformer를 학습하는 부담을 고려했습니다.
- **정규화와 구조 조정:** dropout과 weight decay를 적용하고, convolution block을 3개에서 4개로 늘리며 ReLU를 GELU로 변경했습니다.
- **Feature dimension 확대:** 초기 256차원 feature에서 prefix를 늘리면 training loss는 낮아졌지만, 실제 생성에서는 자극과 관계없는 문장과 반복 표현이 나타났습니다. Encoder와 Bridge 사이의 중간 표현을 768차원으로 확대했으며, 당시 실험에서는 이 변경이 생성 결과 개선의 주요 전환점이었습니다.
- **Prefix 수 조정:** prefix 추가에 따른 개선과 전체 학습 시간을 고려해 최종 3개로 설정했습니다.

## 결과 — HCP 분류와 NSD 캡션 생성

### 1. HCP 19개 task condition 분류

![HCP classification loss curve](https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/encoder-loss.png)

**1,520개 sample을 8:1:1로 분할한 결과, test Macro-F1 0.8667을 기록했습니다.** Sample 단위 무작위 분할 기준이며, 피험자별로 분리한 평가는 아닙니다.

![HCP 19개 task condition confusion matrix](https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/encoder-cm.png)

[Confusion matrix와 분류 결과](https://github.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/blob/main/Results/Classification/Results.md)

### 2. NSD 캡션 생성과 ablation

![NSD caption decoding loss curve](https://raw.githubusercontent.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/main/Docs/Images/btt-loss.png)

아래 reference는 COCO 이미지 설명의 한국어 요약이며, 생성 문장은 실제 출력입니다. 피험자의 발화를 복원하는 과제는 아닙니다.

| Reference 요약 | Generated caption |
|---|---|
| 침대 두 개가 있는 방 | A large room with a bed and a desk. |
| 음식이 놓인 테이블에 앉은 여성 | A woman is sitting on a bench with her hands on her hips. |
| 들판에 서 있는 얼룩말 | A man is standing on a tree in the middle of the forest. |

방·침대처럼 겹치는 개념이 나타난 사례도 있었지만, 대상 불일치와 반복도 남았습니다. Zero·random 입력과 직접 전처리한 NSD 입력도 비교했으며, random 입력에서는 서로 같은 문장이 생성됐습니다.

[캡션 생성 예시와 ablation](https://github.com/minsungg/fMRI-Representation-Learning-and-Brain-to-Text-Decoding/blob/main/Results/Caption_examples/example.md)

## 나의 기여

개인 프로젝트로 **HCP·NSD 전처리, 입력 길이 통일, encoder 구조 설계·변경, Bridge와 GPT-2 연결, 모델 학습 및 분류·생성 결과와 ablation 분석**을 수행했습니다.

## 회고

- fMRI data 전처리 중 메모리 제약으로 일부 과정을 생략했습니다. 하지만 나중에 돌아보니 중간 배열의 보관 방식과 처리 순서를 바꾸면 메모리를 더 아낄 수 있었을 것 같습니다. 다음에는 같은 이유로 필요한 과정을 생략하지 않도록 Python의 메모리 관리 방식인 reference counting과 GC 등을 공부하고 있습니다.
- 모델에 dropout, weight decay 등 여러 방법론을 적용하는 것만으로 문제가 해결되지는 않았습니다. 모델의 어느 부분에 문제가 있는지 추정하고, 그에 맞는 해결 방법을 찾아 여러 파라미터를 바꿔보는 시행착오를 통해 모델을 개선하는 과정을 배웠습니다.
