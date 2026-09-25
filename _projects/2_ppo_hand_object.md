---
layout: page
title: PPO 기반 손-물체 상호작용 모방 강화학습
description: Reward Function 설계 및 학습 로그 분석을 통한 손-물체 상호작용 모방 강화학습
importance: 2
category: Research
giscus_comments: true
---

## 프로젝트 개요

Isaac Lab 기반의 PPO 강화학습 환경에서 손-물체 상호작용 모방 과제를 수행했습니다.

제공된 PPO 알고리즘과 기본 강화학습 환경을 활용하여, 에이전트가 사람의 손 동작을 모방하면서 동시에 특정 물체에 접근하고 조작하도록 유도하는 **Reward Function을 설계하고 조정했습니다.**

구체적으로 손의 움직임, 물체의 위치와 회전, 손가락과 물체의 거리, 접촉 여부 등을 각각의 reward로 구성하고, reward 간의 weight와 exponential scale을 조정했습니다. 이후 reward log를 분석하여 에이전트가 어떤 행동을 학습하고 있는지 추정하고, Sequence별 학습 양상의 차이를 비교했습니다.

> 본 프로젝트에서는 PPO 알고리즘 자체를 구현하거나 기본 환경을 처음부터 설계하기보다는, **제공된 환경에서 reward를 설계하고 학습 결과를 분석하는 과정에 집중했습니다.**

**사용 기술:** Python, PyTorch, Isaac Lab

## 주요 담당 업무

- Hand imitation reward 설계
- Object position tracking reward 설계
- Object rotation tracking reward 설계
- Finger reach reward 설계
- Contact reward 설계
- Action regularization penalty 적용
- Reward별 weight 및 exponential scale 조정
- Observation 구성 및 rotation representation 변경
- Reward log 분석
- Sequence별 학습 양상 비교 및 분석
- Reward 설계에 따른 agent 행동 변화 관찰

## 1. 환경 및 Observation 구성

강화학습 환경에서 agent에게 제공되는 observation은 손과 물체의 현재 상태뿐만 아니라 reference의 상태와 일부 미래 상태를 포함하도록 구성되어 있었습니다.

Observation에는 다음과 같은 물리량이 포함됩니다.

- 현재 hand의 위치
- 현재 hand의 회전
- 현재 hand의 속도
- 현재 object의 위치
- 현재 object의 회전
- Reference hand/object의 물리량
- 다음 frame의 reference 물리량
- Fingertip 위치

기존 환경에서 회전 정보는 quaternion 형태로 표현되어 있었습니다. 저는 quaternion을 `6D rotation representation`으로 변환하여 observation에 포함했습니다. Quaternion 대신 6D representation을 사용하여 회전 표현의 연속성을 고려하고자 했습니다.

## 2. Intermediate Value 계산

Reward를 계산하기 전에 학습에 필요한 주요 물리량을 각 environment별로 계산하여 저장하도록 구현했습니다.

특히 다음과 같은 값을 계산했습니다.

- 현재 object 위치와 reference object 위치 사이의 거리
- 현재 hand keypoint와 reference hand keypoint 사이의 거리
- Object tracking에 필요한 거리 정보
- Hand imitation에 필요한 거리 정보

또한 기존 환경에서 `early_terminate`와 관련된 변수는 업데이트하는 부분이 명확하지 않았기 때문에, object tracking 거리 또는 hand keypoint tracking 거리가 0.3m보다 커지는 상황을 감지하도록 구현했습니다.

## 3. Reward Function 설계

본 프로젝트에서 가장 중점적으로 수행한 부분입니다.

에이전트가 단순히 손의 움직임만 모방하는 것이 아니라, **손 동작을 모방하면서 물체를 실제로 접근하고 조작하도록 유도하는 것**을 목표로 reward를 구성했습니다.

### 3.1 Hand Imitation Reward

현재 손의 keypoint와 reference 손 keypoint 사이의 L2 distance를 계산하여 두 손의 움직임이 가까워지도록 reward를 구성했습니다.

손의 각 keypoint가 reference에 가까워질수록 더 높은 reward를 얻도록 하여 에이전트가 사람의 손 동작을 모방하도록 유도했습니다.

### 3.2 Object Position Tracking Reward

현재 object의 위치와 reference object의 위치 사이의 L2 distance를 이용하여 reward를 구성했습니다.

전체 task에서 물체의 위치를 정확하게 따라가는 것이 중요한 목표라고 판단하여 높은 weight를 부여했습니다.

초기에는 exponential function의 scale을 -1보다 작은 값으로 설정했습니다. 그러나 이 경우 object와의 거리가 어느 정도 이상 떨어져 있으면 reward가 지나치게 작아져, agent가 물체에 가까워지기 전까지 유의미한 reward signal을 얻기 어려웠습니다.

이를 개선하기 위해 exponential scale을 조정했고, 최종적으로 `-20`을 사용했습니다.

### 3.3 Object Rotation Tracking Reward

현재 object와 reference object의 회전 상태를 비교하여 두 상태가 일치하도록 reward를 구성했습니다.

Object rotation을 quaternion으로 표현하고 두 회전 상태의 차이를 계산하여, reference와 가까운 회전 상태를 갖도록 유도했습니다.

### 3.4 Finger Reach Reward

손가락 끝과 현재 object 사이의 L2 distance를 이용하여 손가락이 물체에 접근하도록 reward를 구성했습니다.

초기에는 exponential function의 scale을 -1보다 작은 값으로 설정했지만, 물체에 접근할수록 더 높은 reward를 얻는다는 관계를 agent가 학습하기까지 오랜 시간이 걸렸습니다.

따라서 scale을 조정하여 최종적으로 `-10`을 사용했습니다.

### 3.5 Contact Reward

손가락 끝에 가해지는 contact force를 이용하여 물체와의 접촉을 유도하는 reward를 구성했습니다.

손가락 끝에 가해지는 힘이 클수록 더 높은 reward를 제공하도록 구성했으며, 특히 Sequence 3과 같이 면적이 넓은 물체를 안정적으로 잡는 상황에서 활용했습니다.

### 3.6 Action Regularization Penalty

Agent가 지나치게 큰 action을 사용하거나 손을 과도하게 움직이는 현상을 줄이기 위해 action regularization penalty를 적용했습니다.

주된 목적은 학습 과정에서 발생하는 손의 **jittering을 완화하는 것**이었습니다.

| Weight | 관찰 결과 |
| --- | --- |
| 0.01 | Jittering 억제가 충분하지 않음 |
| 0.03 | 최종 선택 |
| 0.05 | Penalty가 과도하여 학습이 제한됨 |

Weight가 0.01인 경우 penalty가 충분히 크지 않아 jittering이 계속 나타났습니다. 반대로 0.05에서는 action에 대한 penalty가 지나치게 커져 학습 자체가 제대로 진행되지 않았습니다.

최종적으로 0.03을 사용했으나, jittering을 완전히 해결하지는 못했습니다.

## 4. Reward 계산 및 Scaling

각 reward와 penalty를 적절한 weight와 함께 결합하여 최종 reward를 계산했습니다.

거리 기반 reward의 경우 distance의 norm을 계산한 뒤 음의 값을 곱하고 exponential function에 넣는 형태를 사용했습니다. 이러한 형태는 distance가 작아질수록 1에 가까워지고, distance가 커질수록 0에 가까워지는 특성을 갖기 때문에 거리 기반 task에서 reward scale을 조정하기 용이했습니다.

Contact reward에는 `tanh`를 사용했고, Action regularization에는 norm 자체를 penalty로 사용했습니다.

또한 여러 reward와 penalty를 결합하는 과정에서 reward가 음수가 되지 않도록 `clamp`를 적용했습니다.

초기 실험에서는 과도한 penalty로 인해 negative reward가 발생하면서 agent가 positive reward를 얻기보다 negative reward를 피하는 방향으로 행동하는 문제가 관찰되었습니다. 이를 완화하기 위해 penalty의 크기를 줄이고 reward를 0 이상으로 제한했습니다.

## 5. Reward Log 분석

Reward Function을 수정한 이후에는 최종 reward 값만 확인하지 않고, 각 reward 항목의 변화를 별도로 기록하여 학습 과정을 분석했습니다.

특히 다음 reward의 변화 양상을 비교했습니다.

- Hand imitation
- Object position tracking
- Object rotation tracking
- Finger reach
- Contact
- Action regularization

이를 통해 특정 reward가 상승하는 시점과 다른 reward가 상승하는 시점을 비교하고, agent가 어떤 행동을 학습하고 있는지 추정했습니다.

### Sequence 1

초기 약 2000 epoch까지는 여러 물리량과 reward에서 큰 진폭이 나타났지만 뚜렷한 상승세가 나타나지 않았습니다.

특히 Finger reach reward가 낮은 반면 Hand imitation reward가 상대적으로 높게 나타나는 현상을 확인했습니다. 이를 통해 agent가 물체에 접근하기보다는 제자리에서 손의 움직임을 따라하는 행동에 머무르고 있을 가능성을 추정했습니다.

약 2000~2500 epoch 구간에서는 여러 reward가 동시에 상승하기 시작했습니다. 특히 Contact와 Finger reach reward가 함께 증가하는 모습을 통해 agent가 물체에 접근하기 시작한 것으로 추정했습니다.

반면 이 시점에서 Hand imitation reward는 감소하는 모습을 보였습니다. 이를 통해 기존에 손 동작 자체를 모방하는 데 집중하던 행동에서 벗어나 물체에 접근하고 조작하는 방향으로 학습이 변화한 것으로 해석했습니다.

약 2500 epoch 이후에는 Object position 및 Object rotation reward가 다른 항목에 비해 더욱 가파르게 상승하는 모습을 확인했습니다. 이를 통해 해당 구간에서는 agent가 물체의 위치와 회전 상태를 reference에 맞추는 데 집중하는 것으로 추정했습니다.

### Sequence 2

Sequence 2에서는 Sequence 1보다 초기부터 물체에 빠르게 접근하는 양상이 나타났습니다.

초반부터 Contact와 Object position reward가 빠르게 증가하는 모습을 통해 agent가 비교적 빠르게 물체에 접근하고 물체를 조작하는 행동을 학습한 것으로 추정했습니다.

약 2000~4800 epoch에서는 reward가 전체적으로 평탄해지는 구간이 나타났지만, reward의 변동성이 크게 나타났습니다.

반면 Hand imitation reward는 낮은 수준에 머무르는 모습을 보여, 물체를 조작하는 행동에 비해 사람의 손 동작을 정확하게 모방하는 능력은 상대적으로 부족한 것으로 추정했습니다.

Sequence 2는 5000 epoch까지 학습하지 못하고 4800 epoch에서 종료되었습니다. AWS instance의 사용 시간이 만료되면서 학습이 중단되었기 때문입니다.

### Sequence 3

Sequence 3에서는 초기 약 1000 epoch까지 Finger reach reward를 통해 물체에 접근하는 행동이 빈번하게 나타났습니다.

그러나 Object position reward가 낮은 상황에서도 Hand imitation reward가 상대적으로 높게 나타났습니다. 이를 통해 agent가 물체 근처까지 접근한 이후 실제 물체 조작보다는 손 동작을 따라하는 행동에 머무르는 구간이 있었던 것으로 추정했습니다.

약 1000 epoch 이후에는 이러한 경향이 감소하면서 Object position 및 Contact reward가 상승하기 시작했습니다. 이를 통해 agent가 물체를 안정적으로 잡고 조작하는 방향으로 학습하기 시작한 것으로 추정했습니다.

특히 Sequence 3에서는 Sequence 2보다 Contact reward가 크게 나타났습니다. Sequence 3의 물체가 태블릿 형태로 넓은 면적을 갖고 있었으며, 이를 안정적으로 잡기 위해 상대적으로 강한 접촉이 필요했기 때문일 가능성이 있다고 분석했습니다.

반면 Object rotation reward는 약 4000 epoch 부근에서야 상승하기 시작했습니다. 이를 통해 태블릿 형태의 물체에서는 위치를 조정하는 것보다 회전하거나 뒤집는 동작이 더 어려웠을 가능성이 있다고 분석했습니다.

## 6. 주요 실험 및 관찰

### Reward Scale 조정

Object position tracking과 Finger reach reward의 exponential scale을 조정하면서 초기 상태에서 reward signal이 지나치게 작아지는 문제를 개선했습니다.

특히 물체와의 거리가 큰 초기 상태에서도 agent가 물체에 접근하는 행동을 학습할 수 있도록 reward scale을 조정했습니다.

### Action Regularization Weight 조정

Action regularization weight를 0.01, 0.03, 0.05로 비교했습니다.

- 너무 작은 penalty → jittering 억제 부족
- 적절한 penalty → 학습 진행과 action 안정성 사이의 절충
- 너무 큰 penalty → action 자체가 제한되어 학습 저해

최종적으로 0.03을 사용했습니다.

### Reward Log 기반 행동 분석

Reward 항목 간의 상승 및 감소 시점을 비교하여 agent의 행동 변화를 추정했습니다.

특히 Hand imitation reward와 Finger reach/Object tracking reward 사이의 차이를 통해, agent가 손 동작을 모방하는 것과 실제 물체를 조작하는 것 사이에서 어떤 행동을 학습하고 있는지 분석했습니다.

## 7. 한계

본 프로젝트에서는 PPO 알고리즘과 기본 환경을 직접 구현하기보다는 **제공된 PPO 환경에서 Reward Function을 설계하고 조정하는 역할에 집중했습니다.**

따라서 PPO의 optimizer, policy architecture, rollout mechanism 등의 알고리즘 자체를 개발한 프로젝트는 아닙니다.

또한 Action regularization을 적용하여 jittering을 완화하고자 했지만 이를 완전히 해결하지는 못했습니다.

Reward log를 통해 agent의 행동 양상을 추정할 수 있었지만, 로그만으로 특정 행동의 원인을 직접적으로 증명하기에는 한계가 있었습니다.

예를 들어 Hand imitation reward가 높고 Object tracking reward가 낮다는 현상으로부터 agent가 손의 자세 모방에 치우쳐 있을 가능성을 추정할 수는 있었지만, reward log만으로 agent 내부의 학습 원인을 확정할 수는 없었습니다.

## 8. 회고

본 프로젝트를 통해 강화학습에서 Reward Function이 단순히 agent의 성능을 평가하는 지표가 아니라, **agent가 어떤 행동을 학습하도록 유도하는지를 결정하는 핵심적인 설계 요소**라는 점을 경험했습니다.

특히 하나의 task를 여러 개의 reward로 분해하는 과정에서 각 reward가 서로 독립적으로 작동하는 것이 아니라 서로 경쟁하거나 보완할 수 있다는 점을 관찰했습니다.

예를 들어 Hand imitation reward를 높이는 행동이 반드시 Object tracking reward를 높이는 행동과 일치하지 않았으며, 실제 학습 과정에서 agent가 손 동작을 모방하는 데 집중하면서 물체에 접근하지 않는 양상이 나타나기도 했습니다.

또한 reward의 weight나 exponential scale을 단순히 수치적으로 조정하는 것에 그치지 않고, reward log의 변화 양상을 확인하면서 **왜 특정 행동이 나타났는지를 추정하고 다음 실험의 방향을 결정하는 과정**을 경험했습니다.

이를 통해 강화학습 실험에서 단순히 최종 reward가 높은 모델을 선택하는 것뿐만 아니라, 각 reward가 agent의 행동에 어떤 영향을 미치는지 분석하는 것이 중요하다는 점을 배웠습니다.
