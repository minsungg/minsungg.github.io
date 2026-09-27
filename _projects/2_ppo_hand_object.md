---
layout: page
title: PPO 기반 손-물체 상호작용 모방 강화학습
description: Reward Function 설계 및 학습 로그 분석을 통한 손-물체 상호작용 모방 강화학습
img:
importance: 2
category: Main
--

**사용 기술:** Python, PyTorch, Isaac Lab

## 프로젝트 개요

**수업 프로젝트**로, 제공된 Isaac Lab 기반 PPO 환경과 기본 알고리즘을 활용하여 손-물체 상호작용 모방 과제를 수행했습니다.

본 프로젝트에서 제가 주로 담당한 부분은 **Observation 구성, Reward Function 설계 및 조정, 학습 로그 분석**입니다. PPO 알고리즘이나 기본 환경 자체를 처음부터 구현한 프로젝트는 아닙니다.

---

## 1. Observation 구성

Agent가 손과 물체의 상태를 reference와 비교할 수 있도록 다음 정보를 observation에 포함했습니다.

- Hand 위치·회전·속도
- Object 위치·회전
- Reference hand/object 상태
- 다음 frame의 reference 상태
- Fingertip 위치

기존 quaternion 회전 표현을 **6D rotation representation**으로 변환하여 observation에 사용했습니다.

또한 object 위치와 hand keypoint가 reference에서 크게 벗어나는 경우를 감지하기 위해 관련 거리를 계산하고, 0.3m를 초과하면 early termination하도록 구성했습니다.

---

## 2. Reward Function 설계

손의 움직임뿐 아니라 물체를 실제로 접근하고 조작하도록 유도하기 위해 여러 reward를 구성했습니다.

### Hand imitation
현재 hand keypoint와 reference keypoint 사이의 L2 distance를 이용해 손 동작을 모방하도록 구성했습니다.

### Object tracking
Object의 reference 대비 위치와 회전을 추적하도록 reward를 구성했습니다.

### Finger reach
Fingertip과 object 사이의 거리를 이용해 손가락이 물체에 접근하도록 유도했습니다.

### Contact
Fingertip contact force를 이용해 물체와의 접촉을 유도했습니다.

### Action regularization
과도한 action으로 인한 jittering을 줄이기 위해 action penalty를 추가했습니다.

Reward 간의 weight와 exponential scale을 조정하며 학습 양상을 비교했습니다. 특히 object position과 finger reach reward의 scale을 각각 **-20, -10**으로 조정했고, action regularization weight는 **0.01, 0.03, 0.05**를 비교하여 최종적으로 0.03을 사용했습니다.

---

## 3. 학습 결과 분석

최종 reward만 비교하는 대신 reward별 로그를 기록하여 agent의 행동 변화를 분석했습니다.

예를 들어 다음과 같은 관계를 비교했습니다.

- Hand imitation ↑ / Object tracking ↓
- Finger reach ↑ / Contact ↑
- Object position 및 rotation reward의 상승 시점

이를 통해 agent가 손 동작을 모방하는 단계에서 물체에 접근하고 조작하는 단계로 행동을 변화시키는 양상을 추정했습니다.

Sequence별로 학습 양상도 비교했습니다. 일부 sequence에서는 물체 접근과 contact가 빠르게 증가한 반면, 다른 sequence에서는 hand imitation에 비해 object tracking이 늦게 증가하는 등의 차이가 나타났습니다.

다만 reward log만으로 agent의 내부 학습 원인을 직접 증명할 수는 없으므로, 분석 결과는 **관찰된 reward 변화에 기반한 추정**으로 해석했습니다.

---

## 4. 한계

본 프로젝트의 주요 한계는 다음과 같습니다.

- PPO 알고리즘 자체를 구현하지 않고 제공된 알고리즘을 사용
- Isaac Lab의 기본 환경 및 데이터가 제공된 상태에서 실험 수행
- Action regularization을 적용했지만 jittering을 완전히 해결하지 못함
- Reward log만으로 agent 행동의 원인을 직접적으로 증명하기 어려움

이를 통해 강화학습에서는 reward의 최종 값뿐 아니라 **각 reward가 agent의 행동을 어떻게 유도하는지 분석하는 과정이 중요하다**는 점을 확인했습니다.
