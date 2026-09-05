---
layout: page
title: fMRI 생체 신호-자연어 간 멀티모달(Encoder-LLM) 번역 모델 구축
description: fMRI 영상 전처리 및 Encoder 구조 설계를 통한 뇌 활성화 영역 자연어 변환
img: assets/img/fmri_thumbnail.png
importance: 1
category: Research / AIrelated_publications: true
---

## Executive Summary
- **기간 / 성격:** 2025.04 ~ 2025.06
- **주요 역할:** fMRI 데이터 전처리 파이프라인 구축, Encoder-LLM 연동 및 파인튜닝
- **Key Tech Stack:** PyTorch, Hugging Face Transformers, Nilearn, Python, HCP Dataset, NSD Dataset

---

## 1. Problem (문제 정의 및 연구 배경)
- fMRI 생체 신호 데이터는 고차원(High-dimensional) 스파스 데이터로, 강한 노이즈와 개인별 차이가 존재하여 직접적인 자연어 변환에 한계가 있음.
- 이를 극복하기 위해 생체 신호의 특징 추출(Feature Extraction)을 위한 전용 Encoder 파이프라인을 설계하고, 사전 학습된 LLM의 latent space와 정렬(Alignment)하는 구조를 기획함.

## 2. Action (엔지니어링 & 구현 과정)
### ① fMRI 생체 신호 전처리 파이프라인
- Signal-to-Noise Ratio(SNR) 개선을 위해 temporal filtering 및 spatial smoothing 적용.
- 대용량 3D/4D fMRI voxel 데이터의 메모리 병목을 해소하기 위해 PyTorch DataLoader 기반 커스텀 텐서 파이프라인 구현.

### ② Encoder-LLM 크로스 도메인 아키텍처 설계
- 뇌 활성화 영역의 시공간 패턴을 압축 표현하는 Encoder 모델 구조 구현.
- Projection Layer를 통해 Encoder의 output 스페이스를 LLM의 토큰 임베딩 영역으로 맵핑.

## 3. Result & Insight (성과 및 한계점)
- 피실험자의 특정 시각/언어 자극에 따른 뇌 활성 영역을 구체적인 문장(자연어)으로 재구성.
- **배운 점:** 이종(Heterogeneous) 데이터 간 latent alignment 시 발생할 수 있는 Representation Collapse 현상을 경험하고, 콘트라스티브 손실 함수(Contrastive Loss) 적용을 통해 안정화하는 노하우 습득.

---
## 🔗 Links & Code
- [GitHub Repository](https://github.com/minsungg/DataScienceApplicationII)
