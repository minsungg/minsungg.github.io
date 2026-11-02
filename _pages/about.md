---
layout: about
title: Minsung Ryu
permalink: /
subtitle: Korea University · Data Science
profile:
  align: right
  image: My_Face.jpg
  image_circular: false
selected_papers: false
social: true
announcements:
  enabled: false
latest_posts:
  enabled: false
nav: false
nav_order: 1
---

### Mathematics × Machine Learning

**수학적 이해를 바탕으로 모델을 구현하고, 실험으로 검증하는 데이터사이언스 전공 유민성입니다.** AI/ML 모델 연구·개발 분야의 겨울 인턴십을 희망합니다.

부동소수점 환경에서의 신경망 표현력을 연구하며, fMRI encoder와 GPT-2를 연결한 캡션 생성, 양자화 LLM을 활용한 리뷰 분석을 수행했습니다. 모델이 왜 작동하는지 이해하는 데서 출발해, 직접 구현하고 실험 결과와 한계를 분석하는 데 관심이 있습니다.

---

## Research Interests

<div class="mb-4">
  <h4>ML on Discrete Structures</h4>
  <p>
    ML 이론에서 표현력과 generalization bound는 주로 실수 연산을 가정해 연구됩니다.
    반면 실제 컴퓨터는 Fixed-point 및 Floating-point과 같이 유한한 정밀도를 가진 이산 구조에서 연산합니다.
    이러한 구조에서 머신러닝 모델이 어떻게 작동하는지, 표현력과 계산 효율성에 어떤 차이가 나타나는지 연구하고 있습니다.
  </p>
</div>

---

## Selected Projects

<div class="row row-cols-1 row-cols-md-2">
  <div class="col mb-4">
    <div class="card h-100">
      <div class="card-body">
        <h5 class="card-title">fMRI Representation Learning &amp; Brain-to-Text Decoding</h5>
        <p class="card-text">
          HCP task classification으로 학습한 1D CNN encoder를 GPT-2에 연결해 NSD 이미지 캡션 생성으로 확장한 프로젝트
        </p>
        <a href="{{ '/projects/1_fmri_llm/' | relative_url }}">View project →</a>
      </div>
    </div>
  </div>

  <div class="col mb-4">
    <div class="card h-100">
      <div class="card-body">
        <h5 class="card-title">부동소수점 신경망의 표현력</h5>
        <a href="{{ '/projects/3_floating_point_nn/' | relative_url }}">View project →</a>
      </div>
    </div>
  </div>
</div>

<div class="text-center mt-2">
  <a href="{{ '/projects/' | relative_url }}">View all projects →</a>
</div>
