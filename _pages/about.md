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

머신러닝의 수학적 이론과 데이터 기반 문제 해결에 관심이 있는 데이터사이언스 전공 학생입니다.

모델을 단순히 사용하는 데 그치기보다, **왜 특정한 구조가 작동하는지, 어떤 조건에서 더 나은 표현과 학습이 가능한지**를 수학적으로 이해하는 데 관심이 있습니다.

---

## Research Interests

<div class="mb-4">
  <h4>ML on discrete structures</h4>
  <p>
    머신러닝의 표현력(expressivity)과 일반화 경계(generalization bound)는 주로 실수 연산을 가정해 분석됩니다.
    반면 실제 컴퓨터는 고정소수점(fixed-point) 및 부동소수점(floating-point)과 같이 유한한 정밀도를 가진 이산 구조에서 연산합니다.
  </p>
  <p>
    이러한 구조에서도 머신러닝 모델이 어떻게 작동하는지, 표현력과 계산 효율성에 어떤 차이가 나타나는지 연구하고 있습니다.
  </p>
</div>

<div class="mb-4">
  <h4>Theoretical Foundations of Machine Learning</h4>
  <p>
    머신러닝의 표현력(expressivity), 최적화(optimization), 일반화(generalization)를 이해하기 위해 수학 공부와 이론 연구를 병행하고 있습니다.
  </p>
  <p>
    특히 Universal Approximation Theorem과 double descent phenomenom 등을 수학적으로 이해하는 것을 목표로 합니다.
  </p>
</div>

---

## Selected Projects

<div class="row row-cols-1 row-cols-md-2">
  <div class="col mb-4">
    <div class="card h-100">
      <div class="card-body">
        <h5 class="card-title">fMRI 기반 인지 조건 분류</h5>
        <p class="card-text">
          fMRI beta map을 활용한 CNN 기반 encoder 설계 및 실험.
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
