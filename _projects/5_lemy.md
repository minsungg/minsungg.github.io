---
layout: page
title: Lemy - 게임 개발
description: 레거시 코드 개선, 기능 구현 및 성능 최적화
img:
importance: 5
category: Project
---

## Lemy
**기간:** 2025.01 \~ 2026.09
**팀 규모:** 10명
**담당:** 레거시 코드 개선, 기능 구현 및 성능 최적화
**Tech Stack:** C#, Unity 2D

### 대량 장판 오브젝트로 인한 프레임 성능 저하 개선
#### 문제
2D 탑뷰 액션 게임에서 몬스터와 장판 오브젝트가 동시에 많이 생성될 때 프레임 드랍이 발생했습니다.
기존에는 `GameObject`와 `Collider2D`, `OnTriggerStay2D`를 이용해 장판의 충돌을 처리하고 있었으며, 대량의 장판이 생성되는 상황에서 물리 처리 및 오브젝트 관리 비용이 증가했습니다.
#### 해결
- 기존 `GameObject` 및 `Collider2D` 기반 장판 처리 구조를 데이터 기반 방식으로 변경
- `BurningFieldData` 구조체를 활용해 장판의 위치, 지속 시간 등 필요한 데이터 관리
- 물리 트리거 대신 거리 제곱(`sqrMagnitude`)을 활용한 직접 충돌 판정 구현
- 개별 장판 오브젝트 대신 `ParticleSystem.Emit`을 활용해 시각 효과 처리
#### 결과
동일한 테스트 환경에서 변경 전후 성능을 측정했습니다.
<table header-row="true">
<tr>
<td>측정 항목</td>
<td>변경 전</td>
<td>변경 후</td>
<td>변화</td>
</tr>
<tr>
<td>Physics2D 처리 시간(평균)</td>
<td>6.41ms</td>
<td>2.55ms</td>
<td>약 60.2% 감소</td>
</tr>
<tr>
<td>전체 프레임 시간(평균)</td>
<td>40.02ms</td>
<td>31.73ms</td>
<td>약 20.7% 감소</td>
</tr>
<tr>
<td>GC Alloc(평균)</td>
<td>34.59KB</td>
<td>28.17KB</td>
<td>약 18.6% 감소</td>
</tr>
</table>
#### 배운 점
물리 엔진의 편리한 기능을 그대로 사용하는 것보다, 게임 내 오브젝트의 처리 특성에 맞춰 충돌 판정과 데이터 관리를 직접 설계하는 것이 성능에 영향을 줄 수 있음을 경험했습니다.
또한 프로파일러를 활용해 성능 문제를 추측에 의존하지 않고 수치로 비교하는 과정을 익혔습니다.
