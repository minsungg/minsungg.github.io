---
layout: page
title: projects
permalink: /projects/
description: 프로젝트 및 연구 활동
nav: true
nav_order: 2
horizontal: false
---

<div class="projects">

<h2>Main Projects</h2>
<div class="row row-cols-1 row-cols-md-3">
{% for project in site.projects %}
  {% if project.category == "Main" %}
    {% include projects.liquid %}
  {% endif %}
{% endfor %}
</div>

<h2 class="mt-5">Sub Projects</h2>
<div class="row row-cols-1 row-cols-md-3">
{% for project in site.projects %}
  {% if project.category == "Sub" %}
    {% include projects.liquid %}
  {% endif %}
{% endfor %}
</div>

</div>
