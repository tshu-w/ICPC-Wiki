---
title: Binary Search
toc: true
date: 2017-09-04 19:47:57
math: true
categories: Basics
tags:
  - binary-search
---


## 最小化最大值

区间设定左开右闭——(lb, ub]

```c++
while (ub - lb > 1) {
    int mid = (ub - lb) / 2 + lb;
    if (C(mid)) ub = mid;
    else lb = mid;
}
return ub;
```


## 最大化最小值

区间设定左闭右开——[lb, ub)

```c++
while (ub - lb > 1) {
    int mid = (ub - lb) / 2 + lb;
    if (C(mid)) lb = mid;
    else ub = mid;
}
return lb;
```


## 浮点数

浮点数二分一般直接指定循环次数（100次）作为终止条件。1次循环可以把区间的范围缩小一半，100次的循环则可以达到 $2^{−100}≈10^{−30}$ 的精度范围。（如果设置终止条件为 $ub - lb > eps$ 可能因为 $eps$ 取太小浮点小数精度的原因导致陷入死循环。)

```c++
for (int i = 0; i < 100; i++) {
    double mid = (lb + ub) / 2;
    if (C(mid)) lb = mid;
    else ub = mid;
}
return lb;
```
