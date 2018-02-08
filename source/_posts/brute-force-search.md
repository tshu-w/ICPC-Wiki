---
title: Brute-force Search
toc: false
date: 2017-09-12 18:31:22
categories: Basics
tags:
  - brute-force
  - search
---

## 排列枚举

```c++
do {

} while (next_permutation(A, A + n)); // prev_permutation
```


## 子集枚举

```c++
int sub = sup;
do {
    sub = (sub - 1) & sup;
} while (sub != sup); // -1 & sup = sup;
```


## 势为 k 的集合枚举
```c++
int comb = (1 << k) - 1;
while (comb < 1 << n) {
    int x = comb & -comb, y = comb + x;
    comb = ((comb & ~y) / x >> 1) | y;
}
```

## 高维前缀和(子集/超集和)

```c++
// 子集和
for (int i = 0; i < k; i++)
    for (int s = 0; s < 1 << k; s++)
        if (s >> i & 1) cnt[s] += cnt[s ^ (1 << i)];
```

```c++
// 超集和
for (int i = 0; i < k; i++)
    for (int s = 0; s < 1 << k; s++)
        if (!(s >> i & 1)) cnt[s] += cnt[s | (1 << i)];
```
