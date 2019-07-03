---
title: Mo's Algorithm
toc: false
date: 2018-08-02 21:12:09
categories: Others
tags:
  - mo
---

转载自[莫队算法学习笔记](https://blog.sengxian.com/algorithms/mo-s-algorithm)

## 概述

莫队算法是由莫涛提出的算法，可以解决一类离线区间询问问题，适用性极为广泛。同时将其加以扩展，便能轻松处理树上路径询问以及支持修改操作。

## 形式

#### 普通莫队

对于序列上的区间询问问题，如果从 $[l, r]$ 的答案能够 $O(1)$ 扩展到 $[l-1, r], [l+1, r], [l, r+1], [l, r-1]$ 的答案，那么可以在 $O(n\sqrt n)$ 的复杂度内求出所有询问的答案。

**实现：**离线后排序，顺序处理每个询问，暴力从上一个区间的答案转移到下一个区间答案。

**排序方法：**设定块的长度为 $S$ ，按照二元组 $(\lfloor \frac{l}{S} \rfloor, r)$ 从小到大排序。

**复杂度分析：**设序列长度为 $n$，询问个数为 $m$。可以发现从 $(l_1, r_1)$ 转移到 $(l_2, r_2)$ 的代价为他们之间的曼哈顿距离。对于每一个询问序列中的每一个块（第一关键字相同），整个块内纵坐标最多变化 $n$ 长度（纵坐标必然单调不减），对于每个询问，横坐标最多变化 $S$ 。一共有 $\frac{n}{S}$ 个块，相邻块之间转移的复杂度为 $O(n)$ ，所以复杂度为 $O(\frac{n^{2}}{S}+mS+\frac{n^{2}}{S})$，不妨让 $n, m$ 同阶，取 $S=\sqrt{n}$ 时可达到最优复杂度 $O(n\sqrt{n})$。

```cpp
int l = 0, r = 0, cur = 0, SIZE;

inline void move(int x, int y) {
    // update current answer
}
void solve() {
    SIZE = int(ceil(pow(n, 0.5)));
    sort(querys, querys + m);
    for (int i = 0; i < m; ++i) {
        const query &q = querys[i];
        while (l > q.l) move(-1, 0);
        while (r < q.r) move(0, 1);
        while (l < q.l) move(1, 0);
        while (r > q.r) move(0, -1);
        ans[q.id] = cur;
    }
}
```

[BZOJ 2038](http://www.lydsy.com/JudgeOnline/problem.php?id=2038)

#### 带修改莫队

考虑普通莫队加入修改操作，如果修改操作可以 $O(1)$ 的应用以及撤销（同时也要维护当前区间的答案），那么可以在 $O(n^\frac{5}{3})$ 的复杂度内求出所有询问的答案。

**实现**：离线后排序，顺序遍历询问，先将时间转移到当前询问的时间，然后再像普通莫队一样转移区间。

**排序方法：**设定块的长度为 $S_{1}$ 和 $S_{2}$，按照 $(\lfloor \frac{l}{S_{1}} \rfloor, \lfloor \frac{r}{S_{2}} \rfloor,t)$ 的三元组小到大排序，其中 $t$ 表示这个询问的时刻之前经历过了几次修改操作。

**复杂度分析：**考虑询问序列中的每个小块，小块内每个询问的一二关键字相同。在这个小块内，显然 $t$ 最多变化 $m$ ，对于每个询问，$l,r$最多变化 $S_{1}$ 和 $S_{2}$，一共有 $\frac{n^{2}}{S_{1}S_{2}}$ 个这样的块，相邻块之间转移的复杂度为 $O(n)$，总复杂度就是 $O(mS_{1}+mS_{2}+\frac{n^{2}m}{S_{1}S_{2}}+\frac{n^{2}m}{S_{1}S_{2}})$，不妨设 $n, m$ 同阶，取 $S_{1}=S_{2}=n^{\frac{2}{3}}$ 时可达到最优复杂度 $O(n^{\frac{5}{3}})$。

```cpp
int l = 0, r = 0, t = 0, cur = 0;

inline void move(int x, int y) {
    // update current answer
}

inline void moveTime(int t, int sign) {
    // apply or revoke modification
    // update current answer
}

void solve() {
    BLOCK_SIZE = int(ceil(pow(n, 2.0 / 3)));
    sort(querys, querys + m);
    for (int i = 0; i < q1; ++i) {
        const query &q = querys[i];
        while (t < q.t) moveTime(t++, 1);
        while (t > q.t) moveTime(--t, -1);
        while (l < q.l) move(1, 0);
        while (l > q.l) move(-1, 0);
        while (r < q.r) move(0, 1);
        while (r > q.r) move(0, -1);
        ans[q.id] = cur;
    }
}
```

[BZOJ 2120](http://www.lydsy.com/JudgeOnline/problem.php?id=2120)

#### 树上莫队

#### 树上带修改莫队
