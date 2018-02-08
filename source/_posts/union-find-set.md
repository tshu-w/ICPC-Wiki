---
title: Union-find Set
toc: false
date: 2017-09-12 11:17:45
categories: Date-Structure
tags:
  - union-find-set
---

```c++
int par[maxn], rnk[maxn];
void init(int n) {
    for (int i = 0; i < n; i++) {
        par[i] = i; rnk[i] = 0;
    }
}
int find(int x) {
    return x == par[x]? x : par[x] = find(par[x]);
}
bool same(int x, int y) {
    return find(x) == find(y);
}
void unite(int x, int y) {
    x = find(x); y = find(y);
    if (x == y) return;
    if (rnk[x] < rnk[y]) par[x] = y;
    else {
        par[y] = x;
        if (rnk[x] == rnk[y]) rnk[x]++;
    }
}
```
