---
title: Topological Sorting
toc: true
date: 2017-09-06 18:33:35
categories: Graph
tags:
  - topological-sorting
---

## DFS

```c++
bool dfs(int u) {
    c[u] = -1;
    for (int v = 1; v <= n; v++)
        if (G[u][v]) {
            if (c[v] < 0) return false;
            else if (!c[v] && !dfs(v)) return false;
        }
    c[u] = 1; topo[--t] = u;
    return true;
}

bool toposort() {
    t = n;
    memset(c, 0, sizeof c);
    for (int i = 1; i <= n; i++)
        if (!c[i]) if (!dfs(i)) return false;
    return true;
}
```

## Indegree

算法描述：

1. 选择一个入度为 0 的顶点并输出；
2. 然后从 AOV 网中删除此顶点及以此结点为起点的所有关联边（更新终点的入度）；
3. 重复 1，2 两步直到不存在入度为 0 的顶点为止；
4. 若输出的顶点小于网络中的顶点数，则有回路，否则输出结点序列为拓扑序列。
