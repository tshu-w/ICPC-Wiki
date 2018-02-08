---
title: binary-indexed-tree
toc: false
date: 2017-10-08 11:01:37
categories: Date-Structure
tags:
---

## lowbit

`x & -x`
转为二进制后，最后一个 1 的位置所代表的数值。

## 一维 bit

从 1 开始

### 单点修改/查询

```c++
int bit[maxn];
int sum(int i) {
    int s = 0;
    while (i > 0) {
        s += bit[i];
        i -= i & -i;
    }
    return s;
}
void add(int i, int x) {
    while (i <= n) {
        bit[i] += x;
        i += i & -i;
    }
}
```

### 区间修改/查询

```c++
struct bit {
    int bit[maxn];
    int sum(int i) {
        int s = 0;
        while (i > 0) {
            s += bit[i];
            i -= i & -i;
        }
        return s;
    }
    void add(int i, int x) {
        while (i <= n) {
            bit[i] += x;
            i += i & -i;
        }
   }
}a, b;
inline void add(int l, int r, int t) {
    a.add(l,t); a.add(r+1,-t);
    b.add(l,-t*(l-1)); b.add(r+1,t*r);
}
inline int get(int i) {
    return a.sum(i)*i+b.sum(i);
}
inline int get(int l, int r) {
    return get(r)-get(l - 1);
}
```

## 二维 bit

### 单点修改/查询

```c++
int bit[maxn][maxn];
int sum(int x, int y) {
    int res = 0;
    for (int i = x; i > 0; i -= i & -i)
        for (int j = y; j > 0; j -= j & -j)
            res += bit[i][j];
    return res;
}
void add(int x, int y, int k) {
    for (int i = x; i <= n; i += i & -i)
        for (int j = y; j <= n; j += j & -j)
            bit[i][j] += k;
}
```

### 区间修改/查询

```c++
struct bit {
    int a[maxn][maxn];
    inline int lowbit(int x) {
        return x&(-x);
    }
    inline void add(int x,int y,int t) {
        int i,j;
        for(i=x;i<maxn;i+=lowbit(i)) {
            for(j=y;j<maxn;j+=lowbit(j))a[i][j]+=t;
        }
    }
    inline int get(int x,int y) {
        int ans=0;
        int i,j;
        for(i=x;i>0;i-=lowbit(i)) {
            for(j=y;j>0;j-=lowbit(j))ans+=a[i][j];
        }
        return ans;
    }
}a,b,c,d;
inline void add(int x1,int y1,int x2,int y2,int t) {
    a.add(x1,y1,t),a.add(x1,y2+1,-t);
    a.add(x2+1,y1,-t),a.add(x2+1,y2+1,t);

    b.add(x1,y1,t*x1); b.add(x2+1,y1,-t*(x2+1));
    b.add(x1,y2+1,-t*x1); b.add(x2+1,y2+1,t*(x2+1));

    c.add(x1,y1,t*y1); c.add(x2+1,y1,-t*y1);
    c.add(x1,y2+1,-t*(y2+1)); c.add(x2+1,y2+1,t*(y2+1));

    d.add(x1,y1,t*x1*y1); d.add(x2+1,y1,-t*(x2+1)*y1);
    d.add(x1,y2+1,-t*x1*(y2+1)); d.add(x2+1,y2+1,t*(x2+1)*(y2+1));
}
inline int get(int x,int y) {
    return a.get(x,y)*(x+1)*(y+1)-b.get(x,y)*(y+1)-(x+1)*c.get(x,y)+d.get(x,y);
}
inline int get(int x1,int y1,int x2,int y2) {
    return get(x2,y2)-get(x2,y1-1)-get(x1-1,y2)+get(x1-1,y1-1);
}
```

bzoj3132-上帝造题的七分钟
