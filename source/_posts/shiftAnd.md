---
title: shiftAnd
toc: false
date: 2017-10-02 17:21:32
categories: String
tags:
---

## shiftAnd

主串 s[0...n] 模式串 t[0..m]
bitset D 中 D[j] = 1 表示模式串前缀 $t_0,...,t_j$ 是主串 $s_0,...,s_i$ 的后缀。
D = (D << 1 | 1) & B[s[i + 1]]

```c++
bitset<maxm> D, S[256];
void shiftAnd(int n, int m) {
    D.reset();
    for (int i = 0; i < n; i++) {
        D <<= 1; D.set(0);
        D &= B[s[i]];
        if (D[m - 1]) {
            char tmp = s[i + 1];
            s[i + 1] = '\0';
            puts(s + (i - n + 1));
            s[i + 1] = tmp;
        }
    }
}
```

## shiftOr

为减少位运算， bitset D 中 D[j] = 0 表示模式串前缀 $t_0,...,t_j$ 是主串 $s_0,...,s_i$ 的后缀
