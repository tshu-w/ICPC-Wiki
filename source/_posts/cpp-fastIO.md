---
title: cpp-fastIO
toc: false
date: 2017-08-30 13:27:02
categories: Others
tags:
- fastIO
---

## 关同步
```cpp
#define IOS std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr);
#define endl "\n"
```
{% alert danger %}关同步后 C IO（scanf, printf, getchar, putchar, fgets, puts, etc.） 与 C++ IO（cin, cout, etc.） 不可同时使用{% endalert %}

[Reference](http://codeforces.com/blog/entry/5217)

## 读入挂

### getchar 版
```cpp
inline void read(int &x) { // 可根据情况去掉负数
	int t = 1;
	char ch = getchar();
	while (ch < '0' || ch > '9') { if (ch == '-') t = -1; ch = getchar();}
	x = 0;
	while (ch >= '0' && ch <= '9') { x = x * 10 + ch -'0'; ch = getchar();}
	x *= t;
}
void print(int i){
	if(i < 10) {
		putchar('0' + i);
		return ;
	}
	print(i / 10);
	putchar('0' + i % 10);
}
```

### freed 版
```cpp
namespace fastIO {
#define BUF_SIZE 100000 // 本地小数据测试改为1
    //fread -> read
    bool IOerror = 0;
    inline char nc() {
        static char buf[BUF_SIZE], *p1 = buf + BUF_SIZE, *pend = buf + BUF_SIZE;
        if(p1 == pend) {
            p1 = buf;
            pend = buf + fread(buf, 1, BUF_SIZE, stdin);
            if(pend == p1) {
                IOerror = 1;
                return -1;
            }
        }
        return *p1++;
    }
    inline bool blank(char ch) {
        return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t';
    }
    inline void read(int &x) {
        char ch;
        while(blank(ch = nc()));
        if(IOerror)
            return;
        for(x = ch - '0'; (ch = nc()) >= '0' && ch <= '9'; x = x * 10 + ch - '0');
    }
#undef BUF_SIZE
};
using namespace fastIO;
// while (read(n), !fastIO::IOerror) {}
```
