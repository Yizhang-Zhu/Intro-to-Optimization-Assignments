# 给定一个函数f(x)=8e^(1-x)+7log(x)，利用黄金分割法把区间压缩到长度只有0.23，需给出所有中间结果。

import math

# 搜索区间上下界初始化
lower = 0
upper = 0

# 存储中间结果
alist = []
blist = []
xlist = []

# 待处理函数
def f(x):
    # y = x**2-7*x+10
    y = 8*(math.e**(1-x))+7*math.log10(x)
    return y

# 进退法确定搜索区间
x0 = 1 # 初始点为0
h0 = 1 # 初始步长为1
x1 = x0+h0
x2 = x1+h0
if f(x1) <= f(x0):
    while f(x1) > f(x2):
        x0 = x1
        x1 = x2
        h0 = h0*2 # 每次放大2倍
        x2 = x1+h0
        if h0 > 1024:
            print('Can not find')
            break # 找不到，判定为递增
else:
    while f(x1) > f(x2):
        x1 = x0
        x2 = x1
        h0 = h0/2
        x1 = x2-h0
        if h0 < 0.001:
            print('Can not find')
            break # 找不到，判定为递减

# 进退法搜索区间
lower = x0 # 搜索区间下界
upper = x2 # 搜索区间上界

# 黄金分割法: 三个参数 a, b, e
# a b表示区间边界，e表示精度要求
def GoldenSection(a, b, e):
    a1 = b-0.618*(b-a)
    a2 = a+0.618*(b-a)
    f1 = f(a1)
    f2 = f(a2)
    while b-a > e: # 未达到所需精度
        if f1 < f2:
            b = a2
            a2 = a1
            f2 = f1
            a1 = b-0.618*(b-a)
            f1 = f(a1)
        else:
            a = a1
            a1 = a2
            f1 = f2
            a2 = a+0.618*(b-a)
            f2 = f(a2)
        alist.append(a) # 存放a的中间结果
        blist.append(b) # 存放b的中间结果
        xlist.append((a+b)/2) # 存放x的中间结果
    a_star = (a+b)/2 # a*为近似极小值
    return a_star

# 近似极小值，精度取0.23
approximateMin = GoldenSection(lower, upper, 0.23)

# 输出
print('进退法: 搜索区间下界 =', lower, '; 搜索区间上界 =', upper)
print('a 中间值:', alist)
print('b 中间值:', blist)
print('x 中间值:', xlist)
print('近似极小值对应x =' ,approximateMin)
print('函数近似极小值y =' ,f(approximateMin))
