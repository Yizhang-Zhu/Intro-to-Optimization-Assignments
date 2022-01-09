# 给定一个函数f(x)=60-10x1-4x2+x1^2+x2^2-x1x2，利用牛顿法求解该函数的最小值，需给出中间结果

import math

# 目标函数
def f(x1, x2):
    return 60-10*x1-4*x2+x1**2+x2**2-x1*x2

# 偏导数
# 一阶偏导
def fx1(x1, x2):
    return -10+2*x1-x2
def fx2(x1, x2):
    return -4+2*x2-x1
# 二阶偏导
def fx1x1(x1, x2):
    return 2
def fx2x2(x1, x2):
    return 2

# 初始化a0
a0 = [1,1]
# xlist存放(x1,x2)中间结果
xlist = []
# ylist存放y的中间结果
ylist = []

# 牛顿迭代法
def NewtonFunc(a0, e):
    while True:
        # 一阶导数
        f1 = [fx1(a0[0], a0[1]), fx2(a0[0], a0[1])]
        # 二阶导数
        f2 = [fx1x1(a0[0], a0[1]), fx2x2(a0[0], a0[1])]
        # 初始化a1
        a1 = [0, 0]
        a1[0] = a0[0] - f1[0]/f2[0]
        a1[1] = a0[1] - f1[1]/f2[1]
        # 计算a1与a0之间的距离
        if(math.sqrt((a1[0]-a0[0])**2+(a1[1]-a0[1])**2) <= e):
            break        # 如果达到要求精度，推出while循环
        xlist.append(a1) # (x1，x2)中间结果记录在xlist中
        ylist.append(f(a1[0],a1[1])) # y记录中间结果记录在ylist中
        a0 = a1          # 准备下一轮while循环的迭代
    return a1            # 此时a1为近似解a*

# 输出求得近似解和中间结果,精度要求0.001
res = NewtonFunc([1,1], 0.001)
print('近似解时(x1,x2)取:' ,res)
print('函数近似极小值:', f(res[0], res[1]))
print('(x1,x2)中间值:')
for i in xlist:
    print(i)
print('y的中间值:')
for i in ylist:
    print(i)


