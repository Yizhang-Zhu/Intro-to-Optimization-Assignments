# 给定一个函数f(x)=60-10x1-4x2+x1^2+x2^2-x1x2，利用牛顿法求解该函数的最小值，需给出中间结果

import math
import numpy as np

# 目标函数
def f(x):
    return 60-10*x[0]-4*x[1]+x[0]**2+x[1]**2-x[0]*x[1]

# 偏导数
# 一阶偏导
def fx1(x):
    return -10+2*x[0]-x[1]
def fx2(x):
    return -4+2*x[1]-x[0]

# 梯度
def grad(x):
    return np.array([fx1(x), fx2(x)])

# hessian矩阵
def hessian(x):
    return np.array()