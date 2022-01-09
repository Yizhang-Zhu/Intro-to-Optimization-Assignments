n = int(input('输入台阶数：')) # 输入的台阶数
dp = [] # DP表
# 初始化
for i in range(n+1):
    dp.append(0)
dp[1] = 1
dp[2] = 2
# 开始DP
for i in range(3, n+1):
    dp[i] = dp[i-1] + dp[i-2]
# DP表最后一个即为所求结果
print('爬完楼梯的方式数：' ,dp[-1])