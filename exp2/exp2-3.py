'''
本题可以转化成0-1背包问题的模型
总人数：背包容量
每个矿场收益：value
每个矿场用人：weight
'''
c = 10 # 背包容量
value = [0, 200, 300, 350, 400, 500]
weight = [0, 3, 4, 3, 5, 5]
maxValue = 0 # 所求的最大收益
mineList = [[]] # 记录路径，某个矿挖不挖

# DP数组初始化
# DP是二维数组，表示前i个物品装入容量j的背包下的最优解
dp = [[]]
for i in range(6):
    dp.append([])
    mineList.append([])
    for j in range(11):
        dp[i].append(0)
        mineList[i].append(0)

# 开始DP
for i in range(1, 6):
    for j in range(0, 11):
        if i == 0 or j == 0:
            dp[i][j] = 0
        else:
            dp[i][j] = dp[i-1][j]
            if j >= weight[i]: # 背包剩余容量可以容纳物品i
                # 第i物品放不放，比较选择收益较大的
                if dp[i-1][j] < dp[i-1][j-weight[i]]+value[i]:
                    dp[i][j] = dp[i-1][j-weight[i]]+value[i]
                    mineList[i][j] = 1 # 这个物品要放入背包（这个矿要挖）
                else:
                    dp[i][j] = dp[i-1][j]
                    mineList[i][j] = 0 # 这个物品不放入背包（这个矿不挖）
                # dp[i][j] = max(dp[i-1][j], dp[i-1][j-weight[i]]+value[i])
                maxValue = max(maxValue, dp[i][j]) # 记录最大值

# 输出
print("最大收益：", maxValue)
print("最大收益下，应该挖以下几座矿：")
# 输出选择的矿场
i = 5
j = c
while i>0 and j>0:
    if mineList[i][j] == 1:
        print('矿场',i, "  此矿场用人：", weight[i], " 收益：", value[i])
        j = j-weight[i]
    i = i-1
