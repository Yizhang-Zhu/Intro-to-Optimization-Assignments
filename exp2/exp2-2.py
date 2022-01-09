# 输入数字代表每天股票价格，用空格分隔
inputList = input("")
# 把输入的数字放入priceList数组
priceList = [int(n) for n in inputList.split()]
n = len(priceList)
# 二维DP表，dp[i][j]中i表示天数；j表示当天是否持有股票，0为没有，1为持有
dp = [[]]

# 初始化
for i in range(n):
    dp.append([])
    for j in range(2):
        dp[i].append(-99999)

dp[0][0] = 0
dp[0][1] = -priceList[0] # 购入股票

# 开始DP
for i in range(1, n):
    # 当天不持有股票：前一天也没有；昨天持有，今天卖了
    dp[i][0] = max(dp[i-1][0], dp[i-1][1]+priceList[i])
    # 当天持有股票：前一天也持有；昨天不持有，今天买的
    dp[i][1] = max(dp[i-1][1], dp[i-1][0]-priceList[i])

print("最大收益：", dp[n-1][0])
print("DP表：")
for i in range(n):
    print(dp[i])
