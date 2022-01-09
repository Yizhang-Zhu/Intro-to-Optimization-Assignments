#-*-coding:gb2312-*-
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import math

# 计算两个城市之间的距离
def distance(city1, city2):
    return np.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

# 导入城市坐标数据的csv文件
def loadFile(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        cities = []
        for row in reader:
           longtitude = float(row[1])
           latitude = float(row[2])
           cities.append((longtitude,latitude))
    return cities

# Adjacency matrix


'''
遗传算法
'''

# 交叉、繁衍
def crossAndBreed(chro1, chro2):
    # 染色体a b
    a = np.array(chro1).copy()
    b = np.array(chro2).copy()
    # 0-9随机生成交换序列的起始点下标和终止点下标，起始点下标小于终止点下标
    x = random.randint(0, 9)
    y = random.randint(0, 9) 
    begin = max(x, y)
    end = min(x, y)
    # 为防止染色体序列交换导致同一染色体内元素重复，建立映射关系
    castMap = {}
    for i in range(begin, end+1):
        if a[i] not in castMap.keys():
            castMap[a[i]] = [] # 新建映射 
        if b[i] not in castMap.keys():
            castMap[b[i]] = [] # 新建映射
        # 建立映射关系
        castMap[a[i]].append(b[i])
        castMap[b[i]].append(a[i])
    # 开始交配，交换起始点到终止点之间的染色体片段
    temp = a[begin:end+1].copy()
    a[begin:end+1] = b[begin:end+1]
    b[begin:end+1] = temp
    # 映射、交换除了起始点到终止点之外的染色体片段
    remainChro = list(range(0, begin))
    remainChro.extend(range(end+1, len(a)))
    aExchange = a[begin:end+1]
    bExchange = b[begin:end+1]
    for i in remainChro:
        if a[i] in castMap.keys():
            for j in castMap[a[i]]:
                if j not in aExchange:
                    a[i] = j # 避免同一条染色体中元素重复
                    break
        if b[i] in castMap.keys():
            for j in castMap[b[i]]:
                if j not in bExchange:
                    b[i] = j # 避免同一条染色体中元素重复
                    break
    return a, b

# 基因突变，随机选择亮点进行位置对换
def mutate(chro):
    pos1 = random.randint(0, 9)
    pos2 = random.randint(0, 9)
    while pos1 == pos2:
        pos1 = random.randint(0, 9)
    chro[pos1], chro[pos2] = chro[pos2], chro[pos1]
    return chro

# 适应度函数
def fitnessFunction(chro):
    # 读取城市坐标的csv文件
    cities = loadFile('C:\\Users\\admin\\Desktop\\cities.csv')
    n = len(cities)
    loss = 0
    for i in range(1, n):
        loss += distance(cities[chro[i-1]], cities[chro[i]])
    loss += distance(cities[chro[0]], cities[chro[-1]])
    return -loss # 取负数，距离越小越好

# 排序规则：按照适应度排序
def sortByAdapt(list):
    return list[2]

# 计算个体适应度、选择概率
def adaption(population):
    # 二维列表存放个体信息：个体下标、适应度、选择概率、累计概率
    # 累计概率方便轮盘选择
    individualInfo = []
    populationFitenss = 0
    index = 0
    # 计算适应度
    for individual in population:
        individualFitness = math.exp(fitnessFunction(individual))
        populationFitenss += individualFitness
        # 添加个体下标和个体适应度
        individualInfo.append([index])
        individualInfo[index].append(individualFitness)
        index += 1
    # 计算生存概率
    for individual in individualInfo:
        individual.append(individual[1]/populationFitenss)
        individual.append(individual[2])
    # 根据个体适应度排序由
    individualInfo.sort(key = sortByAdapt, reverse = True)
    # 计算累计概率
    n = len(individualInfo)
    for i in range(1, n):
        p = individualInfo[i][3] + individualInfo[i-1][3]
        individualInfo[i][3] = p
    return individualInfo

# 根据个体信息列表individualInfo轮盘选择，自然选择
def rouletteSelection(individualInfo):
    chooseList = []
    # 选择次数
    epoch = 20
    n = len(individualInfo)
    for i in range(epoch):
        p = random.random()
        if individualInfo[0][3] >= p:
            chooseList.append(individualInfo[0][0])
        else:
            for j in range(1, n):
                if individualInfo[j][3] >= p and individualInfo[j-1][3] < p:
                    chooseList.append(individualInfo[j][0]) 
                    break
    return chooseList

# 交叉变异操作
def crossMutate(chooseList, population):
    crossPossibility = 0.7 # 交叉率
    mutatePossibility = 0.3 # 变异率
    # 交叉变异
    chooseNum = len(chooseList)
    couplesNum = chooseNum//2 # 配偶对数
    for i in range(couplesNum):
        index1, index2 = random.sample(chooseList, 2)
        # 参与交叉的父母节点
        mom = population[index1]
        dad = population[index2]
        # 交配的从chooseList中删除
        chooseList.remove(index1)
        chooseList.remove(index2)
        # 交叉
        p = random.random()
        if crossPossibility >= p:
            child1, child2 = crossAndBreed(mom, dad)
            p1 = random.random()
            p2 = random.random()
            if mutatePossibility > p1:
                child1 = mutate(child1)
            if mutatePossibility > p2:
                child2 = mutate(child2)
            population.append(list(child1))
            population.append(list(child2))
    return population

# 某次遗传过程
def genetic(population):
    individualInfo = adaption(population)
    # 选择
    selection = rouletteSelection(individualInfo)
    # 交叉变异
    population = crossMutate(selection, population)
    return population

# 自然选择过程
def natureSelection(population):
    path = []   # 记录路径
    loss = []   # 记录损失值
    epoch = 51 # 迭代次数
    cnt = 0     # 计数器
    while cnt < epoch:
        individualInfo = []
        # 计算适应度
        for individual in population:
            individualFitness = fitnessFunction(individual)
            individualInfo.append(individualFitness)
        # 遗传算法更新种群
        population = genetic(population)
        # 寻找最小损失值
        minLoss = max(individualInfo)
        if cnt % 10 == 0:
            print('epoch %d: loss = %.2f' % (cnt, -minLoss)) # 隔十次输出一下
        loss.append([cnt, -minLoss])
        cnt += 1
        if cnt == epoch:
            # 迭代结束
            n = len(population)
            for i in range(n):
                if individualInfo[i] == minLoss:
                    path = population[i] # 记录最佳路径
                    print('最小代价:', -minLoss)
                    break
    # 绘制损失函数曲线
    loss = np.array(loss)
    plt.plot(loss[:, 0], loss[:, 1])
    plt.title('Cost Function loss-epoch Chart')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    return path

# 随机初始化
def randomInit():
    # 初始化，随机生成种群，规模为50
    population = []
    chro1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    chro2 = [5, 4, 6, 9, 2, 1, 7, 8, 3, 0]
    chro3 = [0, 1, 2, 3, 7, 8, 9, 4, 5, 6]
    chro4 = [1, 2, 3, 4, 5, 0, 7, 6, 8, 9]
    population = [chro1, chro2, chro3, chro4]
    chro = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    for i in range(46):
        random.shuffle(chro)
        population.append(chro) 
    return population

# 打印最佳路径
def printPath(path):
    for i in range(10):
        if path[i] == 0:
            print('北京', end = ' ')
        elif path[i] == 1:
            print('天津', end = ' ')
        elif path[i] == 2:
            print('上海', end = ' ')
        elif path[i] == 3:
            print('重庆', end = ' ')
        elif path[i] == 4:
            print('拉萨', end = ' ')
        elif path[i] == 5:
            print('乌鲁木齐', end = ' ')
        elif path[i] == 6:
            print('银川', end = ' ')
        elif path[i] == 7:
            print('呼和浩特', end = ' ')
        elif path[i] == 8:
            print('南宁', end = ' ')
        elif path[i] == 9:
            print('哈尔滨', end = ' ')
        else:
            print()

# 绘图
def drawGraph(path):
    cities = loadFile('C:\\Users\\admin\\Desktop\\cities.csv')
    x = []
    y = []
    for i in range(len(path)):
        city = path[i]
        x.append(cities[city][0])
        y.append(cities[city][1])
    x.append(x[0])
    y.append(y[0])
    plt.scatter(x, y, color = 'r')
    plt.plot(x, y)
    plt.title('Path Picture')
    plt.show()

if __name__ == "__main__":
    # 随机初始化
    population = randomInit()
    # 调用
    path = natureSelection(population)
    drawGraph(path)
    # 打印路径
    printPath(path)