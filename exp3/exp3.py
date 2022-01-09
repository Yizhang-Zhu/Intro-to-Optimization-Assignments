#-*-coding:gb2312-*-
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import math

# ������������֮��ľ���
def distance(city1, city2):
    return np.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

# ��������������ݵ�csv�ļ�
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
�Ŵ��㷨
'''

# ���桢����
def crossAndBreed(chro1, chro2):
    # Ⱦɫ��a b
    a = np.array(chro1).copy()
    b = np.array(chro2).copy()
    # 0-9������ɽ������е���ʼ���±����ֹ���±꣬��ʼ���±�С����ֹ���±�
    x = random.randint(0, 9)
    y = random.randint(0, 9) 
    begin = max(x, y)
    end = min(x, y)
    # Ϊ��ֹȾɫ�����н�������ͬһȾɫ����Ԫ���ظ�������ӳ���ϵ
    castMap = {}
    for i in range(begin, end+1):
        if a[i] not in castMap.keys():
            castMap[a[i]] = [] # �½�ӳ�� 
        if b[i] not in castMap.keys():
            castMap[b[i]] = [] # �½�ӳ��
        # ����ӳ���ϵ
        castMap[a[i]].append(b[i])
        castMap[b[i]].append(a[i])
    # ��ʼ���䣬������ʼ�㵽��ֹ��֮���Ⱦɫ��Ƭ��
    temp = a[begin:end+1].copy()
    a[begin:end+1] = b[begin:end+1]
    b[begin:end+1] = temp
    # ӳ�䡢����������ʼ�㵽��ֹ��֮���Ⱦɫ��Ƭ��
    remainChro = list(range(0, begin))
    remainChro.extend(range(end+1, len(a)))
    aExchange = a[begin:end+1]
    bExchange = b[begin:end+1]
    for i in remainChro:
        if a[i] in castMap.keys():
            for j in castMap[a[i]]:
                if j not in aExchange:
                    a[i] = j # ����ͬһ��Ⱦɫ����Ԫ���ظ�
                    break
        if b[i] in castMap.keys():
            for j in castMap[b[i]]:
                if j not in bExchange:
                    b[i] = j # ����ͬһ��Ⱦɫ����Ԫ���ظ�
                    break
    return a, b

# ����ͻ�䣬���ѡ���������λ�öԻ�
def mutate(chro):
    pos1 = random.randint(0, 9)
    pos2 = random.randint(0, 9)
    while pos1 == pos2:
        pos1 = random.randint(0, 9)
    chro[pos1], chro[pos2] = chro[pos2], chro[pos1]
    return chro

# ��Ӧ�Ⱥ���
def fitnessFunction(chro):
    # ��ȡ���������csv�ļ�
    cities = loadFile('C:\\Users\\admin\\Desktop\\cities.csv')
    n = len(cities)
    loss = 0
    for i in range(1, n):
        loss += distance(cities[chro[i-1]], cities[chro[i]])
    loss += distance(cities[chro[0]], cities[chro[-1]])
    return -loss # ȡ����������ԽСԽ��

# ������򣺰�����Ӧ������
def sortByAdapt(list):
    return list[2]

# ���������Ӧ�ȡ�ѡ�����
def adaption(population):
    # ��ά�б��Ÿ�����Ϣ�������±ꡢ��Ӧ�ȡ�ѡ����ʡ��ۼƸ���
    # �ۼƸ��ʷ�������ѡ��
    individualInfo = []
    populationFitenss = 0
    index = 0
    # ������Ӧ��
    for individual in population:
        individualFitness = math.exp(fitnessFunction(individual))
        populationFitenss += individualFitness
        # ��Ӹ����±�͸�����Ӧ��
        individualInfo.append([index])
        individualInfo[index].append(individualFitness)
        index += 1
    # �����������
    for individual in individualInfo:
        individual.append(individual[1]/populationFitenss)
        individual.append(individual[2])
    # ���ݸ�����Ӧ��������
    individualInfo.sort(key = sortByAdapt, reverse = True)
    # �����ۼƸ���
    n = len(individualInfo)
    for i in range(1, n):
        p = individualInfo[i][3] + individualInfo[i-1][3]
        individualInfo[i][3] = p
    return individualInfo

# ���ݸ�����Ϣ�б�individualInfo����ѡ����Ȼѡ��
def rouletteSelection(individualInfo):
    chooseList = []
    # ѡ�����
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

# ����������
def crossMutate(chooseList, population):
    crossPossibility = 0.7 # ������
    mutatePossibility = 0.3 # ������
    # �������
    chooseNum = len(chooseList)
    couplesNum = chooseNum//2 # ��ż����
    for i in range(couplesNum):
        index1, index2 = random.sample(chooseList, 2)
        # ���뽻��ĸ�ĸ�ڵ�
        mom = population[index1]
        dad = population[index2]
        # ����Ĵ�chooseList��ɾ��
        chooseList.remove(index1)
        chooseList.remove(index2)
        # ����
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

# ĳ���Ŵ�����
def genetic(population):
    individualInfo = adaption(population)
    # ѡ��
    selection = rouletteSelection(individualInfo)
    # �������
    population = crossMutate(selection, population)
    return population

# ��Ȼѡ�����
def natureSelection(population):
    path = []   # ��¼·��
    loss = []   # ��¼��ʧֵ
    epoch = 51 # ��������
    cnt = 0     # ������
    while cnt < epoch:
        individualInfo = []
        # ������Ӧ��
        for individual in population:
            individualFitness = fitnessFunction(individual)
            individualInfo.append(individualFitness)
        # �Ŵ��㷨������Ⱥ
        population = genetic(population)
        # Ѱ����С��ʧֵ
        minLoss = max(individualInfo)
        if cnt % 10 == 0:
            print('epoch %d: loss = %.2f' % (cnt, -minLoss)) # ��ʮ�����һ��
        loss.append([cnt, -minLoss])
        cnt += 1
        if cnt == epoch:
            # ��������
            n = len(population)
            for i in range(n):
                if individualInfo[i] == minLoss:
                    path = population[i] # ��¼���·��
                    print('��С����:', -minLoss)
                    break
    # ������ʧ��������
    loss = np.array(loss)
    plt.plot(loss[:, 0], loss[:, 1])
    plt.title('Cost Function loss-epoch Chart')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    return path

# �����ʼ��
def randomInit():
    # ��ʼ�������������Ⱥ����ģΪ50
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

# ��ӡ���·��
def printPath(path):
    for i in range(10):
        if path[i] == 0:
            print('����', end = ' ')
        elif path[i] == 1:
            print('���', end = ' ')
        elif path[i] == 2:
            print('�Ϻ�', end = ' ')
        elif path[i] == 3:
            print('����', end = ' ')
        elif path[i] == 4:
            print('����', end = ' ')
        elif path[i] == 5:
            print('��³ľ��', end = ' ')
        elif path[i] == 6:
            print('����', end = ' ')
        elif path[i] == 7:
            print('���ͺ���', end = ' ')
        elif path[i] == 8:
            print('����', end = ' ')
        elif path[i] == 9:
            print('������', end = ' ')
        else:
            print()

# ��ͼ
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
    # �����ʼ��
    population = randomInit()
    # ����
    path = natureSelection(population)
    drawGraph(path)
    # ��ӡ·��
    printPath(path)