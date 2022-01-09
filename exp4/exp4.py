import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
import gzip
import numpy as np


# 处理文件数据的class
class DealDataset():
    def __init__(self, folder, data_name, label_name, transform=None):
        (train_set, train_labels) = load_data(folder, data_name, label_name)
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


# 载入mnist数据集的函数
def load_data(data_folder, data_name, label_name):
    # 使用gzip打开压缩文件
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset = 8)
    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset = 16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


batchSize = 512

# 训练集
trainDataset = DealDataset(r'C:\Users\admin\Desktop\mnist', "train-images-idx3-ubyte.gz",
                           "train-labels-idx1-ubyte.gz", transform = transforms.ToTensor())
trainData = DataLoader(dataset = trainDataset,
                       batch_size = batchSize, shuffle = True)

# 测试集
testDataset = DealDataset(r'C:\Users\admin\Desktop\mnist', "t10k-images-idx3-ubyte.gz",
                          "t10k-labels-idx1-ubyte.gz", transform = transforms.ToTensor())
testData = DataLoader(dataset = testDataset, batch_size = batchSize, shuffle = True)

# # 预览数据
# images,labels=next(iter(trainData))
# print(labels)
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(images[i][0])
# plt.show()


# 建立神经网络模型
class NetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.conv1(x)            # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 1* 10 * 12 * 12
        out = self.conv2(out)          # 1* 20 * 10 * 10
        out = F.relu(out)
        out = out.view(-1, 2000)       # 1 * 2000
        out = self.fc1(out)            # 1 * 500
        out = F.relu(out)
        out = self.fc2(out)            # 1 * 10
        out = F.log_softmax(out, dim = 1)
        return out


# 模型实例化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NetworkModel().to(device)

# 优化器 optimizer
# 使用随机梯度下降法SGD
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

# 损失函数
loss_function = nn.CrossEntropyLoss()  # 此交叉熵包括取对数，以类的形式存在，所以要实例化

# 存放历次batch_idx损失值和历次epoch准确度的数组
lossList = []
accList = []


# 训练集数据训练模型
def train(model, device, trainData, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainData):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)  # 以函数形式存在，无对数的交叉熵
        loss.backward()
        optimizer.step()
        lossList.append(loss.item())
        # 中间输出几次，以方便查看进度和中间值
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainData.dataset),
                100. * batch_idx / len(trainData), loss.item()))


# 测试集数据测试准确度
def test(model, device, testData):
    model.eval()
    total = 0
    correct = 0
    for images, labels in testData:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("correct1: ", correct)
    print("Test acc: {0}".format(correct.item() / len(testDataset)))
    accList.append(correct.item() / len(testDataset))


# 训练的总epoch数
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    train(model,  device, trainData, optimizer, epoch)
    test(model, device, testData)


# 绘制损失曲线与准确度曲线
plt.subplot(121)
plt.plot(lossList, color = 'b')
plt.xlabel('batch_idx')
plt.ylabel('loss')
plt.title('Loss Chart')

plt.subplot(122)
plt.plot(accList, color = 'r')
plt.ylim(ymin = 0.6, ymax = 1.00)
plt.xlabel('epoch')
plt.ylabel('Test Accuracy')
plt.title('Accuracy Chart')

plt.show()
