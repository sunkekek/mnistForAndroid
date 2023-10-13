from LeNet5Model import *

def train():
    # 学习率0.001
    learning_rate = 1e-3
    # 单次大小
    batch_size = 100
    # 总的循环
    epoches = 50
    lenet = LeNet()
 
    # 1、数据集准备
    # 这个函数包括了两个操作：transforms.ToTensor()将图片转换为张量，transforms.Normalize()将图片进行归一化处理
    trans_img = transforms.Compose([transforms.ToTensor()])
    # path = './data/'数据集下载后保存的目录，下载训练集
    trainset = MNIST('./data', train=True, transform=trans_img, download=True)
    # 构建数据集的DataLoader,
    # Pytorch自提供了DataLoader的方法来进行训练，该方法自动将数据集打包成为迭代器，能够让我们很方便地进行后续的训练处理
    # 迭代器(iterable)是一个超级接口! 是可以遍历集合的对象,
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
 
    # 2、构建迭代器与损失函数
    criterian = nn.CrossEntropyLoss(reduction='sum')  # loss（损失函数）
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)  # optimizer（迭代器）
 
    # 如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #lenet.to("cpu")
 
    # 3、训练
    for i in range(epoches):
        running_loss = 0.
        running_acc = 0.
        for (img, label) in trainloader:  # 将图像和标签传输进device中
            optimizer.zero_grad()  # 求梯度之前对梯度清零以防梯度累加
            output=lenet(img)  # 对模型进行前向推理
            loss=criterian(output,label)  # 计算本轮推理的Loss值
            loss.backward()    # loss反传存到相应的变量结构当中
            optimizer.step()   # 使用计算好的梯度对参数进行更新
            running_loss+=loss.item()
            #print(output)
            _,predict=torch.max(output,1)  # 计算本轮推理的准确率
            correct_num=(predict==label).sum()
            running_acc+=correct_num.item()
 
        running_loss/=len(trainset)
        running_acc/=len(trainset)
        print("[%d/%d] Loss: %.5f, Acc: %.2f" % (i + 1, epoches, running_loss,100 * running_acc))
 
    return lenet