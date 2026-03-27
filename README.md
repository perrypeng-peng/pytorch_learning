# PyTorch理论学习+操练练习计划（含学习资源+报错解决方案+实操案例代码）

核心目标：从基础入门到能够独立搭建简单深度学习模型，掌握PyTorch核心API用法、模型构建逻辑及训练流程，理论与实操紧密结合，避免“只看不动手”“只会调包不懂原理”的问题。计划总周期建议8-10周（可根据自身基础调整节奏），每天学习+练习时长建议1.5-2.5小时，周末可适当延长至3-4小时，重点进行综合实操。

## 第一阶段：基础入门（1-2周）—— 搭建环境，掌握PyTorch核心基础

### 一、理论学习内容（每天30-40分钟）

- 第1-2天：PyTorch简介与环境搭建
核心理论：PyTorch的优势、适用场景（对比TensorFlow）；Anaconda、PyTorch的安装流程（CPU/GPU版本选择，GPU版本需了解CUDA、cuDNN的作用）；PyCharm或VS Code配置PyTorch开发环境。
补充资源：1. 官方英文文档（最权威）：https://pytorch.org/docs/stable/index.html ；2. 官方中文入门教程（适合零基础）：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html （中文翻译版可参考ApacheCN社区）[4]；3. 环境配置避坑：使用Docker镜像（如pytorch-cuda:v2.7）可快速搭建环境，启动时需加--gpus all参数启用GPU，避免CUDA识别失败。

- 第3-5天：Tensor（张量）核心操作
核心理论：Tensor的定义（与Numpy数组的区别与联系）；Tensor的基本属性（shape、dtype、device）；Tensor的创建方法（直接创建、从Numpy转换、随机创建等）；Tensor的常用操作（索引、切片、拼接、拆分、转置）；Tensor的数学运算（加减乘除、矩阵运算、激活函数的张量实现）。
补充资源：1. 莫烦Python PyTorch教程（图文+视频，通俗易懂）：https://mofanpy.com/tutorials/machine-learning/torch/ ；2. 张量操作快速查询：PyCharm中按住Ctrl+Q（Windows）/Ctrl+J（Mac）可查看API详细说明。

- 第6-10天：Autograd自动求导机制
核心理论：自动求导的原理（计算图的概念、前向传播与反向传播）；requires_grad参数的作用；backward()方法的使用（标量求导、向量求导）；梯度清零（zero_grad()）的必要性；detach()与with torch.no_grad()的区别及使用场景。
补充资源：1. 官方60分钟快速入门（含自动求导实操）：https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html ；2. 常见报错：梯度计算报错可优先检查requires_grad设置及张量是否为标量（backward()默认对标量求导）。

- 第11-14天：PyTorch数据加载（Dataset与DataLoader）
核心理论：Dataset类的作用与自定义方法（__init__、__getitem__、__len__）；DataLoader的核心参数（batch_size、shuffle、num_workers）；常用数据集（torchvision.datasets中的MNIST、CIFAR-10等）的使用；数据预处理（transforms的常用操作：归一化、 resize、翻转等）。
补充资源：1. DataLoader避坑：num_workers>0时加载大型数据集报错“共享内存溢出”，可通过docker run --shm-size=8g增大共享内存，或临时设num_workers=0（单线程）；2. 自定义Dataset实操案例：参考CSDN博客中模型构建相关实操代码。

### 二、对应操练练习（每天1-2小时）

- 第1-2天：环境搭建实操
1. 安装Anaconda，配置虚拟环境；2. 安装PyTorch（根据自身设备选择CPU/GPU版本）；3. 验证安装成功（编写简单代码，打印torch版本、判断是否支持GPU）；4. 配置PyCharm，关联虚拟环境，运行测试代码。
常见报错及解决方案：1. GPU版本安装后torch.cuda.is_available()返回False：检查Docker启动参数、NVIDIA驱动版本，或重新安装对应CUDA版本的PyTorch；2. 虚拟环境关联失败：PyCharm中手动选择Anaconda虚拟环境的python.exe路径。

- 第3-5天：Tensor操作实操
1. 用多种方法创建Tensor（如torch.tensor()、torch.zeros()、torch.randn()、torch.from_numpy()），查看其属性；2. 实现Tensor的索引、切片、拼接（cat、stack）、拆分（split、chunk）操作；3. 完成Tensor的加减乘除、矩阵乘法（mm、matmul）、幂运算等；4. 练习常用激活函数（sigmoid、relu、tanh）的张量运算，观察输出结果。
常见报错及解决方案：1. 张量维度不匹配：使用shape查看各张量维度，通过view()、reshape()调整；2. 设备不兼容（CPU/GPU张量混用）：统一用.to(device)指定设备，避免Tensor.cuda()未赋值导致的报错。

- 第6-10天：自动求导实操
1. 定义简单的张量函数（如y = x² + 2x + 1），设置requires_grad=True，计算梯度；2. 练习多变量求导（如z = x*y + x²，分别求x、y的梯度）；3. 模拟模型训练中的梯度清零操作，观察梯度累积与清零的区别；4. 使用detach()和with torch.no_grad()，对比两者对梯度计算的影响。
常见报错及解决方案：1. 向量求导报错：需指定backward()的gradient参数，或先将向量转为标量（如sum()）；2. 梯度累积异常：训练循环中未调用zero_grad()，导致梯度叠加偏差。

- 第11-14天：数据加载实操
1. 自定义一个简单的Dataset（如加载本地文件夹中的图片和标签），实现__init__、__getitem__、__len__方法；2. 使用DataLoader加载自定义Dataset，设置不同的batch_size和shuffle参数，观察输出结果；3. 加载torchvision.datasets.MNIST数据集，使用transforms进行预处理（归一化、转为Tensor）；4. 可视化加载的数据（如打印MNIST数据集的图片和对应标签）。
常见报错及解决方案：1. __getitem__返回值异常：确保返回格式为（数据，标签），且数据类型为Tensor；2. 数据预处理报错：检查transforms操作顺序，确保输入维度符合要求。

练习要求：每道题编写完整代码，运行并查看结果，遇到报错（如环境配置、张量维度不匹配），记录并解决，整理报错原因与解决方案。

## 第二阶段：核心进阶（3-6周）—— 模型构建与训练，掌握深度学习核心流程

### 一、理论学习内容（每天40-50分钟）

- 第1-7天：神经网络基础与nn.Module
核心理论：神经网络的基本结构（神经元、层、激活函数）；PyTorch中nn.Module类的核心作用（模型继承、forward方法）；常用网络层（nn.Linear、nn.Conv2d、nn.MaxPool2d、nn.Flatten、nn.Dropout）的原理与参数；激活函数的选择（ReLU及其变体、Sigmoid、Softmax）；损失函数（nn.MSELoss、nn.CrossEntropyLoss、nn.BCELoss）的适用场景。
补充资源：1. 模型构建三种方式详解（含代码）：https://blog.csdn.net/2201_75691511/article/details/152037841 ；2. nn.Module核心用法：可通过PyCharm跳转到源码定义，深入理解其工作机制。

- 第8-14天：模型训练流程
核心理论：模型训练的完整步骤（初始化模型、定义损失函数、定义优化器）；优化器的原理与选择（SGD、Adam、RMSprop，参数调整：lr、momentum等）；训练循环的编写（前向传播、计算损失、反向传播、参数更新）；验证集与测试集的作用，模型评估指标（准确率、召回率、F1-score）。
补充资源：1. 模型训练完整流程实战：https://blog.csdn.net/220_75691511/article/details/152037841 （含前向传播、反向传播代码）；2. 评估指标计算：参考PyTorch官方文档中torchmetrics库的使用方法。

- 第15-21天：经典模型实战理论（分类+回归）
核心理论：线性回归模型的原理与PyTorch实现；逻辑回归模型的原理（二分类、多分类）；CNN卷积神经网络的核心原理（卷积操作、池化操作、特征提取）；简单CNN模型（LeNet-5）的结构与原理；循环神经网络（RNN、LSTM）的基础原理（适用于序列数据）。
补充资源：1. LeNet-5模型实现代码：参考CSDN模型构建实战博客；2. 经典模型讲解：B站“李沐老师”PyTorch实战课程（免费），通俗易懂。

- 第23-28天：模型保存与加载、迁移学习基础
核心理论：模型保存的两种方式（保存整个模型、保存模型参数）；模型加载的方法（load_state_dict()的使用）；迁移学习的原理（预训练模型的作用、微调的流程）；torchvision.models中预训练模型（ResNet、VGG）的使用方法。
补充资源：1. 模型保存与加载避坑：多卡训练保存的模型加载报错，需去除state_dict中的module.前缀；2. ResNet预训练模型使用案例：torchvision官方文档教程。

### 二、对应操练练习（每天1.5-2小时）

- 第1-7天：神经网络层与Module实操
1. 继承nn.Module，自定义一个简单的线性模型（单隐藏层），实现forward方法；2. 练习使用nn.Conv2d、nn.MaxPool2d构建简单的卷积层，观察输入输出维度变化；3. 分别使用不同的损失函数（MSELoss、CrossEntropyLoss）计算损失，对比适用场景；4. 编写代码，查看模型的参数（parameters()、named_parameters()），理解参数的维度。
补充实操代码参考：自定义模型可参考nn.Module继承式构建案例，如：
class MyModel(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 26 * 26, 10)):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
 self.fc = nn.Linear(32 * 26 * 26, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 。

- 第8-14天：模型训练流程实操
1. 基于第一阶段的MNIST数据集，搭建简单的线性分类模型，编写完整训练循环（前向传播→计算损失→反向传播→参数更新）；2. 分别使用SGD和Adam优化器，对比两种优化器的训练效果（学习率相同，观察损失下降速度）；3. 划分训练集与验证集，在训练过程中计算验证集准确率，绘制损失曲线和准确率曲线；4. 解决训练中可能出现的问题（过拟合、梯度消失/爆炸，尝试使用Dropout、调整学习率）。
常见报错及解决方案：1. 过拟合：增加Dropout层、扩大数据集或使用数据增强；2. 梯度消失：使用ReLU激活函数、加入BatchNorm层。

- 第15-21天：经典模型实操
1. 实现线性回归模型（预测连续值，如房价预测简化版），使用MSELoss作为损失函数，训练并评估模型；2. 基于MNIST数据集，实现逻辑回归模型，完成手写数字分类，计算准确率；3. 搭建LeNet-5模型，用于MNIST手写数字分类，对比线性模型与CNN模型的准确率；4. （可选）使用RNN/LSTM模型，处理简单的序列数据（如文本分类简化版），熟悉序列数据的处理流程。
补充资源：1. LeNet-5完整代码：参考CSDN经典模型实战博客；2. 序列数据处理：莫烦Python RNN/LSTM教程。

- 第23-28天：模型保存与迁移学习实操
1. 保存训练好的CNN模型（两种方式），然后加载模型，用测试集验证模型性能；2. 使用torchvision.models中的ResNet18预训练模型，微调用于CIFAR-10数据集分类；3. 练习模型的断点续训（保存训练过程中的模型参数和优化器状态，下次继续训练）；4. 将训练好的模型用于预测，输入新的数据（如自己手写的数字图片），查看预测结果。
常见报错及解决方案：1. 模型加载报错“key不匹配”：检查保存的state_dict与当前模型结构是否一致，多卡训练保存的模型需去除module.前缀；2. 预训练模型微调报错：确保输入数据维度与模型要求一致。

练习要求：每一个模型都要完整编写“数据加载→模型定义→训练循环→模型评估→保存加载”的代码，记录训练过程中的关键参数（学习率、batch_size、训练轮次）和结果，对比不同模型、不同参数的效果。

## 第三阶段：综合提升（7-10周）—— 实战项目+问题排查，巩固所学知识

### 一、理论学习内容（每天30-40分钟）

- 第1-7天：PyTorch高级特性（含并行计算、图模式）
核心理论：nn.Sequential的使用（简化模型定义）；自定义损失函数的方法；学习率调度器（lr_scheduler）的使用（调整训练过程中的学习率）；多GPU训练的基础原理（DataParallel的使用）；PyTorch并行计算核心（多GPU分布式训练基础、DataParallel与DistributedDataParallel区别）；Torch图模式（eager mode与graph mode区别、torch.jit的使用场景与原理，静态图与动态图的优劣）。
补充资源：1. nn.Sequential简化模型案例：model = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 26 * 26, 10)) ；2. 多GPU训练避坑：参考CSDN多卡训练常见问题博客；3. 图模式学习：PyTorch官方文档torch.jit教程（https://pytorch.org/docs/stable/jit.html）；4. 并行计算实操参考：B站“李沐老师”分布式训练讲解。

- 第8-14天：常见问题排查与优化
核心理论：模型训练中常见问题（过拟合、欠拟合、梯度消失、梯度爆炸）的原因与解决方案；数据增强的高级方法（albumentations库的使用）；模型性能优化（批量归一化BatchNorm、梯度裁剪clip_grad_norm_）。
补充资源：1. PyTorch常见坑与解决方案：https://blog.csdn.net/weixin_42400643/article/details/156303487 ；2. albumentations库使用教程：CSDN相关实操博客。

- 第15-28天：综合项目理论补充
根据选定的实战项目，补充相关理论知识（如目标检测基础、语义分割基础、自然语言处理基础等），了解项目的核心逻辑和实现思路。
补充资源：1. 实战项目案例：GitHub上“PyTorch实战项目合集”；2. 目标检测/语义分割基础：B站“同济子豪兄”PyTorch实战课程。

### 二、对应操练练习（每天2-2.5小时，周末可延长）

- 第1-7天：高级特性实操（含并行计算、图模式实操）
1. 使用nn.Sequential简化之前编写的CNN模型，对比两种模型定义方式的差异；2. 自定义一个损失函数（如基于MSELoss的改进损失），并用于线性回归模型训练；3. 使用lr_scheduler（如StepLR、ReduceLROnPlateau），在训练过程中调整学习率，观察损失变化；4. （可选）配置多GPU环境，使用DataParallel实现多GPU训练，对比单GPU与多GPU的训练速度，尝试使用DistributedDataParallel搭建分布式训练框架；5. 图模式实操：将之前编写的CNN模型转为torch.jit脚本模式（script）和追踪模式（trace），对比图模式与 eager 模式的运行速度，排查图模式转换中的常见报错。
补充实操参考：结合nn.Module与模型容器（如nn.ModuleList）构建灵活模型，参考CSDN模型构建案例；并行计算实操：编写简单多GPU训练代码，测试数据并行效果；图模式实操：使用torch.jit.trace追踪模型，保存并加载脚本模型，验证模型性能。
常见报错及解决方案：1. 多GPU训练报错“进程启动失败”：检查分布式训练环境配置，确保各进程端口不冲突；2. 图模式转换报错“不支持动态操作”：修改模型中动态分支（如if-else），或使用script模式替代trace模式。

- 第8-14天：问题排查与优化实操
1. 故意设计一个过拟合的模型（如复杂CNN、无Dropout），然后通过增加Dropout、数据增强、早停（Early Stopping）等方法解决过拟合；2. 练习使用albumentations库进行数据增强（如随机翻转、缩放、高斯模糊），应用于CIFAR-10数据集；3. 在模型训练中加入BatchNorm和梯度裁剪，解决梯度消失/爆炸问题，对比优化前后的训练效果；4. 排查代码中的常见报错（如维度不匹配、设备不兼容、数据类型错误），整理解决方案。
常见报错及解决方案：汇总前两阶段报错，重点掌握设备兼容、维度匹配、模型保存加载三类核心报错的解决方法，参考PyTorch常见坑博客。

- 第15-28天：综合实战项目（二选一或全做）
项目1：手写数字识别进阶（CNN+迁移学习）
- 目标：基于MNIST数据集，搭建更优的CNN模型，结合迁移学习，实现99%以上的准确率；
- 要求：完成数据增强、模型优化、超参数调优（学习率、batch_size、轮次），编写完整的项目代码，包含训练、验证、测试、预测全流程，生成模型报告。
项目2：图像分类实战（CIFAR-10）
- 目标：使用ResNet预训练模型，微调实现CIFAR-10数据集分类，准确率达到85%以上；
- 要求：使用数据增强、学习率调度器、批量归一化等优化方法，解决过拟合问题，保存最佳模型，编写项目文档（包含模型结构、训练过程、结果分析）。

- （可选）项目3：简单回归任务（如气温预测）
- 目标：基于公开数据集，使用PyTorch搭建线性回归或LSTM回归模型，预测未来气温；
- 要求：完成数据预处理（缺失值处理、归一化）、模型构建、训练优化，评估模型预测效果。
补充资源：项目代码参考GitHub开源项目，报错排查参考CSDN、Stack Overflow相关问题解答。

## 整体学习注意事项

- 理论与实操同步：每天先学习理论，再进行对应练习，避免“只看不动手”，实操中遇到的问题，回头再巩固理论知识，形成闭环。

- 注重基础：前两周的Tensor、Autograd、数据加载是核心基础，务必熟练掌握，否则后续模型构建会遇到大量障碍。

- 记录与总结：每完成一个阶段的学习和练习，整理笔记（核心理论、API用法、常见报错），每周复盘一次，查漏补缺。

- 循序渐进：不要急于求成，尤其是零基础学习者，可适当延长第一、二阶段的时间，确保每个知识点都能理解并熟练运用。

- 善用资源：遇到问题可查阅PyTorch官方文档、CSDN、B站相关教程，也可参考开源项目（GitHub上的PyTorch实战项目），学习他人的代码思路；优先使用官方英文文档保证准确性，中文资源作为入门辅助；并行计算、图模式相关问题可重点参考PyTorch官方分布式训练和torch.jit教程，结合实操案例理解。

## 并行计算+图模式 单独实操案例代码（可直接复制运行）

### 一、并行计算实操案例（DataParallel + DistributedDataParallel）

```python

# 案例1：DataParallel 单节点多GPU训练（简单易懂，适合入门）
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# 1. 配置设备（自动识别多GPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")

# 2. 定义简单CNN模型（复用之前的模型结构，便于衔接）
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 28 * 28, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# 3. 初始化模型、损失函数、优化器
model = MyCNN()
# 多GPU包装模型
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 加载数据（MNIST数据集，复用之前的配置）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 5. 训练循环（与单GPU训练几乎一致，DataParallel自动分配数据）
epochs = 3
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播、计算损失、反向传播、参数更新
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('DataParallel 多GPU训练完成！')

# 案例2：DistributedDataParallel 分布式训练（进阶，适合多节点/多GPU高效训练）
# 注意：需在终端用 torchrun 启动（如：torchrun --nproc_per_node=2 ddp_train.py）
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def ddp_train():
    # 1. 初始化分布式环境
    dist.init_process_group(backend='nccl')  # nccl是GPU分布式常用后端
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # 2. 定义模型（与案例1一致）
    model = MyCNN().to(device)
    # DDP包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 3. 数据加载（需使用DistributedSampler分配数据，避免多GPU重复加载）
    train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    
    # 4. 损失函数、优化器（与案例1一致）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. 训练循环（需注意sampler的set_epoch）
    epochs = 3
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # 确保每轮数据打乱不一致
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99 and local_rank == 0:  # 只在主进程打印
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    # 销毁分布式环境
    dist.destroy_process_group()
    if local_rank == 0:
        print('DistributedDataParallel 分布式训练完成！')

if __name__ == "__main__":
    ddp_train()

```

### 二、Torch图模式实操案例（torch.jit trace + script）

```python

import torch
import torch.nn as nn
import time

# 1. 定义复用的CNN模型（与之前一致，保证衔接性）
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 28 * 28, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# 2. 初始化模型并切换为评估模式（图模式常用于推理加速）
model = MyCNN().eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 准备输入数据（模拟MNIST输入：[batch_size, channel, height, width]）
input_tensor = torch.randn(32, 1, 28, 28).to(device)

# 案例1：torch.jit.trace 追踪模式（静态图，适合无动态分支的模型）
print("=== 测试 trace 模式 ===")
# 追踪模型（用输入数据“引导”模型生成图）
traced_model = torch.jit.trace(model, input_tensor)
# 保存追踪模型（可用于部署）
traced_model.save("traced_cnn.pt")
# 加载模型
loaded_traced_model = torch.jit.load("traced_cnn.pt").to(device)

# 对比 eager 模式与 trace 模式的运行速度
# eager 模式
start_time = time.time()
for _ in range(100):
    model(input_tensor)
eager_time = time.time() - start_time

# trace 模式
start_time = time.time()
for _ in range(100):
    loaded_traced_model(input_tensor)
trace_time = time.time() - start_time

print(f"Eager 模式耗时: {eager_time:.4f}s")
print(f"Trace 模式耗时: {trace_time:.4f}s")
print("Trace 模式加速比: {:.2f}x".format(eager_time / trace_time))

# 案例2：torch.jit.script 脚本模式（静态图，支持动态分支，更灵活）
print("\n=== 测试 script 模式 ===")
# 定义一个带动态分支的模型（模拟实际场景中if-else逻辑）
class DynamicModel(nn.Module):
    def __init__(self):
        super(DynamicModel, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*28*28, 50)
        self.fc2 = nn.Linear(32*28*28, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    
    def forward(self, x, threshold=0.5):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        # 动态分支：根据输入均值判断走哪个全连接层
        if x.mean() > threshold:
            return self.fc1(x)
        else:
            return self.fc2(x)

# 初始化动态模型
dynamic_model = DynamicModel().eval().to(device)
# 脚本模式转换（支持动态分支，无需输入数据引导）
script_model = torch.jit.script(dynamic_model)
# 保存与加载
script_model.save("script_dynamic_model.pt")
loaded_script_model = torch.jit.load("script_dynamic_model.pt").to(device)

# 测试脚本模型运行（输入不同数据，验证动态分支生效）
input1 = torch.randn(32, 1, 28, 28).to(device)  # 均值大概率大于0.5
input2 = torch.randn(32, 1, 28, 28) * 0.1 - 0.6  # 均值大概率小于0.5
input2 = input2.to(device)

output1 = loaded_script_model(input1)
output2 = loaded_script_model(input2)
print(f"输入1输出维度（fc1）: {output1.shape}")  # 应该是 [32, 50]
print(f"输入2输出维度（fc2）: {output2.shape}")  # 应该是 [32, 10]

# 对比 script 模式与 eager 模式速度
start_time = time.time()
for _ in range(100):
    dynamic_model(input1)
    dynamic_model(input2)
eager_dynamic_time = time.time() - start_time

start_time = time.time()
for _ in range(100):
    loaded_script_model(input1)
    loaded_script_model(input2)
script_time = time.time() - start_time

print(f"动态模型 Eager 模式耗时: {eager_dynamic_time:.4f}s")
print(f"动态模型 Script 模式耗时: {script_time:.4f}s")

# 常见报错解决：trace模式遇到动态分支报错
# 解决方案1：改用script模式（如本案例）
# 解决方案2：修改模型，移除动态分支（如将if-else改为矩阵运算）

```

实操说明：1. 并行计算案例需在多GPU环境（或Docker多GPU容器）中运行，单GPU环境可跳过DDP案例，重点练习DataParallel；2. 图模式案例可直接在单GPU/CPU环境运行，重点对比两种模式的差异和运行速度，掌握模型保存与加载方法；3. 代码中均添加详细注释，可直接复制到PyCharm运行，遇到报错可参考之前的报错解决方案。
> （注：文档部分内容可能由 AI 生成）