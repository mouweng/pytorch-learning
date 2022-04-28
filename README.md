# pytorch学习笔记

- [PyTorch深度学习快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN?spm_id_from=333.337.search-card.all.click)

## pytorch环境搭建

- 安装conda并搭建python环境

```shell
conda create -n pytorch python=3.6
```

- 安装pytorch，在[pytorch官网](https://pytorch.org/get-started/locally/)选择版本

```shell
conda install pytorch torchvision torchaudio -c pytorch
```

- 安装jupyter

```shell
conda install jupyter notebook
```

python文件是以所有行为块执行，python控制台以每一行为块执行，jupyter以任意行为块执行。

## python法宝函数

- `dir(torch)`

- `help(torch.cuda.is_available)`

## pytorch加载数据

### Dataset

> 提供一种方式去获取数据及其label

- 如何获取每一个数据及其label
- 告诉我们总共有多少的数据

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)
    
root_dir = "./hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_ds = MyData(root_dir, ants_label_dir)
bees_ds = MyData(root_dir, bees_label_dir)
print(len(ants_ds))
print(len(bees_ds))
img, label = ants_ds[0]
display(img)
```

### Dataloader

>  为后面的网络提供不同的数据形式

## Tensorboard使用

> TensorBoard 是用于提供机器学习工作流程期间所需的测量和可视化的工具。 它使您能够跟踪实验指标，例如损失和准确性，可视化模型图，将嵌入物投影到较低维度的空间等等。

### conda安装

```shell
conda install tensorboard
```

### 使用add_scalar

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")

# y = x
for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()
```

在终端里面运行

```shell
tensorboard --logdir=logs --port=6006
```

### 使用add_image

 ```python
 from torch.utils.tensorboard import SummaryWriter
 import numpy as np
 from PIL import Image
 
 writer = SummaryWriter("logs")
 image_path = "hymenoptera_data/train/ants/0013035.jpg"
 image_PIL = Image.open(image_path)
 img_array = np.array(image_PIL)
 print(type(img_array))
 print(img_array.shape)
 
 writer.add_image("test", img_array, 2, dataformats='HWC')
 
 writer.close()
 ```

在终端里面运行

```shell
tensorboard --logdir=logs --port=6006
```

## Transforms使用

### 常用API

- ToTensor：把图片类型（PIL IMAGE）和 数组类型（numpy.ndarray）转换成tensor
- Normalize：主要用于图片的归一化
- Resize：用于图片更改大小
- Compose：将几个transform组合在一起，后面一个参数需要的输入，和前面一个参数的输出是要一致的！
- RandomCrop：随机裁剪

### transforms的结构和用法

![transforms的结构和用法](https://cdn.jsdelivr.net/gh/mouweng/FigureBed/img/202204261027759.jpg)

**为什么需要tensor的数据类型？**

tensor是一种强大的表示方向和空间的数学方法。其实这些东西就是numpy，转成tensor是为了加速计算。

### transforms使用

#### ToTensor使用

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "./dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(img_tensor)
writer.add_image("ToTensor", img_tensor)
writer.close()
```

#### Normalize

 ```python
 from PIL import Image
 from torch.utils.tensorboard import SummaryWriter
 from torchvision import transforms
 
 writer = SummaryWriter("logs")
 img_path = "./dataset/train/ants_image/0013035.jpg"
 img = Image.open(img_path)
 print(img)
 
 trans_totensor = transforms.ToTensor()
 img_tensor = trans_totensor(img)
 
 trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
 img_norm = trans_norm(img_tensor)
 writer.add_image("Normalize", img_norm)
 writer.close()
 ```

#### Resize

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "./dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)

trans_totensor = transforms.ToTensor()
img_resize_tensor = trans_totensor(img_resize)

writer.add_image("Resize", img_resize_tensor)
writer.close()
```

#### Compose

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "./dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

trans_resize_2 = transforms.Resize((256, 256))
trans_totensor = transforms.ToTensor()
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
print(img_resize_2)

writer.add_image("Resize2", img_resize_2)
writer.close()
```

#### RandomCrop

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "./dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

trans_random = transforms.RandomCrop(512)
trans_totensor = transforms.ToTensor()
trans_compose = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_random = trans_compose(img)
    writer.add_image("random", img_random, i)
writer.close()
```

## torchvision的数据集使用

> torchvision中提供了很多数据集，可以在官网搜索到

### 下载数据集

```python
import torchvision

## 下载数据集
train_set = torchvision.datasets.CIFAR10(root="./tv_dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./tv_dataset", train=False, download=True)

# print输出看一看
print(test_set[0])
print(test_set.classes)
img, target = test_set[0]
print(img)
print(target)
display(img)
```

### 下载数据集的时候转换成tensor

```python
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 下载数据集并转换为tensor
train_set = torchvision.datasets.CIFAR10(root="./tv_dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./tv_dataset", train=False, transform=dataset_transform, download=True)

# 输出到tensorboard里面
writer = SummaryWriter("logs")
for i in range(10):
    image, target = test_set[i] 
    writer.add_image("test_set", image, i)
writer.close()
```

## DataLoader使用

### 基本参数

- batch_size:每个batch有多少个样本
- shuffle:在每个epoch开始的时候，对数据进行重新打乱
- num_workers:这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
- drop_last:如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。

### 使用

```python
import torchvision
from torch.utils.data import DataLoader

# 数据集
test_set = torchvision.datasets.CIFAR10(root="./tv_dataset", train=False, 
                                        transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, 
                         num_workers=0, drop_last=False)

# 测试数据集第一张样本
img, target = test_set[0]
print(img.shape)
print(target)

for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
```

## torch_nn的使用

### 定义模型

定义一个很简单的加一的神经网络

```python
from torch import nn
import torch

class AddOne(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        output = input + 1
        return output
      
add = AddOne()
x = torch.tensor(1.0)
output = add(x)
print(output)
```

## 卷积层

### 常用卷积

- nn.Conv1d：代表一维
- nn.Conv2d：代表二维
- nn.Conv3d：代表三维

### torch.nn.functional

这里以torch.nn.functional为样例，但是掌握torch.nn就行

- stride=n 每次卷积窗口走移动n步

  ![conv](https://cdn.jsdelivr.net/gh/mouweng/FigureBed/img/202204270946370.jpg)

```python
import torch
import torch.nn.functional as F
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)
print(output)
output2 = F.conv2d(input, kernel, stride=2)
print(output2)
'''
[output]:
torch.Size([1, 1, 5, 5])
torch.Size([1, 1, 3, 3])
tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])
tensor([[[[10, 12],
          [13,  3]]]])
'''
```

- padding=1  矩阵上下左右扩展一行一列

  ![](https://cdn.jsdelivr.net/gh/mouweng/FigureBed/img/202204271001587.jpg)

```python
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
'''
[output]:
tensor([[[[ 1,  3,  4, 10,  8],
          [ 5, 10, 12, 12,  6],
          [ 7, 18, 16, 16,  8],
          [11, 13,  9,  3,  4],
          [14, 13,  9,  7,  4]]]])
'''
```

### Conv2d

- in_channels：输入图像的channels
- out_channels：输出的channels
- kernel_size：卷积核大小，其中的数不需要我们设置
- stride：移动步数
- padding：是否扩展

```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch import nn
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root="./tv_dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        return x
    
tudui = Tudui()

step = 0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs, targes = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("outputImage",output, step)
    step += 1

writer.close()
```

输出结果

![](https://cdn.jsdelivr.net/gh/mouweng/FigureBed/img/202204271030392.jpg)

## 池化层

### MaxPool2d

为什么要进行最大池化：保持数据特征但是减小数据量

- **kernel_size** – the size of the window to take a max over
- **stride** – the stride of the window. Default value is `kernel_size`
- **padding** – implicit zero padding to be added on both sides
- **ceil_mode** – when True, will use ceil instead of floor to compute the output shape

![](https://cdn.jsdelivr.net/gh/mouweng/FigureBed/img/202204271039504.jpg)

 **处理矩阵**

```python
import torch
from torch.nn import MaxPool2d
from torch import nn

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
        
    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()
output = tudui(input)
print(output)

'''
tensor([[[[2., 3.],
          [5., 1.]]]])
'''
```

**处理图片**（相当于把图片给缩小，但是保存了一定的图片特征）

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter("logs")

img_path = "./dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(img_tensor)
writer.add_image("pool_pre", img_tensor)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
        
    def forward(self, input):
        output = self.maxpool1(input)
        return output


img_pool = tudui(img_tensor)
writer.add_image("pool_end", img_pool)
writer.close()
```

## 非线性激活

### ReLU

![](https://cdn.jsdelivr.net/gh/mouweng/FigureBed/img/202204271056066.jpg)

```python
from torch.nn import ReLU
import torch

input = torch.tensor([-2, -1, 0, 1, 2])
m = ReLU()
print(input)
output = m(input)
print(output)

'''
tensor([-2, -1,  0,  1,  2])
tensor([0, 0, 0, 1, 2])
'''
```

### Sigmoid

![](https://cdn.jsdelivr.net/gh/mouweng/FigureBed/img/202204271102246.jpg)

```python
from torch.nn import Sigmoid
import torch

input = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float)
m = Sigmoid()
print(input)
output = m(input)
print(output)

'''
tensor([-2., -1.,  0.,  1.,  2.])
tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
'''
```

## 线性层

```python
from torch.nn import Linear
import torch

m = Linear(4, 2)
input = torch.tensor([[1,2,3,4]], dtype=torch.float)
output = m(input)
print(input)
print(output)

'''
tensor([[1., 2., 3., 4.]])
tensor([[-1.8285,  0.9838]], grad_fn=<AddmmBackward>)
'''
```

## 搭建神经网络

![神经网络结构](https://cdn.jsdelivr.net/gh/mouweng/FigureBed/img/202204271338545.jpg)

### 普通写法

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=5, padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(32, 32, kernel_size=5, padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(32, 64, kernel_size=5, padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
tudui = Tudui()
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)
```

### Sequential简化写法

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x
    
tudui = Tudui()
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)
```

## 损失函数与反向传播

- 计算实际输出和目标之间的差距
- 为我们更新输出提供一定的依据（反向传播）

### L1Loss

[公式详见文档](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss)

```python
import torch
from torch.nn import L1Loss

inputs = torch.tensor([1, 2, 3], dtype=torch.float)
targets = torch.tensor([1, 2, 5], dtype=torch.float)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(inputs, targets)
print(result)
'''
tensor(0.6667)
[(1-1)+(2-2)+(5-3)]/3 = 0.6667 
'''
```

### MSELoss

[公式详见文档](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)

```python
import torch
from torch.nn import MSELoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float)
targets = torch.tensor([1, 2, 5], dtype=torch.float)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = MSELoss()
result = loss(inputs, targets)
print(result)
'''
tensor(1.3333)
[0+0+2*2]/3 = 1.3333
'''
```

### CrossEntropyLoss

交叉商损失函数-主要用作分类问题 [详见文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)

```python
import torch
from torch.nn import CrossEntropyLoss

'''
person(0) dog(1) cat(2)
  0.1      0.2    0.3
'''

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])

x = torch.reshape(x, (1, 3))

loss = CrossEntropyLoss()
result = loss(x, y)
print(result)

'''
tensor(1.1019)
'''
```

### 神经网络计算loss

```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss

dataset = torchvision.datasets.CIFAR10(root="./tv_dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x

    
loss = CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs, targets)
    print(result_loss)
```

## 优化器

```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss

dataset = torchvision.datasets.CIFAR10(root="./tv_dataset", 
                                       train=False, 
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x

    
loss = CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)

'''
tensor(360.9059, grad_fn=<AddBackward0>)
tensor(356.4598, grad_fn=<AddBackward0>)
tensor(340.2340, grad_fn=<AddBackward0>)
tensor(321.7190, grad_fn=<AddBackward0>)
tensor(312.0399, grad_fn=<AddBackward0>)
tensor(303.5707, grad_fn=<AddBackward0>)
tensor(293.7194, grad_fn=<AddBackward0>)
tensor(285.3713, grad_fn=<AddBackward0>)
tensor(278.6652, grad_fn=<AddBackward0>)
tensor(272.7400, grad_fn=<AddBackward0>)
tensor(267.5092, grad_fn=<AddBackward0>)
tensor(262.5300, grad_fn=<AddBackward0>)
tensor(257.5721, grad_fn=<AddBackward0>)
tensor(252.6479, grad_fn=<AddBackward0>)
tensor(247.9521, grad_fn=<AddBackward0>)
tensor(243.6132, grad_fn=<AddBackward0>)
tensor(239.6037, grad_fn=<AddBackward0>)
tensor(235.8739, grad_fn=<AddBackward0>)
tensor(232.3826, grad_fn=<AddBackward0>)
tensor(229.0876, grad_fn=<AddBackward0>)
'''
```

## 现有网络模型修改及使用

```python
import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

# 如何利用现有的网络来进行添加
print(vgg16_true)
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 如何利用现有的网络进行修改
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
```

## 模型的保存和加载

#### 保存/加载-方式1

```python
import torch
import torchvision
vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1，保存模型结构+参数
torch.save(vgg16, "vgg16_method1.pth")
# 加载方式1
model = torch.load("vgg16_method1.pth")
print(model)
```

#### 保存/加载-方式2

```python
import torch
import torchvision
vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式2, 保存参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
# 加载方式2
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
```

## 完整的模型训练套路

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss


# 获取数据集
train_data = torchvision.datasets.CIFAR10(root="./tv_dataset", train=True, transform=ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./tv_dataset", train=False, transform=ToTensor(), download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集长度 : {}, 测试集长度 : {}".format(train_data_size, test_data_size))

# 使用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = CrossEntropyLoss()

# 定义优化器
learning_rate = 1e-2
optim = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 训练轮数
epoch = 10
# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0

# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print ("-----第 {} 轮训练开始-----".format(i + 1))
    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1 
        if total_train_step % 100 == 0:
            print("训练次数: {}, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        
    
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
            
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1
    
    torch.save(tudui, "./train_model/tudui_{}.pth".format(i + 1))
    print("模型已保存")

writer.close()
```

## 模型验证

```python
import torchvision
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss

# 加载图片
image_path = "./dataset/my_test_image/dog.jpg"
image = Image.open(image_path)
# 图片格式化处理
transform = Compose([Resize((32, 32)),ToTensor()])
image = transform(image)
image = torch.reshape(image, (1,3,32,32))
print(image.shape)

# 搭建神经模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x

# 预测
model = torch.load("./train_model/tudui_10.pth")
print(model)
output = model(image)
print(output)

print(output.argmax(1))
```
