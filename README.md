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

