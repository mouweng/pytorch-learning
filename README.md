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

transforms的结构和用法