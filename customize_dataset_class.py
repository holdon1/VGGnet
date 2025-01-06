from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

"""
:param root: 数据集的根目录,指定的下载路径
:param train: 是否加载训练集
:param transform: 数据增强/预处理方法
"""

# 自定义cifar10的数据集类
class Cifar10(Dataset):
    def __init__(self, root, train=True, transform=None):
        # 初始化属性，实例化这个类之后，数据集就加载到data中
        self.data = CIFAR10(root=root, train=train, transform=transform, download=True)

    def __len__(self):
        return len(self.data)

    # 返回指定数据的图像张量和标签
    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label
