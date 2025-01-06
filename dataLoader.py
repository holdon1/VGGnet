import customize_dataset_class

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# Cifar10数据的预处理
cifar10_train_and_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 适应vgg模型输入尺寸
    transforms.ToTensor(),  # 转换张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# Cifar10训练集加载
cifar10_train_dataset = customize_dataset_class.Cifar10('./data',train=True,transform=cifar10_train_and_test_transform)
cifar10_train_dataloader = DataLoader(cifar10_train_dataset,batch_size=64,shuffle=True)

# Cifar10测试集加载
cifar10_test_dataset = customize_dataset_class.Cifar10('./data',train=False,transform=cifar10_train_and_test_transform)
cifar10_test_dataloader = DataLoader(cifar10_test_dataset,batch_size=64,shuffle=True)

