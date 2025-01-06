import model_vgg_11
import train
import dataLoader
import customize_dataset_class
import torch
import torch.nn as nn
import torch.optim as optim
import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model = model_vgg_11.VggNet11(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)
epochs = 1
save_path = 'vgg.pth'
if __name__ == '__main__':
    train.cifar10_train(epochs, criterion, dataLoader.cifar10_train_dataloader, model, optimizer, device)
    # model.load_state_dict(torch.load(save_path))
    # test.cifar10_test(model,dataLoader.cifar10_test_dataloader,device)