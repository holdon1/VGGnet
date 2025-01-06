"""
测试函数需要一准确率返回值，相当于把最佳模型跑一次测试集，并且关注准确率
"""''
import torch


def cifar10_test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct = (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'测试集准确率：{accuracy:.4f}')
    return accuracy
