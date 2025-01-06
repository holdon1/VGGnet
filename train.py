from tqdm import tqdm
import time
import torch
"""
:param epochs: 训练轮次
:param criterion: 损失函数
:param dataloader: 数据加载器
:param model: 模型
:param optimizer 优化器
:param device 设备（Gpu 还是 Cpu）
"""


def cifar10_train(epochs, criterion, dataloader, model, optimizer, device, save_path):
    """
    训练逻辑：
    使用批次加载训练数据。
    通过前向传播获得预测结果。
    通过反向传播调整模型权重。
    累计和记录训练过程的损失和准确率。 
    打印每个 epoch 的性能指标，方便观察训练效果。
    """""
    best_accuracy = 0.0  # 初始化最佳准确率
    for epoch in range(epochs):
        start_time = time.time()
        print(f"当前轮次：{epoch + 1}")
        # 训练模式
        model.to(device)
        model.train()
        train_loss = 0  # 该轮次所有样本的损失值
        correct = 0  # 计算预测正确的个数
        total = 0  # 总样本个数
        # 按批次训练数据
        # 在每个轮次内使用 tqdm 包装整个 dataloader
        with tqdm(total=len(dataloader), desc="Training", leave=True) as pbar:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累加损失和准确率
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # 计算轮次的平均损失和准确率
            avg_loss = train_loss / len(dataloader)
            accuracy = 100. * correct / total

            # 更新进度条并显示轮次的损失和准确率
            pbar.set_postfix(loss=avg_loss, accuracy=accuracy)
            pbar.update(len(dataloader))
        # 计算平均损失和准确率
        train_loss = train_loss / total
        train_accuracy = correct / total * 100
        # 结束时间
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Time: {epoch_time:.2f}s")
        # 保存最佳模型
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型，准确率：{best_accuracy:.2f}%")