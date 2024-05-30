from typing import List, Tuple
import torch
from torch.utils.data import DataLoader

from fedlab.utils.functional import AverageMeter

def detail_evaluate(model: torch.nn.Module, criterion, test_loader: DataLoader, num_classes: int):
    """计算详细的准确率，包括每个类别的准确率和总体准确率
    
    Returns:
        result: list, 包含每个类别的准确率,总体准确率
    """
    model.eval()
    device = next(model.parameters()).device
    
    total_acc = AverageMeter()
    detail_acc_: List[AverageMeter] = [AverageMeter() for _ in range(num_classes)]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            
            _, pred = torch.max(outputs, 1)
            total_acc.update(torch.sum(pred.eq(labels)).item(), len(labels))
            for label, pred in zip(labels, pred):
                detail_acc_[label].update(int(label == pred))
    
    result = []
    # 计算每个类别的准确率
    for i in range(num_classes):
        result.append(round(detail_acc_[i].avg, 4))
    # 计算总体准确率
    result.append(round(total_acc.avg, 4))
    
    return result


def err_tolerate_evaluate(model: torch.nn.Module, criterion, test_loader: DataLoader) -> Tuple[float, float, float]:
    model.eval()
    device = next(model.parameters()).device

    correct_1 = 0
    correct_5 = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()

            # compute top 1
            correct_1 += correct[:, :1].sum()

            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    top1_acc = correct_1 / len(test_loader.dataset)
    top5_acc = correct_5 / len(test_loader.dataset)
    return avg_loss, top1_acc.item(), top5_acc.item()


def evaluate(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: torch.nn.CrossEntropyLoss, 
             device: str = "cpu"):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        num_samples = 0
        correct = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels)
            pred = torch.softmax(outputs, dim=-1).argmax(-1)
            # print(pred, labels)
            correct += torch.eq(pred, labels).int().sum()
            num_samples += inputs.size(-1)
    
    model.train()
    return total_loss, correct / num_samples
