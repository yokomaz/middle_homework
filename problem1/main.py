from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import itertools

# 标准划分：每类 30 张训练，其余测试
def split_caltech101(dataset, train_per_class=30):
    targets = dataset.targets
    class_indices = {}
    for idx, label in enumerate(targets):
        class_indices.setdefault(label, []).append(idx)

    train_idx, test_idx = [], []
    for indices in class_indices.values():
        random.shuffle(indices)
        train_idx.extend(indices[:train_per_class])
        test_idx.extend(indices[train_per_class:])

    return Subset(dataset, train_idx), Subset(dataset, test_idx)

def train_model(model, train_loader, val_loader, device, writer, lr=1e-3, num_epochs=25):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    best_val_accuracy = 0.0
    best_accuracy_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batchs = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for inputs, labels in loop:
            # print(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_batchs += inputs.size(0)
            loop.set_postfix(loss=running_loss / total_batchs, lr=lr)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device=device)
        generate_loss = running_loss/len(train_loader.dataset)
        train_loss_list.append(generate_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        writer.add_scalar('Loss/Training set', generate_loss, epoch)
        writer.add_scalar('Loss/Validation set', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation set', val_acc, epoch)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_accuracy_model = model.state_dict()
        print(f"Epoch {epoch+1}, Train Loss: {generate_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    writer.close()
    return (best_val_accuracy, best_accuracy_model)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    correct = 0.
    total_loss = 0.
    total_sample = 0.
    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"evaluating")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total_sample += labels.size(0)
    avg_loss = total_loss / total_sample
    avg_correct = 100 * correct / total_sample
    return avg_loss, avg_correct

def init_model(pretrained: bool):
    if pretrained:
        model = models.resnet18(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 101)
    if not pretrained:
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 101)
    
    return model

def main():
    # 初始化SummaryWriter
    # 设置超参数
    print(f"Setting hyperparameters")
    lr_list = [1e-3, 1e-4, 1e-5]
    bs_list = [64]#[32, 64]
    num_epochs_list = [500]#[50, 100, 200]
    # num_epochs_list = [3]
    hyper_list = list(itertools.product(lr_list, bs_list, num_epochs_list))
    # print(f"Hyperparameters are learning rate={lr}, batch size={batch_size}, number of epochs={num_epochs}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # 数据预处理
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])
    # 加载数据集
    data_dir = './caltech-101/101_ObjectCategories'
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_data, val_data = split_caltech101(dataset)

    # 用于记录最佳模型的参数
    best_val_accuracy_pretrained = 0.0
    best_val_accuracy_unpretrained = 0.0
    best_model_pretrained = None
    best_model_unpretrained = None

    for index, (lr, batch_size, num_epochs) in enumerate(hyper_list):
        print(f"Num {index} Hyperparameters grid {lr, batch_size, num_epochs}")
        print("Preparing model")
        model_pretrained = init_model(pretrained=True)
        model_unpretrained = init_model(pretrained=False)

        print(f"Preparing train/val dataloader")
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        print(f"Start training")
        writer = SummaryWriter(log_dir=f"run/pretrained_lr_{lr}_batch_size_{batch_size}_num_epochs_{num_epochs}")
        print(f"Training pretrained model")
        accuracy_pretrained, model_pret = train_model(model_pretrained, train_loader, val_loader, writer=writer, lr=lr, device=device, num_epochs=num_epochs)
        if accuracy_pretrained > best_val_accuracy_pretrained:
            best_val_accuracy_pretrained = accuracy_pretrained
            best_model_pretrained = model_pret
        
        writer = SummaryWriter(log_dir=f"run/unpretrained_lr_{lr}_batch_size_{batch_size}_num_epochs_{num_epochs}")
        print(f"Training unpretrained model")
        accuracy_unpretrained, model_unpret = train_model(model_unpretrained, train_loader, val_loader, writer=writer, lr=lr, device=device, num_epochs=num_epochs)
        if accuracy_unpretrained > best_val_accuracy_unpretrained:
            best_val_accuracy_unpretrained = accuracy_unpretrained
            best_model_unpretrained = model_unpret
    
    torch.save(best_model_pretrained, "best_model_pretrained.pth")
    torch.save(best_model_unpretrained, "best_model_unpretrained.pth")
    print("model saved")

if __name__ == "__main__":
    main()