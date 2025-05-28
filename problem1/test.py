import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import random
from tqdm import tqdm

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

def init_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 101)
    return model

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Testing")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_model_unpretrained.pth"  # 修改为另一个模型文件也可以

    # 加载数据
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    data_dir = './caltech-101/101_ObjectCategories'
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # 固定随机种子，保证与训练集划分一致
    random.seed(42)
    _, test_data = split_caltech101(dataset)

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 初始化模型结构并加载参数
    model = init_model().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 测试
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
