import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# 设置超参数
batch_size = 64
epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 准备数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# 加载数据集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 定义简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x



# TensorBoard
writer = {
    'SGD_loss': SummaryWriter("./logs/SGD_loss"),  # 必须要不同的writer
    'SGD_acc': SummaryWriter("./logs/SGD_acc"),
    'SGD_precision': SummaryWriter("./logs/SGD_precision"),
    'SGD_recall': SummaryWriter("./logs/SGD_recall"),
    'SGD_f1': SummaryWriter("./logs/SGD_f1"),

    'SGD-low_loss': SummaryWriter("./logs/SGD-low_loss"),  # 必须要不同的writer
    'SGD-low_acc': SummaryWriter("./logs/SGD-low_acc"),
    'SGD-low_precision': SummaryWriter("./logs/SGD-low_precision"),
    'SGD-low_recall': SummaryWriter("./logs/SGD-low_recall"),
    'SGD-low_f1': SummaryWriter("./logs/SGD-low_f1"),

    'SGD-momentum_loss': SummaryWriter("./logs/SGD-momentum_loss"),  # 必须要不同的writer
    'SGD-momentum_acc': SummaryWriter("./logs/SGD-momentum_acc"),
    'SGD-momentum_precision': SummaryWriter("./logs/SGD-momentum_precision"),
    'SGD-momentum_recall': SummaryWriter("./logs/SGD-momentum_recall"),
    'SGD-momentum_f1': SummaryWriter("./logs/SGD-momentum_f1"),

    'SGD-momentum-low_loss': SummaryWriter("./logs/SGD-momentum-low_loss"),  # 必须要不同的writer
    'SGD-momentum-low_acc': SummaryWriter("./logs/SGD-momentum-low_acc"),
    'SGD-momentum-low_precision': SummaryWriter("./logs/SGD-momentum-low_precision"),
    'SGD-momentum-low_recall': SummaryWriter("./logs/SGD-momentum-low_recall"),
    'SGD-momentum-low_f1': SummaryWriter("./logs/SGD-momentum-low_f1"),

    'Adam_loss': SummaryWriter("./logs/Adam_loss"),
    'Adam_acc': SummaryWriter("./logs/Adam_acc"),
    'Adam_precision': SummaryWriter("./logs/Adam_precision"),
    'Adam_recall': SummaryWriter("./logs/Adam_recall"),
    'Adam_f1': SummaryWriter("./logs/Adam_f1"),

    'RMSprop_loss': SummaryWriter("./logs/RMSprop_loss"),
    'RMSprop_acc': SummaryWriter("./logs/RMSprop_acc"),
    'RMSprop_precision': SummaryWriter("./logs/RMSprop_precision"),
    'RMSprop_recall': SummaryWriter("./logs/RMSprop_recall"),
    'RMSprop_f1': SummaryWriter("./logs/RMSprop_f1"),

    'Adagrad_loss': SummaryWriter("./logs/Adagrad_loss"),
    'Adagrad_acc': SummaryWriter("./logs/Adagrad_acc"),
    'Adagrad_precision': SummaryWriter("./logs/Adagrad_precision"),
    'Adagrad_recall': SummaryWriter("./logs/Adagrad_recall"),
    'Adagrad_f1': SummaryWriter("./logs/Adagrad_f1"),

    'Adagrad-low_loss': SummaryWriter("./logs/Adagrad-low_loss"),
    'Adagrad-low_acc': SummaryWriter("./logs/Adagrad-low_acc"),
    'Adagrad-low_precision': SummaryWriter("./logs/Adagrad-low_precision"),
    'Adagrad-low_recall': SummaryWriter("./logs/Adagrad-low_recall"),
    'Adagrad-low_f1': SummaryWriter("./logs/Adagrad-low_f1"),

    'model': SummaryWriter("./logs/model"),

}

# 训练和评估函数
def train_and_evaluate(optimizer_name,learning_rate):
    model = SimpleCNN().to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # 优化器
    optimizer = None
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD-low':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD-momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    elif optimizer_name == 'SGD-momentum-low':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adagrad-low':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

    train_loss, test_acc, precision_list, recall_list,f1_list = [], [], [], [], []

    if optimizer_name == 'SGD':  # 记录一次模型结构即可
        inputs = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-10 的单个示例输入
        writer['model'].add_graph(model, inputs)

    for epoch in range(epochs):
        print(f"\n[{optimizer_name}] Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if (batch_idx + 1) % 100 == 0 or batch_idx == len(train_loader) - 1:
            #     print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_loss.append(epoch_loss / len(train_loader))

        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # test_acc.append(  correct / total)
        test_acc.append(accuracy_score(all_labels,all_preds))
        precision_list.append(precision_score(all_labels, all_preds, average='macro', zero_division=0))
        recall_list.append(recall_score(all_labels, all_preds, average='macro'))
        f1_list.append(f1_score(all_labels, all_preds, average='macro'))

        print(f"Epoch {epoch + 1} - Accuracy: {test_acc[-1]:.16f}, Precision: {precision_list[-1]:.4f}, Recall: {recall_list[-1]:.16f},F1: {f1_list[-1]:.4f}")

        if optimizer_name == 'SGD':
            writer['SGD_loss'].add_scalar("Loss", loss.item(), epoch)     # 要想显示在一张图 表格名字要一样
            writer['SGD_acc'].add_scalar("Value", test_acc[-1], epoch)
            writer['SGD_precision'].add_scalar("Value", precision_list[-1], epoch)
            writer['SGD_recall'].add_scalar("Value", recall_list[-1], epoch)
            writer['SGD_f1'].add_scalar("Value", f1_list[-1], epoch)
        elif optimizer_name == 'SGD-low':
            writer['SGD-low_loss'].add_scalar("Loss", loss.item(), epoch)  # 要想显示在一张图 表格名字要一样
            writer['SGD-low_acc'].add_scalar("Value", test_acc[-1], epoch)
            writer['SGD-low_precision'].add_scalar("Value", precision_list[-1], epoch)
            writer['SGD-low_recall'].add_scalar("Value", recall_list[-1], epoch)
            writer['SGD-low_f1'].add_scalar("Value", f1_list[-1], epoch)
        elif optimizer_name == 'SGD-momentum':
            writer['SGD-momentum_loss'].add_scalar("Loss", loss.item(), epoch)  # 要想显示在一张图 表格名字要一样
            writer['SGD-momentum_acc'].add_scalar("Value", test_acc[-1], epoch)
            writer['SGD-momentum_precision'].add_scalar("Value", precision_list[-1], epoch)
            writer['SGD-momentum_recall'].add_scalar("Value", recall_list[-1], epoch)
            writer['SGD-momentum_f1'].add_scalar("Value", f1_list[-1], epoch)
        elif optimizer_name == 'SGD-momentum-low':
            writer['SGD-momentum-low_loss'].add_scalar("Loss", loss.item(), epoch)  # 要想显示在一张图 表格名字要一样
            writer['SGD-momentum-low_acc'].add_scalar("Value", test_acc[-1], epoch)
            writer['SGD-momentum-low_precision'].add_scalar("Value", precision_list[-1], epoch)
            writer['SGD-momentum-low_recall'].add_scalar("Value", recall_list[-1], epoch)
            writer['SGD-momentum-low_f1'].add_scalar("Value", f1_list[-1], epoch)
        elif optimizer_name == 'Adam':
            writer['Adam_loss'].add_scalar("Loss", loss.item(), epoch)  # 要想显示在一张图 表格名字要一样
            writer['Adam_acc'].add_scalar("Value", test_acc[-1], epoch)
            writer['Adam_precision'].add_scalar("Value", precision_list[-1], epoch)
            writer['Adam_recall'].add_scalar("Value", recall_list[-1], epoch)
            writer['Adam_f1'].add_scalar("Value", f1_list[-1], epoch)
        elif optimizer_name == 'RMSprop':
            writer['RMSprop_loss'].add_scalar("Loss", loss.item(), epoch)  # 要想显示在一张图 表格名字要一样
            writer['RMSprop_acc'].add_scalar("Value", test_acc[-1], epoch)
            writer['RMSprop_precision'].add_scalar("Value", precision_list[-1], epoch)
            writer['RMSprop_recall'].add_scalar("Value", recall_list[-1], epoch)
            writer['RMSprop_f1'].add_scalar("Value", f1_list[-1], epoch)
        elif optimizer_name == 'Adagrad':
            writer['Adagrad_loss'].add_scalar("Loss", loss.item(), epoch)  # 要想显示在一张图 表格名字要一样
            writer['Adagrad_acc'].add_scalar("Value", test_acc[-1], epoch)
            writer['Adagrad_precision'].add_scalar("Value", precision_list[-1], epoch)
            writer['Adagrad_recall'].add_scalar("Value", recall_list[-1], epoch)
            writer['Adagrad_f1'].add_scalar("Value", f1_list[-1], epoch)
        elif optimizer_name == 'Adagrad-low':
            writer['Adagrad-low_loss'].add_scalar("Loss", loss.item(), epoch)  # 要想显示在一张图 表格名字要一样
            writer['Adagrad-low_acc'].add_scalar("Value", test_acc[-1], epoch)
            writer['Adagrad-low_precision'].add_scalar("Value", precision_list[-1], epoch)
            writer['Adagrad-low_recall'].add_scalar("Value", recall_list[-1], epoch)
            writer['Adagrad-low_f1'].add_scalar("Value", f1_list[-1], epoch)

    return train_loss, test_acc, precision_list, recall_list

# 比较不同优化器
optimizers = ['SGD','SGD-low','SGD-momentum','SGD-momentum-low', 'Adam', 'RMSprop','Adagrad','Adagrad-low']
#设置初始学习率
learning_rates = {
    'SGD': 0.005,
    'SGD-low': 0.001,#
    'SGD-momentum': 0.005,#
    'SGD-momentum-low': 0.001,#
    'Adam': 0.001,
    'RMSprop': 0.001,
    'Adagrad': 0.01,
    'Adagrad-low': 0.001#
}
results = {}

for opt in optimizers:
    print(f"\n=== Training with optimizer: {opt} ===")
    lr = learning_rates[opt]
    results[opt] = train_and_evaluate(opt,lr)

# 保存模型
# torch.save(SimpleCNN, "./SimpleCNN_model.pth")

for key, writer_instance in writer.items():
    writer_instance.close()
