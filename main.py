import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import thop
import numpy as np
import random
from medmnist import *
from sklearn.metrics import roc_auc_score


#  1. Set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Accelerate convolutions


set_seed(42)

#  2. Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  3. Data preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


#  4. MedMNIST data loading
trainset = PathMNIST(root='./data', split='train', transform=transform, download=True)
testset = PathMNIST(root='./data', split='test', transform=transform, download=True)
train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# Get number of classes and channels
num_classes = 14
n_channels = 1

#  5. ViT-Lite Block
class ViTLiteBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super(ViTLiteBlock, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # Depthwise Conv
        self.norm = nn.BatchNorm2d(dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim * 4, dropout=0.1)

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = self.norm(x)
        x = x * self.se(x)
        x = self.proj(x)

        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1)  # (N, B, C)
        x = self.transformer(x)
        x = x.permute(1, 2, 0).view(b, c, h, w)  # Restore original shape

        return x + res


#  6. ViT-Lite
class ViTLite(nn.Module):
    def __init__(self, num_classes=10, n_channels=3):
        super(ViTLite, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            ViTLiteBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.stage2 = nn.Sequential(
            ViTLiteBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.stage3 = nn.Sequential(
            ViTLiteBlock(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return x


#  7. Compute FLOPs & Params
def compute_model_metrics(model, n_channels=n_channels):
    inputs = torch.randn(1, n_channels, 32, 32).to(device)
    flops, params = thop.profile(model, inputs=(inputs,), verbose=False)
    print(f" Model FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")
    return flops, params


#  8. Training & Evaluation
def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, criterion, epochs=200):
    best_test_acc1, best_test_acc5 = 0.0, 0.0
    train_losses = []  # List to record training loss for each epoch
    test_losses = []  # List to record test loss for each epoch
    all_labels = []
    all_preds = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Ensure labels are 1D tensor
            labels = labels.squeeze()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()

        #  Evaluation
        model.eval()
        correct_top1, correct_top5, total = 0, 0, 0
        epoch_labels = []
        epoch_preds = []
        epoch_test_loss = 0.0  # To calculate test loss for each epoch
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Ensure labels are 1D tensor
                labels = labels.squeeze()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item()

                _, predicted = outputs.topk(5, 1, True, True)
                correct_top1 += (predicted[:, 0] == labels).sum().item()
                correct_top5 += (predicted == labels.view(-1, 1)).sum().item()
                total += labels.size(0)

                # Collect labels and predictions for ROC calculation
                epoch_labels.append(labels.cpu().numpy())
                epoch_preds.append(F.softmax(outputs, dim=1).cpu().numpy())

        test_acc1 = 100. * correct_top1 / total
        test_acc5 = 100. * correct_top5 / total
        best_test_acc1 = max(best_test_acc1, test_acc1)
        best_test_acc5 = max(best_test_acc5, test_acc5)

        test_losses.append(epoch_test_loss / len(test_loader))  # Record test loss for each epoch

        all_labels.extend(np.concatenate(epoch_labels, axis=0))
        all_preds.extend(np.concatenate(epoch_preds, axis=0))

    # Only take the probability of the positive class (class 1)
    # binary_preds = np.array(all_preds)[:, 1]
    # Compute AUC
    auc_score = roc_auc_score(np.array(all_labels), np.array(all_preds), multi_class='ovr', average='macro')
    # auc_score = roc_auc_score(np.array(all_labels), binary_preds)
    print(f'AUC Score (Macro-average): {auc_score:.4f}')

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Train Loss', color='blue')
    # plt.plot(range(epochs), test_losses, label='Test Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n Best Test Top-1: {best_test_acc1:.2f}%, Best Test Top-5: {best_test_acc5:.2f}%")


#  9. Run the model
if __name__ == '__main__':
    model = ViTLite(num_classes=num_classes, n_channels=n_channels).to(device)
    compute_model_metrics(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss()
    train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, criterion, epochs=200)
