from SSR.dataset import BrainLesionDataset
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from monai.networks.nets import UNETR, BasicUNet
from monai.losses import DiceCELoss
from SSR.pipeline import train_epoch, val_epoch
import mlflow
from IPython.display import clear_output

# 定义图像和掩膜的预处理与增强
base_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 创建完整数据集实例
csv_file = 'Dataset_BUSI_with_GT/metadata.csv'
full_dataset = BrainLesionDataset(csv_file=csv_file, transform=base_transform)

# 将数据集划分为训练集、验证集和测试集
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# 定义批量大小
batch_size = 8  # 可根据硬件资源调整批量大小

# 创建DataLoader，用于批量加载训练、验证和测试数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

# 示例：检查训练集中的图像和掩膜大小
for images, masks, image_paths, mask_paths in train_loader:
    print(images.shape, masks.shape)
    break



# 配置设备、模型和损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BasicUNet(
    in_channels=1,
    out_channels=1,
    features=(32, 32, 64, 128, 256, 32),
    # img_size=256,
    # dropout_rate=0.25,
    spatial_dims=2,
).to(device)

optimizer = Adam(model.parameters(), lr=0.001)
criterion = DiceCELoss()

# 定义训练参数
num_epochs = 30
confidence_threshold = 0.6  # 低信心阈值

# 启动 MLflow 实验
mlflow.set_experiment("Breast Ultrasound Segmentation")  # 设置实验名称

with mlflow.start_run(run_name="BasicUNet Baseline"):
    # 记录参数
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("num_epochs", num_epochs)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # 1. 训练阶段
        train_loss, train_dice, _ = train_epoch(
            model=model, 
            train_loader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            confidence_threshold=confidence_threshold
        )
        print(f"Training Loss: {train_loss:.4f}, Training Dice: {train_dice:.4f}")

        # 2. 验证阶段
        val_loss, val_dice = val_epoch(
            model=model, 
            val_loader=val_loader, 
            criterion=criterion, 
            device=device
        )
        print(f"Validation Loss: {val_loss:.4f}, Validation Dice Score: {val_dice:.4f}")

        # 记录指标
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_dice", train_dice, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_dice", val_dice, step=epoch)

        print("-" * 40)
        # clear_output()
    # 保存模型
    mlflow.pytorch.log_model(model, "model")
