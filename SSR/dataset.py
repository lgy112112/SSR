import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

# 自定义 PyTorch Dataset，用于加载脑部病变图像和掩膜
class BrainLesionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # 读取CSV文件，包含图像和掩膜的路径信息
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像和掩膜路径
        image_path = self.data.iloc[idx]['image_path']
        mask_path = self.data.iloc[idx]['mask_path']
        
        # 打开图像和掩膜，并转换为灰度图像
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # 如果有transform，对图像和掩膜进行变换
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 返回图像、掩膜及路径信息
        return image, mask.long(), image_path, mask_path  # 确保掩膜为整数类型

# 定义图像和掩膜的预处理与增强
base_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 创建完整数据集实例
csv_file = 'lgg-mri-segmentation/kaggle_3m/metadata.csv'  # 替换为你的CSV文件路径
full_dataset = BrainLesionDataset(csv_file=csv_file, transform=base_transform)

# 将数据集划分为训练集、验证集和测试集
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# 定义批量大小
batch_size = 8  # 可根据硬件资源调整批量大小

# 创建DataLoader，用于批量加载训练、验证和测试数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 示例：检查训练集中的图像和掩膜大小
for images, masks, image_paths, mask_paths in train_loader:
    print(images.shape, masks.shape)
    break

# 定义复习数据集类，用于低信心样本的复习
class ReviewDataset(Dataset):
    def __init__(self, samples, transform=None):
        # 使用传入的低信心样本列表创建DataFrame
        self.data = pd.DataFrame(samples)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图像和掩膜路径
        image_path = self.data.iloc[idx]['image_path']
        mask_path = self.data.iloc[idx]['mask_path']
        
        # 打开图像和掩膜，并转换为灰度图像
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # 应用数据增强变换
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # 返回图像和掩膜
        return image, mask.long()

# 定义复习阶段的数据增强变换
review_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
