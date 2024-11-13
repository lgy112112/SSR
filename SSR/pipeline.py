from monai.metrics import DiceMetric
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化 DiceMetric
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

def train_epoch(model, train_loader, optimizer, criterion, device, confidence_threshold=0.6):
    model.train()
    running_loss = 0.0
    low_confidence_samples = []  # 存储低信心样本
    dice_metric.reset()  # 每个 epoch 重置 DiceMetric

    print("Starting training epoch...")
    for batch_idx, (images, masks, image_paths, mask_paths) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        outputs = torch.sigmoid(model(images))
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate batch loss
        running_loss += loss.item()
        
        # Update dice metric with batch predictions and masks
        preds = (outputs > 0.5).float()
        dice_metric(y_pred=preds, y=masks)  # 累积计算当前批次的 Dice score
        batch_dice_score = dice_metric.aggregate(reduction="mean").item()  # 获取当前批次的 Dice score

        # 检查低信心样本
        low_conf_sample_count = 0
        for i in range(images.size(0)):
            single_pred = preds[i].unsqueeze(0)
            single_mask = masks[i].unsqueeze(0)
            single_dice_score = dice_metric(y_pred=single_pred, y=single_mask)  # 计算单个样本的 Dice score
            dice_value = single_dice_score.mean().item()  # 将 dice 转换为标量
            if dice_value < confidence_threshold:
                low_confidence_samples.append({'image_path': image_paths[i], 'mask_path': mask_paths[i]})
                low_conf_sample_count += 1

        # 显示当前批次的损失和 Dice，及低信心样本数量
        print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}, Dice: {batch_dice_score:.4f}, "
              f"Low confidence samples: {low_conf_sample_count}")

    # Epoch metrics
    avg_loss = running_loss / len(train_loader)
    avg_dice_score = dice_metric.aggregate().item()  # 获取整个 epoch 的平均 Dice score
    dice_metric.reset()  # 重置 Dice metric，以便用于下一轮 epoch

    print(f"Training epoch completed - Average Loss: {avg_loss:.4f}, Average Dice: {avg_dice_score:.4f}")
    
    return avg_loss, avg_dice_score, low_confidence_samples




def val_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_metric.reset()  # 每个 epoch 重置 DiceMetric

    print("Starting validation epoch...")
    with torch.no_grad():
        for batch_idx, (images, masks, _, _) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Calculate Dice score for validation metrics
            preds = (outputs > 0.5).float()
            dice_metric(y_pred=preds, y=masks)  # 累积当前批次的 Dice score
            batch_dice_score = dice_metric.aggregate(reduction="mean").item()  # 获取当前批次的 Dice score
            
            print(f"Validation Batch {batch_idx + 1}/{len(val_loader)} - Loss: {loss.item():.4f}, Dice: {batch_dice_score:.4f}")

    # Aggregate the Dice score for the entire validation epoch
    avg_val_dice = dice_metric.aggregate().item()  # 获取平均 Dice score
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation epoch completed - Average Loss: {avg_val_loss:.4f}, Average Dice Score: {avg_val_dice:.4f}")

    return avg_val_loss, avg_val_dice


def review_epoch(model, review_loader, optimizer, criterion, device):
    model.train()
    review_loss = 0.0
    dice_metric.reset()  # 每个 review epoch 重置 DiceMetric

    print("Starting review epoch on low confidence samples...")
    for batch_idx, (images, masks) in enumerate(review_loader):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = torch.sigmoid(model(images))
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track review loss
        review_loss += loss.item()

        # Calculate Dice score for review metrics
        preds = (outputs > 0.5).float()
        dice_metric(y_pred=preds, y=masks)  # 累积当前批次的 Dice score
        batch_dice_score = dice_metric.aggregate(reduction="mean").item()  # 获取当前批次的 Dice score
        
        print(f"Review Batch {batch_idx + 1}/{len(review_loader)} - Loss: {loss.item():.4f}, Dice: {batch_dice_score:.4f}")

    # Aggregate the Dice score for the entire review epoch
    avg_review_dice = dice_metric.aggregate().item()  # 获取平均 Dice score
    avg_review_loss = review_loss / len(review_loader)
    print(f"Review epoch completed - Average Loss: {avg_review_loss:.4f}, Average Dice: {avg_review_dice:.4f}")

    return avg_review_loss, avg_review_dice
