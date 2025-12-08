"""
Focal Loss损失函数实现
用于处理类别不平衡问题，特别适用于极端不平衡的数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss损失函数
    
    Focal Loss通过降低易分类样本的权重，使模型更关注难分类样本
    对于类别不平衡问题特别有效
    """
    
    def __init__(self, 
                 alpha: float = 1.0, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean',
                 pos_weight: Optional[float] = None):
        """
        初始化Focal Loss
        
        Args:
            alpha: 平衡因子，用于平衡正负样本 (默认: 1.0)
            gamma: 调节因子，控制难易样本的权重 (默认: 2.0)
            reduction: 损失聚合方式 ('none', 'mean', 'sum')
            pos_weight: 正样本权重，用于进一步平衡类别
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 模型预测值 (logits)
            targets: 真实标签 (0或1)
            
        Returns:
            Focal Loss值
        """
        # 计算概率
        probs = torch.sigmoid(inputs)
        
        # 计算交叉熵
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        
        # 计算p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 计算alpha_t
        if self.pos_weight is not None:
            alpha_t = self.alpha * targets * self.pos_weight + \
                     self.alpha * (1 - targets)
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BalancedFocalLoss(nn.Module):
    """
    平衡的Focal Loss，自动计算类别权重
    """
    
    def __init__(self, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean',
                 auto_weight: bool = True):
        """
        初始化平衡Focal Loss
        
        Args:
            gamma: 调节因子
            reduction: 损失聚合方式
            auto_weight: 是否自动计算类别权重
        """
        super(BalancedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.auto_weight = auto_weight
        self.pos_weight = None
        
    def set_class_weights(self, targets: torch.Tensor):
        """
        根据标签分布设置类别权重
        
        Args:
            targets: 标签张量
        """
        if not self.auto_weight:
            return
            
        # 计算类别频率
        pos_count = (targets == 1).sum().float()
        neg_count = (targets == 0).sum().float()
        total_count = pos_count + neg_count
        
        # 计算权重 (逆频率)
        pos_weight = neg_count / (pos_count + 1e-8)
        neg_weight = pos_count / (neg_count + 1e-8)
        
        # 归一化权重
        self.pos_weight = pos_weight / (pos_weight + neg_weight)
        self.neg_weight = neg_weight / (pos_weight + neg_weight)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算平衡Focal Loss
        """
        if self.auto_weight and self.pos_weight is None:
            self.set_class_weights(targets)
        
        # 计算概率
        probs = torch.sigmoid(inputs)
        
        # 计算交叉熵
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        
        # 计算p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 计算alpha_t (类别权重)
        if self.auto_weight and self.pos_weight is not None:
            alpha_t = self.pos_weight * targets + self.neg_weight * (1 - targets)
        else:
            alpha_t = targets + 0.5 * (1 - targets)  # 默认平衡
        
        # 计算Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    加权二元交叉熵损失
    简单但有效的类别不平衡处理方法
    """
    
    def __init__(self, pos_weight: Optional[float] = None, reduction: str = 'mean'):
        """
        初始化加权BCE Loss
        
        Args:
            pos_weight: 正样本权重
            reduction: 损失聚合方式
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算加权BCE Loss
        """
        if self.pos_weight is not None:
            # 使用PyTorch内置的pos_weight参数
            loss = F.binary_cross_entropy_with_logits(
                inputs, targets.float(), 
                pos_weight=torch.tensor(self.pos_weight, device=inputs.device),
                reduction=self.reduction
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                inputs, targets.float(), reduction=self.reduction
            )
        
        return loss


def create_loss_function(loss_type: str = 'focal', **kwargs) -> nn.Module:
    """
    工厂函数：创建损失函数
    
    Args:
        loss_type: 损失函数类型 ('focal', 'balanced_focal', 'weighted_bce', 'bce')
        **kwargs: 损失函数参数
        
    Returns:
        损失函数实例
    """
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'balanced_focal':
        return BalancedFocalLoss(**kwargs)
    elif loss_type == 'weighted_bce':
        return WeightedBCELoss(**kwargs)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


__all__ = [
    'FocalLoss', 
    'BalancedFocalLoss', 
    'WeightedBCELoss', 
    'create_loss_function'
]