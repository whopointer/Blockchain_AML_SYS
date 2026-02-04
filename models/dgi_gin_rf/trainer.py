import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time
import os
from tqdm import tqdm


class MetricsTracker:
    """
    训练指标跟踪器
    """
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'train_auc': [],
            'val_loss': [],
            'val_auc': [],
            'val_ap': [],
            'learning_rate': []
        }
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def update(self, epoch: int, train_loss: float, train_auc: float,
               val_loss: float, val_auc: float, val_ap: float, lr: float):
        """更新指标"""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_auc'].append(train_auc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_auc'].append(val_auc)
        self.metrics['val_ap'].append(val_ap)
        self.metrics['learning_rate'].append(lr)
        
        # 更新最佳模型
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_epoch = epoch
            self.patience_counter = 0
            return True  # 新的最佳模型
        else:
            self.patience_counter += 1
            return False

    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'best_val_auc': self.best_val_auc,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else 0,
            'final_val_auc': self.metrics['val_auc'][-1] if self.metrics['val_auc'] else 0
        }


class ImprovedTrainer:
    """
    改进的训练器，支持多种训练策略和高级功能
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[Any] = None,
                 checkpoint_dir: str = 'checkpoints',
                 patience: int = 10,
                 gradient_clip_value: float = 1.0):
        
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.gradient_clip_value = gradient_clip_value
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 指标跟踪器
        self.metrics_tracker = MetricsTracker()
        
        # 训练状态
        self.current_epoch = 0
        self.training_history = []

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_scores = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, data in enumerate(progress_bar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # 前向传播
            if hasattr(self.model, 'compute_loss'):
                # DGI模型
                pos_score = self.model(data.x, data.edge_index, data.batch)
                neg_edge_index = self.model.corruption(data.edge_index, data.num_nodes, data.batch)
                neg_score = self.model(data.x, neg_edge_index, data.batch)
                
                # 计算损失
                loss = self.model.compute_loss(pos_score, neg_score)
                
                # 收集分数和标签用于AUC计算
                all_scores.extend(pos_score.cpu().detach().numpy())
                all_scores.extend(neg_score.cpu().detach().numpy())
                all_labels.extend([1] * len(pos_score))
                all_labels.extend([0] * len(neg_score))
            else:
                # 标准分类模型
                output = self.model(data.x, data.edge_index, data.batch)
                loss = F.cross_entropy(output, data.y)
                
                # 收集预测和标签
                pred = F.softmax(output, dim=1)[:, 1]  # 正类概率
                all_scores.extend(pred.cpu().detach().numpy())
                all_labels.extend(data.y.cpu().numpy())
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # 计算训练AUC
        train_auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
        
        return total_loss / len(train_loader), train_auc

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                
                if hasattr(self.model, 'compute_loss'):
                    # DGI模型
                    pos_score = self.model(data.x, data.edge_index, data.batch)
                    neg_edge_index = self.model.corruption(data.edge_index, data.num_nodes, data.batch)
                    neg_score = self.model(data.x, neg_edge_index, data.batch)
                    
                    loss = self.model.compute_loss(pos_score, neg_score)
                    
                    all_scores.extend(pos_score.cpu().numpy())
                    all_scores.extend(neg_score.cpu().numpy())
                    all_labels.extend([1] * len(pos_score))
                    all_labels.extend([0] * len(neg_score))
                else:
                    # 标准分类模型
                    output = self.model(data.x, data.edge_index, data.batch)
                    loss = F.cross_entropy(output, data.y)
                    
                    pred = F.softmax(output, dim=1)[:, 1]
                    all_scores.extend(pred.cpu().numpy())
                    all_labels.extend(data.y.cpu().numpy())
                
                total_loss += loss.item()
        
        # 计算指标
        val_loss = total_loss / len(val_loader)
        val_auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
        val_ap = average_precision_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
        
        return val_loss, val_auc, val_ap

    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int,
              save_best: bool = True) -> Dict[str, Any]:
        """完整训练流程"""
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 训练
            train_loss, train_auc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_auc, val_ap = self.evaluate(val_loader)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 更新指标
            is_best = self.metrics_tracker.update(
                epoch, train_loss, train_auc, val_loss, val_auc, val_ap, current_lr
            )
            
            # 保存最佳模型
            if is_best and save_best:
                self.save_checkpoint('best_model.pth', epoch, val_loss, val_auc)
            
            # 打印训练信息
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Best Val AUC: {self.metrics_tracker.best_val_auc:.4f} (Epoch {self.metrics_tracker.best_epoch + 1})")
            print("-" * 50)
            
            # 早停检查
            if self.metrics_tracker.patience_counter >= self.patience:
                print(f"早停触发！在epoch {epoch + 1}停止训练")
                break
        
        total_time = time.time() - start_time
        print(f"训练完成！总时间: {total_time:.2f}秒")
        
        # 加载最佳模型
        if save_best:
            self.load_checkpoint('best_model.pth')
        
        return self.metrics_tracker.get_summary()

    def save_checkpoint(self, filename: str, epoch: int, loss: float, auc: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'auc': auc,
            'metrics': self.metrics_tracker.metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))

    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"成功加载检查点: {checkpoint_path}")
        else:
            print(f"检查点文件不存在: {checkpoint_path}")


# 向后兼容的函数
def train(model, data_loader, optimizer, device):
    """向后兼容的训练函数"""
    trainer = ImprovedTrainer(model, optimizer, device)
    return trainer.train_epoch(data_loader)[0]


def evaluate(model, data_loader, device):
    """向后兼容的评估函数"""
    trainer = ImprovedTrainer(model, None, device)
    val_loss, val_auc, _ = trainer.evaluate(data_loader)
    return val_auc


def create_trainer(model: torch.nn.Module,
                  learning_rate: float = 0.001,
                  weight_decay: float = 1e-5,
                  scheduler_type: str = 'cosine',
                  device: Optional[torch.device] = None,
                  **kwargs) -> ImprovedTrainer:
    """
    工厂函数：创建训练器
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = None
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    return ImprovedTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        **kwargs
    )