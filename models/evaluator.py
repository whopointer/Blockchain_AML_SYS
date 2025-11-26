import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
from torch_geometric.data import Data, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
from datetime import datetime


class ModelEvaluator:
    """
    模型评估器：提供全面的模型评估功能
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: torch.device,
                 threshold: float = 0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()
        
        # 评估历史
        self.evaluation_history = []
        
    def evaluate_dgi_model(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        评估DGI模型
        """
        all_scores = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                
                # 正样本分数
                pos_score = self.model(data.x, data.edge_index, data.batch)
                
                # 负样本分数
                neg_edge_index = self.model.corruption(
                    data.edge_index, data.num_nodes, data.batch
                )
                neg_score = self.model(data.x, neg_edge_index, data.batch)
                
                # 计算损失
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(pos_score, neg_score)
                    total_loss += loss.item()
                
                # 收集分数和标签
                all_scores.extend(pos_score.cpu().numpy())
                all_scores.extend(neg_score.cpu().numpy())
                all_labels.extend([1] * len(pos_score))
                all_labels.extend([0] * len(neg_score))
        
        # 计算指标
        scores_array = np.array(all_scores)
        labels_array = np.array(all_labels)
        
        auc = roc_auc_score(labels_array, scores_array)
        ap = average_precision_score(labels_array, scores_array)
        
        metrics = {
            'auc': auc,
            'average_precision': ap,
            'loss': total_loss / len(data_loader) if total_loss > 0 else 0,
            'num_samples': len(all_scores)
        }
        
        return metrics
    
    def evaluate_classification_model(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        评估分类模型
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                
                # 前向传播
                outputs = self.model(data.x, data.edge_index, data.batch)
                
                # 计算损失
                loss = F.cross_entropy(outputs, data.y)
                total_loss += loss.item()
                
                # 获取预测和概率
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 正类概率
                all_labels.extend(data.y.cpu().numpy())
        
        # 转换为numpy数组
        predictions_array = np.array(all_predictions)
        probabilities_array = np.array(all_probabilities)
        labels_array = np.array(all_labels)
        
        # 计算各种指标
        metrics = self._compute_classification_metrics(
            labels_array, predictions_array, probabilities_array
        )
        metrics['loss'] = total_loss / len(data_loader)
        metrics['num_samples'] = len(all_labels)
        
        return metrics
    
    def _compute_classification_metrics(self, 
                                       y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       y_prob: np.ndarray) -> Dict[str, float]:
        """
        计算分类指标
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            'average_precision': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        # 添加每个类别的指标
        if len(np.unique(y_true)) > 1:
            for i in range(len(np.unique(y_true))):
                class_precision = precision_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
                class_recall = recall_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
                class_f1 = f1_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
                
                metrics[f'class_{i}_precision'] = class_precision
                metrics[f'class_{i}_recall'] = class_recall
                metrics[f'class_{i}_f1'] = class_f1
        
        return metrics
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        自动选择评估方法
        """
        if hasattr(self.model, 'compute_loss'):
            return self.evaluate_dgi_model(data_loader)
        else:
            return self.evaluate_classification_model(data_loader)
    
    def generate_confusion_matrix(self, data_loader: DataLoader) -> Tuple[np.ndarray, List[str]]:
        """
        生成混淆矩阵
        """
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                
                if hasattr(self.model, 'compute_loss'):
                    # DGI模型不适用混淆矩阵
                    return None, None
                
                outputs = self.model(data.x, data.edge_index, data.batch)
                predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_predictions)
        class_names = [f'Class_{i}' for i in range(len(np.unique(all_labels)))]
        
        return cm, class_names
    
    def plot_roc_curve(self, data_loader: DataLoader, save_path: Optional[str] = None):
        """
        绘制ROC曲线
        """
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                
                if hasattr(self.model, 'compute_loss'):
                    # DGI模型
                    pos_score = self.model(data.x, data.edge_index, data.batch)
                    neg_edge_index = self.model.corruption(data.edge_index, data.num_nodes, data.batch)
                    neg_score = self.model(data.x, neg_edge_index, data.batch)
                    
                    all_probabilities.extend(pos_score.cpu().numpy())
                    all_probabilities.extend(neg_score.cpu().numpy())
                    all_labels.extend([1] * len(pos_score))
                    all_labels.extend([0] * len(neg_score))
                else:
                    # 分类模型
                    outputs = self.model(data.x, data.edge_index, data.batch)
                    probabilities = F.softmax(outputs, dim=1)[:, 1]
                    
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_labels.extend(data.y.cpu().numpy())
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        auc = roc_auc_score(all_labels, all_probabilities)
        
        # 绘图
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, data_loader: DataLoader, save_path: Optional[str] = None):
        """
        绘制Precision-Recall曲线
        """
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                
                if hasattr(self.model, 'compute_loss'):
                    pos_score = self.model(data.x, data.edge_index, data.batch)
                    neg_edge_index = self.model.corruption(data.edge_index, data.num_nodes, data.batch)
                    neg_score = self.model(data.x, neg_edge_index, data.batch)
                    
                    all_probabilities.extend(pos_score.cpu().numpy())
                    all_probabilities.extend(neg_score.cpu().numpy())
                    all_labels.extend([1] * len(pos_score))
                    all_labels.extend([0] * len(neg_score))
                else:
                    outputs = self.model(data.x, data.edge_index, data.batch)
                    probabilities = F.softmax(outputs, dim=1)[:, 1]
                    
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_labels.extend(data.y.cpu().numpy())
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(all_labels, all_probabilities)
        ap = average_precision_score(all_labels, all_probabilities)
        
        # 绘图
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR曲线已保存到: {save_path}")
        
        plt.show()
    
    def print_metrics(self, metrics: Dict[str, float], title: str = "Evaluation Metrics"):
        """
        打印评估指标
        """
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:.4f}")
            else:
                print(f"{key:25s}: {value}")
        
        print(f"{'='*50}\n")
    
    def save_evaluation_report(self, 
                              metrics: Dict[str, float],
                              save_path: str,
                              additional_info: Optional[Dict] = None):
        """
        保存评估报告
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'DGI' if hasattr(self.model, 'compute_loss') else 'Classification',
            'device': str(self.device),
            'threshold': self.threshold,
            'metrics': metrics
        }
        
        if additional_info:
            report.update(additional_info)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"评估报告已保存到: {save_path}")
    
    def compare_models(self, 
                       models: List[torch.nn.Module],
                       model_names: List[str],
                       data_loader: DataLoader) -> pd.DataFrame:
        """
        比较多个模型的性能
        """
        results = []
        
        for model, name in zip(models, model_names):
            # 临时替换当前模型
            original_model = self.model
            self.model = model
            model.eval()
            
            # 评估
            metrics = self.evaluate(data_loader)
            metrics['model_name'] = name
            
            results.append(metrics)
            
            # 恢复原始模型
            self.model = original_model
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        df = df.set_index('model_name')
        
        return df


# 向后兼容的函数
def evaluate(model, data, device):
    """向后兼容的评估函数"""
    from torch_geometric.data import DataLoader
    
    # 创建单个数据的DataLoader
    loader = DataLoader([data], batch_size=1)
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(loader)
    
    return metrics.get('auc', 0.0)


def create_evaluator(model: torch.nn.Module,
                    device: Optional[torch.device] = None,
                    threshold: float = 0.5) -> ModelEvaluator:
    """
    工厂函数：创建评估器
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return ModelEvaluator(model, device, threshold)