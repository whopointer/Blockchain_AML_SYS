"""
GNN + DGI + 随机森林联合训练模型
用于区块链AML反洗钱检测的端到端解决方案
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .gnn_model import ImprovedGNNModel
from .dgi import ImprovedDGI
from .random_forest_classifier import DownstreamRandomForest
from .trainer import ImprovedTrainer


class GNNDGIRandomForest:
    """
    GNN + DGI + 随机森林联合训练模型
    
    训练流程:
    1. 使用DGI进行自监督预训练，学习图结构表示
    2. 提取DGI学习到的节点嵌入
    3. 使用随机森林在嵌入空间进行下游分类任务
    """
    
    def __init__(self,
                 num_features: int,
                 num_classes: int = 2,
                 hidden_channels: int = 64,
                 gnn_layers: int = 3,
                 dropout: float = 0.1,
                 dgi_pooling: str = 'mean',
                 rf_n_estimators: int = 200,
                 rf_max_depth: int = 15,
                 device: str = 'auto',
                 checkpoint_dir: str = 'checkpoints',
                 experiment_name: str = 'gnn_dgi_rf'):
        """
        初始化联合训练模型
        """
        # 设备设置
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        
        # 创建检查点目录
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 1. 创建GNN编码器
        self.gnn_model = ImprovedGNNModel(
            num_features=num_features,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            num_layers=gnn_layers,
            dropout=dropout,
            use_multi_scale=False,
            use_attention_pooling=False
        ).to(self.device)
        
        # 2. 创建DGI模型
        self.dgi_model = ImprovedDGI(
            gnn_model=self.gnn_model,
            hidden_channels=hidden_channels,
            pooling_strategy=dgi_pooling,
            corruption_method='shuffle',
            temperature=0.5
        ).to(self.device)
        
        # 3. 创建随机森林分类器
        self.rf_classifier = DownstreamRandomForest(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 训练状态
        self.dgi_trained = False
        self.rf_trained = False
        self.training_history = {}
        
        # 日志设置
        self.logger = logging.getLogger(f"{__name__}.{experiment_name}")
    
    def train_dgi(self, 
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  num_epochs: int = 100,
                  learning_rate: float = 0.001,
                  patience: int = 15) -> Dict[str, Any]:
        """
        第一阶段：使用DGI进行自监督预训练
        """
        self.logger.info("开始DGI自监督预训练...")
        
        # 创建DGI优化器
        dgi_optimizer = optim.Adam(
            self.dgi_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 创建训练器
        dgi_trainer = ImprovedTrainer(
            model=self.dgi_model,
            optimizer=dgi_optimizer,
            device=self.device,
            checkpoint_dir=str(self.checkpoint_dir / "dgi"),
            patience=patience,
            gradient_clip_value=1.0
        )
        
        # 训练DGI
        dgi_results = dgi_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            save_best=True
        )
        
        self.dgi_trained = True
        self.training_history['dgi'] = dgi_results
        
        # 保存DGI模型
        self.save_dgi_model()
        
        self.logger.info(f"DGI预训练完成，最佳验证AUC: {dgi_results['best_val_auc']:.4f}")
        return dgi_results
    
    def extract_embeddings(self, 
                          data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        从训练好的DGI模型中提取节点嵌入
        """
        if not self.dgi_trained:
            raise ValueError("DGI模型尚未训练，请先调用train_dgi方法")
        
        self.logger.info("提取节点嵌入...")
        
        self.dgi_model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx % 10 == 0:
                    self.logger.info(f"处理批次 {batch_idx+1}/{len(data_loader)}")
                
                batch = batch.to(self.device)
                
                # 获取节点嵌入
                embeddings = self.dgi_model.get_embeddings(batch.x, batch.edge_index)
                
                # 如果是批处理图，需要保持节点到图的映射
                if hasattr(batch, 'batch'):
                    all_embeddings.append(embeddings.cpu().numpy())
                    all_labels.append(batch.y.cpu().numpy())
                else:
                    all_embeddings.append(embeddings.cpu().numpy())
                    all_labels.append(batch.y.cpu().numpy())
        
        # 合并所有批次
        embeddings = np.vstack(all_embeddings)
        labels = np.hstack(all_labels)
        
        self.logger.info(f"嵌入提取完成，形状: {embeddings.shape}")
        self.logger.info(f"标签分布: {np.bincount(labels.astype(int))}")
        
        return embeddings, labels
    
    def train_random_forest(self,
                           train_embeddings: np.ndarray,
                           train_labels: np.ndarray,
                           val_embeddings: np.ndarray,
                           val_labels: np.ndarray,
                           tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        第二阶段：使用随机森林进行下游分类训练
        """
        self.logger.info("开始随机森林分类训练...")
        
        if tune_hyperparameters:
            # 超参数调优
            tuning_results = self.rf_classifier.hyperparameter_tuning(
                train_embeddings, train_labels, self.logger
            )
            self.logger.info(f"最佳超参数: {tuning_results['best_params']}")
        
        # 训练随机森林
        rf_results = self.rf_classifier.train(
            train_embeddings, train_labels,
            val_embeddings, val_labels,
            self.logger
        )
        
        self.rf_trained = True
        self.training_history['random_forest'] = rf_results
        
        # 保存随机森林模型
        self.save_rf_model()
        
        self.logger.info(f"随机森林训练完成，验证AUC: {rf_results['val_auc']:.4f}")
        return rf_results
    
    def end_to_end_train(self,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        dgi_epochs: int = 100,
                        rf_hyperparameter_tuning: bool = False,
                        learning_rate: float = 0.001,
                        patience: int = 15) -> Dict[str, Any]:
        """
        端到端训练流程
        """
        self.logger.info("开始端到端训练...")
        
        # 第一阶段：DGI预训练
        dgi_results = self.train_dgi(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=dgi_epochs,
            learning_rate=learning_rate,
            patience=patience
        )
        
        # 提取嵌入
        train_embeddings, train_labels = self.extract_embeddings(train_loader)
        val_embeddings, val_labels = self.extract_embeddings(val_loader)
        
        # 第二阶段：随机森林训练
        rf_results = self.train_random_forest(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            tune_hyperparameters=rf_hyperparameter_tuning
        )
        
        # 保存完整的训练结果
        self.save_training_results()
        
        combined_results = {
            'dgi': dgi_results,
            'random_forest': rf_results,
            'overall_performance': {
                'dgi_val_auc': dgi_results['best_val_auc'],
                'rf_val_auc': rf_results['val_auc'],
                'rf_val_ap': rf_results['val_ap']
            }
        }
        
        self.logger.info("端到端训练完成！")
        self.logger.info(f"DGI验证AUC: {dgi_results['best_val_auc']:.4f}")
        self.logger.info(f"随机森林验证AUC: {rf_results['val_auc']:.4f}")
        self.logger.info(f"随机森林验证AP: {rf_results['val_ap']:.4f}")
        
        return combined_results
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用训练好的模型进行预测
        """
        if not self.rf_trained:
            raise ValueError("模型尚未训练完成，请先调用end_to_end_train方法")
        
        # 提取嵌入
        embeddings, _ = self.extract_embeddings(data_loader)
        
        # 随机森林预测
        predictions = self.rf_classifier.predict(embeddings)
        probabilities = self.rf_classifier.predict_proba(embeddings)
        
        return predictions, probabilities
    
    def save_dgi_model(self):
        """保存DGI模型"""
        dgi_path = self.checkpoint_dir / f"{self.experiment_name}_dgi_model.pth"
        torch.save({
            'dgi_state_dict': self.dgi_model.state_dict(),
            'gnn_state_dict': self.gnn_model.state_dict(),
            'model_config': {
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'hidden_channels': self.hidden_channels
            }
        }, dgi_path)
        self.logger.info(f"DGI模型已保存到: {dgi_path}")
    
    def save_rf_model(self):
        """保存随机森林模型"""
        rf_path = self.checkpoint_dir / f"{self.experiment_name}_random_forest.joblib"
        self.rf_classifier.save_model(str(rf_path))
        self.logger.info(f"随机森林模型已保存到: {rf_path}")
    
    def save_training_results(self):
        """保存训练结果"""
        results_path = self.checkpoint_dir / f"{self.experiment_name}_training_results.json"
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        serializable_history = {}
        for model_name, results in self.training_history.items():
            serializable_history[model_name] = {}
            for key, value in results.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_history[model_name][key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_history[model_name][key] = value.tolist()
                else:
                    serializable_history[model_name][key] = value
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"训练结果已保存到: {results_path}")
    
    def load_dgi_model(self, model_path: str):
        """加载DGI模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.dgi_model.load_state_dict(checkpoint['dgi_state_dict'])
        self.gnn_model.load_state_dict(checkpoint['gnn_state_dict'])
        self.dgi_trained = True
        self.logger.info(f"DGI模型已从 {model_path} 加载")
    
    def load_rf_model(self, model_path: str):
        """加载随机森林模型"""
        self.rf_classifier.load_model(model_path)
        self.rf_trained = True
        self.logger.info(f"随机森林模型已从 {model_path} 加载")
    
    def load_full_model(self, experiment_name: str = None):
        """加载完整的训练模型"""
        if experiment_name:
            self.experiment_name = experiment_name
        
        # 加载DGI模型
        dgi_path = self.checkpoint_dir / f"{self.experiment_name}_dgi_model.pth"
        if dgi_path.exists():
            self.load_dgi_model(str(dgi_path))
        
        # 加载随机森林模型
        rf_path = self.checkpoint_dir / f"{self.experiment_name}_random_forest.joblib"
        if rf_path.exists():
            self.load_rf_model(str(rf_path))
        
        # 加载训练结果
        results_path = self.checkpoint_dir / f"{self.experiment_name}_training_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)
            self.logger.info(f"训练结果已从 {results_path} 加载")


def create_gnn_dgi_rf_model(**kwargs) -> GNNDGIRandomForest:
    """
    工厂函数：创建GNN+DGI+随机森林联合训练模型
    """
    return GNNDGIRandomForest(**kwargs)