"""
两阶段DGI+随机森林模型
第一阶段：DGI+GIN自监督学习节点嵌入
第二阶段：随机森林监督分类（使用嵌入+原始特征）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .dgi import DGIWithGIN, create_dgi_with_gin
from .random_forest_classifier import DownstreamRandomForest
from .trainer import ImprovedTrainer
from .focal_loss import create_loss_function


class TwoStageDGIRandomForest:
    """
    两阶段DGI+随机森林模型
    
    阶段1：使用DGI+GIN进行自监督学习，获取节点嵌入
    阶段2：将嵌入与原始特征拼接，训练随机森林分类器
    """
    
    def __init__(self,
                 num_features: int,
                 num_classes: int = 2,
                 hidden_channels: int = 128,
                 gnn_layers: int = 3,
                 rf_n_estimators: int = 200,
                 rf_max_depth: int = 15,
                 device: str = 'auto',
                 checkpoint_dir: str = 'checkpoints',
                 experiment_name: str = 'two_stage_dgi_rf',
                 balance_strategy: str = 'undersample',
                 loss_type: str = 'balanced_focal'):
        """
        初始化两阶段模型
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
        
        # 阶段1：DGI+GIN模型
        self.dgi_model = DGIWithGIN(
            num_features=num_features,
            hidden_channels=hidden_channels,
            num_layers=gnn_layers,
            pooling_strategy='mean',
            corruption_method='feature_shuffle'
        ).to(self.device)
        
        # 阶段2：随机森林分类器
        self.rf_classifier = DownstreamRandomForest(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 训练状态
        self.stage1_trained = False
        self.stage2_trained = False
        self.training_history = {}
        
        # 平衡策略和损失函数
        self.balance_strategy = balance_strategy
        self.loss_type = loss_type
        
        # 日志设置
        self.logger = logging.getLogger(f"{__name__}.{experiment_name}")
    
    def stage1_self_supervised_training(self, 
                                      train_loader: DataLoader,
                                      val_loader: DataLoader,
                                      num_epochs: int = 100,
                                      learning_rate: float = 0.001,
                                      patience: int = 15) -> Dict[str, Any]:
        """
        第一阶段：DGI自监督训练
        """
        self.logger.info("开始第一阶段：DGI+GIN自监督训练...")
        
        # 创建优化器
        optimizer = optim.Adam(
            self.dgi_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 创建学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.dgi_model.train()
            total_train_loss = 0.0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # 前向传播
                pos_scores, neg_scores = self.dgi_model(
                    batch.x, batch.edge_index, batch.batch
                )
                
                # 计算损失
                loss = self.dgi_model.compute_dgi_loss(pos_scores, neg_scores)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dgi_model.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # 验证阶段
            self.dgi_model.eval()
            total_val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    
                    pos_scores, neg_scores = self.dgi_model(
                        batch.x, batch.edge_index, batch.batch
                    )
                    
                    loss = self.dgi_model.compute_dgi_loss(pos_scores, neg_scores)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_stage1_model()
            else:
                patience_counter += 1
            
            # 日志记录
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )
            
            # 早停
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 冻结GIN编码器
        self.dgi_model.freeze_encoder()
        self.stage1_trained = True
        
        # 保存训练结果
        stage1_results = {
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'final_train_loss': avg_train_loss
        }
        
        self.training_history['stage1'] = stage1_results
        self.logger.info(f"第一阶段训练完成，最佳验证损失: {best_val_loss:.4f}")
        
        return stage1_results
    
    def extract_embeddings(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        提取节点嵌入（用于第二阶段训练）
        
        Returns:
            embeddings: 节点嵌入
            features: 原始特征
            labels: 标签
        """
        if not self.stage1_trained:
            raise ValueError("第一阶段训练未完成，请先进行自监督训练")
        
        self.logger.info("提取节点嵌入...")
        
        self.dgi_model.eval()
        all_embeddings = []
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx % 10 == 0:
                    self.logger.info(f"处理批次 {batch_idx+1}/{len(data_loader)}")
                
                batch = batch.to(self.device)
                
                # 获取DGI嵌入
                embeddings = self.dgi_model.gin_encoder.get_node_embeddings(batch.x, batch.edge_index)
                
                # 获取原始特征
                features = batch.x
                
                # 收集标签
                if batch.y is not None:
                    labels = batch.y.cpu().numpy()
                else:
                    labels = np.zeros(len(embeddings))  # 如果没有标签，用0填充
                
                # 转换为numpy并保存
                all_embeddings.append(embeddings.cpu().numpy())
                all_features.append(features.cpu().numpy())
                all_labels.append(labels)
        
        # 合并所有批次
        embeddings = np.vstack(all_embeddings)
        features = np.vstack(all_features)
        labels = np.hstack(all_labels)
        
        self.logger.info(f"嵌入提取完成，形状: {embeddings.shape}")
        self.logger.info(f"标签分布: {np.bincount(labels.astype(int))}")
        
        return embeddings, features, labels
    
    def stage2_supervised_training(self,
                                train_loader: DataLoader,
                                val_loader: DataLoader,
                                learning_rate: float = 0.001,
                                patience: int = 15,
                                tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        第二阶段：随机森林监督训练
        """
        self.logger.info("开始第二阶段：随机森林监督训练...")
        
        # 提取嵌入和特征
        train_embeddings, train_features, train_labels = self.extract_embeddings(train_loader)
        val_embeddings, val_features, val_labels = self.extract_embeddings(val_loader)
        
        # 应用数据平衡策略
        if self.balance_strategy != 'none':
            self.logger.info(f"应用数据平衡策略: {self.balance_strategy}")
            from data.data_balancer import DataBalancer
            
            # 为训练数据创建临时Data对象以应用平衡
            import torch
            from torch_geometric.data import Data
            
            # 创建包含所有数据的Data对象
            train_data = Data(
                x=torch.tensor(train_features, dtype=torch.float),
                y=torch.tensor(train_labels, dtype=torch.long)
            )
            
            balancer = DataBalancer(strategy=self.balance_strategy)
            balanced_train_data = balancer.balance_graph_data(train_data)
            
            # 获取平衡后的数据索引
            # 需要重新实现平衡逻辑以保持索引一致性
            original_indices = np.arange(len(train_labels))
            balanced_indices = balancer._get_balanced_indices(train_labels)
            
            # 使用平衡后的索引重新提取所有数据
            train_features = train_features[balanced_indices]
            train_labels = train_labels[balanced_indices]
            train_embeddings = train_embeddings[balanced_indices]
            
            self.logger.info(f"平衡后训练数据形状: 特征 {train_features.shape}, 嵌入 {train_embeddings.shape}")
            self.logger.info(f"平衡后标签分布: {np.bincount(train_labels)}")
        
        # 拼接嵌入和原始特征
        train_combined = np.hstack([train_embeddings, train_features])
        val_combined = np.hstack([val_embeddings, val_features])
        
        self.logger.info(f"训练特征维度: {train_combined.shape[1]} "
                        f"(嵌入: {train_embeddings.shape[1]}, "
                        f"原始: {train_features.shape[1]})")
        
        # 训练随机森林
        rf_results = self.rf_classifier.train(
            train_combined, train_labels,
            val_combined, val_labels,
            self.logger,
            tune_hyperparameters=tune_hyperparameters
        )
        
        self.stage2_trained = True
        self.training_history['stage2'] = rf_results
        
        # 保存模型
        self.save_stage2_model()
        
        self.logger.info(f"第二阶段训练完成，验证AUC: {rf_results['val_auc']:.4f}")
        
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
        self.logger.info("开始端到端两阶段训练...")
        
        # 第一阶段：DGI自监督训练
        stage1_results = self.stage1_self_supervised_training(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=dgi_epochs,
            learning_rate=learning_rate,
            patience=patience
        )
        
        # 提取嵌入
        train_embeddings, train_features, train_labels = self.extract_embeddings(train_loader)
        val_embeddings, val_features, val_labels = self.extract_embeddings(val_loader)
        
        # 第二阶段：随机森林监督训练
        stage2_results = self.stage2_supervised_training(
            train_loader=train_loader,
            val_loader=val_loader,
            tune_hyperparameters=rf_hyperparameter_tuning
        )
        
        # 保存完整训练结果
        self.save_training_results()
        
        combined_results = {
            'stage1': stage1_results,
            'stage2': stage2_results,
            'overall_performance': {
                'stage1_val_loss': stage1_results['best_val_loss'],
                'stage2_val_auc': stage2_results['val_auc'],
                'stage2_val_ap': stage2_results['val_ap']
            }
        }
        
        self.logger.info("端到端训练完成！")
        self.logger.info(f"第一阶段最佳损失: {stage1_results['best_val_loss']:.4f}")
        self.logger.info(f"第二阶段验证AUC: {stage2_results['val_auc']:.4f}")
        self.logger.info(f"第二阶段验证AP: {stage2_results['val_ap']:.4f}")
        
        return combined_results
    
    def predict(self, data_loader, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            data_loader: 数据加载器
            threshold: 分类阈值，如果为None则使用模型保存的最优阈值
            
        Returns:
            predictions: 预测结果
            probabilities: 预测概率
        """
        self.dgi_model.eval()
        
        all_embeddings = []
        all_original_features = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                # 获取节点嵌入
                embeddings = self.dgi_model.gin_encoder.get_node_embeddings(batch.x, batch.edge_index)
                all_embeddings.append(embeddings.cpu().numpy())
                all_original_features.append(batch.x.cpu().numpy())
        
        # 合并所有批次
        embeddings = np.vstack(all_embeddings)
        original_features = np.vstack(all_original_features)
        
        # 拼接嵌入和原始特征
        combined_features = np.hstack([embeddings, original_features])
        
        # 随机森林预测（使用阈值）
        if hasattr(self.rf_classifier, 'predict_with_threshold'):
            predictions = self.rf_classifier.predict_with_threshold(combined_features, threshold)
        else:
            predictions = self.rf_classifier.predict(combined_features)
        
        probabilities = self.rf_classifier.predict_proba(combined_features)
        
        return predictions, probabilities
    
    def find_optimal_threshold(self, val_loader, metric: str = 'f1') -> float:
        """
        在验证集上寻找最优分类阈值
        
        Args:
            val_loader: 验证数据加载器
            metric: 优化指标
            
        Returns:
            最优阈值
        """
        self.dgi_model.eval()
        
        all_embeddings = []
        all_original_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # 获取节点嵌入
                embeddings = self.dgi_model.gin_encoder.get_node_embeddings(batch.x, batch.edge_index)
                all_embeddings.append(embeddings.cpu().numpy())
                all_original_features.append(batch.x.cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())
        
        # 合并所有批次
        embeddings = np.vstack(all_embeddings)
        original_features = np.vstack(all_original_features)
        labels = np.hstack(all_labels)
        
        # 拼接嵌入和原始特征
        combined_features = np.hstack([embeddings, original_features])
        
        # 寻找最优阈值
        optimal_threshold = self.rf_classifier.find_optimal_threshold(
            combined_features, labels, metric, self.logger
        )
        
        return optimal_threshold
    
    def save_stage1_model(self):
        """保存第一阶段模型"""
        stage1_path = self.checkpoint_dir / f"{self.experiment_name}_stage1_dgi.pth"
        torch.save({
            'dgi_state_dict': self.dgi_model.state_dict(),
            'model_config': {
                'num_features': self.num_features,
                'hidden_channels': self.hidden_channels,
                'num_layers': self.dgi_model.gin_encoder.num_layers
            }
        }, stage1_path)
        self.logger.info(f"第一阶段模型已保存到: {stage1_path}")
    
    def save_stage2_model(self):
        """保存第二阶段模型"""
        stage2_path = self.checkpoint_dir / f"{self.experiment_name}_stage2_rf.joblib"
        self.rf_classifier.save_model(str(stage2_path))
        self.logger.info(f"第二阶段模型已保存到: {stage2_path}")
    
    def save_training_results(self):
        """保存训练结果"""
        results_path = self.checkpoint_dir / f"{self.experiment_name}_training_results.json"
        
        # 转换numpy类型为Python原生类型
        serializable_history = {}
        for stage_name, results in self.training_history.items():
            serializable_history[stage_name] = {}
            for key, value in results.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_history[stage_name][key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_history[stage_name][key] = value.tolist()
                else:
                    serializable_history[stage_name][key] = value
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"训练结果已保存到: {results_path}")
    
    def load_stage1_model(self, model_path: str):
        """加载第一阶段模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.dgi_model.load_state_dict(checkpoint['dgi_state_dict'])
        self.dgi_model.freeze_encoder()
        self.stage1_trained = True
        self.logger.info(f"第一阶段模型已从 {model_path} 加载")
    
    def load_stage2_model(self, model_path: str):
        """加载第二阶段模型"""
        self.rf_classifier.load_model(model_path)
        self.stage2_trained = True
        self.logger.info(f"第二阶段模型已从 {model_path} 加载")
    
    def load_full_model(self, experiment_name: str = None):
        """加载完整的两阶段模型"""
        if experiment_name:
            self.experiment_name = experiment_name
        
        # 加载第一阶段模型
        stage1_path = self.checkpoint_dir / f"{self.experiment_name}_stage1_dgi.pth"
        if stage1_path.exists():
            self.load_stage1_model(str(stage1_path))
        
        # 加载第二阶段模型
        stage2_path = self.checkpoint_dir / f"{self.experiment_name}_stage2_rf.joblib"
        if stage2_path.exists():
            self.load_stage2_model(str(stage2_path))
        
        # 加载训练结果
        results_path = self.checkpoint_dir / f"{self.experiment_name}_training_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)
            self.logger.info(f"训练结果已从 {results_path} 加载")


def create_two_stage_dgi_rf(**kwargs) -> TwoStageDGIRandomForest:
    """
    工厂函数：创建两阶段DGI+随机森林模型
    """
    return TwoStageDGIRandomForest(**kwargs)