"""
随机森林分类器模块
用于DGI预训练后的下游分类任务
"""

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import json
import logging
from typing import Tuple, Dict, Any, Optional


class DownstreamRandomForest:
    """
    下游随机森林分类器
    用于基于DGI嵌入的异常检测
    """
    
    def __init__(self, 
                 n_estimators: int = 200,
                 max_depth: int = 15,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 class_weight: str = None,  # 移除自动平衡，使用自定义权重
                 **kwargs):
        """
        初始化随机森林分类器
        """
        # 使用balanced_subsample而不是balanced，以减少对多数类的偏向
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
            bootstrap=True,
            max_features='sqrt',
            **kwargs
        )
        
        self.is_trained = False
        self.feature_importance = None
        self.training_stats = {}
        self.optimal_threshold = 0.5  # 默认阈值
        
    def extract_embeddings(self, 
                          dgi_model: torch.nn.Module, 
                          data_loader, 
                          device: torch.device,
                          logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从DGI模型中提取节点嵌入
        """
        dgi_model.eval()
        all_embeddings = []
        all_labels = []
        
        try:
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if logger:
                        logger.info(f"处理批次 {i+1}/{len(data_loader)}")
                    
                    batch = batch.to(device)
                    # 获取节点嵌入
                    embeddings = dgi_model.gnn_model.get_node_embeddings(batch.x, batch.edge_index)
                    all_embeddings.append(embeddings.cpu().numpy())
                    # 确保标签是整数类型
                    labels = batch.y.cpu().numpy().astype(int)
                    all_labels.append(labels)
                    
                    # 释放内存
                    del embeddings, labels
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # 合并所有批次的嵌入和标签
            embeddings = np.vstack(all_embeddings)
            labels = np.hstack(all_labels)
            
            if logger:
                logger.info(f"标签分布: {np.bincount(labels)}")
                logger.info(f"嵌入统计: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
            
            return embeddings, labels
        
        except Exception as e:
            if logger:
                logger.error(f"提取嵌入时出错: {e}")
            raise e
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              X_val: np.ndarray, 
              y_val: np.ndarray,
              logger: Optional[logging.Logger] = None,
              tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        训练随机森林分类器
        """
        if logger:
            logger.info("开始训练随机森林分类器...")
        
        # 检查类别分布
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            error_msg = f"训练数据只有一个类别: {unique_classes}"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)
        
        if logger:
            logger.info(f"训练数据类别分布: {np.bincount(y_train)}")
            logger.info(f"验证数据类别分布: {np.bincount(y_val)}")
        
        # 超参数调优
        if tune_hyperparameters:
            if logger:
                logger.info("开始随机森林超参数调优...")
            
            # 进行超参数调优
            tuning_results = self._hyperparameter_tuning(X_train, y_train, logger)
            
            # 更新分类器参数
            if 'best_params' in tuning_results:
                self.classifier.set_params(**tuning_results['best_params'])
                if logger:
                    logger.info(f"最佳超参数: {tuning_results['best_params']}")
        
        # 计算自定义类别权重以平衡数据
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # 更新分类器的类别权重
        self.classifier.class_weight = class_weight_dict
        
        if logger:
            logger.info(f"计算的类别权重: {class_weight_dict}")
        
        # 训练模型
        if logger:
            logger.info("训练随机森林模型...")
        
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # 验证集预测
        val_pred = self.classifier.predict(X_val)
        val_pred_proba = self.classifier.predict_proba(X_val)[:, 1]
        
        # 计算指标
        try:
            val_auc = roc_auc_score(y_val, val_pred_proba) if len(set(y_val)) > 1 else 0.0
            val_ap = average_precision_score(y_val, val_pred_proba) if len(set(y_val)) > 1 else 0.0
        except Exception as e:
            if logger:
                logger.warning(f"计算AUC/AP时出错: {e}")
            val_auc = 0.0
            val_ap = 0.0
        
        # 特征重要性
        self.feature_importance = self.classifier.feature_importances_
        top_features = np.argsort(self.feature_importance)[-10:][::-1]
        
        # 训练统计
        self.training_stats = {
            'val_auc': val_auc,
            'val_ap': val_ap,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0],
            'top_features': top_features.tolist(),
            'feature_importance': self.feature_importance[top_features].tolist()
        }
        
        if logger:
            logger.info(f"随机森林验证AUC: {val_auc:.4f}")
            logger.info(f"随机森林验证AP: {val_ap:.4f}")
            logger.info(f"Top 10 重要特征索引: {top_features}")
            logger.info(f"Top 10 特征重要性: {self.feature_importance[top_features]}")
            
            # 打印分类报告
            try:
                logger.info("验证集分类报告:")
                report = classification_report(y_val, val_pred, target_names=['正常(0)', '异常(1)'], zero_division=0)
                logger.info(f"\n{report}")
            except Exception as e:
                logger.warning(f"生成分类报告时出错: {e}")
        
        return self.training_stats
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        return self.classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        return self.classifier.predict_proba(X)
    
    def find_optimal_threshold(self, 
                              X_val: np.ndarray, 
                              y_val: np.ndarray,
                              metric: str = 'f1',
                              logger: Optional[logging.Logger] = None) -> float:
        """
        寻找最优分类阈值
        
        Args:
            X_val: 验证集特征
            y_val: 验证集标签
            metric: 优化指标，可选'f1', 'precision', 'recall'
            logger: 日志记录器
            
        Returns:
            最优阈值
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # 获取预测概率
        y_proba = self.predict_proba(X_val)[:, 1]
        
        # 测试不同阈值，扩大搜索范围并降低步长
        thresholds = np.arange(0.05, 0.95, 0.02)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred, average='binary', zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred, average='binary', zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred, average='binary', zero_division=0)
            else:
                raise ValueError(f"不支持的指标: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        
        if logger:
            logger.info(f"最优阈值: {best_threshold:.3f} (基于{metric}指标: {best_score:.4f})")
        
        return best_threshold
    
    def predict_with_threshold(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        使用指定阈值进行预测
        
        Args:
            X: 输入特征
            threshold: 分类阈值，如果为None则使用self.optimal_threshold
            
        Returns:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        if threshold is None:
            threshold = self.optimal_threshold
        
        # 获取预测概率
        y_proba = self.predict_proba(X)[:, 1]
        
        # 应用阈值
        y_pred = (y_proba >= threshold).astype(int)
        
        return y_pred
    
    def save_model(self, filepath: str):
        """
        保存模型
        """
        model_data = {
            'classifier': self.classifier,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance,
            'training_stats': self.training_stats
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        加载模型
        """
        model_data = joblib.load(filepath)
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        self.feature_importance = model_data.get('feature_importance')
        self.training_stats = model_data.get('training_stats', {})
    
    def hyperparameter_tuning(self, 
                             X_train: np.ndarray, 
                             y_train: np.ndarray,
                             logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """
        超参数调优
        """
        if logger:
            logger.info("开始随机森林超参数调优...")
        
        # 定义参数网格（减少搜索空间以提高速度）
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt']
        }
        
        # 网格搜索
        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 使用RandomizedSearchCV减少搜索次数
        from sklearn.model_selection import RandomizedSearchCV
        grid_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=20,  # 只尝试20种参数组合
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        
        # 更新模型
        self.classifier = grid_search.best_estimator_
        self.is_trained = True
        
        if logger:
            logger.info(f"最佳参数: {grid_search.best_params_}")
            return grid_search.best_score_


def _hyperparameter_tuning(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    私有方法：超参数调优
    """
    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt']
    }
    
    # 创建基础随机森林
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # 使用RandomizedSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    grid_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # 执行搜索
    grid_search.fit(X_train, y_train)
    
    if logger:
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳分数: {grid_search.best_score_:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
    
    def _hyperparameter_tuning(self, 
                              X_train: np.ndarray, 
                              y_train: np.ndarray,
                              logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """
        私有方法：超参数调优
        """
        # 定义参数网格
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt']
        }
        
        # 创建基础随机森林
        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 使用RandomizedSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        grid_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # 执行搜索
        grid_search.fit(X_train, y_train)
        
        if logger:
            logger.info(f"最佳参数: {grid_search.best_params_}")
            logger.info(f"最佳分数: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }


def create_random_forest_classifier(**kwargs) -> DownstreamRandomForest:
    """
    工厂函数：创建随机森林分类器
    """
    return DownstreamRandomForest(**kwargs)