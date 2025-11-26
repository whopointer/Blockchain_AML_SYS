import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import os


class InferenceEngine:
    """
    推理引擎：用于模型推理和结果分析
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: torch.device,
                 threshold: float = 0.5):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()
        
        # 推理历史
        self.inference_history = []
        
    def predict_node_embeddings(self, data: Data) -> torch.Tensor:
        """
        预测节点嵌入
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'get_node_embeddings'):
                embeddings = self.model.get_node_embeddings(data.x, data.edge_index)
            elif hasattr(self.model, 'gnn_model'):
                embeddings = self.model.gnn_model.get_node_embeddings(data.x, data.edge_index)
            else:
                # 直接前向传播获取嵌入
                embeddings = self.model(data.x, data.edge_index)
        
        return embeddings.cpu()
    
    def predict_risk_scores(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测风险分数
        返回: (风险分数, 风险标签)
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                outputs = self.model(data.x, data.edge_index, data.batch)
                
                if outputs.size(1) == 2:  # 二分类
                    probabilities = F.softmax(outputs, dim=1)
                    risk_scores = probabilities[:, 1]  # 正类概率
                    risk_labels = (risk_scores > self.threshold).long()
                else:
                    # 回归或单输出
                    risk_scores = outputs.squeeze(-1)
                    risk_labels = (risk_scores > self.threshold).long()
            else:
                # 使用嵌入计算风险分数
                embeddings = self.predict_node_embeddings(data)
                risk_scores = self._compute_risk_from_embeddings(embeddings)
                risk_labels = (risk_scores > self.threshold).long()
        
        return risk_scores.cpu(), risk_labels.cpu()
    
    def _compute_risk_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        从嵌入计算风险分数（示例实现）
        """
        # 这里可以实现更复杂的风险计算逻辑
        # 例如：基于嵌入的异常检测、聚类等
        
        # 简单示例：使用嵌入的L2范数作为风险指标
        norms = torch.norm(embeddings, dim=1)
        
        # 归一化到[0,1]范围
        if norms.max() > norms.min():
            risk_scores = (norms - norms.min()) / (norms.max() - norms.min())
        else:
            risk_scores = torch.zeros_like(norms)
        
        return risk_scores
    
    def detect_anomalies(self, 
                        embeddings: torch.Tensor, 
                        method: str = 'dbscan',
                        **kwargs) -> Dict[str, Any]:
        """
        基于嵌入的异常检测
        """
        embeddings_np = embeddings.numpy()
        
        if method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clusterer.fit_predict(embeddings_np)
            
            # 异常点标记为-1
            anomaly_mask = cluster_labels == -1
            
        elif method == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 8)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings_np)
            
            # 计算到聚类中心的距离
            centers = clusterer.cluster_centers_
            distances = np.array([
                np.linalg.norm(embeddings_np[i] - centers[cluster_labels[i]])
                for i in range(len(embeddings_np))
            ])
            
            # 距离最远的点作为异常
            threshold = np.percentile(distances, 95)  # 95%分位数
            anomaly_mask = distances > threshold
            
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        return {
            'anomaly_mask': anomaly_mask,
            'cluster_labels': cluster_labels,
            'anomaly_indices': np.where(anomaly_mask)[0].tolist(),
            'num_anomalies': int(np.sum(anomaly_mask)),
            'anomaly_ratio': float(np.sum(anomaly_mask) / len(anomaly_mask))
        }
    
    def compute_similarity_matrix(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        计算节点相似度矩阵
        """
        embeddings_np = embeddings.numpy()
        similarity_matrix = cosine_similarity(embeddings_np)
        return similarity_matrix
    
    def find_similar_nodes(self, 
                          embeddings: torch.Tensor, 
                          target_node_idx: int,
                          top_k: int = 10) -> List[Tuple[int, float]]:
        """
        找到与目标节点最相似的节点
        """
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        target_similarities = similarity_matrix[target_node_idx]
        
        # 排除自身
        target_similarities[target_node_idx] = -1
        
        # 获取top-k最相似的节点
        top_indices = np.argsort(target_similarities)[-top_k:][::-1]
        top_similarities = target_similarities[top_indices]
        
        return list(zip(top_indices.tolist(), top_similarities.tolist()))
    
    def batch_inference(self, 
                       data_loader: DataLoader,
                       save_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量推理
        """
        results = []
        
        for batch_idx, data in enumerate(data_loader):
            # 预测嵌入
            embeddings = self.predict_node_embeddings(data)
            
            # 预测风险分数
            risk_scores, risk_labels = self.predict_risk_scores(data)
            
            # 异常检测
            anomaly_results = self.detect_anomalies(embeddings)
            
            batch_result = {
                'batch_idx': batch_idx,
                'num_nodes': data.num_nodes,
                'embeddings': embeddings.numpy(),
                'risk_scores': risk_scores.numpy(),
                'risk_labels': risk_labels.numpy(),
                'anomaly_results': anomaly_results,
                'high_risk_nodes': torch.where(risk_labels == 1)[0].tolist(),
                'avg_risk_score': float(torch.mean(risk_scores)),
                'max_risk_score': float(torch.max(risk_scores))
            }
            
            results.append(batch_result)
        
        # 保存结果
        if save_path:
            self.save_inference_results(results, save_path)
        
        self.inference_history.extend(results)
        return results
    
    def save_inference_results(self, 
                              results: List[Dict[str, Any]], 
                              save_path: str):
        """
        保存推理结果
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_path.endswith('.json'):
            # 保存为JSON（包含基本统计信息）
            json_results = []
            for result in results:
                json_result = {
                    k: v for k, v in result.items()
                    if k not in ['embeddings']  # 排除大型数组
                }
                json_results.append(json_result)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        elif save_path.endswith('.pkl'):
            # 保存为pickle（包含所有数据）
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
        
        print(f"推理结果已保存到: {save_path}")
    
    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成推理报告
        """
        total_nodes = sum(r['num_nodes'] for r in results)
        total_anomalies = sum(r['anomaly_results']['num_anomalies'] for r in results)
        total_high_risk = sum(len(r['high_risk_nodes']) for r in results)
        
        avg_risk_scores = [r['avg_risk_score'] for r in results]
        max_risk_scores = [r['max_risk_score'] for r in results]
        
        report = {
            'summary': {
                'total_batches': len(results),
                'total_nodes': total_nodes,
                'total_anomalies': total_anomalies,
                'total_high_risk_nodes': total_high_risk,
                'anomaly_rate': total_anomalies / total_nodes if total_nodes > 0 else 0,
                'high_risk_rate': total_high_risk / total_nodes if total_nodes > 0 else 0
            },
            'risk_statistics': {
                'avg_risk_score': np.mean(avg_risk_scores),
                'std_risk_score': np.std(avg_risk_scores),
                'max_risk_score': np.max(max_risk_scores),
                'min_risk_score': np.min(avg_risk_scores)
            },
            'batch_details': results
        }
        
        return report


# 向后兼容的函数
def inference(model, data, device):
    """向后兼容的推理函数"""
    engine = InferenceEngine(model, device)
    return engine.predict_node_embeddings(data)


def create_inference_engine(model: torch.nn.Module,
                           device: Optional[torch.device] = None,
                           threshold: float = 0.5) -> InferenceEngine:
    """
    工厂函数：创建推理引擎
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return InferenceEngine(model, device, threshold)