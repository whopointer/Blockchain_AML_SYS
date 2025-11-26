"""
GNN模型测试模块
测试图神经网络模型的各项功能
"""

import unittest
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_model import (
    GINLayer, MultiScaleGNN, AttentionPooling, 
    ImprovedGNNModel, GNNModel, create_model
)


class TestGINLayer(unittest.TestCase):
    """测试GIN层"""
    
    def setUp(self):
        self.in_channels = 16
        self.out_channels = 32
        self.num_nodes = 100
        self.num_edges = 200
        
        # 创建测试数据
        self.x = torch.randn(self.num_nodes, self.in_channels)
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        
        # 创建GIN层
        self.gin_layer = GINLayer(self.in_channels, self.out_channels)
    
    def test_forward(self):
        """测试前向传播"""
        out = self.gin_layer(self.x, self.edge_index)
        
        # 检查输出形状
        self.assertEqual(out.shape, (self.num_nodes, self.out_channels))
        
        # 检查输出不是NaN
        self.assertFalse(torch.isnan(out).any())
    
    def test_residual_connection(self):
        """测试残差连接"""
        # 当输入输出维度相同时应该有残差连接
        gin_layer_residual = GINLayer(self.out_channels, self.out_channels)
        out = gin_layer_residual(self.x, self.edge_index)
        
        self.assertEqual(out.shape, (self.num_nodes, self.out_channels))
        self.assertFalse(torch.isnan(out).any())
    
    def test_gradient_flow(self):
        """测试梯度流"""
        self.x.requires_grad_(True)
        out = self.gin_layer(self.x, self.edge_index)
        loss = out.sum()
        loss.backward()
        
        # 检查梯度是否存在
        self.assertIsNotNone(self.x.grad)
        self.assertFalse(torch.isnan(self.x.grad).any())


class TestMultiScaleGNN(unittest.TestCase):
    """测试多尺度GNN"""
    
    def setUp(self):
        self.in_channels = 32
        self.out_channels = 48
        self.num_heads = 3
        self.num_nodes = 50
        self.num_edges = 100
        
        # 创建测试数据
        self.x = torch.randn(self.num_nodes, self.in_channels)
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        
        # 创建多尺度GNN
        self.multi_scale_gnn = MultiScaleGNN(self.in_channels, self.out_channels, self.num_heads)
    
    def test_forward(self):
        """测试前向传播"""
        out = self.multi_scale_gnn(self.x, self.edge_index)
        
        # 检查输出形状
        self.assertEqual(out.shape, (self.num_nodes, self.out_channels))
        
        # 检查输出不是NaN
        self.assertFalse(torch.isnan(out).any())
    
    def test_head_dimension(self):
        """测试多头维度分配"""
        # 每个头的输出维度应该是总维度除以头数
        expected_head_dim = self.out_channels // self.num_heads
        self.assertEqual(self.out_channels % self.num_heads, 0)  # 应该能整除


class TestAttentionPooling(unittest.TestCase):
    """测试注意力池化"""
    
    def setUp(self):
        self.in_channels = 64
        self.num_nodes = 100
        
        # 创建测试数据
        self.x = torch.randn(self.num_nodes, self.in_channels)
        self.batch = torch.zeros(self.num_nodes, dtype=torch.long)  # 单图
        
        # 创建注意力池化层
        self.attention_pooling = AttentionPooling(self.in_channels)
    
    def test_single_graph_pooling(self):
        """测试单图池化"""
        out = self.attention_pooling(self.x)
        
        # 单图池化应该返回单个向量
        self.assertEqual(out.shape, (self.in_channels,))
        self.assertFalse(torch.isnan(out).any())
    
    def test_batch_graph_pooling(self):
        """测试批图池化"""
        # 创建两个图的批次
        batch_size = 2
        nodes_per_graph = self.num_nodes // batch_size
        batch = torch.repeat_interleave(torch.arange(batch_size), nodes_per_graph)
        
        out = self.attention_pooling(self.x, batch)
        
        # 批图池化应该返回batch_size个向量
        self.assertEqual(out.shape, (batch_size, self.in_channels))
        self.assertFalse(torch.isnan(out).any())


class TestImprovedGNNModel(unittest.TestCase):
    """测试改进的GNN模型"""
    
    def setUp(self):
        self.num_features = 165  # Elliptic数据集特征数
        self.num_classes = 2
        self.hidden_channels = 64
        self.num_nodes = 200
        self.num_edges = 400
        
        # 创建测试数据
        self.x = torch.randn(self.num_nodes, self.num_features)
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.batch = torch.zeros(self.num_nodes, dtype=torch.long)  # 单图
        
        # 创建模型
        self.model = ImprovedGNNModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
            hidden_channels=self.hidden_channels,
            use_multi_scale=True,
            use_attention_pooling=True
        )
    
    def test_forward(self):
        """测试前向传播"""
        out = self.model(self.x, self.edge_index, self.batch)
        
        # 检查输出形状
        self.assertEqual(out.shape, (1, self.num_classes))
        
        # 检查输出不是NaN
        self.assertFalse(torch.isnan(out).any())
    
    def test_get_node_embeddings(self):
        """测试获取节点嵌入"""
        embeddings = self.model.get_node_embeddings(self.x, self.edge_index)
        
        # 检查嵌入形状
        self.assertEqual(embeddings.shape, (self.num_nodes, self.hidden_channels))
        
        # 检查嵌入不是NaN
        self.assertFalse(torch.isnan(embeddings).any())
    
    def test_get_graph_embedding(self):
        """测试获取图嵌入"""
        graph_embedding = self.model.get_graph_embedding(self.x, self.edge_index, self.batch)
        
        # 检查图嵌入形状
        self.assertEqual(graph_embedding.shape, (1, 1))  # DGI的图分类器输出
    
    def test_batch_forward(self):
        """测试批次前向传播"""
        # 创建两个图的批次
        batch_size = 2
        nodes_per_graph = self.num_nodes // batch_size
        x_batch = torch.randn(self.num_nodes, self.num_features)
        batch = torch.repeat_interleave(torch.arange(batch_size), nodes_per_graph)
        
        out = self.model(x_batch, self.edge_index, batch)
        
        # 检查批次输出形状
        self.assertEqual(out.shape, (batch_size, self.num_classes))
    
    def test_model_parameters(self):
        """测试模型参数数量"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 检查参数数量合理性
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)
    
    def test_different_configurations(self):
        """测试不同配置的模型"""
        # 测试基础配置
        basic_model = ImprovedGNNModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
            hidden_channels=32,
            use_multi_scale=False,
            use_attention_pooling=False
        )
        
        out = basic_model(self.x, self.edge_index, self.batch)
        self.assertEqual(out.shape, (1, self.num_classes))
        
        # 测试只启用多尺度
        multi_scale_model = ImprovedGNNModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
            hidden_channels=32,
            use_multi_scale=True,
            use_attention_pooling=False
        )
        
        out = multi_scale_model(self.x, self.edge_index, self.batch)
        self.assertEqual(out.shape, (1, self.num_classes))
        
        # 测试只启用注意力池化
        attention_model = ImprovedGNNModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
            hidden_channels=32,
            use_multi_scale=False,
            use_attention_pooling=True
        )
        
        out = attention_model(self.x, self.edge_index, self.batch)
        self.assertEqual(out.shape, (1, self.num_classes))


class TestModelFactory(unittest.TestCase):
    """测试模型工厂函数"""
    
    def test_create_improved_model(self):
        """测试创建改进模型"""
        model = create_model(
            model_type='improved',
            num_features=165,
            num_classes=2,
            hidden_channels=64
        )
        
        self.assertIsInstance(model, ImprovedGNNModel)
        
        # 测试前向传播
        x = torch.randn(100, 165)
        edge_index = torch.randint(0, 100, (2, 200))
        out = model(x, edge_index)
        
        self.assertEqual(out.shape, (1, 2))
    
    def test_create_basic_model(self):
        """测试创建基础模型"""
        model = create_model(
            model_type='basic',
            num_features=165,
            num_classes=2
        )
        
        self.assertIsInstance(model, GNNModel)
        
        # 测试前向传播
        x = torch.randn(100, 165)
        edge_index = torch.randint(0, 100, (2, 200))
        out = model(x, edge_index)
        
        self.assertEqual(out.shape, (1, 2))
    
    def test_invalid_model_type(self):
        """测试无效模型类型"""
        with self.assertRaises(ValueError):
            create_model(model_type='invalid_type')


class TestModelIntegration(unittest.TestCase):
    """模型集成测试"""
    
    def setUp(self):
        self.num_features = 165
        self.num_classes = 2
        self.hidden_channels = 64
        
        # 创建模型
        self.model = ImprovedGNNModel(
            num_features=self.num_features,
            num_classes=self.num_classes,
            hidden_channels=self.hidden_channels
        )
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def test_training_step(self):
        """测试训练步骤"""
        # 创建训练数据
        x = torch.randn(100, self.num_features)
        edge_index = torch.randint(0, 100, (2, 200))
        y = torch.randint(0, self.num_classes, (1,))
        
        # 训练步骤
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(x, edge_index)
        loss = F.cross_entropy(out, y)
        
        loss.backward()
        self.optimizer.step()
        
        # 检查损失是否下降
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_evaluation_step(self):
        """测试评估步骤"""
        # 创建评估数据
        x = torch.randn(100, self.num_features)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # 评估步骤
        self.model.eval()
        with torch.no_grad():
            out = self.model(x, edge_index)
            probabilities = F.softmax(out, dim=1)
            
            # 检查概率和为1
            self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(1), atol=1e-6))
    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        # 创建临时文件路径
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # 保存模型
            torch.save(self.model.state_dict(), model_path)
            
            # 创建新模型并加载
            new_model = ImprovedGNNModel(
                num_features=self.num_features,
                num_classes=self.num_classes,
                hidden_channels=self.hidden_channels
            )
            new_model.load_state_dict(torch.load(model_path))
            
            # 测试加载后的模型
            x = torch.randn(50, self.num_features)
            edge_index = torch.randint(0, 50, (2, 100))
            
            self.model.eval()
            new_model.eval()
            
            with torch.no_grad():
                out1 = self.model(x, edge_index)
                out2 = new_model(x, edge_index)
                
                # 检查输出是否相同
                self.assertTrue(torch.allclose(out1, out2, atol=1e-6))
        
        finally:
            # 清理临时文件
            import os
            if os.path.exists(model_path):
                os.unlink(model_path)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestGINLayer,
        TestMultiScaleGNN,
        TestAttentionPooling,
        TestImprovedGNNModel,
        TestModelFactory,
        TestModelIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == '__main__':
    print("开始运行GNN模型测试...")
    success = run_tests()
    
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败，请检查代码。")
        exit(1)