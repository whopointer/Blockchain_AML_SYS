# 导入必要的库
import dgl.function as fn  # DGL图神经网络库的函数操作
import scipy  # 科学计算库，用于贝塔函数等特殊函数
import sympy  # 符号数学库，用于计算多项式系数
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch的函数操作
from torch import nn  # PyTorch的神经网络模块
from torch.nn import init  # PyTorch的参数初始化
import math  # 数学函数库


class PolyConv(nn.Module):
    """
    多项式卷积层类
    实现基于多项式滤波的图卷积操作，用于在图上传播特征信息
    """
    def __init__(self,
                 in_feats,      # 输入特征维度
                 out_feats,    # 输出特征维度
                 theta,         # 多项式系数参数
                 activation=F.leaky_relu,  # 激活函数，默认LeakyReLU
                 lin=False,     # 是否使用线性变换
                 bias=False):   # 是否使用偏置项
        # 调用父类nn.Module的初始化方法
        super(PolyConv, self).__init__()

        # 存储多项式系数参数
        self._theta = theta
        # 多项式的阶数（长度-1）
        self._k = len(self._theta)
        # 保存输入和输出特征维度
        self._in_feats = in_feats
        self._out_feats = out_feats
        # 保存激活函数
        self.activation = activation
        # 创建线性变换层，将输入特征映射到输出特征空间
        self.linear = nn.Linear(in_feats, out_feats, bias)

        # 保存是否使用线性变换的标志
        self.lin = lin
        # 注释掉的参数重置方法调用
        # self.reset_parameters()
        # 注释掉的第二个线性层
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        """重置模型参数"""
        # 如果线性层有权重参数，使用Xavier初始化
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        # 如果线性层有偏置参数，初始化为零
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        """
        前向传播函数

        Args:
            graph: DGL图对象，包含图的拓扑结构
            feat: 节点特征矩阵，形状为(num_nodes, in_feats)

        Returns:
            更新后的节点特征矩阵，形状为(num_nodes, out_feats)
        """
        def unnLaplacian(feat, D_invsqrt, graph):
            """
            执行图拉普拉斯正则化操作
            计算 D^(-1/2) * A * D^(-1/2) * feat
            这是图信号处理中的标准化拉普拉斯算子

            Args:
                feat: 节点特征
                D_invsqrt: 度的逆平方根矩阵
                graph: 图对象
            """
            # 将特征与度的逆平方根相乘进行标准化
            graph.ndata['h'] = feat * D_invsqrt
            # 在图上进行消息传递：对邻居节点特征求和
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            # 返回标准化后的特征
            return feat - graph.ndata.pop('h') * D_invsqrt

        # 使用图的局部作用域，避免影响其他计算
        with graph.local_scope():
            # 计算度的逆平方根，用于标准化
            # 对每个节点的入度求逆平方根，避免除零错误（最小值设为1）
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)

            # 初始化输出特征为零
            h = self._theta[0] * feat
            # 对多项式的每一阶进行计算
            for k in range(1, self._k):
                # 对特征进行拉普拉斯变换
                feat = unnLaplacian(feat, D_invsqrt, graph)
                # 累加到输出特征中
                h += self._theta[k] * feat

        # 如果需要线性变换
        if self.lin:
            # 应用线性变换
            h = self.linear(h)
            # 应用激活函数
            h = self.activation(h)

        # 返回处理后的特征
        return h


def calculate_theta2(d):
    """
    计算多项式卷积的系数参数

    Args:
        d: 多项式的阶数

    Returns:
        thetas: 多项式系数的列表
    """
    thetas = []  # 存储计算出的系数
    x = sympy.symbols('x')  # 创建符号变量x

    # 对每一阶多项式进行计算
    for i in range(d + 1):
        # 构建多项式：(x/2)^i * (1-x/2)^(d-i) / beta(i+1, d+1-i)
        # 这是基于Chebyshev多项式的图滤波器设计
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        # 获取多项式的所有系数
        coeff = f.all_coeffs()
        inv_coeff = []  # 存储反转后的系数

        # 系数反转处理
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)

    return thetas  # 返回系数列表


class Encoder(nn.Module):
    """
    图编码器类
    将图节点特征编码为低维潜在表示
    """
    def __init__(self, g, in_feats, h_feats, d):
        """
        初始化编码器

        Args:
            g: DGL图对象
            in_feats: 输入特征维度
            h_feats: 隐藏层特征维度
            d: 多项式阶数
        """
        super(Encoder, self).__init__()
        # 保存图对象和特征维度信息
        self.g = g
        self.in_feats = in_feats
        self.h_feats = h_feats
        # 注释掉的潜在空间维度
        # self.z_dim = int(h_feats/2)

        # 计算多项式系数
        self.thetas = calculate_theta2(d=d)
        # 激活函数
        self.act = nn.ReLU()
        # 卷积层列表
        self.conv = []

        # 第一个线性变换层：输入特征 -> 隐藏特征
        self.linear1 = nn.Linear(in_feats, h_feats)

        # 创建多个多项式卷积层
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(in_feats, h_feats, self.thetas[i], lin=False))

        # 第二个线性变换层：拼接后的特征 -> 隐藏特征
        self.linear2 = nn.Linear(h_feats * len(self.conv), h_feats)

    def forward(self, features, corrupt=False):
        """
        编码器前向传播

        Args:
            features: 输入节点特征，形状为(num_nodes, in_feats)
            corrupt: 是否进行特征损坏（用于对比学习）

        Returns:
            编码后的节点特征，形状为(num_nodes, h_feats)
        """
        # 如果需要损坏特征（用于对比学习中的负样本）
        if corrupt:
            # 生成随机排列的索引
            perm = torch.randperm(self.g.number_of_nodes())
            # 根据随机索引重排列特征
            features = features[perm]

        # 第一个线性变换
        features = self.linear1(features)
        # 应用ReLU激活函数
        features = self.act(features)

        # 初始化最终特征矩阵（空列）
        features_final = torch.zeros([len(features), 0]).to(features.device)

        # 对每个多项式卷积层进行处理
        for conv in self.conv:
            # 应用多项式卷积
            h0 = conv(self.g, features)
            # 将处理后的特征拼接到最终特征中
            features_final = torch.cat([features_final, h0], -1)

        # 第二个线性变换：处理拼接后的特征
        features = self.linear2(features_final)

        return features  # 返回编码后的特征


class Discriminator(nn.Module):
    """
    判别器类（用于DGI）
    区分真实的节点表示和损坏的节点表示
    """
    def __init__(self, n_hidden):
        """
        初始化判别器

        Args:
            n_hidden: 隐藏层特征维度
        """
        super(Discriminator, self).__init__()
        # 创建可学习的权重参数
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        # 重置参数
        self.reset_parameters()

    def uniform(self, size, tensor):
        """
        均匀分布初始化

        Args:
            size: 参数大小
            tensor: 要初始化的张量
        """
        # 计算均匀分布的边界值
        bound = 1.0 / math.sqrt(size)
        # 如果张量存在，则进行均匀初始化
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        """重置判别器参数"""
        size = self.weight.size(0)
        # 使用均匀分布重置权重
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        """
        判别器前向传播

        Args:
            features: 节点特征矩阵
            summary: 图级特征摘要（所有节点特征的均值）

        Returns:
            判别器输出分数
        """
        # 计算判别分数：features * weight * summary
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features  # 返回判别分数


class Decoder(nn.Module):
    """
    图解码器类
    将潜在表示解码回图结构或节点特征
    """
    def __init__(self, g, in_feats, h_feats, d):
        """
        初始化解码器

        Args:
            g: DGL图对象
            in_feats: 输入特征维度
            h_feats: 隐藏层特征维度
            d: 多项式阶数
        """
        super(Decoder, self).__init__()
        # 保存图对象和特征维度信息
        self.g = g
        self.in_feats = in_feats
        self.h_feats = h_feats
        # 计算多项式系数
        self.thetas = calculate_theta2(d=d)
        # 激活函数
        self.act = nn.ReLU()
        # 卷积层列表
        self.conv = []

        # 第一个线性变换层
        self.linear1 = nn.Linear(in_feats, h_feats)

        # 创建多个多项式卷积层
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(in_feats, h_feats, self.thetas[i], lin=False))

        # 第二个线性变换层
        self.linear2 = nn.Linear(h_feats * len(self.conv), h_feats)

    def forward(self, features):
        """
        解码器前向传播

        Args:
            features: 输入特征（通常是潜在表示）

        Returns:
            解码后的特征
        """
        # 第一个线性变换
        features = self.linear1(features)
        # 应用ReLU激活函数
        features = self.act(features)

        # 初始化最终特征矩阵
        features_final = torch.zeros([len(features), 0]).to(features.device)

        # 对每个多项式卷积层进行处理
        for conv in self.conv:
            # 应用多项式卷积
            h0 = conv(self.g, features)
            # 拼接特征
            features_final = torch.cat([features_final, h0], -1)

        # 第二个线性变换
        features = self.linear2(features_final)

        return features  # 返回解码后的特征


class LG_VGAE(nn.Module):
    """
    Label-Graph变分图自编码器主模型类
    结合了VGAE和DGI的思想，实现半监督的图表示学习
    """
    def __init__(self, g, in_feats, n_hidden, z_dim, d, b=0.2):
        """
        初始化LG-VGAE模型

        Args:
            g: DGL图对象
            in_feats: 输入特征维度
            n_hidden: 隐藏层维度
            z_dim: 潜在空间维度
            d: 多项式阶数
            b: VGAE和DGI损失的平衡常数（默认0.2）
        """
        super(LG_VGAE, self).__init__()

        # 创建编码器和解码器
        self.encoder = Encoder(g, in_feats, n_hidden, d)
        self.decoder = Decoder(g, n_hidden, in_feats, d)

        # VGAE损失计算相关组件
        self.z_dim = z_dim  # 潜在空间维度
        self.linear_rep = nn.Linear(n_hidden, z_dim)  # 编码潜在分布参数（均值和对数方差）
        self.linear_rec = nn.Linear(z_dim, n_hidden)  # 潜在向量到隐藏维度的映射

        # DGI损失计算相关组件
        self.discriminator = Discriminator(n_hidden)  # 判别器
        self.lossDGI = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失

        # 联合损失平衡常数
        self.b = b  # 控制VGAE和DGI损失的权重
        # 注释掉的dropout层
        # self.dropout = nn.Dropout(p=dropout)

    def reparameterize(self, mu, log_var):
        """
        重参数化技巧：从潜在分布中采样

        Args:
            mu: 潜在分布的均值
            log_var: 潜在分布的对数方差

        Returns:
            从潜在分布中采样的向量
        """
        # 计算标准差：exp(log_var/2)
        std = torch.exp(log_var/2)
        # 生成与std相同形状的标准正态分布噪声
        eps = torch.randn_like(std)
        # 重参数化采样：mu + eps * std
        return mu + eps * std    # 返回潜在空间样本

    def forward(self, features):
        """
        LG-VGAE前向传播

        Args:
            features: 输入节点特征

        Returns:
            联合损失值
        """
        b = self.b  # 获取平衡常数
        # 注释掉的dropout
        # features = self.dropout(features)

        # 编码器前向传播：生成正样本（真实特征）和负样本（损坏特征）
        positive = self.encoder(features, corrupt=False)  # 真实编码
        negative = self.encoder(features, corrupt=True)  # 损坏编码

        # 使用DGI计算对比学习损失
        # 计算图级特征摘要（所有节点特征的均值）
        summary = torch.sigmoid(positive.mean(dim=0))
        # 判别器对正样本和负样本的评分
        positive_dis = self.discriminator(positive, summary)
        negative_dis = self.discriminator(negative, summary)
        # 计算DGI损失：正样本接近1，负样本接近0
        loss_dgi_1 = self.lossDGI(positive_dis, torch.ones_like(positive_dis))
        loss_dgi_2 = self.lossDGI(negative_dis, torch.zeros_like(negative_dis))
        dgi_loss = loss_dgi_1 + loss_dgi_2  # 总DGI损失

        # VAE解码部分：计算潜在分布参数并采样
        mu, log_var = self.linear_rep(positive), self.linear_rep(positive)  # 编码均值和对数方差
        z = self.reparameterize(mu, log_var)  # 重参数化采样潜在向量
        x_reconst = self.decoder(self.linear_rec(z))  # 解码重构输入

        # 计算重构损失（MSE损失）
        reconst_loss = F.mse_loss(x_reconst, features, reduction='sum')
        # 计算KL散度损失（正则化项）
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        vgae_loss = reconst_loss + kl_div  # 总VGAE损失

        # 联合损失计算：使用平衡常数b权衡VGAE和DGI损失
        joint_loss = b * dgi_loss / (dgi_loss/vgae_loss).detach() + (1 - b) * vgae_loss

        return joint_loss  # 返回联合损失


class Classifier(nn.Module):
    """
    分类器类
    对编码后的节点特征进行分类预测
    """
    def __init__(self, n_hidden, n_classes, dropout=0.3):
        """
        初始化分类器

        Args:
            n_hidden: 隐藏层维度
            n_classes: 分类类别数
            dropout: Dropout概率（默认0.3）
        """
        super(Classifier, self).__init__()
        self.mid_feat = None  # 存储中间特征（用于特征提取）

        # 第一个全连接层：多层感知机，逐步降维
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),  # Dropout正则化
            nn.Linear(n_hidden, n_hidden),  # 输入层到隐藏层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(n_hidden, int(n_hidden / 2)),  # 隐藏层到半隐藏层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(int(n_hidden / 2), int(n_hidden / 4))  # 半隐藏层到四分之一隐藏层
        )

        # 第二个全连接层：输出层
        self.fc2 = nn.Sequential(
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(int(n_hidden / 4), n_classes)  # 四分之一隐藏层到输出层
        )
        # 注释掉的简化版本
        # self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        """重置分类器参数"""
        # 注释掉的条件检查
        if isinstance(self, nn.Conv2d) or isinstance(self, nn.Linear):
            self.reset_parameters()
        # 注释掉的简化版本重置
        # self.fc.reset_parameters()

    def forward(self, features):
        """
        分类器前向传播

        Args:
            features: 输入特征

        Returns:
            分类的对数概率
        """
        # 第一个全连接层
        features = self.fc1(features)
        # 保存中间特征（用于下游任务）
        self.mid_feat = features
        # 第二个全连接层
        features = self.fc2(features)
        # 返回对数softmax概率
        return torch.log_softmax(features, dim=-1)
