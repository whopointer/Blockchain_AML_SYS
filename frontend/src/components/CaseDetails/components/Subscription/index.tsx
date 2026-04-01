import React, { useState, useCallback } from "react";
import { Card, Tabs, Button, Row, Col, message } from "antd";
import {
  PlusOutlined,
  BellOutlined,
  WalletOutlined,
  SwapOutlined,
} from "@ant-design/icons";
import dayjs from "dayjs";
import NodeSubscription from "./NodeSubscription";
import TransactionSubscription from "./TransactionSubscription";
import SubscriptionModal from "./SubscriptionModal";
import {
  SubscribedNode,
  SubscribedTransaction,
  SubscriptionFilter,
} from "../../types";
import "./Subscription.css";

// 模拟数据
const mockNodes: SubscribedNode[] = [
  {
    id: "node-001",
    address: "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
    label: "可疑交易所地址",
    riskLevel: "HIGH",
    tags: ["交易所", "可疑"],
    remark: "该地址涉及多起可疑交易，需要持续监控",
    subscribedAt: dayjs().subtract(7, "day"),
    lastActivity: dayjs().subtract(1, "day"),
    alertEnabled: true,
    relatedCases: ["CASE-2024-001"],
  },
  {
    id: "node-002",
    address: "0x8ba1f109551bD432803012645Hac136c82C3e8C",
    label: "混币器地址",
    riskLevel: "CRITICAL",
    tags: ["混币器", "高风险"],
    remark: "已知的混币服务地址",
    subscribedAt: dayjs().subtract(30, "day"),
    lastActivity: dayjs().subtract(2, "hour"),
    alertEnabled: true,
    relatedCases: ["CASE-2024-002"],
  },
  {
    id: "node-003",
    address: "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0b",
    label: "正常监控地址",
    riskLevel: "LOW",
    tags: ["监控"],
    remark: "常规监控地址",
    subscribedAt: dayjs().subtract(60, "day"),
    lastActivity: dayjs().subtract(5, "day"),
    alertEnabled: false,
    relatedCases: [],
  },
];

const mockTransactions: SubscribedTransaction[] = [
  {
    id: "tx-001",
    txHash: "0xabc123def456789012345678901234567890123456789012345678901234abcd",
    fromAddress: "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
    toAddress: "0x8ba1f109551bD432803012645Hac136c82C3e8C",
    amount: "150.5",
    token: "ETH",
    riskLevel: "HIGH",
    tags: ["大额转账", "可疑"],
    remark: "大额转账到混币器地址",
    subscribedAt: dayjs().subtract(3, "day"),
    txTime: dayjs().subtract(3, "day"),
    alertEnabled: true,
    relatedCases: ["CASE-2024-001"],
  },
  {
    id: "tx-002",
    txHash: "0xdef789abc1234567890123456789012345678901234567890123456789012def",
    fromAddress: "0x1111111111111111111111111111111111111111",
    toAddress: "0x2222222222222222222222222222222222222222",
    amount: "50000",
    token: "USDT",
    riskLevel: "MEDIUM",
    tags: ["稳定币", "监控"],
    remark: "USDT大额转账监控",
    subscribedAt: dayjs().subtract(10, "day"),
    txTime: dayjs().subtract(10, "day"),
    alertEnabled: false,
    relatedCases: [],
  },
];

const Subscription: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"nodes" | "transactions">("nodes");
  const [nodes, setNodes] = useState<SubscribedNode[]>(mockNodes);
  const [transactions, setTransactions] = useState<SubscribedTransaction[]>(mockTransactions);
  const [modalVisible, setModalVisible] = useState(false);
  const [modalType, setModalType] = useState<"node" | "transaction">("node");
  const [editingItem, setEditingItem] = useState<SubscribedNode | SubscribedTransaction | null>(null);

  // 统计数据
  const nodeStats = {
    total: nodes.length,
    critical: nodes.filter((n) => n.riskLevel === "CRITICAL").length,
    high: nodes.filter((n) => n.riskLevel === "HIGH").length,
    alertEnabled: nodes.filter((n) => n.alertEnabled).length,
  };

  const txStats = {
    total: transactions.length,
    critical: transactions.filter((t) => t.riskLevel === "CRITICAL").length,
    high: transactions.filter((t) => t.riskLevel === "HIGH").length,
    alertEnabled: transactions.filter((t) => t.alertEnabled).length,
  };

  const currentStats = activeTab === "nodes" ? nodeStats : txStats;

  // 筛选处理
  const handleFilterNodes = useCallback((filters: SubscriptionFilter) => {
    let filtered = [...mockNodes];

    if (filters.keyword) {
      const keyword = filters.keyword.toLowerCase();
      filtered = filtered.filter(
        (n) =>
          n.address.toLowerCase().includes(keyword) ||
          n.label?.toLowerCase().includes(keyword) ||
          n.remark.toLowerCase().includes(keyword)
      );
    }

    if (filters.riskLevel) {
      filtered = filtered.filter((n) => n.riskLevel === filters.riskLevel);
    }

    if (filters.tags && filters.tags.length > 0) {
      filtered = filtered.filter((n) =>
        filters.tags.some((tag) => n.tags.includes(tag))
      );
    }

    if (filters.alertOnly) {
      filtered = filtered.filter((n) => n.alertEnabled);
    }

    setNodes(filtered);
  }, []);

  const handleFilterTransactions = useCallback((filters: SubscriptionFilter) => {
    let filtered = [...mockTransactions];

    if (filters.keyword) {
      const keyword = filters.keyword.toLowerCase();
      filtered = filtered.filter(
        (t) =>
          t.txHash.toLowerCase().includes(keyword) ||
          t.fromAddress.toLowerCase().includes(keyword) ||
          t.toAddress.toLowerCase().includes(keyword) ||
          t.remark.toLowerCase().includes(keyword)
      );
    }

    if (filters.riskLevel) {
      filtered = filtered.filter((t) => t.riskLevel === filters.riskLevel);
    }

    if (filters.tags && filters.tags.length > 0) {
      filtered = filtered.filter((t) =>
        filters.tags.some((tag) => t.tags.includes(tag))
      );
    }

    if (filters.alertOnly) {
      filtered = filtered.filter((t) => t.alertEnabled);
    }

    setTransactions(filtered);
  }, []);

  // 添加订阅
  const handleAddSubscription = (values: any) => {
    if (modalType === "node") {
      const newNode: SubscribedNode = {
        id: `node-${Date.now()}`,
        address: values.address,
        label: values.label,
        riskLevel: values.riskLevel,
        tags: values.tags || [],
        remark: values.remark,
        subscribedAt: dayjs(),
        alertEnabled: values.alertEnabled,
        relatedCases: values.relatedCases || [],
      };
      setNodes([newNode, ...nodes]);
    } else {
      const newTx: SubscribedTransaction = {
        id: `tx-${Date.now()}`,
        txHash: values.txHash,
        fromAddress: values.fromAddress,
        toAddress: values.toAddress,
        amount: values.amount,
        token: values.token,
        riskLevel: values.riskLevel,
        tags: values.tags || [],
        remark: values.remark,
        subscribedAt: dayjs(),
        txTime: dayjs(),
        alertEnabled: values.alertEnabled,
        relatedCases: values.relatedCases || [],
      };
      setTransactions([newTx, ...transactions]);
    }
    setModalVisible(false);
    message.success("订阅添加成功");
  };

  // 编辑订阅
  const handleEditSubscription = (values: any) => {
    if (!editingItem) return;

    if (modalType === "node") {
      const updated = nodes.map((n) =>
        n.id === editingItem.id ? { ...n, ...values } : n
      );
      setNodes(updated);
    } else {
      const updated = transactions.map((t) =>
        t.id === editingItem.id ? { ...t, ...values } : t
      );
      setTransactions(updated);
    }
    setEditingItem(null);
    setModalVisible(false);
    message.success("订阅更新成功");
  };

  // 删除订阅
  const handleDeleteNode = (id: string) => {
    setNodes(nodes.filter((n) => n.id !== id));
    message.success("节点订阅已删除");
  };

  const handleDeleteTransaction = (id: string) => {
    setTransactions(transactions.filter((t) => t.id !== id));
    message.success("交易订阅已删除");
  };

  // 切换告警
  const handleToggleNodeAlert = (id: string) => {
    const updated = nodes.map((n) =>
      n.id === id ? { ...n, alertEnabled: !n.alertEnabled } : n
    );
    setNodes(updated);
  };

  const handleToggleTxAlert = (id: string) => {
    const updated = transactions.map((t) =>
      t.id === id ? { ...t, alertEnabled: !t.alertEnabled } : t
    );
    setTransactions(updated);
  };

  // 打开添加弹窗
  const openAddModal = (type: "node" | "transaction") => {
    setModalType(type);
    setEditingItem(null);
    setModalVisible(true);
  };

  // 打开编辑弹窗
  const openEditModal = (item: SubscribedNode | SubscribedTransaction, type: "node" | "transaction") => {
    setModalType(type);
    setEditingItem(item);
    setModalVisible(true);
  };

  // 获取所有标签
  const allNodeTags = Array.from(new Set(mockNodes.flatMap((n) => n.tags)));
  const allTxTags = Array.from(new Set(mockTransactions.flatMap((t) => t.tags)));

  const tabItems = [
    {
      key: "nodes",
      label: (
        <span>
          <WalletOutlined />
          节点订阅 ({nodes.length})
        </span>
      ),
      children: (
        <NodeSubscription
          nodes={nodes}
          allTags={allNodeTags}
          onFilter={handleFilterNodes}
          onDelete={handleDeleteNode}
          onToggleAlert={handleToggleNodeAlert}
          onEdit={(node) => openEditModal(node, "node")}
        />
      ),
    },
    {
      key: "transactions",
      label: (
        <span>
          <SwapOutlined />
          交易订阅 ({transactions.length})
        </span>
      ),
      children: (
        <TransactionSubscription
          transactions={transactions}
          allTags={allTxTags}
          onFilter={handleFilterTransactions}
          onDelete={handleDeleteTransaction}
          onToggleAlert={handleToggleTxAlert}
          onEdit={(tx) => openEditModal(tx, "transaction")}
        />
      ),
    },
  ];

  return (
    <div className="subscription-container">
      {/* 统计卡片 */}
      <Row gutter={16} className="subscription-stats-row">
        <Col span={6}>
          <Card className="subscription-stat-card">
            <div className="subscription-stat-value">{currentStats.total}</div>
            <div className="subscription-stat-label">总订阅数</div>
          </Card>
        </Col>
        <Col span={6}>
          <Card className="subscription-stat-card critical">
            <div className="subscription-stat-value" style={{ color: "#ff4d4f" }}>
              {currentStats.critical}
            </div>
            <div className="subscription-stat-label">极高风险</div>
          </Card>
        </Col>
        <Col span={6}>
          <Card className="subscription-stat-card high">
            <div className="subscription-stat-value" style={{ color: "#faad14" }}>
              {currentStats.high}
            </div>
            <div className="subscription-stat-label">高风险</div>
          </Card>
        </Col>
        <Col span={6}>
          <Card className="subscription-stat-card">
            <div className="subscription-stat-value" style={{ color: "#52c41a" }}>
              {currentStats.alertEnabled}
            </div>
            <div className="subscription-stat-label">告警开启</div>
          </Card>
        </Col>
      </Row>

      {/* 标签页 */}
      <Card className="subscription-tabs">
        <Tabs
          activeKey={activeTab}
          onChange={(key) => setActiveTab(key as "nodes" | "transactions")}
          items={tabItems}
          tabBarExtraContent={
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => openAddModal(activeTab === "nodes" ? "node" : "transaction")}
            >
              添加{activeTab === "nodes" ? "节点" : "交易"}订阅
            </Button>
          }
        />
      </Card>

      {/* 添加/编辑弹窗 */}
      <SubscriptionModal
        visible={modalVisible}
        type={modalType}
        isEdit={!!editingItem}
        initialValues={editingItem}
        onCancel={() => {
          setModalVisible(false);
          setEditingItem(null);
        }}
        onSubmit={editingItem ? handleEditSubscription : handleAddSubscription}
      />
    </div>
  );
};

export default Subscription;
