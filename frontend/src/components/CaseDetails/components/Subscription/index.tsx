import React, { useState, useCallback, useEffect } from "react";
import { Card, Tabs, Button, Row, Col, message, Modal } from "antd";
import { PlusOutlined, WalletOutlined, SwapOutlined } from "@ant-design/icons";
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
import { subscriptionApi } from "@/services/case/api";

const parseBackendStringArray = (value: any): string[] => {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter((item) => item);
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item);
  }
  return [];
};

// 后端节点数据转换为前端类型
const convertBackendNodeToFrontend = (backendNode: any): SubscribedNode => {
  return {
    id: backendNode.id,
    address: backendNode.address,
    label: backendNode.label,
    riskLevel:
      backendNode.riskLevel === "CRITICAL"
        ? "HIGH"
        : backendNode.riskLevel === "HIGH"
          ? "HIGH"
          : backendNode.riskLevel === "MEDIUM"
            ? "MEDIUM"
            : "LOW",
    tags: parseBackendStringArray(backendNode.tags),
    remark: backendNode.remark || "",
    subscribedAt: dayjs(backendNode.subscribedAt),
    lastActivity: backendNode.lastActivity
      ? dayjs(backendNode.lastActivity)
      : undefined,
    alertEnabled: backendNode.alertEnabled,
    relatedCases: parseBackendStringArray(backendNode.relatedCases),
  };
};

// 后端交易数据转换为前端类型
const convertBackendTxToFrontend = (backendTx: any): SubscribedTransaction => {
  return {
    id: backendTx.id,
    txHash: backendTx.txHash,
    fromAddress: backendTx.fromAddress,
    toAddress: backendTx.toAddress,
    amount: backendTx.amount,
    token: backendTx.token,
    riskLevel:
      backendTx.riskLevel === "CRITICAL"
        ? "HIGH"
        : backendTx.riskLevel === "HIGH"
          ? "HIGH"
          : backendTx.riskLevel === "MEDIUM"
            ? "MEDIUM"
            : "LOW",
    tags: parseBackendStringArray(backendTx.tags),
    remark: backendTx.remark || "",
    subscribedAt: dayjs(backendTx.subscribedAt),
    txTime: backendTx.txTime ? dayjs(backendTx.txTime) : undefined,
    alertEnabled: backendTx.alertEnabled,
    relatedCases: parseBackendStringArray(backendTx.relatedCases),
  };
};

// 前端节点转换为后端格式
const convertFrontendNodeToBackend = (
  frontendNode: Partial<SubscribedNode>,
): any => {
  return {
    address: frontendNode.address,
    label: frontendNode.label,
    riskLevel: frontendNode.riskLevel,
    tags: frontendNode.tags?.join(","),
    remark: frontendNode.remark,
    alertEnabled: frontendNode.alertEnabled,
    relatedCases: frontendNode.relatedCases?.join(","),
  };
};

// 前端交易转换为后端格式
const convertFrontendTxToBackend = (
  frontendTx: Partial<SubscribedTransaction>,
): any => {
  return {
    txHash: frontendTx.txHash,
    fromAddress: frontendTx.fromAddress,
    toAddress: frontendTx.toAddress,
    amount: frontendTx.amount,
    token: frontendTx.token,
    riskLevel: frontendTx.riskLevel,
    tags: frontendTx.tags?.join(","),
    remark: frontendTx.remark,
    alertEnabled: frontendTx.alertEnabled,
    relatedCases: frontendTx.relatedCases?.join(","),
  };
};

const Subscription: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"nodes" | "transactions">("nodes");
  const [nodes, setNodes] = useState<SubscribedNode[]>([]);
  const [transactions, setTransactions] = useState<SubscribedTransaction[]>([]);
  const [allNodes, setAllNodes] = useState<SubscribedNode[]>([]);
  const [allTransactions, setAllTransactions] = useState<
    SubscribedTransaction[]
  >([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [modalType, setModalType] = useState<"node" | "transaction">("node");
  const [editingItem, setEditingItem] = useState<
    SubscribedNode | SubscribedTransaction | null
  >(null);
  const [loading, setLoading] = useState(false);

  // 加载节点订阅数据
  const loadNodeSubscriptions = useCallback(async () => {
    try {
      const response = await subscriptionApi.getAllNodeSubscriptions();
      if (response.success) {
        const convertedNodes = response.data.map(convertBackendNodeToFrontend);
        setAllNodes(convertedNodes);
        setNodes(convertedNodes);
      } else {
        message.error(response.msg || "加载节点订阅失败");
        setAllNodes([]);
        setNodes([]);
      }
    } catch (error) {
      console.error("加载节点订阅失败:", error);
      message.error("加载节点订阅失败");
      setAllNodes([]);
      setNodes([]);
    }
  }, []);

  // 加载交易订阅数据
  const loadTransactionSubscriptions = useCallback(async () => {
    try {
      const response = await subscriptionApi.getAllTransactionSubscriptions();
      if (response.success) {
        const convertedTxs = response.data.map(convertBackendTxToFrontend);
        setAllTransactions(convertedTxs);
        setTransactions(convertedTxs);
      } else {
        message.error(response.msg || "加载交易订阅失败");
        setAllTransactions([]);
        setTransactions([]);
      }
    } catch (error) {
      console.error("加载交易订阅失败:", error);
      message.error("加载交易订阅失败");
      setAllTransactions([]);
      setTransactions([]);
    }
  }, []);

  // 初始化加载
  useEffect(() => {
    setLoading(true);
    Promise.all([
      loadNodeSubscriptions(),
      loadTransactionSubscriptions(),
    ]).finally(() => {
      setLoading(false);
    });
  }, [loadNodeSubscriptions, loadTransactionSubscriptions]);

  // 统计数据
  const nodeStats = {
    total: nodes.length,
    high: nodes.filter((n) => n.riskLevel === "HIGH").length,
    alertEnabled: nodes.filter((n) => n.alertEnabled).length,
  };

  const txStats = {
    total: transactions.length,
    high: transactions.filter((t) => t.riskLevel === "HIGH").length,
    alertEnabled: transactions.filter((t) => t.alertEnabled).length,
  };

  const currentStats = activeTab === "nodes" ? nodeStats : txStats;

  // 筛选处理
  const handleFilterNodes = useCallback(
    (filters: SubscriptionFilter) => {
      let filtered = [...allNodes];

      if (filters.keyword) {
        const keyword = filters.keyword.toLowerCase();
        filtered = filtered.filter(
          (n) =>
            n.address.toLowerCase().includes(keyword) ||
            n.label?.toLowerCase().includes(keyword) ||
            n.remark.toLowerCase().includes(keyword),
        );
      }

      if (filters.riskLevel) {
        filtered = filtered.filter((n) => n.riskLevel === filters.riskLevel);
      }

      if (filters.tags && filters.tags.length > 0) {
        filtered = filtered.filter((n) =>
          filters.tags.some((tag) => n.tags.includes(tag)),
        );
      }

      if (filters.alertOnly) {
        filtered = filtered.filter((n) => n.alertEnabled);
      }

      setNodes(filtered);
    },
    [allNodes],
  );

  const handleFilterTransactions = useCallback(
    (filters: SubscriptionFilter) => {
      let filtered = [...allTransactions];

      if (filters.keyword) {
        const keyword = filters.keyword.toLowerCase();
        filtered = filtered.filter(
          (t) =>
            t.txHash.toLowerCase().includes(keyword) ||
            t.fromAddress.toLowerCase().includes(keyword) ||
            t.toAddress.toLowerCase().includes(keyword) ||
            t.remark.toLowerCase().includes(keyword),
        );
      }

      if (filters.riskLevel) {
        filtered = filtered.filter((t) => t.riskLevel === filters.riskLevel);
      }

      if (filters.tags && filters.tags.length > 0) {
        filtered = filtered.filter((t) =>
          filters.tags.some((tag) => t.tags.includes(tag)),
        );
      }

      if (filters.alertOnly) {
        filtered = filtered.filter((t) => t.alertEnabled);
      }

      setTransactions(filtered);
    },
    [allTransactions],
  );

  // 添加订阅
  const handleAddSubscription = async (values: any) => {
    try {
      if (modalType === "node") {
        const backendData = convertFrontendNodeToBackend({
          address: values.address,
          label: values.label,
          riskLevel: values.riskLevel,
          tags: values.tags || [],
          remark: values.remark,
          alertEnabled: values.alertEnabled,
          relatedCases: values.relatedCases || [],
        });
        const response =
          await subscriptionApi.createNodeSubscription(backendData);
        if (response.success) {
          const newNode = convertBackendNodeToFrontend(response.data);
          setAllNodes([newNode, ...allNodes]);
          setNodes([newNode, ...nodes]);
          setModalVisible(false);
          message.success("节点订阅添加成功");
        } else {
          message.error(response.msg || "添加节点订阅失败");
        }
      } else {
        const backendData = convertFrontendTxToBackend({
          txHash: values.txHash,
          fromAddress: values.fromAddress,
          toAddress: values.toAddress,
          amount: values.amount,
          token: values.token,
          riskLevel: values.riskLevel,
          tags: values.tags || [],
          remark: values.remark,
          alertEnabled: values.alertEnabled,
          relatedCases: values.relatedCases || [],
        });
        const response =
          await subscriptionApi.createTransactionSubscription(backendData);
        if (response.success) {
          const newTx = convertBackendTxToFrontend(response.data);
          setAllTransactions([newTx, ...allTransactions]);
          setTransactions([newTx, ...transactions]);
          setModalVisible(false);
          message.success("交易订阅添加成功");
        } else {
          message.error(response.msg || "添加交易订阅失败");
        }
      }
    } catch (error) {
      console.error("添加订阅失败:", error);
      message.error("添加订阅失败");
    }
  };

  // 编辑订阅
  const handleEditSubscription = async (values: any) => {
    if (!editingItem) return;

    try {
      if (modalType === "node") {
        const backendData = convertFrontendNodeToBackend({
          address: values.address,
          label: values.label,
          riskLevel: values.riskLevel,
          tags: values.tags || [],
          remark: values.remark,
          alertEnabled: values.alertEnabled,
          relatedCases: values.relatedCases || [],
        });
        const response = await subscriptionApi.updateNodeSubscription(
          editingItem.id,
          backendData,
        );
        if (response.success) {
          const updatedNode = convertBackendNodeToFrontend(response.data);
          const updated = allNodes.map((n) =>
            n.id === editingItem.id ? updatedNode : n,
          );
          setAllNodes(updated);
          setNodes(updated);
          setEditingItem(null);
          setModalVisible(false);
          message.success("节点订阅更新成功");
        } else {
          message.error(response.msg || "更新节点订阅失败");
        }
      } else {
        const backendData = convertFrontendTxToBackend({
          txHash: values.txHash,
          fromAddress: values.fromAddress,
          toAddress: values.toAddress,
          amount: values.amount,
          token: values.token,
          riskLevel: values.riskLevel,
          tags: values.tags || [],
          remark: values.remark,
          alertEnabled: values.alertEnabled,
          relatedCases: values.relatedCases || [],
        });
        const response = await subscriptionApi.updateTransactionSubscription(
          editingItem.id,
          backendData,
        );
        if (response.success) {
          const updatedTx = convertBackendTxToFrontend(response.data);
          const updated = allTransactions.map((t) =>
            t.id === editingItem.id ? updatedTx : t,
          );
          setAllTransactions(updated);
          setTransactions(updated);
          setEditingItem(null);
          setModalVisible(false);
          message.success("交易订阅更新成功");
        } else {
          message.error(response.msg || "更新交易订阅失败");
        }
      }
    } catch (error) {
      console.error("更新订阅失败:", error);
      message.error("更新订阅失败");
    }
  };

  // 删除订阅
  const handleDeleteNode = async (id: string) => {
    Modal.confirm({
      title: "确认删除",
      content: "您确定要删除这个节点订阅吗？此操作无法撤销。",
      okText: "删除",
      okType: "danger",
      cancelText: "取消",
      onOk: async () => {
        try {
          const response = await subscriptionApi.deleteNodeSubscription(id);
          if (response.success) {
            const updated = allNodes.filter((n) => n.id !== id);
            setAllNodes(updated);
            setNodes(updated);
            message.success("节点订阅已删除");
          } else {
            message.error(response.msg || "删除节点订阅失败");
          }
        } catch (error) {
          console.error("删除节点订阅失败:", error);
          message.error("删除节点订阅失败");
        }
      },
    });
  };

  const handleDeleteTransaction = async (id: string) => {
    Modal.confirm({
      title: "确认删除",
      content: "您确定要删除这个交易订阅吗？此操作无法撤销。",
      okText: "删除",
      okType: "danger",
      cancelText: "取消",
      onOk: async () => {
        try {
          const response =
            await subscriptionApi.deleteTransactionSubscription(id);
          if (response.success) {
            const updated = allTransactions.filter((t) => t.id !== id);
            setAllTransactions(updated);
            setTransactions(updated);
            message.success("交易订阅已删除");
          } else {
            message.error(response.msg || "删除交易订阅失败");
          }
        } catch (error) {
          console.error("删除交易订阅失败:", error);
          message.error("删除交易订阅失败");
        }
      },
    });
  };

  // 切换告警
  const handleToggleNodeAlert = async (id: string) => {
    try {
      const response = await subscriptionApi.toggleNodeSubscriptionStatus(id);
      if (response.success) {
        const updatedNode = convertBackendNodeToFrontend(response.data);
        const updated = allNodes.map((n) => (n.id === id ? updatedNode : n));
        setAllNodes(updated);
        setNodes(updated);
        message.success("告警状态已更新");
      } else {
        message.error(response.msg || "更新告警状态失败");
      }
    } catch (error) {
      console.error("更新告警状态失败:", error);
      message.error("更新告警状态失败");
    }
  };

  const handleToggleTxAlert = async (id: string) => {
    try {
      const response =
        await subscriptionApi.toggleTransactionSubscriptionStatus(id);
      if (response.success) {
        const updatedTx = convertBackendTxToFrontend(response.data);
        const updated = allTransactions.map((t) =>
          t.id === id ? updatedTx : t,
        );
        setAllTransactions(updated);
        setTransactions(updated);
        message.success("告警状态已更新");
      } else {
        message.error(response.msg || "更新告警状态失败");
      }
    } catch (error) {
      console.error("更新告警状态失败:", error);
      message.error("更新告警状态失败");
    }
  };

  // 打开添加弹窗
  const openAddModal = (type: "node" | "transaction") => {
    setModalType(type);
    setEditingItem(null);
    setModalVisible(true);
  };

  // 打开编辑弹窗
  const openEditModal = (
    item: SubscribedNode | SubscribedTransaction,
    type: "node" | "transaction",
  ) => {
    setModalType(type);
    setEditingItem(item);
    setModalVisible(true);
  };

  // 获取所有标签
  const allNodeTags = Array.from(new Set(allNodes.flatMap((n) => n.tags)));
  const allTxTags = Array.from(new Set(allTransactions.flatMap((t) => t.tags)));

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
          loading={loading}
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
          loading={loading}
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
          <Card className="subscription-stat-card high">
            <div
              className="subscription-stat-value"
              style={{ color: "#faad14" }}
            >
              {currentStats.high}
            </div>
            <div className="subscription-stat-label">高风险</div>
          </Card>
        </Col>
        <Col span={6}>
          <Card className="subscription-stat-card high">
            <div
              className="subscription-stat-value"
              style={{ color: "#faad14" }}
            >
              {currentStats.high}
            </div>
            <div className="subscription-stat-label">高风险</div>
          </Card>
        </Col>
        <Col span={6}>
          <Card className="subscription-stat-card">
            <div
              className="subscription-stat-value"
              style={{ color: "#52c41a" }}
            >
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
              onClick={() =>
                openAddModal(activeTab === "nodes" ? "node" : "transaction")
              }
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
