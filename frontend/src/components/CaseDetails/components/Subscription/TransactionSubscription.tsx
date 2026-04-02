import React, { useState, useEffect } from "react";
import {
  Input,
  Select,
  Row,
  Col,
  Button,
  Form,
  Switch,
  Card,
  Tag,
  Tooltip,
  Skeleton,
} from "antd";
import dayjs from "dayjs";
import {
  SearchOutlined,
  ClearOutlined,
  BellOutlined,
  DeleteOutlined,
  EditOutlined,
  ArrowRightOutlined,
} from "@ant-design/icons";
import { SubscribedTransaction, SubscriptionFilter } from "../../types";

interface TransactionSubscriptionProps {
  transactions: SubscribedTransaction[];
  allTags: string[];
  loading?: boolean;
  onFilter: (filters: SubscriptionFilter) => void;
  onDelete: (id: string) => void;
  onToggleAlert: (id: string) => void;
  onEdit: (tx: SubscribedTransaction) => void;
}

const TransactionSubscription: React.FC<TransactionSubscriptionProps> = ({
  transactions,
  allTags,
  loading = false,
  onFilter,
  onDelete,
  onToggleAlert,
  onEdit,
}) => {
  const [form] = Form.useForm();
  const [filters, setFilters] = useState<SubscriptionFilter>({
    keyword: "",
    riskLevel: "",
    tags: [],
    alertOnly: false,
  });

  useEffect(() => {
    const timer = setTimeout(() => {
      onFilter(filters);
    }, 300);
    return () => clearTimeout(timer);
  }, [filters, onFilter]);

  const handleClear = () => {
    const resetFilters: SubscriptionFilter = {
      keyword: "",
      riskLevel: "",
      tags: [],
      alertOnly: false,
    };
    setFilters(resetFilters);
    form.resetFields();
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case "CRITICAL":
      case "HIGH":
        return "red";
      case "MEDIUM":
        return "orange";
      case "LOW":
        return "green";
      default:
        return "default";
    }
  };

  const getRiskLevelLabel = (level: string) => {
    switch (level) {
      case "CRITICAL":
      case "HIGH":
        return "高风险";
      case "MEDIUM":
        return "中风险";
      case "LOW":
        return "低风险";
      default:
        return level;
    }
  };

  const truncateHash = (hash: string) => {
    if (hash.length <= 20) return hash;
    return `${hash.slice(0, 10)}...${hash.slice(-10)}`;
  };

  const truncateAddress = (address: string) => {
    if (address.length <= 16) return address;
    return `${address.slice(0, 6)}...${address.slice(-6)}`;
  };

  const renderSkeleton = () => (
    <div style={{ padding: "20px" }}>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Skeleton.Input active style={{ width: "100%" }} />
        </Col>
        <Col span={6}>
          <Skeleton.Input active style={{ width: "100%" }} />
        </Col>
        <Col span={6}>
          <Skeleton.Input active style={{ width: "100%" }} />
        </Col>
        <Col span={6}>
          <Skeleton.Input active style={{ width: "100%" }} />
        </Col>
      </Row>
      <Row gutter={[16, 16]}>
        {[1, 2].map((item) => (
          <Col xs={24} lg={12} key={item}>
            <Card className="subscription-card" bodyStyle={{ padding: 0 }}>
              <div style={{ padding: 16 }}>
                <div className="subscription-card-header">
                  <div>
                    <div className="subscription-card-title">
                      <Skeleton.Input
                        active
                        style={{ width: 120, height: 20 }}
                      />
                    </div>
                    <div
                      className="subscription-card-subtitle"
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        marginTop: 8,
                      }}
                    >
                      <Skeleton.Input
                        active
                        style={{ width: 80, height: 16 }}
                      />
                      <Skeleton.Button active size="small" shape="circle" />
                      <Skeleton.Input
                        active
                        style={{ width: 80, height: 16 }}
                      />
                    </div>
                  </div>
                  <div className="subscription-card-actions">
                    <Skeleton.Button active size="small" shape="circle" />
                    <Skeleton.Button active size="small" shape="circle" />
                    <Skeleton.Button active size="small" shape="circle" />
                  </div>
                </div>

                <div className="subscription-card-content">
                  <div className="subscription-card-item">
                    <span className="subscription-card-label">金额:</span>
                    <Skeleton.Input
                      active
                      style={{ width: 80, height: 16, marginLeft: 8 }}
                    />
                  </div>
                  <div className="subscription-card-item">
                    <span className="subscription-card-label">风险等级:</span>
                    <Skeleton.Input
                      active
                      style={{ width: 60, height: 24, marginLeft: 8 }}
                    />
                  </div>
                  <div className="subscription-card-item">
                    <span className="subscription-card-label">交易时间:</span>
                    <Skeleton.Input
                      active
                      style={{ width: 100, height: 16, marginLeft: 8 }}
                    />
                  </div>
                </div>

                <div
                  style={{
                    background: "#f5f7fa",
                    padding: 12,
                    borderRadius: 6,
                    marginBottom: 12,
                  }}
                >
                  <Skeleton.Input
                    active
                    style={{ width: "100%", height: 40 }}
                  />
                </div>

                <div className="subscription-card-footer">
                  <div className="subscription-card-tags">
                    <Skeleton.Input
                      active
                      style={{ width: 50, height: 24, marginRight: 8 }}
                    />
                    <Skeleton.Input active style={{ width: 50, height: 24 }} />
                  </div>
                  <div className="subscription-card-time">
                    <Skeleton.Input active style={{ width: 80, height: 16 }} />
                  </div>
                </div>
              </div>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );

  return (
    <div>
      {/* 筛选栏 */}
      <div className="subscription-filter-section">
        <Form form={form} layout="vertical">
          <Row gutter={[16, 16]} align="bottom">
            <Col xs={24} sm={12} md={6}>
              <Form.Item label="关键词" style={{ marginBottom: 0 }}>
                <Input
                  placeholder="搜索交易哈希/地址"
                  prefix={<SearchOutlined />}
                  value={filters.keyword}
                  onChange={(e) =>
                    setFilters({ ...filters, keyword: e.target.value })
                  }
                  allowClear
                />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12} md={5}>
              <Form.Item label="风险等级" style={{ marginBottom: 0 }}>
                <Select
                  placeholder="全部等级"
                  value={filters.riskLevel || undefined}
                  onChange={(value) =>
                    setFilters({ ...filters, riskLevel: value })
                  }
                  allowClear
                >
                  <Select.Option value="HIGH">高风险</Select.Option>
                  <Select.Option value="MEDIUM">中风险</Select.Option>
                  <Select.Option value="LOW">低风险</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} sm={12} md={5}>
              <Form.Item label="标签" style={{ marginBottom: 0 }}>
                <Select
                  mode="multiple"
                  placeholder="选择标签"
                  value={filters.tags}
                  onChange={(value) => setFilters({ ...filters, tags: value })}
                  maxTagCount={1}
                  allowClear
                >
                  {allTags.map((tag) => (
                    <Select.Option key={tag} value={tag}>
                      {tag}
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col xs={24} sm={12} md={4}>
              <Form.Item label="仅显示告警" style={{ marginBottom: 0 }}>
                <Switch
                  checked={filters.alertOnly}
                  onChange={(checked) =>
                    setFilters({ ...filters, alertOnly: checked })
                  }
                  checkedChildren="开启"
                  unCheckedChildren="全部"
                />
              </Form.Item>
            </Col>
            <Col xs={24} sm={12} md={4}>
              <Button
                icon={<ClearOutlined />}
                onClick={handleClear}
                style={{ width: "100%" }}
              >
                清空筛选
              </Button>
            </Col>
          </Row>
        </Form>
      </div>

      {/* 交易列表 */}
      <div className="subscription-list-section">
        {loading && transactions.length === 0 ? (
          renderSkeleton()
        ) : (
          <Row gutter={[16, 16]}>
            {transactions.length === 0 ? (
              <Col span={24}>
                <div style={{ textAlign: "center", padding: "60px 0" }}>
                  <div style={{ fontSize: "48px", marginBottom: "16px" }}>
                    🔔
                  </div>
                  <div style={{ color: "#999", fontSize: "14px" }}>
                    暂无交易订阅
                  </div>
                  <div
                    style={{
                      color: "#bbb",
                      fontSize: "12px",
                      marginTop: "8px",
                    }}
                  >
                    点击右上角按钮添加新的交易订阅
                  </div>
                </div>
              </Col>
            ) : (
              transactions.map((tx) => (
                <Col xs={24} lg={12} key={tx.id}>
                  <Card
                    className="subscription-card"
                    bodyStyle={{ padding: 0 }}
                  >
                    <div style={{ padding: 16 }}>
                      <div className="subscription-card-header">
                        <div>
                          <div className="subscription-card-title">
                            <Tooltip title={tx.txHash}>
                              <span className="subscription-address-text">
                                {truncateHash(tx.txHash)}
                              </span>
                            </Tooltip>
                          </div>
                          <div
                            className="subscription-card-subtitle"
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 8,
                              marginTop: 8,
                            }}
                          >
                            <Tooltip title={tx.fromAddress}>
                              <span>{truncateAddress(tx.fromAddress)}</span>
                            </Tooltip>
                            <ArrowRightOutlined style={{ color: "#8c8c8c" }} />
                            <Tooltip title={tx.toAddress}>
                              <span>{truncateAddress(tx.toAddress)}</span>
                            </Tooltip>
                          </div>
                        </div>
                        <div className="subscription-card-actions">
                          <Tooltip
                            title={tx.alertEnabled ? "关闭告警" : "开启告警"}
                          >
                            <Button
                              type={tx.alertEnabled ? "primary" : "default"}
                              size="small"
                              icon={<BellOutlined />}
                              onClick={() => onToggleAlert(tx.id)}
                              danger={tx.alertEnabled}
                            />
                          </Tooltip>
                          <Tooltip title="编辑">
                            <Button
                              type="text"
                              size="small"
                              icon={<EditOutlined />}
                              onClick={() => onEdit(tx)}
                            />
                          </Tooltip>
                          <Tooltip title="删除">
                            <Button
                              type="text"
                              danger
                              size="small"
                              icon={<DeleteOutlined />}
                              onClick={() => onDelete(tx.id)}
                            />
                          </Tooltip>
                        </div>
                      </div>
                      <div className="subscription-card-content">
                        <div className="subscription-card-item">
                          <span className="subscription-card-label">金额:</span>
                          <span
                            className="subscription-card-value"
                            style={{ color: "#667eea", fontWeight: 600 }}
                          >
                            {tx.amount} {tx.token}
                          </span>
                        </div>
                        <div className="subscription-card-item">
                          <span className="subscription-card-label">
                            风险等级:
                          </span>
                          <Tag color={getRiskLevelColor(tx.riskLevel)}>
                            {getRiskLevelLabel(tx.riskLevel)}
                          </Tag>
                        </div>
                        {tx.txTime && (
                          <div className="subscription-card-item">
                            <span className="subscription-card-label">
                              交易时间:
                            </span>
                            <span className="subscription-card-value">
                              {typeof tx.txTime === "string"
                                ? dayjs(tx.txTime).format("YYYY-MM-DD HH:mm")
                                : tx.txTime.format("YYYY-MM-DD HH:mm")}
                            </span>
                          </div>
                        )}
                      </div>
                      {tx.remark && (
                        <div
                          style={{
                            background: "#f5f7fa",
                            padding: 12,
                            borderRadius: 6,
                            marginBottom: 12,
                            fontSize: 13,
                            color: "#4b5563",
                          }}
                        >
                          {tx.remark}
                        </div>
                      )}
                      <div className="subscription-card-footer">
                        <div className="subscription-card-tags">
                          {tx.tags.map((tag) => (
                            <Tag key={tag} color="blue">
                              {tag}
                            </Tag>
                          ))}
                        </div>
                        <div className="subscription-card-time">
                          订阅于{" "}
                          {typeof tx.subscribedAt === "string"
                            ? dayjs(tx.subscribedAt).format("YYYY-MM-DD")
                            : tx.subscribedAt.format("YYYY-MM-DD")}
                        </div>
                      </div>
                    </div>
                  </Card>
                </Col>
              ))
            )}
          </Row>
        )}
      </div>
    </div>
  );
};

export default TransactionSubscription;
