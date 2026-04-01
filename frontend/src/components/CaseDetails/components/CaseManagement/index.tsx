import React, { useState, useCallback } from "react";
import {
  Card,
  Row,
  Col,
  Button,
  Tag,
  Space,
  Drawer,
  Descriptions,
  message,
  Popconfirm,
  Avatar,
  List,
  Input,
  Divider,
} from "antd";
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  InboxOutlined,
  UndoOutlined,
  FileTextOutlined,
} from "@ant-design/icons";
import dayjs from "dayjs";
import { Case, CaseComment } from "../../types";
import CaseFilter from "./CaseFilter";
import CaseList from "./CaseList";
import CreateCaseModal from "./CreateCaseModal";
import "./CaseManagement.css";

// 模拟案件数据
const mockCases: Case[] = [
  {
    id: "CASE-2024-001",
    title: "可疑交易调查 - 0x1234...5678",
    description: "发现多笔大额异常交易，涉及多个可疑地址",
    status: "ACTIVE",
    riskLevel: "HIGH",
    priority: "URGENT",
    tags: ["可疑交易", "大额转账", "多地址关联"],
    createTime: dayjs().subtract(3, "day"),
    updateTime: dayjs().subtract(1, "day"),
    assignedTo: "张三",
    relatedSnapshots: ["snapshot-1", "snapshot-2"],
    comments: [
      {
        id: "1",
        author: "张三",
        content: "已初步分析交易路径，发现3层关联地址",
        createdAt: dayjs().subtract(2, "day").format("YYYY-MM-DD HH:mm:ss"),
      },
    ],
  },
  {
    id: "CASE-2024-002",
    title: "地址风险监控 - 0xabcd...efgh",
    description: "该地址被标记为高风险，需要持续监控",
    status: "ACTIVE",
    riskLevel: "MEDIUM",
    priority: "HIGH",
    tags: ["地址监控", "风险标记"],
    createTime: dayjs().subtract(7, "day"),
    updateTime: dayjs().subtract(2, "day"),
    assignedTo: "李四",
    relatedSnapshots: ["snapshot-3"],
  },
  {
    id: "CASE-2024-003",
    title: "常规审查 - 交易所钱包",
    description: "对交易所热钱包进行常规风险审查",
    status: "CLOSED",
    riskLevel: "LOW",
    priority: "LOW",
    tags: ["常规审查", "交易所"],
    createTime: dayjs().subtract(30, "day"),
    updateTime: dayjs().subtract(25, "day"),
    assignedTo: "王五",
    relatedSnapshots: [],
  },
];

const CaseManagement: React.FC = () => {
  const [cases, setCases] = useState<Case[]>(mockCases);
  const [filteredCases, setFilteredCases] = useState<Case[]>(mockCases);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [editingCase, setEditingCase] = useState<Case | null>(null);
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false);
  const [selectedCase, setSelectedCase] = useState<Case | null>(null);
  const [commentInput, setCommentInput] = useState("");

  // 统计数据
  const stats = {
    total: cases.length,
    active: cases.filter((c) => c.status === "ACTIVE").length,
    highRisk: cases.filter((c) => c.riskLevel === "HIGH").length,
    urgent: cases.filter((c) => c.priority === "URGENT").length,
  };

  // 筛选处理
  const handleFilter = useCallback(
    (filters: any) => {
      let filtered = [...cases];

      if (filters.keyword) {
        const keyword = filters.keyword.toLowerCase();
        filtered = filtered.filter(
          (c) =>
            c.title.toLowerCase().includes(keyword) ||
            c.description.toLowerCase().includes(keyword) ||
            c.id.toLowerCase().includes(keyword),
        );
      }

      if (filters.status && filters.status !== "ALL") {
        filtered = filtered.filter((c) => c.status === filters.status);
      }

      if (filters.riskLevel) {
        filtered = filtered.filter((c) => c.riskLevel === filters.riskLevel);
      }

      if (filters.priority) {
        filtered = filtered.filter((c) => c.priority === filters.priority);
      }

      if (filters.tags && filters.tags.length > 0) {
        filtered = filtered.filter((c) =>
          filters.tags.some((tag: string) => c.tags.includes(tag)),
        );
      }

      if (filters.dateRange?.[0] && filters.dateRange?.[1]) {
        const start = filters.dateRange[0].startOf("day").valueOf();
        const end = filters.dateRange[1].endOf("day").valueOf();
        filtered = filtered.filter((c) => {
          const time = dayjs(c.createTime).valueOf();
          return time >= start && time <= end;
        });
      }

      setFilteredCases(filtered);
    },
    [cases],
  );

  // 创建案件
  const handleCreateCase = (values: any) => {
    const newCase: Case = {
      id: `CASE-${dayjs().format("YYYY")}-${String(cases.length + 1).padStart(3, "0")}`,
      title: values.title,
      description: values.description,
      status: "ACTIVE",
      riskLevel: values.riskLevel,
      priority: values.priority,
      tags: values.tags || [],
      createTime: dayjs(),
      updateTime: dayjs(),
      assignedTo: values.assignedTo,
      relatedSnapshots: values.relatedSnapshots || [],
    };
    setCases([newCase, ...cases]);
    setFilteredCases([newCase, ...filteredCases]);
    setCreateModalVisible(false);
    message.success("案件创建成功");
  };

  // 编辑案件
  const handleEditCase = (values: any) => {
    if (!editingCase) return;
    const updated = cases.map((c) =>
      c.id === editingCase.id ? { ...c, ...values, updateTime: dayjs() } : c,
    );
    setCases(updated);
    setFilteredCases(updated);
    setEditingCase(null);
    setCreateModalVisible(false);
    message.success("案件更新成功");
  };

  // 删除案件
  const handleDeleteCase = (caseId: string) => {
    const updated = cases.filter((c) => c.id !== caseId);
    setCases(updated);
    setFilteredCases(updated);
    if (selectedCase?.id === caseId) {
      setDetailDrawerVisible(false);
      setSelectedCase(null);
    }
    message.success("案件已删除");
  };

  // 归档/取消归档
  const handleToggleArchive = (caseItem: Case) => {
    const newStatus: "ACTIVE" | "ARCHIVED" =
      caseItem.status === "ARCHIVED" ? "ACTIVE" : "ARCHIVED";
    const updated = cases.map((c) =>
      c.id === caseItem.id
        ? { ...c, status: newStatus, updateTime: dayjs() }
        : c,
    );
    setCases(updated);
    setFilteredCases(updated);
    if (selectedCase?.id === caseItem.id) {
      setSelectedCase({ ...selectedCase, status: newStatus });
    }
    message.success(newStatus === "ARCHIVED" ? "案件已归档" : "已取消归档");
  };

  // 查看详情
  const handleViewCase = (caseItem: Case) => {
    setSelectedCase(caseItem);
    setDetailDrawerVisible(true);
  };

  // 打开编辑弹窗
  const handleOpenEdit = (caseItem: Case) => {
    setEditingCase(caseItem);
    setCreateModalVisible(true);
  };

  // 添加评论
  const handleAddComment = () => {
    if (!commentInput.trim() || !selectedCase) return;
    const newComment: CaseComment = {
      id: Date.now().toString(),
      author: "当前用户",
      content: commentInput.trim(),
      createdAt: dayjs().format("YYYY-MM-DD HH:mm:ss"),
    };
    const updated = cases.map((c) =>
      c.id === selectedCase.id
        ? { ...c, comments: [...(c.comments || []), newComment] }
        : c,
    );
    setCases(updated);
    setFilteredCases(updated);
    setSelectedCase({
      ...selectedCase,
      comments: [...(selectedCase.comments || []), newComment],
    });
    setCommentInput("");
    message.success("评论已添加");
  };

  // 获取所有标签
  const allTags = Array.from(new Set(cases.flatMap((c) => c.tags)));

  return (
    <div className="case-management-container">
      {/* 统计卡片 */}
      <Row gutter={16} className="case-stats-row">
        <Col span={6}>
          <Card className="case-stat-card">
            <div className="case-stat-value">{stats.total}</div>
            <div className="case-stat-label">总案件数</div>
          </Card>
        </Col>
        <Col span={6}>
          <Card className="case-stat-card">
            <div className="case-stat-value" style={{ color: "#1890ff" }}>
              {stats.active}
            </div>
            <div className="case-stat-label">进行中</div>
          </Card>
        </Col>
        <Col span={6}>
          <Card className="case-stat-card high-risk">
            <div className="case-stat-value">{stats.highRisk}</div>
            <div className="case-stat-label">高风险</div>
          </Card>
        </Col>
        <Col span={6}>
          <Card className="case-stat-card medium-risk">
            <div className="case-stat-value">{stats.urgent}</div>
            <div className="case-stat-label">紧急案件</div>
          </Card>
        </Col>
      </Row>

      {/* 筛选栏 */}
      <CaseFilter onFilter={handleFilter} allTags={allTags} />

      {/* 案件列表 */}
      <Card
        title="案件列表"
        extra={
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => {
              setEditingCase(null);
              setCreateModalVisible(true);
            }}
          >
            创建案件
          </Button>
        }
        className="case-list-section"
      >
        <CaseList
          cases={filteredCases}
          loading={loading}
          onView={handleViewCase}
          onEdit={handleOpenEdit}
          onDelete={handleDeleteCase}
          onToggleArchive={handleToggleArchive}
        />
      </Card>

      {/* 创建/编辑弹窗 */}
      <CreateCaseModal
        visible={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false);
          setEditingCase(null);
        }}
        onSubmit={editingCase ? handleEditCase : handleCreateCase}
        initialValues={editingCase}
        isEdit={!!editingCase}
      />

      {/* 详情抽屉 */}
      <Drawer
        title={
          <Space>
            <FileTextOutlined />
            案件详情
          </Space>
        }
        placement="right"
        width={600}
        onClose={() => setDetailDrawerVisible(false)}
        open={detailDrawerVisible}
        className="case-detail-drawer"
      >
        {selectedCase && (
          <>
            {/* 操作按钮 */}
            <Space>
              <Button
                icon={<EditOutlined />}
                onClick={() => {
                  setDetailDrawerVisible(false);
                  handleOpenEdit(selectedCase);
                }}
              >
                编辑案件
              </Button>
              <Button
                icon={
                  selectedCase.status === "ARCHIVED" ? (
                    <UndoOutlined />
                  ) : (
                    <InboxOutlined />
                  )
                }
                onClick={() => handleToggleArchive(selectedCase)}
              >
                {selectedCase.status === "ARCHIVED" ? "取消归档" : "归档案件"}
              </Button>
              <Popconfirm
                title="确认删除"
                description="确定要删除此案件吗？此操作不可撤销。"
                onConfirm={() => handleDeleteCase(selectedCase.id)}
                okText="删除"
                cancelText="取消"
                okButtonProps={{ danger: true }}
              >
                <Button danger icon={<DeleteOutlined />}>
                  删除案件
                </Button>
              </Popconfirm>
            </Space>

            <Divider />
            
            <Descriptions
              bordered
              column={1}
              size="small"
              style={{ marginBottom: 24 }}
            >
              <Descriptions.Item label="案件编号">
                {selectedCase.id}
              </Descriptions.Item>
              <Descriptions.Item label="案件标题">
                {selectedCase.title}
              </Descriptions.Item>
              <Descriptions.Item label="案件描述">
                {selectedCase.description}
              </Descriptions.Item>
              <Descriptions.Item label="风险等级">
                <Tag
                  color={
                    selectedCase.riskLevel === "HIGH"
                      ? "red"
                      : selectedCase.riskLevel === "MEDIUM"
                        ? "orange"
                        : "green"
                  }
                >
                  {selectedCase.riskLevel === "HIGH"
                    ? "高风险"
                    : selectedCase.riskLevel === "MEDIUM"
                      ? "中风险"
                      : "低风险"}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="优先级">
                <Tag
                  color={
                    selectedCase.priority === "URGENT"
                      ? "red"
                      : selectedCase.priority === "HIGH"
                        ? "orange"
                        : selectedCase.priority === "MEDIUM"
                          ? "blue"
                          : "default"
                  }
                >
                  {selectedCase.priority === "URGENT"
                    ? "紧急"
                    : selectedCase.priority === "HIGH"
                      ? "高"
                      : selectedCase.priority === "MEDIUM"
                        ? "中"
                        : "低"}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <span
                  className={`case-status-badge case-status-${selectedCase.status.toLowerCase()}`}
                >
                  <span className="case-status-dot" />
                  {selectedCase.status === "ACTIVE"
                    ? "进行中"
                    : selectedCase.status === "ARCHIVED"
                      ? "已归档"
                      : "已关闭"}
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="负责人">
                {selectedCase.assignedTo || "未分配"}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {dayjs(selectedCase.createTime).format("YYYY-MM-DD HH:mm")}
              </Descriptions.Item>
              <Descriptions.Item label="更新时间">
                {dayjs(selectedCase.updateTime).format("YYYY-MM-DD HH:mm")}
              </Descriptions.Item>
              <Descriptions.Item label="标签">
                <Space wrap>
                  {selectedCase.tags.map((tag) => (
                    <Tag key={tag} color="blue">
                      {tag}
                    </Tag>
                  ))}
                </Space>
              </Descriptions.Item>
              <Descriptions.Item label="关联快照">
                {selectedCase.relatedSnapshots?.length || 0} 个
              </Descriptions.Item>
            </Descriptions>

            <Divider />

            {/* 评论区域 */}
            <div className="case-detail-section">
              <div className="case-detail-section-title">
                协同评论 ({selectedCase.comments?.length || 0})
              </div>
              <List
                className="case-comments-list"
                dataSource={selectedCase.comments || []}
                locale={{ emptyText: "暂无评论" }}
                renderItem={(comment) => (
                  <div className="case-comment-item" key={comment.id}>
                    <div className="case-comment-header">
                      <Avatar size="small" style={{ marginRight: 8 }}>
                        {comment.author[0]}
                      </Avatar>
                      <span className="case-comment-author">
                        {comment.author}
                      </span>
                      <span className="case-comment-time">
                        {comment.createdAt}
                      </span>
                    </div>
                    <div className="case-comment-content">
                      {comment.content}
                    </div>
                  </div>
                )}
              />
              <Input.TextArea
                rows={3}
                value={commentInput}
                onChange={(e) => setCommentInput(e.target.value)}
                placeholder="添加评论..."
                style={{ marginTop: 16 }}
              />
              <Button
                type="primary"
                onClick={handleAddComment}
                disabled={!commentInput.trim()}
                style={{ marginTop: 8 }}
              >
                添加评论
              </Button>
            </div>
          </>
        )}
      </Drawer>
    </div>
  );
};

export default CaseManagement;
