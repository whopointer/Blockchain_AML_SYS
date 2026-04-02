import React, { useState, useCallback, useEffect } from "react";
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
  Skeleton,
  Modal,
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
import { caseApi } from "../../../../services/case/api";

const parseBackendTags = (tags: any): string[] => {
  if (Array.isArray(tags)) {
    return tags.map((tag) => String(tag).trim()).filter((t) => t);
  }
  if (typeof tags === "string") {
    return tags
      .split(",")
      .map((tag) => tag.trim())
      .filter((t) => t);
  }
  return [];
};

// 后端数据转换为前端Case类型
const convertBackendCaseToFrontend = (backendCase: any): Case => {
  return {
    id: backendCase.id.toString(), // 确保id是字符串
    title: backendCase.title || backendCase.caseName || "未命名案件",
    description: backendCase.description || "",
    status: mapBackendStatusToStatus(backendCase.status),
    riskLevel: mapBackendRiskLevelToRiskLevel(backendCase.riskLevel),
    tags: Array.isArray(backendCase.tags)
      ? backendCase.tags
      : parseBackendTags(backendCase.tags),
    createTime: dayjs(backendCase.createTime),
    updateTime: dayjs(backendCase.updateTime),
    assignedTo: backendCase.assignedTo,
    priority: mapBackendPriorityToPriority(backendCase.priority),
    relatedSnapshots: [],
    comments: [],
  };
};

// 将后端状态映射到前端状态
const mapBackendStatusToStatus = (
  backendStatus: string,
): "ACTIVE" | "ARCHIVED" | "CLOSED" => {
  switch (backendStatus) {
    case "NEW":
      return "ACTIVE";
    case "IN_PROGRESS":
      return "ACTIVE";
    case "ARCHIVED":
      return "ARCHIVED";
    case "CLOSED":
      return "CLOSED";
    default:
      return "ACTIVE"; // 默认为ACTIVE
  }
};

// 将后端风险等级映射到前端风险等级
const mapBackendRiskLevelToRiskLevel = (
  backendRiskLevel: string,
): "LOW" | "MEDIUM" | "HIGH" => {
  switch (backendRiskLevel) {
    case "HIGH":
      return "HIGH";
    case "MEDIUM":
      return "MEDIUM";
    case "LOW":
      return "LOW";
    default:
      return "LOW"; // 默认为LOW
  }
};

// 将后端优先级映射到前端优先级
const mapBackendPriorityToPriority = (
  backendPriority: string,
): "LOW" | "MEDIUM" | "HIGH" | "URGENT" => {
  switch (backendPriority) {
    case "URGENT":
      return "URGENT";
    case "HIGH":
      return "HIGH";
    case "MEDIUM":
      return "MEDIUM";
    case "LOW":
      return "LOW";
    default:
      return "MEDIUM"; // 默认为MEDIUM
  }
};

// 前端Case转换为后端数据格式
const convertFrontendCaseToBackend = (frontendCase: Partial<Case>): any => {
  return {
    caseName: frontendCase.title,
    description: frontendCase.description,
    status: frontendCase.status,
    riskLevel: frontendCase.riskLevel,
    tags: frontendCase.tags?.join(","),
    assignedTo: frontendCase.assignedTo,
    priority: frontendCase.priority,
  };
};

const CaseManagement: React.FC = () => {
  const [cases, setCases] = useState<Case[]>([]);
  const [filteredCases, setFilteredCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [editingCase, setEditingCase] = useState<Case | null>(null);
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false);
  const [selectedCase, setSelectedCase] = useState<Case | null>(null);
  const [commentInput, setCommentInput] = useState("");

  // 加载案件数据
  const loadCases = useCallback(async () => {
    setLoading(true);
    try {
      const response = await caseApi.getAllCases();
      if (response.success) {
        const convertedCases = response.data.map(convertBackendCaseToFrontend);
        setCases(convertedCases);
        setFilteredCases(convertedCases);
      } else {
        message.error(response.msg || "加载案件失败");
        setCases([]);
        setFilteredCases([]);
      }
    } catch (error) {
      console.error("加载案件失败:", error);
      message.error("加载案件失败");
      setCases([]);
      setFilteredCases([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // 初始化加载
  useEffect(() => {
    loadCases();
  }, [loadCases]);

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
  const handleCreateCase = async (values: any) => {
    try {
      const backendData = convertFrontendCaseToBackend({
        title: values.title,
        description: values.description,
        status: "ACTIVE",
        riskLevel: values.riskLevel,
        priority: values.priority,
        tags: values.tags || [],
        assignedTo: values.assignedTo,
      });
      const response = await caseApi.createCase(backendData);
      if (response.success) {
        const newCase = convertBackendCaseToFrontend(response.data);
        setCases([newCase, ...cases]);
        setFilteredCases([newCase, ...filteredCases]);
        setCreateModalVisible(false);
        message.success("案件创建成功");
      } else {
        message.error(response.msg || "创建案件失败");
      }
    } catch (error) {
      console.error("创建案件失败:", error);
      message.error("创建案件失败");
    }
  };

  // 编辑案件
  const handleEditCase = async (values: any) => {
    if (!editingCase) return;
    try {
      const backendData = convertFrontendCaseToBackend({
        title: values.title,
        description: values.description,
        riskLevel: values.riskLevel,
        priority: values.priority,
        tags: values.tags || [],
        assignedTo: values.assignedTo,
      });
      const response = await caseApi.updateCase(editingCase.id, backendData);
      if (response.success) {
        const updatedCase = convertBackendCaseToFrontend(response.data);
        const updated = cases.map((c) =>
          c.id === editingCase.id ? updatedCase : c,
        );
        setCases(updated);
        setFilteredCases(updated);
        setEditingCase(null);
        setCreateModalVisible(false);
        message.success("案件更新成功");
      } else {
        message.error(response.msg || "更新案件失败");
      }
    } catch (error) {
      console.error("更新案件失败:", error);
      message.error("更新案件失败");
    }
  };

  // 删除案件
  const handleDeleteCase = async (caseId: string) => {
    Modal.confirm({
      title: "确认删除",
      content: "您确定要删除这个案件吗？此操作无法撤销。",
      okText: "删除",
      okType: "danger",
      cancelText: "取消",
      onOk: async () => {
        try {
          const response = await caseApi.deleteCase(caseId);
          if (response.success) {
            const updated = cases.filter((c) => c.id !== caseId);
            setCases(updated);
            setFilteredCases(updated);
            if (selectedCase?.id === caseId) {
              setDetailDrawerVisible(false);
              setSelectedCase(null);
            }
            message.success("案件已删除");
          } else {
            message.error(response.msg || "删除案件失败");
          }
        } catch (error) {
          console.error("删除案件失败:", error);
          message.error("删除案件失败");
        }
      },
    });
  };

  // 归档/取消归档
  const handleToggleArchive = async (caseItem: Case) => {
    const newStatus = caseItem.status === "ARCHIVED" ? "ACTIVE" : "ARCHIVED";
    try {
      const response = await caseApi.updateCaseStatus(caseItem.id, newStatus);
      if (response.success) {
        const updatedCase = convertBackendCaseToFrontend(response.data);
        const updated = cases.map((c) =>
          c.id === caseItem.id ? updatedCase : c,
        );
        setCases(updated);
        setFilteredCases(updated);
        if (selectedCase?.id === caseItem.id) {
          setSelectedCase(updatedCase);
        }
        message.success(newStatus === "ARCHIVED" ? "案件已归档" : "已取消归档");
      } else {
        message.error(response.msg || "更新案件状态失败");
      }
    } catch (error) {
      console.error("更新案件状态失败:", error);
      message.error("更新案件状态失败");
    }
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
        {loading ? (
          // 骨架屏
          <>
            <Col span={6}>
              <Card className="case-stat-card">
                <Skeleton active paragraph={{ rows: 2 }} />
              </Card>
            </Col>
            <Col span={6}>
              <Card className="case-stat-card">
                <Skeleton active paragraph={{ rows: 2 }} />
              </Card>
            </Col>
            <Col span={6}>
              <Card className="case-stat-card">
                <Skeleton active paragraph={{ rows: 2 }} />
              </Card>
            </Col>
            <Col span={6}>
              <Card className="case-stat-card">
                <Skeleton active paragraph={{ rows: 2 }} />
              </Card>
            </Col>
          </>
        ) : (
          // 实际数据
          <>
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
          </>
        )}
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
