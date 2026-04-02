import React from "react";
import { Table, Space, Button, Tag, Tooltip, Skeleton, Row, Col } from "antd";
import {
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  InboxOutlined,
  UndoOutlined,
} from "@ant-design/icons";
import dayjs from "dayjs";
import { Case } from "../../types";

interface CaseListProps {
  cases: Case[];
  loading: boolean;
  onView: (caseItem: Case) => void;
  onEdit: (caseItem: Case) => void;
  onDelete: (caseId: string) => void;
  onToggleArchive: (caseItem: Case) => void;
}

const CaseList: React.FC<CaseListProps> = ({
  cases,
  loading,
  onView,
  onEdit,
  onDelete,
  onToggleArchive,
}) => {
  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case "HIGH":
        return "red";
      case "MEDIUM":
        return "orange";
      case "LOW":
        return "green";
      default:
        return "blue";
    }
  };

  const getRiskLevelLabel = (level: string) => {
    switch (level) {
      case "HIGH":
        return "高风险";
      case "MEDIUM":
        return "中风险";
      case "LOW":
        return "低风险";
      default:
        return "未知";
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "URGENT":
        return "red";
      case "HIGH":
        return "orange";
      case "MEDIUM":
        return "blue";
      case "LOW":
        return "default";
      default:
        return "default";
    }
  };

  const getPriorityLabel = (priority: string) => {
    switch (priority) {
      case "URGENT":
        return "紧急";
      case "HIGH":
        return "高";
      case "MEDIUM":
        return "中";
      case "LOW":
        return "低";
      default:
        return "未知";
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case "ACTIVE":
        return (
          <span style={{ color: "#52c41a" }}>
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#52c41a",
                marginRight: 6,
              }}
            />
            进行中
          </span>
        );
      case "ARCHIVED":
        return (
          <span style={{ color: "#8c8c8c" }}>
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#8c8c8c",
                marginRight: 6,
              }}
            />
            已归档
          </span>
        );
      case "CLOSED":
        return (
          <span style={{ color: "#ff4d4f" }}>
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#ff4d4f",
                marginRight: 6,
              }}
            />
            已关闭
          </span>
        );
      default:
        return status;
    }
  };

  const columns = [
    {
      title: "案件编号",
      dataIndex: "id",
      key: "id",
      width: 140,
      render: (text: string) => (
        <span style={{ fontFamily: "monospace", fontSize: 13 }}>{text}</span>
      ),
    },
    {
      title: "案件标题",
      dataIndex: "title",
      key: "title",
      ellipsis: true,
      render: (text: string, record: Case) => (
        <Tooltip title={text}>
          <span style={{ fontWeight: 500 }}>{text}</span>
        </Tooltip>
      ),
    },
    {
      title: "风险等级",
      dataIndex: "riskLevel",
      key: "riskLevel",
      width: 100,
      render: (level: string) => (
        <Tag color={getRiskLevelColor(level)}>{getRiskLevelLabel(level)}</Tag>
      ),
    },
    {
      title: "优先级",
      dataIndex: "priority",
      key: "priority",
      width: 90,
      render: (priority: string) => (
        <Tag color={getPriorityColor(priority)}>
          {getPriorityLabel(priority)}
        </Tag>
      ),
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (status: string) => getStatusLabel(status),
    },
    {
      title: "负责人",
      dataIndex: "assignedTo",
      key: "assignedTo",
      width: 100,
      render: (text: string) => text || "-",
    },
    {
      title: "标签",
      dataIndex: "tags",
      key: "tags",
      width: 180,
      ellipsis: true,
      render: (tags: string[]) => (
        <Space size={4} wrap>
          {tags.slice(0, 3).map((tag) => (
            <Tag key={tag} color="blue">
              {tag}
            </Tag>
          ))}
          {tags.length > 3 && <Tag color="default">+{tags.length - 3}</Tag>}
        </Space>
      ),
    },
    {
      title: "创建时间",
      dataIndex: "createTime",
      key: "createTime",
      width: 160,
      render: (time: any) => dayjs(time).format("YYYY-MM-DD HH:mm"),
    },
    {
      title: "操作",
      key: "action",
      width: 160,
      fixed: "right" as const,
      render: (_: any, record: Case) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => onView(record)}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => onEdit(record)}
            />
          </Tooltip>
          <Tooltip title={record.status === "ARCHIVED" ? "取消归档" : "归档"}>
            <Button
              type="text"
              size="small"
              icon={
                record.status === "ARCHIVED" ? (
                  <UndoOutlined />
                ) : (
                  <InboxOutlined />
                )
              }
              onClick={() => onToggleArchive(record)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              type="text"
              danger
              size="small"
              icon={<DeleteOutlined />}
              onClick={() => onDelete(record.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

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
      <Skeleton active paragraph={{ rows: 8 }} />
    </div>
  );

  return (
    <div style={{ position: "relative" }}>
      {loading && cases.length === 0 ? (
        renderSkeleton()
      ) : (
        <Table
          columns={columns}
          dataSource={cases}
          rowKey="id"
          loading={{
            spinning: loading && cases.length > 0,
            tip: "案件数据加载中...",
            size: "large",
          }}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`,
          }}
          scroll={{ x: 1200 }}
          locale={{
            emptyText: (
              <div style={{ padding: "40px 0", textAlign: "center" }}>
                <div style={{ fontSize: "48px", marginBottom: "16px" }}>📁</div>
                <div style={{ color: "#999", fontSize: "14px" }}>
                  暂无案件数据
                </div>
                <div
                  style={{ color: "#bbb", fontSize: "12px", marginTop: "8px" }}
                >
                  点击右上角按钮创建新案件
                </div>
              </div>
            ),
          }}
        />
      )}
    </div>
  );
};

export default CaseList;
