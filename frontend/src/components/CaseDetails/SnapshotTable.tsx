import React from "react";
import {
  Table,
  Button,
  Space,
  Tag,
  Input,
  Select,
  Row,
  Col,
} from "antd";
import {
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  ClearOutlined,
} from "@ant-design/icons";
import { GraphSnapshot, SnapshotTableProps } from "./types";

const { Search } = Input;

const SnapshotTable: React.FC<SnapshotTableProps> = ({
  snapshots,
  filteredSnapshots,
  loading,
  filterConfig,
  allTags,
  onFilterChange,
  onViewSnapshot,
  onDeleteSnapshot,
  onDownloadSnapshot,
  onClearFilters,
}) => {
  const getRiskLevelColor = (riskLevel: "low" | "medium" | "high"): string => {
    switch (riskLevel) {
      case "high":
        return "red";
      case "medium":
        return "orange";
      case "low":
        return "green";
      default:
        return "blue";
    }
  };

  const getRiskLevelLabel = (riskLevel: "low" | "medium" | "high"): string => {
    switch (riskLevel) {
      case "high":
        return "高风险";
      case "medium":
        return "中风险";
      case "low":
        return "低风险";
      default:
        return "未知";
    }
  };

  const columns = [
    {
      title: "快照标题",
      dataIndex: "title",
      key: "title",
      width: "25%",
      ellipsis: true,
      render: (text: string) => <strong>{text}</strong>,
    },
    {
      title: "主地址",
      dataIndex: "mainAddress",
      key: "mainAddress",
      width: "25%",
      ellipsis: true,
      render: (address: string) => (
        <span style={{ fontSize: 12, fontFamily: "monospace" }}>{address}</span>
      ),
    },
    {
      title: "风险等级",
      dataIndex: "riskLevel",
      key: "riskLevel",
      width: "12%",
      render: (riskLevel: "low" | "medium" | "high") => (
        <Tag color={getRiskLevelColor(riskLevel)}>
          {getRiskLevelLabel(riskLevel)}
        </Tag>
      ),
    },
    {
      title: "创建时间",
      dataIndex: "createTime",
      key: "createTime",
      width: "15%",
      render: (date: any) => {
        // 处理日期格式
        if (typeof date === "string") {
          return new Date(date).toLocaleString();
        }
        return date.format ? date.format("YYYY-MM-DD HH:mm") : date.toString();
      },
    },
    {
      title: "操作",
      key: "action",
      width: "12%",
      align: "center" as const,
      render: (_: any, record: GraphSnapshot) => (
        <Space size="small">
          <Button
            type="text"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => onViewSnapshot(record)}
            title="查看详情"
          />
          <Button
            type="text"
            size="small"
            icon={<DownloadOutlined />}
            onClick={() => onDownloadSnapshot(record)}
            title="下载快照"
          />
          <Button
            type="text"
            danger
            size="small"
            icon={<DeleteOutlined />}
            onClick={() => onDeleteSnapshot(record)}
            title="删除快照"
          />
        </Space>
      ),
    },
  ];

  return (
    <div className="case-details-container">
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Search
            placeholder="搜索快照标题"
            value={filterConfig.title}
            onChange={(e) =>
              onFilterChange({ ...filterConfig, title: e.target.value })
            }
            enterButton
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Select
            style={{ width: "100%" }}
            placeholder="风险等级"
            value={filterConfig.riskLevel || undefined}
            allowClear
            onChange={(value) =>
              onFilterChange({ ...filterConfig, riskLevel: value })
            }
          >
            <Select.Option value="high">高风险</Select.Option>
            <Select.Option value="medium">中风险</Select.Option>
            <Select.Option value="low">低风险</Select.Option>
          </Select>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Select
            mode="multiple"
            style={{ width: "100%" }}
            placeholder="标签"
            value={filterConfig.tags}
            onChange={(value) =>
              onFilterChange({ ...filterConfig, tags: value })
            }
          >
            {allTags.map((tag) => (
              <Select.Option key={tag} value={tag}>
                {tag}
              </Select.Option>
            ))}
          </Select>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Button
            type="default"
            icon={<ClearOutlined />}
            onClick={onClearFilters}
            style={{ width: "100%" }}
          >
            清空筛选
          </Button>
        </Col>
      </Row>

      <Table
        dataSource={filteredSnapshots}
        columns={columns}
        rowKey="id"
        loading={loading}
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showQuickJumper: true,
          showTotal: (total) => `共 ${total} 条记录`,
        }}
        scroll={{ x: 800 }}
      />
    </div>
  );
};

export default SnapshotTable;
