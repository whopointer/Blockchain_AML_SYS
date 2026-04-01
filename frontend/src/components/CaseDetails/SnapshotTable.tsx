import React, { useEffect, useState, useRef } from "react";
import {
  Table,
  Button,
  Space,
  Tag,
  Input,
  Select,
  Row,
  Col,
  Skeleton,
} from "antd";
import {
  DeleteOutlined,
  EyeOutlined,
  ClearOutlined,
  FilePdfOutlined,
} from "@ant-design/icons";
import { GraphSnapshot, SnapshotTableProps } from "./types";
import LoadingOverlay from "./LoadingOverlay";

const { Search } = Input;

const SnapshotTable: React.FC<SnapshotTableProps> = ({
  filteredSnapshots,
  loading,
  filterConfig,
  allTags,
  onFilterChange,
  onViewSnapshot,
  onDeleteSnapshot,
  onDownloadSnapshot,
  onClearFilters,
  onExportPDF,
}) => {
  const [searchValue, setSearchValue] = useState(filterConfig.title || "");
  const searchDebounceRef = useRef<number | null>(null);

  useEffect(() => {
    setSearchValue(filterConfig.title || "");
  }, [filterConfig.title]);

  useEffect(() => {
    if (searchDebounceRef.current) {
      window.clearTimeout(searchDebounceRef.current);
    }
    searchDebounceRef.current = window.setTimeout(() => {
      onFilterChange({ ...filterConfig, title: searchValue });
    }, 300);
    return () => {
      if (searchDebounceRef.current) {
        window.clearTimeout(searchDebounceRef.current);
      }
    };
  }, [searchValue, filterConfig, onFilterChange]);
  const getRiskLevelColor = (riskLevel: "LOW" | "MEDIUM" | "HIGH"): string => {
    switch (riskLevel) {
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

  const getRiskLevelLabel = (riskLevel: "LOW" | "MEDIUM" | "HIGH"): string => {
    switch (riskLevel) {
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
      key: "mainAddress",
      width: "25%",
      ellipsis: true,
      render: (_: any, record: GraphSnapshot) => {
        const addressText = record.centerAddress
          ? record.centerAddress
          : record.fromAddress && record.toAddress
            ? `${record.fromAddress} → ${record.toAddress}`
            : "";
        return (
          <span style={{ fontSize: 12, fontFamily: "monospace" }}>
            {addressText}
          </span>
        );
      },
    },
    {
      title: "风险等级",
      dataIndex: "riskLevel",
      key: "riskLevel",
      width: "12%",
      render: (riskLevel: "LOW" | "MEDIUM" | "HIGH") => (
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
            icon={<FilePdfOutlined />}
            onClick={() => onExportPDF && onExportPDF(record)}
            title="导出PDF"
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

  // 骨架屏渲染
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
    <div className="case-details-container" style={{ position: "relative" }}>
      {/* 全屏 loading 遮罩 */}
      <LoadingOverlay
        loading={loading && filteredSnapshots.length === 0}
        text="正在加载案件数据..."
      />

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Search
            placeholder="搜索快照标题/主地址"
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
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
            <Select.Option value="HIGH">高风险</Select.Option>
            <Select.Option value="MEDIUM">中风险</Select.Option>
            <Select.Option value="LOW">低风险</Select.Option>
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

      {loading && filteredSnapshots.length === 0 ? (
        renderSkeleton()
      ) : (
        <Table
          dataSource={filteredSnapshots}
          columns={columns}
          rowKey="id"
          loading={{
            spinning: loading && filteredSnapshots.length > 0,
            tip: "数据加载中...",
            size: "large",
          }}
          locale={{
            emptyText: (
              <div style={{ padding: "40px 0", textAlign: "center" }}>
                <div style={{ fontSize: "48px", marginBottom: "16px" }}>📋</div>
                <div style={{ color: "#999", fontSize: "14px" }}>
                  暂无案件数据
                </div>
                <div
                  style={{ color: "#bbb", fontSize: "12px", marginTop: "8px" }}
                >
                  您可以在交易图谱页面保存快照来创建案件
                </div>
              </div>
            ),
          }}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`,
          }}
          scroll={{ x: 800 }}
        />
      )}
    </div>
  );
};

export default SnapshotTable;
