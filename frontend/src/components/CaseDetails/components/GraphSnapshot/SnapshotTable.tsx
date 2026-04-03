import React, {
  useEffect,
  useState,
  useRef,
  useMemo,
  useCallback,
} from "react";
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
import type { SortOrder } from "antd/es/table/interface";
import {
  DeleteOutlined,
  EyeOutlined,
  ClearOutlined,
  InboxOutlined,
  UndoOutlined,
} from "@ant-design/icons";
import { GraphSnapshot, SnapshotTableProps } from "../../types";

const { Search } = Input;

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

const SnapshotTable: React.FC<SnapshotTableProps> = ({
  filteredSnapshots,
  loading,
  filterConfig,
  statusFilter,
  allTags,
  onFilterChange,
  onStatusFilterChange,
  onViewSnapshot,
  onDeleteSnapshot,
  onToggleArchive,
  onClearFilters,
}) => {
  const [searchValue, setSearchValue] = useState(filterConfig.title || "");
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setSearchValue(filterConfig.title || "");
  }, [filterConfig.title]);

  const debouncedSearch = useCallback(
    (value: string) => {
      if (searchDebounceRef.current) {
        clearTimeout(searchDebounceRef.current);
      }
      searchDebounceRef.current = setTimeout(() => {
        onFilterChange({ ...filterConfig, title: value });
      }, 300);
    },
    [filterConfig, onFilterChange],
  );

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setSearchValue(value);
      debouncedSearch(value);
    },
    [debouncedSearch],
  );

  useEffect(() => {
    return () => {
      if (searchDebounceRef.current) {
        clearTimeout(searchDebounceRef.current);
      }
    };
  }, []);

  const handleViewSnapshot = useCallback(
    (record: GraphSnapshot) => {
      onViewSnapshot(record);
    },
    [onViewSnapshot],
  );

  const handleToggleArchive = useCallback(
    (record: GraphSnapshot) => {
      onToggleArchive(record);
    },
    [onToggleArchive],
  );

  const handleDeleteSnapshot = useCallback(
    (record: GraphSnapshot) => {
      onDeleteSnapshot(record);
    },
    [onDeleteSnapshot],
  );

  const handleRiskLevelChange = useCallback(
    (value: "LOW" | "MEDIUM" | "HIGH" | undefined) => {
      onFilterChange({ ...filterConfig, riskLevel: value || "" });
    },
    [filterConfig, onFilterChange],
  );

  const handleTagsChange = useCallback(
    (value: string[]) => {
      onFilterChange({ ...filterConfig, tags: value });
    },
    [filterConfig, onFilterChange],
  );

  const columns = useMemo(
    () => [
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
        render: (_: unknown, record: GraphSnapshot) => {
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
        sorter: (a: GraphSnapshot, b: GraphSnapshot) => {
          const getTime = (time: GraphSnapshot["createTime"]) => {
            if (!time) return 0;
            if (typeof time === "string") return new Date(time).getTime();
            if (typeof time === "object" && "valueOf" in time)
              return time.valueOf();
            return 0;
          };
          return getTime(a.createTime) - getTime(b.createTime);
        },
        sortDirections: ["descend", "ascend"] as SortOrder[],
        showSorterTooltip: false,
        render: (
          date:
            | string
            | Date
            | { format: (format: string) => string }
            | undefined,
        ) => {
          if (!date) {
            return "";
          }
          if (typeof date === "string") {
            return new Date(date).toLocaleString();
          }
          if (typeof date === "object" && "format" in date) {
            return date.format("YYYY-MM-DD HH:mm");
          }
          return date.toString();
        },
      },
      {
        title: "操作",
        key: "action",
        width: "12%",
        align: "center" as const,
        render: (_: unknown, record: GraphSnapshot) => (
          <Space size="small">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewSnapshot(record)}
              title="查看详情"
            />
            <Button
              type="text"
              size="small"
              icon={record.archived ? <UndoOutlined /> : <InboxOutlined />}
              onClick={() => handleToggleArchive(record)}
              title={record.archived ? "取消归档" : "归档案件"}
            />
            <Button
              type="text"
              danger
              size="small"
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteSnapshot(record)}
              title="删除快照"
            />
          </Space>
        ),
      },
    ],
    [handleViewSnapshot, handleToggleArchive, handleDeleteSnapshot],
  );

  const renderSkeleton = useCallback(
    () => (
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
    ),
    [],
  );

  const emptyText = useMemo(
    () => (
      <div style={{ padding: "40px 0", textAlign: "center" }}>
        <div style={{ fontSize: "48px", marginBottom: "16px" }}>📋</div>
        <div style={{ color: "#999", fontSize: "14px" }}>暂无案件数据</div>
        <div style={{ color: "#bbb", fontSize: "12px", marginTop: "8px" }}>
          您可以在交易图谱页面保存快照来创建案件
        </div>
      </div>
    ),
    [],
  );

  const tableLoading = useMemo(
    () => ({
      spinning: loading && filteredSnapshots.length > 0,
      tip: "数据加载中...",
      size: "large" as const,
    }),
    [loading, filteredSnapshots.length],
  );

  return (
    <div style={{ position: "relative" }}>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Search
            placeholder="搜索快照标题/主地址"
            value={searchValue}
            onChange={handleSearchChange}
            enterButton
          />
        </Col>
        <Col xs={24} sm={12} md={5}>
          <Select
            style={{ width: "100%" }}
            placeholder="风险等级"
            value={filterConfig.riskLevel || undefined}
            allowClear
            onChange={handleRiskLevelChange}
          >
            <Select.Option value="HIGH">高风险</Select.Option>
            <Select.Option value="MEDIUM">中风险</Select.Option>
            <Select.Option value="LOW">低风险</Select.Option>
          </Select>
        </Col>
        <Col xs={24} sm={12} md={5}>
          <Select
            mode="multiple"
            style={{ width: "100%" }}
            placeholder="标签"
            value={filterConfig.tags}
            onChange={handleTagsChange}
          >
            {allTags.map((tag) => (
              <Select.Option key={tag} value={tag}>
                {tag}
              </Select.Option>
            ))}
          </Select>
        </Col>
        <Col xs={24} sm={12} md={5}>
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
          loading={tableLoading}
          locale={{ emptyText }}
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

export default React.memo(SnapshotTable);
