import React, { useState, useEffect } from "react";
import { Input, Select, DatePicker, Row, Col, Button, Form } from "antd";
import { SearchOutlined, ClearOutlined } from "@ant-design/icons";
import dayjs from "dayjs";

const { RangePicker } = DatePicker;

interface CaseFilterProps {
  onFilter: (filters: any) => void;
  allTags: string[];
}

const CaseFilter: React.FC<CaseFilterProps> = ({ onFilter, allTags }) => {
  const [form] = Form.useForm();
  const [filters, setFilters] = useState({
    keyword: "",
    status: "ALL",
    riskLevel: "",
    priority: "",
    tags: [],
    dateRange: [null, null],
  });

  // 防抖处理
  useEffect(() => {
    const timer = setTimeout(() => {
      onFilter(filters);
    }, 300);
    return () => clearTimeout(timer);
  }, [filters, onFilter]);

  const handleClear = () => {
    const resetFilters = {
      keyword: "",
      status: "ALL",
      riskLevel: "",
      priority: "",
      tags: [],
      dateRange: [null, null],
    };
    setFilters(resetFilters);
    form.resetFields();
  };

  return (
    <Form form={form} layout="vertical" className="case-filter-section">
      <Row gutter={[16, 16]} align="bottom">
        <Col xs={24} sm={12} md={6} lg={5}>
          <Form.Item
            label="关键词搜索"
            name="keyword"
            style={{ marginBottom: 0 }}
          >
            <Input
              placeholder="搜索案件标题/编号/描述"
              prefix={<SearchOutlined />}
              value={filters.keyword}
              onChange={(e) =>
                setFilters({ ...filters, keyword: e.target.value })
              }
              allowClear
            />
          </Form.Item>
        </Col>
        <Col xs={24} sm={12} md={6} lg={4}>
          <Form.Item label="案件状态" name="status" style={{ marginBottom: 0 }}>
            <Select
              placeholder="全部状态"
              value={filters.status}
              onChange={(value) => setFilters({ ...filters, status: value })}
              allowClear
            >
              <Select.Option value="ALL">全部</Select.Option>
              <Select.Option value="NEW">新建</Select.Option>
              <Select.Option value="IN_PROGRESS">进行中</Select.Option>
              <Select.Option value="ARCHIVED">已归档</Select.Option>
              <Select.Option value="CLOSED">已关闭</Select.Option>
            </Select>
          </Form.Item>
        </Col>
        <Col xs={24} sm={12} md={6} lg={4}>
          <Form.Item
            label="风险等级"
            name="riskLevel"
            style={{ marginBottom: 0 }}
          >
            <Select
              placeholder="全部等级"
              value={filters.riskLevel || undefined}
              onChange={(value) => setFilters({ ...filters, riskLevel: value })}
              allowClear
            >
              <Select.Option value="HIGH">高风险</Select.Option>
              <Select.Option value="MEDIUM">中风险</Select.Option>
              <Select.Option value="LOW">低风险</Select.Option>
            </Select>
          </Form.Item>
        </Col>
        <Col xs={24} sm={12} md={6} lg={4}>
          <Form.Item label="优先级" name="priority" style={{ marginBottom: 0 }}>
            <Select
              placeholder="全部优先级"
              value={filters.priority || undefined}
              onChange={(value) => setFilters({ ...filters, priority: value })}
              allowClear
            >
              <Select.Option value="URGENT">紧急</Select.Option>
              <Select.Option value="HIGH">高</Select.Option>
              <Select.Option value="MEDIUM">中</Select.Option>
              <Select.Option value="LOW">低</Select.Option>
            </Select>
          </Form.Item>
        </Col>
        <Col xs={24} sm={12} md={6} lg={4}>
          <Form.Item label="标签筛选" name="tags" style={{ marginBottom: 0 }}>
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
        <Col xs={24} sm={12} md={6} lg={3}>
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
  );
};

export default CaseFilter;
