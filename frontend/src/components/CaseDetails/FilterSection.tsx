import React from "react";
import { Card, Row, Col, Input, Select, Button } from "antd";
import { ClearOutlined } from "@ant-design/icons";
import { FilterConfig } from "./types";

interface FilterSectionProps {
  filterConfig: FilterConfig;
  setFilterConfig: React.Dispatch<React.SetStateAction<FilterConfig>>;
  handleClearFilters: () => void;
  allTags: string[];
}

const FilterSection: React.FC<FilterSectionProps> = ({
  filterConfig,
  setFilterConfig,
  handleClearFilters,
  allTags,
}) => {
  return (
    <Card size="small" className="filter-section">
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={24} md={6}>
          <Input
            placeholder="搜索快照标题"
            value={filterConfig.title}
            onChange={(e) =>
              setFilterConfig({
                ...filterConfig,
                title: e.target.value,
              })
            }
            allowClear
          />
        </Col>
        <Col xs={24} sm={24} md={6}>
          <Select
            placeholder="风险等级"
            value={filterConfig.riskLevel || undefined}
            onChange={(value) =>
              setFilterConfig({
                ...filterConfig,
                riskLevel: value,
              })
            }
            style={{ width: "100%" }}
            allowClear
          >
            <Select.Option value="low">低风险</Select.Option>
            <Select.Option value="medium">中风险</Select.Option>
            <Select.Option value="high">高风险</Select.Option>
          </Select>
        </Col>
        <Col xs={24} sm={24} md={6}>
          <Select
            mode="multiple"
            placeholder="选择标签"
            value={filterConfig.tags}
            onChange={(value) =>
              setFilterConfig({
                ...filterConfig,
                tags: value,
              })
            }
            style={{ width: "100%" }}
            allowClear
            maxTagCount="responsive"
          >
            {allTags.map((tag) => (
              <Select.Option key={tag} value={tag}>
                {tag}
              </Select.Option>
            ))}
          </Select>
        </Col>
        <Col xs={24} sm={24} md={6}>
          <Button
            type="primary"
            icon={<ClearOutlined />}
            onClick={handleClearFilters}
            block
          >
            清空过滤
          </Button>
        </Col>
      </Row>
    </Card>
  );
};

export default FilterSection;
