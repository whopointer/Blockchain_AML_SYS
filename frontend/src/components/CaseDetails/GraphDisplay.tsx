import React from "react";
import { Tag, Space, Button, Descriptions, Input, Select } from "antd";
import {
  DownloadOutlined,
  DeleteOutlined,
  EditOutlined,
} from "@ant-design/icons";
import TxGraph from "../TransactionGraph/TxGraph";
import TxGraphFilter from "../TransactionGraph/TxGraphFilter";
import { GraphSnapshot } from "./types";
import dayjs from "dayjs";

interface GraphDisplayProps {
  selectedSnapshot: GraphSnapshot;
  setDrawerVisible: React.Dispatch<React.SetStateAction<boolean>>;
  handleDownloadSnapshot: (snapshot: GraphSnapshot) => void;
  handleDeleteSnapshot: (snapshot: GraphSnapshot) => void;
  editingField: string | null;
  tempValue: any;
  setTempValue: React.Dispatch<React.SetStateAction<any>>;
  startEditing: (field: string, currentValue: any) => void;
  saveEdit: (snapshotId: string, field: string) => void;
  cancelEdit: () => void;
}

const GraphDisplay: React.FC<GraphDisplayProps> = ({
  selectedSnapshot,
  setDrawerVisible,
  handleDownloadSnapshot,
  handleDeleteSnapshot,
  editingField,
  tempValue,
  setTempValue,
  startEditing,
  saveEdit,
  cancelEdit,
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

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <Space>
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            onClick={() => {
              handleDownloadSnapshot(selectedSnapshot);
              setDrawerVisible(false);
            }}
          >
            下载快照
          </Button>
          <Button
            danger
            icon={<DeleteOutlined />}
            onClick={() => {
              handleDeleteSnapshot(selectedSnapshot);
              setDrawerVisible(false);
            }}
          >
            删除快照
          </Button>
        </Space>
      </div>
      <Descriptions
        bordered
        size="small"
        column={1}
        style={{ marginBottom: 20 }}
      >
        <Descriptions.Item label="快照 ID">
          {selectedSnapshot.id}
        </Descriptions.Item>
        <Descriptions.Item label="标题">
          {editingField === "title" ? (
            <div style={{ display: "flex", alignItems: "center" }}>
              <Input
                value={tempValue as string}
                onChange={(e) => setTempValue(e.target.value)}
                onPressEnter={() => saveEdit(selectedSnapshot.id, "title")}
                style={{ flex: 1, marginRight: 8 }}
              />
              <Button
                size="small"
                onClick={cancelEdit}
                style={{ marginRight: 4 }}
              >
                取消
              </Button>
              <Button
                type="primary"
                size="small"
                onClick={() => saveEdit(selectedSnapshot.id, "title")}
              >
                保存
              </Button>
            </div>
          ) : (
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>{selectedSnapshot.title}</span>
              <Button
                type="text"
                size="small"
                icon={<EditOutlined />}
                onClick={() => startEditing("title", selectedSnapshot.title)}
              />
            </div>
          )}
        </Descriptions.Item>

        <Descriptions.Item label="描述">
          {editingField === "description" ? (
            <div style={{ display: "flex", flexDirection: "column" }}>
              <Input.TextArea
                value={tempValue as string}
                onChange={(e) => setTempValue(e.target.value)}
                onPressEnter={(e) => {
                  if (e.ctrlKey || e.metaKey) {
                    saveEdit(selectedSnapshot.id, "description");
                  }
                }}
                rows={4}
                style={{ marginBottom: 8 }}
              />
              <div>
                <Button
                  size="small"
                  onClick={cancelEdit}
                  style={{ marginRight: 4 }}
                >
                  取消
                </Button>
                <Button
                  type="primary"
                  size="small"
                  onClick={() => saveEdit(selectedSnapshot.id, "description")}
                >
                  保存
                </Button>
              </div>
            </div>
          ) : (
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>{selectedSnapshot.description}</span>
              <Button
                type="text"
                size="small"
                icon={<EditOutlined />}
                onClick={() =>
                  startEditing("description", selectedSnapshot.description)
                }
              />
            </div>
          )}
        </Descriptions.Item>

        <Descriptions.Item label="主地址">
          <span className="address-text">{selectedSnapshot.mainAddress}</span>
        </Descriptions.Item>
        <Descriptions.Item label="创建时间">
          {dayjs(selectedSnapshot.createTime).format("YYYY-MM-DD HH:mm:ss")}
        </Descriptions.Item>
        <Descriptions.Item label="风险等级">
          <Tag color={getRiskLevelColor(selectedSnapshot.riskLevel)}>
            {getRiskLevelLabel(selectedSnapshot.riskLevel)}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="标签">
          {editingField === "tags" ? (
            <div style={{ display: "flex", flexDirection: "column" }}>
              <Select
                mode="tags"
                value={tempValue as string[]}
                onChange={(value) => setTempValue(value)}
                style={{ width: "100%", marginBottom: 8 }}
                placeholder="输入标签"
              ></Select>
              <div>
                <Button
                  size="small"
                  onClick={cancelEdit}
                  style={{ marginRight: 4 }}
                >
                  取消
                </Button>
                <Button
                  type="primary"
                  size="small"
                  onClick={() => saveEdit(selectedSnapshot.id, "tags")}
                >
                  保存
                </Button>
              </div>
            </div>
          ) : (
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <Space wrap>
                {selectedSnapshot.tags.map((tag) => (
                  <Tag key={tag} color="blue">
                    {tag}
                  </Tag>
                ))}
              </Space>
              <Button
                type="text"
                size="small"
                icon={<EditOutlined />}
                onClick={() => startEditing("tags", [...selectedSnapshot.tags])}
              />
            </div>
          )}
        </Descriptions.Item>
      </Descriptions>

      {/* 图谱筛选信息和图谱快照 */}
      {selectedSnapshot?.graphData && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ color: "#ffffff", marginBottom: 16 }}>图谱筛选信息</h3>
          <TxGraphFilter
            value={
              selectedSnapshot.filterConfig
                ? {
                    ...selectedSnapshot.filterConfig,
                    startDate: selectedSnapshot.filterConfig.startDate
                      ? dayjs(selectedSnapshot.filterConfig.startDate)
                      : null,
                    endDate: selectedSnapshot.filterConfig.endDate
                      ? dayjs(selectedSnapshot.filterConfig.endDate)
                      : null,
                  }
                : undefined
            }
            onChange={() => {}} // 禁用更改功能，只用于展示
          />

          <h3
            style={{
              color: "#ffffff",
              marginBottom: 16,
              marginTop: 20,
            }}
          >
            图谱快照
          </h3>
          <div
            style={{
              height: "500px",
              border: "1px solid #3a5f7f",
              borderRadius: 8,
            }}
          >
            <TxGraph
              nodes={selectedSnapshot.graphData.nodes}
              links={selectedSnapshot.graphData.links}
              width={680}
              height={480}
              filter={
                selectedSnapshot.filterConfig
                  ? {
                      ...selectedSnapshot.filterConfig,
                      startDate: selectedSnapshot.filterConfig.startDate
                        ? dayjs(
                            selectedSnapshot.filterConfig.startDate
                          ).toDate()
                        : null,
                      endDate: selectedSnapshot.filterConfig.endDate
                        ? dayjs(selectedSnapshot.filterConfig.endDate).toDate()
                        : null,
                    }
                  : {
                      txType: "all" as const,
                      addrType: "all" as const,
                      minAmount: undefined,
                      maxAmount: undefined,
                      startDate: null,
                      endDate: null,
                    }
              }
              onFilterChange={() => {}} // 禁用更改功能，只用于展示
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphDisplay;
