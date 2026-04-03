import React, { useEffect, useState, useCallback } from "react";
import {
  Tag,
  Space,
  Button,
  Descriptions,
  Input,
  Select,
  Spin,
  message,
  List,
  Divider,
  Avatar,
} from "antd";
import {
  CopyOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EditOutlined,
  FilePdfOutlined,
  FileExcelOutlined,
  PictureOutlined,
  InboxOutlined,
  UndoOutlined,
} from "@ant-design/icons";
import TxGraph from "../../../GraphCommon/TxGraph";
import TxGraphFilter from "../../../GraphCommon/TxGraphFilter";
import { GraphSnapshot } from "../../types";
import { transactionApi } from "../../../../services/transaction/api";
import { NodeItem, LinkItem } from "../../../GraphCommon/types";
import dayjs from "dayjs";
import { formatEthValue } from "../../../../utils/ethUtils";
import {
  generatePDFReport,
  exportFullGraphToPNG,
  convertGraphToCSV,
  downloadCSV,
  exportCasePackage,
} from "../../../../utils/exportUtils";
import ErrorPlaceholder from "../../../GraphCommon/ErrorPlaceholder";

interface GraphDisplayProps {
  selectedSnapshot: GraphSnapshot;
  onClose: () => void;
  onDownloadSnapshot: (snapshot: GraphSnapshot) => void;
  onDeleteSnapshot: (snapshot: GraphSnapshot) => void;
  onToggleArchiveSnapshot?: (snapshot: GraphSnapshot) => void;
  onAddComment?: (snapshotId: string, content: string) => void;
  editingField: string | null;
  tempValue: any;
  setTempValue: React.Dispatch<React.SetStateAction<any>>;
  startEditing: (field: string, currentValue: any) => void;
  saveEdit: (snapshotId: string, field: string) => void;
  cancelEdit: () => void;
}

const GraphDisplay: React.FC<GraphDisplayProps> = ({
  selectedSnapshot,
  onClose,
  onDownloadSnapshot,
  onDeleteSnapshot,
  onToggleArchiveSnapshot,
  onAddComment,
  editingField,
  tempValue,
  setTempValue,
  startEditing,
  saveEdit,
  cancelEdit,
}) => {
  const [exportLoading, setExportLoading] = useState<{
    pdf: boolean;
    csv: boolean;
    png: boolean;
    package: boolean;
  }>({ pdf: false, csv: false, png: false, package: false });
  const [commentContent, setCommentContent] = useState<string>("");
  const [graphData, setGraphData] = useState<{
    nodes?: NodeItem[];
    links?: LinkItem[];
  }>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const [filterConfig, setFilterConfig] = useState<{
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
    minAmount?: number;
    maxAmount?: number;
    startDate?: any;
    endDate?: any;
  }>(
    selectedSnapshot.filterConfig || {
      txType: "all",
      addrType: "all",
      minAmount: undefined,
      maxAmount: undefined,
      startDate: null,
      endDate: null,
    },
  );

  useEffect(() => {
    const fetchGraphData = async () => {
      setLoading(true);
      setIsError(false);
      try {
        let response;

        if (
          selectedSnapshot.centerAddress &&
          selectedSnapshot.hops !== undefined
        ) {
          response = await transactionApi.getNhopGraph(
            selectedSnapshot.centerAddress,
            selectedSnapshot.hops,
          );
        } else if (selectedSnapshot.fromAddress && selectedSnapshot.toAddress) {
          response = await transactionApi.getAllPath(
            selectedSnapshot.fromAddress,
            selectedSnapshot.toAddress,
          );
        }

        if (response && response.success && response.data) {
          const { node_list: nodes, edge_list: edges } = response.data;

          const convertedNodes: NodeItem[] = nodes.map((node: any, index) => ({
            id: node.id || node.address,
            label: node.label || node.address,
            title: node.title || node.label || node.address,
            addr: node.addr || node.address,
            layer: node.layer || 0,
            value: node.value || 0,
            malicious: node.malicious || undefined,
            shape: node.shape || undefined,
            image: node.image || undefined,
            expanded: index === 0,
            track: node.track || "one",
            pid: node.pid || undefined,
            color: node.color || undefined,
            exg: node.exg || undefined,
          }));

          const convertedLinks: LinkItem[] = edges.map((edge: any) => {
            const fromNode = convertedNodes.find((n) => n.addr === edge.from);
            const toNode = convertedNodes.find((n) => n.addr === edge.to);
            const rawTime = edge.tx_time || edge.timestamp;
            const parsedTime = parseDateSafely(rawTime);

            let processedVal = edge.val || edge.value || 0;
            let processedLabel = edge.label || "";

            const isEthTransaction =
              edge.from?.startsWith("0x") || edge.to?.startsWith("0x");

            if (isEthTransaction) {
              processedVal = parseFloat(formatEthValue(processedVal));
              processedLabel =
                processedLabel ||
                `${formatEthValue(edge.val || edge.value || 0)} ETH`;
            } else {
              processedLabel = processedLabel || `${processedVal}`;
            }

            return {
              from: fromNode?.id || edge.from,
              to: toNode?.id || edge.to,
              label: processedLabel,
              val: processedVal,
              tx_time:
                parsedTime && parsedTime.isValid()
                  ? parsedTime.format("YYYY-MM-DD HH:mm")
                  : "",
              tx_hash_list: edge.tx_hash_list || [edge.tx_hash],
            };
          });

          setGraphData({ nodes: convertedNodes, links: convertedLinks });
        } else {
          setGraphData({ nodes: [], links: [] });
        }
      } catch (error) {
        console.error("Failed to fetch graph data:", error);
        setIsError(true);
        setGraphData({ nodes: [], links: [] });
      } finally {
        setLoading(false);
      }
    };

    if (selectedSnapshot) {
      fetchGraphData();
    }
  }, [selectedSnapshot]);

  useEffect(() => {
    if (selectedSnapshot) {
      setFilterConfig(
        selectedSnapshot.filterConfig || {
          txType: "all",
          addrType: "all",
          minAmount: undefined,
          maxAmount: undefined,
          startDate: null,
          endDate: null,
        },
      );
    }
  }, [selectedSnapshot]);

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

  const parseDateSafely = (dateValue: any) => {
    if (!dateValue) {
      return null;
    }

    if (typeof dateValue === "number") {
      const strValue = Math.floor(dateValue).toString();
      if (strValue.length === 10 || strValue.length === 13) {
        return dayjs.unix(dateValue);
      } else {
        return dayjs(dateValue);
      }
    }

    if (typeof dateValue === "string") {
      const parsedDate = dayjs(dateValue);
      if (parsedDate.isValid()) {
        return parsedDate;
      }
    }

    return dayjs(dateValue);
  };

  const copyCenterAddress = async () => {
    if (!selectedSnapshot.centerAddress) {
      message.warning("无中心地址可复制");
      return;
    }
    try {
      await navigator.clipboard.writeText(selectedSnapshot.centerAddress);
      message.success("已复制中心地址");
    } catch (error) {
      console.error("复制地址失败:", error);
      message.error("复制地址失败");
    }
  };

  const handleFilterChange = (newFilter: any) => {
    setFilterConfig(newFilter);
  };

  const handleExportPDF = useCallback(async () => {
    setExportLoading((prev) => ({ ...prev, pdf: true }));
    try {
      const success = await generatePDFReport(
        selectedSnapshot,
        graphData.nodes || [],
        graphData.links || [],
        "graph-container",
      );
      if (success) {
        message.success("PDF报告导出成功");
      } else {
        message.error("PDF导出失败");
      }
    } catch (error) {
      console.error("导出PDF失败:", error);
      message.error("导出PDF失败");
    } finally {
      setExportLoading((prev) => ({ ...prev, pdf: false }));
    }
  }, [selectedSnapshot, graphData.nodes, graphData.links]);

  const handleExportCSV = () => {
    setExportLoading((prev) => ({ ...prev, csv: true }));
    try {
      const csvContent = convertGraphToCSV(
        graphData.nodes || [],
        graphData.links || [],
        selectedSnapshot,
      );
      downloadCSV(csvContent, `${selectedSnapshot.title}-数据.csv`);
      message.success("CSV数据导出成功");
    } catch (error) {
      console.error("导出CSV失败:", error);
      message.error("导出CSV失败");
    } finally {
      setExportLoading((prev) => ({ ...prev, csv: false }));
    }
  };

  const handleExportPNG = async () => {
    setExportLoading((prev) => ({ ...prev, png: true }));
    try {
      const container = document.getElementById("graph-container");
      const svgElement = container?.querySelector(
        "svg",
      ) as SVGSVGElement | null;

      const success = await exportFullGraphToPNG(
        svgElement,
        `${selectedSnapshot.title}-图谱.png`,
      );
      if (success) {
        message.success("PNG图片导出成功");
      } else {
        message.error("PNG导出失败");
      }
    } catch (error) {
      console.error("导出PNG失败:", error);
      message.error("导出PNG失败");
    } finally {
      setExportLoading((prev) => ({ ...prev, png: false }));
    }
  };

  const handleExportPackage = async () => {
    setExportLoading((prev) => ({ ...prev, package: true }));
    try {
      const result = await exportCasePackage(
        selectedSnapshot,
        graphData.nodes || [],
        graphData.links || [],
        "graph-container",
      );
      if (result.success) {
        message.success(result.message);
      } else {
        message.error(result.message);
      }
    } catch (error) {
      console.error("导出案件包失败:", error);
      message.error("导出案件包失败");
    } finally {
      setExportLoading((prev) => ({ ...prev, package: false }));
    }
  };

  return (
    <div>
      <div>
        <div style={{ marginBottom: 20 }}>
          <Space wrap>
            <Button
              type="primary"
              icon={<FilePdfOutlined />}
              onClick={handleExportPDF}
              loading={exportLoading.pdf}
            >
              导出PDF
            </Button>
            <Button
              icon={<FileExcelOutlined />}
              onClick={handleExportCSV}
              loading={exportLoading.csv}
            >
              导出CSV
            </Button>
            <Button
              icon={<PictureOutlined />}
              onClick={handleExportPNG}
              loading={exportLoading.png}
            >
              导出PNG
            </Button>
            <Button
              type="dashed"
              icon={<DownloadOutlined />}
              onClick={handleExportPackage}
              loading={exportLoading.package}
            >
              导出完整包
            </Button>
            <Button
              type={selectedSnapshot.archived ? "default" : "dashed"}
              icon={
                selectedSnapshot.archived ? <UndoOutlined /> : <InboxOutlined />
              }
              onClick={() => onToggleArchiveSnapshot?.(selectedSnapshot)}
            >
              {selectedSnapshot.archived ? "取消归档" : "归档案件"}
            </Button>
            <Button
              danger
              icon={<DeleteOutlined />}
              onClick={() => {
                onDeleteSnapshot(selectedSnapshot);
                onClose();
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

          {selectedSnapshot.centerAddress ? (
            <Descriptions.Item label="中心地址">
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <code className="address-text">
                  {selectedSnapshot.centerAddress}
                </code>
                <Button
                  type="text"
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={copyCenterAddress}
                />
              </div>
            </Descriptions.Item>
          ) : (
            <>
              <Descriptions.Item label="起始地址">
                <code className="address-text">
                  {selectedSnapshot.fromAddress || ""}
                </code>
              </Descriptions.Item>
              <Descriptions.Item label="目标地址">
                <code className="address-text">
                  {selectedSnapshot.toAddress || ""}
                </code>
              </Descriptions.Item>
            </>
          )}
          <Descriptions.Item label="创建时间">
            {dayjs(selectedSnapshot.createTime).format("YYYY-MM-DD HH:mm:ss")}
          </Descriptions.Item>
          <Descriptions.Item label="风险等级">
            {editingField === "riskLevel" ? (
              <div style={{ display: "flex", flexDirection: "column" }}>
                <Select
                  value={tempValue as string}
                  onChange={(value) => setTempValue(value)}
                  style={{ width: "100%", marginBottom: 8 }}
                  placeholder="选择风险等级"
                >
                  <Select.Option value="LOW">低风险</Select.Option>
                  <Select.Option value="MEDIUM">中风险</Select.Option>
                  <Select.Option value="HIGH">高风险</Select.Option>
                </Select>
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
                    onClick={() => saveEdit(selectedSnapshot.id, "riskLevel")}
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
                <Tag color={getRiskLevelColor(selectedSnapshot.riskLevel)}>
                  {getRiskLevelLabel(selectedSnapshot.riskLevel)}
                </Tag>
                <Button
                  type="text"
                  size="small"
                  icon={<EditOutlined />}
                  onClick={() =>
                    startEditing("riskLevel", selectedSnapshot.riskLevel)
                  }
                />
              </div>
            )}
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
                >
                  <Select.Option value="洗钱">洗钱</Select.Option>
                  <Select.Option value="欺诈">欺诈</Select.Option>
                  <Select.Option value="可疑交易">可疑交易</Select.Option>
                  <Select.Option value="高风险">高风险</Select.Option>
                  <Select.Option value="调查中">调查中</Select.Option>
                  <Select.Option value="已确认">已确认</Select.Option>
                </Select>
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
                  onClick={() =>
                    startEditing("tags", [...selectedSnapshot.tags])
                  }
                />
              </div>
            )}
          </Descriptions.Item>
          <Descriptions.Item label="可疑地址列表">
            {graphData.nodes?.some((node) => node.malicious === 1) ? (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {graphData.nodes
                  ?.filter((node) => node.malicious === 1)
                  .map((node) => (
                    <Tag key={node.id} color="red">
                      {node.addr || node.title}
                    </Tag>
                  ))}
              </div>
            ) : (
              <span>当前图谱暂无可疑地址</span>
            )}
          </Descriptions.Item>
        </Descriptions>

        <Divider />

        {/* 图谱筛选信息和图谱快照 */}
        <div style={{ marginTop: 20 }}>
          <h4>筛选条件</h4>
          <TxGraphFilter
            value={{
              ...filterConfig,
              startDate: parseDateSafely(filterConfig.startDate),
              endDate: parseDateSafely(filterConfig.endDate),
            }}
            onChange={handleFilterChange}
            links={graphData.links}
          />

          <h4>图谱快照</h4>
          <div
            id="graph-container"
            style={{
              height: "500px",
              backgroundColor: "#ffffff",
              borderRadius: 8,
              position: "relative",
            }}
          >
            {!isError ? (
              <TxGraph
                nodes={graphData.nodes}
                links={graphData.links}
                width={740}
                height={500}
                filter={{
                  ...filterConfig,
                  startDate:
                    parseDateSafely(filterConfig.startDate)?.toDate() || null,
                  endDate:
                    parseDateSafely(filterConfig.endDate)?.toDate() || null,
                }}
                onFilterChange={handleFilterChange}
              />
            ) : (
              <ErrorPlaceholder
                type="network"
                title="图谱数据加载失败"
                description="无法连接到数据服务器，请检查网络连接后重试"
                onRetry={() => {
                  setIsError(false);
                  setLoading(true);
                  window.location.reload();
                }}
              />
            )}
            {loading && (
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  backgroundColor: "rgba(255, 255, 255, 0.7)",
                  zIndex: 10,
                  borderRadius: "8px",
                }}
              >
                <Spin size="large" tip="正在加载交易图谱数据..." />
              </div>
            )}
          </div>
        </div>

        <Divider />

        {/* 评论区域 */}
        <div style={{ marginBottom: 24 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 12,
            }}
          >
            <div style={{ fontSize: 16, fontWeight: 600 }}>协同评论</div>
            <span style={{ color: "#888", fontSize: 12 }}>
              {selectedSnapshot.comments?.length || 0} 条评论
            </span>
          </div>

          <List
            dataSource={selectedSnapshot.comments || []}
            locale={{ emptyText: "暂无评论，开始协同记录调查结论。" }}
            renderItem={(comment) => (
              <List.Item>
                <List.Item.Meta
                  avatar={
                    <Avatar style={{ backgroundColor: "#87d068" }}>
                      {comment.author.slice(0, 1)}
                    </Avatar>
                  }
                  title={
                    <span>
                      {comment.author}
                      <span
                        style={{ marginLeft: 12, color: "#999", fontSize: 12 }}
                      >
                        {comment.createdAt}
                      </span>
                    </span>
                  }
                  description={comment.content}
                />
              </List.Item>
            )}
          />

          <Input.TextArea
            rows={3}
            value={commentContent}
            onChange={(e) => setCommentContent(e.target.value)}
            placeholder="添加协同评论，例如调查结论、疑点提示或复核意见。按 Ctrl+Enter 发送。"
            onPressEnter={(e) => {
              if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                if (commentContent.trim()) {
                  onAddComment?.(selectedSnapshot.id, commentContent.trim());
                  setCommentContent("");
                }
              }
            }}
          />
          <div style={{ textAlign: "right", marginTop: 8 }}>
            <Button
              type="primary"
              disabled={!commentContent.trim()}
              onClick={() => {
                if (commentContent.trim()) {
                  onAddComment?.(selectedSnapshot.id, commentContent.trim());
                  setCommentContent("");
                }
              }}
            >
              添加评论
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GraphDisplay;
