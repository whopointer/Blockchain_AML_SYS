import React, { useEffect, useState, useRef, useCallback } from "react";
import {
  Tag,
  Space,
  Button,
  Descriptions,
  Input,
  Select,
  Spin,
  message,
} from "antd";
import {
  CopyOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EditOutlined,
  FilePdfOutlined,
  FileExcelOutlined,
  PictureOutlined,
} from "@ant-design/icons";
import TxGraph from "../GraphCommon/TxGraph";
import TxGraphFilter from "../GraphCommon/TxGraphFilter";
import { GraphSnapshot } from "./types";
import { transactionApi } from "../../services/transaction/api";
import { NodeItem, LinkItem } from "../GraphCommon/types";
import dayjs from "dayjs";
import { formatEthValue } from "../../utils/ethUtils";
import {
  generatePDFReport,
  exportFullGraphToPNG,
  convertGraphToCSV,
  downloadCSV,
  exportCasePackage,
} from "../../utils/exportUtils";
import ErrorPlaceholder from "../GraphCommon/ErrorPlaceholder";

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
  exportPendingSnapshotId?: string | null;
  onAutoExportComplete?: () => void;
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
  exportPendingSnapshotId,
  onAutoExportComplete,
}) => {
  const [exportLoading, setExportLoading] = useState<{
    pdf: boolean;
    csv: boolean;
    png: boolean;
    package: boolean;
  }>({ pdf: false, csv: false, png: false, package: false });
  const [graphData, setGraphData] = useState<{
    nodes?: NodeItem[];
    links?: LinkItem[];
  }>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [isError, setIsError] = useState<boolean>(false);
  const autoExportTriggeredRef = useRef(false);
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
          // 使用 getNhopGraph 查询图谱
          response = await transactionApi.getNhopGraph(
            selectedSnapshot.centerAddress,
            selectedSnapshot.hops,
          );
        } else if (selectedSnapshot.fromAddress && selectedSnapshot.toAddress) {
          // 使用 getAllPath 查询图谱
          response = await transactionApi.getAllPath(
            selectedSnapshot.fromAddress,
            selectedSnapshot.toAddress,
          );
        }

        if (response && response.success && response.data) {
          // 转换API返回的数据为组件需要的格式
          const { node_list: nodes, edge_list: edges } = response.data;

          // 转换节点数据 - 根据实际API响应结构调整
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
            expanded: index === 0, // 只展开主节点
            track: node.track || "one",
            pid: node.pid || undefined,
            color: node.color || undefined,
            exg: node.exg || undefined,
          }));

          // 转换边数据 - 根据实际API响应结构调整
          const convertedLinks: LinkItem[] = edges.map((edge: any) => {
            // 查找对应的节点ID
            const fromNode = convertedNodes.find((n) => n.addr === edge.from);
            const toNode = convertedNodes.find((n) => n.addr === edge.to);
            const rawTime = edge.tx_time || edge.timestamp;
            const parsedTime = parseDateSafely(rawTime);

            // 处理交易值 - 如果是ETH则从wei转换为eth
            let processedVal = edge.val || edge.value || 0;
            let processedLabel = edge.label || "";

            // 检查是否是ETH交易（根据地址格式或其他特征判断）
            const isEthTransaction =
              edge.from?.startsWith("0x") || edge.to?.startsWith("0x");

            if (isEthTransaction) {
              // 将wei转换为eth
              processedVal = parseFloat(formatEthValue(processedVal));
              processedLabel =
                processedLabel ||
                `${formatEthValue(edge.val || edge.value || 0)} ETH`;
            } else {
              // BTC或其他货币，保持原样
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
      console.error("复制地址失败：", error);
      message.error("复制地址失败");
    }
  };

  const handleFilterChange = (newFilter: any) => {
    setFilterConfig(newFilter);
  };

  // 导出PDF报告
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

  useEffect(() => {
    if (
      exportPendingSnapshotId &&
      selectedSnapshot.id === exportPendingSnapshotId &&
      !loading &&
      !isError &&
      !autoExportTriggeredRef.current
    ) {
      autoExportTriggeredRef.current = true;
      handleExportPDF().finally(() => {
        onAutoExportComplete?.();
      });
    }
  }, [
    exportPendingSnapshotId,
    selectedSnapshot.id,
    loading,
    isError,
    onAutoExportComplete,
    handleExportPDF,
  ]);

  useEffect(() => {
    if (selectedSnapshot.id !== exportPendingSnapshotId) {
      autoExportTriggeredRef.current = false;
    }
  }, [selectedSnapshot.id, exportPendingSnapshotId]);

  // 导出CSV数据
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

  // 导出PNG图片 - 导出完整 SVG 内容
  const handleExportPNG = async () => {
    setExportLoading((prev) => ({ ...prev, png: true }));
    try {
      // 获取 graph-container 中的 SVG 元素
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

  // 导出完整案件包
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
                // 重新获取数据
                const fetchGraphData = async () => {
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
                    } else if (
                      selectedSnapshot.fromAddress &&
                      selectedSnapshot.toAddress
                    ) {
                      response = await transactionApi.getAllPath(
                        selectedSnapshot.fromAddress,
                        selectedSnapshot.toAddress,
                      );
                    }

                    if (response && response.success && response.data) {
                      const { node_list, edge_list } = response.data;
                      setGraphData({
                        nodes: node_list.map((node: any, index: number) => ({
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
                        })),
                        links: edge_list.map((edge: any) => ({
                          from: edge.from,
                          to: edge.to,
                          label: edge.label || `${edge.val || edge.value}`,
                          val: edge.val || edge.value || 0,
                          tx_time: edge.tx_time || "",
                          tx_hash_list: edge.tx_hash_list || [edge.tx_hash],
                        })),
                      });
                    }
                  } catch (error) {
                    console.error("重新加载失败:", error);
                    setIsError(true);
                  } finally {
                    setLoading(false);
                  }
                };
                fetchGraphData();
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
    </div>
  );
};

export default GraphDisplay;
