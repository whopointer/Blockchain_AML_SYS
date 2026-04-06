import React, { useEffect, useState, useCallback, useRef } from "react";
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
import TxGraph from "@/components/GraphCommon/TxGraph";
import TxGraphFilter from "@/components/GraphCommon/TxGraphFilter";
import { GraphSnapshot } from "../../types";
import { transactionApi } from "@/services/transaction/api";
import { NodeItem, LinkItem } from "@/components/GraphCommon/types";
import dayjs from "dayjs";

import {
  generatePDFReport,
  exportFullGraphToPNG,
  convertGraphToCSV,
  downloadCSV,
  exportCasePackage,
} from "@/utils/exportUtils";
import ErrorPlaceholder from "@/components/GraphCommon/ErrorPlaceholder";

interface GraphDisplayProps {
  selectedSnapshot: GraphSnapshot;
  loading: boolean;
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
  const [dimensions, setDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dimensionsTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // 为每个图谱快照单独存储拖拽位移和缩放大小
  const [graphTransform, setGraphTransform] = useState<{
    x: number;
    y: number;
    k: number;
  }>({ x: 0, y: 0, k: 1 });

  // 使用 useRef 来存储所有快照的变换状态，避免每次切换都重置
  const snapshotTransformsRef = useRef<
    Record<string, { x: number; y: number; k: number }>
  >({});

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

  // 当切换快照时，恢复或初始化对应的变换状态
  useEffect(() => {
    if (selectedSnapshot?.id) {
      const snapshotId = selectedSnapshot.id;

      // 如果之前保存过这个快照的变换状态，就恢复它
      if (snapshotTransformsRef.current[snapshotId]) {
        const cachedTransform = snapshotTransformsRef.current[snapshotId];
        // 只在变换实际变化时更新状态
        if (
          graphTransform.x !== cachedTransform.x ||
          graphTransform.y !== cachedTransform.y ||
          graphTransform.k !== cachedTransform.k
        ) {
          setGraphTransform(cachedTransform);
        }
      } else {
        // 否则初始化为默认状态（不在缓存中存储，表示从未交互过）
        // 只在变换实际变化时更新状态
        if (
          graphTransform.x !== 0 ||
          graphTransform.y !== 0 ||
          graphTransform.k !== 1
        ) {
          setGraphTransform({ x: 0, y: 0, k: 1 });
        }
        // 注意：这里不保存到缓存，只有用户实际交互后才保存
      }
    }
  }, [selectedSnapshot?.id, graphTransform.x, graphTransform.y, graphTransform.k]);

  // 使用防抖来减少变换更新的频率
  const transformChangeTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pendingTransformRef = useRef<{ x: number; y: number; k: number } | null>(null);

  // 处理变换状态变化，保存到对应的快照（防抖处理）
  const handleTransformChange = useCallback(
    (transform: { x: number; y: number; k: number }) => {
      if (selectedSnapshot?.id) {
        const snapshotId = selectedSnapshot.id;
        
        // 保存待处理的变换
        pendingTransformRef.current = transform;
        
        // 清除之前的定时器
        if (transformChangeTimeoutRef.current) {
          clearTimeout(transformChangeTimeoutRef.current);
        }
        
        // 设置新的定时器（防抖：延迟50ms后执行，因为现在只在交互结束时触发）
        transformChangeTimeoutRef.current = setTimeout(() => {
          if (pendingTransformRef.current && selectedSnapshot?.id === snapshotId) {
            const finalTransform = pendingTransformRef.current;
            // 检查变换是否实际发生变化，避免不必要的更新
            const currentTransform = snapshotTransformsRef.current[snapshotId];
            if (
              !currentTransform ||
              currentTransform.x !== finalTransform.x ||
              currentTransform.y !== finalTransform.y ||
              currentTransform.k !== finalTransform.k
            ) {
              // 更新当前状态
              setGraphTransform(finalTransform);
              // 保存到缓存（只有用户实际交互后才保存）
              snapshotTransformsRef.current[snapshotId] = finalTransform;
            }
            pendingTransformRef.current = null;
          }
        }, 50); // 50ms防抖延迟，因为现在只在交互结束时触发
      }
    },
    [selectedSnapshot?.id],
  );

  // 清理定时器
  useEffect(() => {
    return () => {
      if (transformChangeTimeoutRef.current) {
        clearTimeout(transformChangeTimeoutRef.current);
      }
    };
  }, []);
  
  // 快照切换时取消未完成的定时器
  useEffect(() => {
    // 取消未完成的定时器
    if (transformChangeTimeoutRef.current) {
      clearTimeout(transformChangeTimeoutRef.current);
      transformChangeTimeoutRef.current = null;
    }
    // 清空待处理的变换
    pendingTransformRef.current = null;
  }, [selectedSnapshot?.id]);

  useEffect(() => {
    const fetchGraphData = async () => {
      setLoading(true);
      setIsError(false);
      try {
        if (
          selectedSnapshot.graphData &&
          selectedSnapshot.dataSource === "snapshot"
        ) {
          const { nodes, links } = selectedSnapshot.graphData;

          const convertedNodes: NodeItem[] = nodes.map((node: any) => ({
            id: node.id || node.addr,
            label: node.label || node.addr,
            title: node.title || node.label || node.addr,
            addr: node.addr,
            layer: node.layer || 0,
            value: node.value || 0,
            malicious: node.malicious || undefined,
            shape: node.shape || undefined,
            image: node.image || undefined,
            expanded: node.expanded || false,
            track: node.track || "one",
            pid: node.pid || undefined,
            color: node.color || undefined,
            exg: node.exg || undefined,
            x: node.x,
            y: node.y,
            fx: node.fx,
            fy: node.fy,
          }));

          const convertedLinks: LinkItem[] = links.map((edge: any) => {
            const fromNode = convertedNodes.find(
              (n) => n.id === edge.from || n.addr === edge.from,
            );
            const toNode = convertedNodes.find(
              (n) => n.id === edge.to || n.addr === edge.to,
            );

            return {
              from: fromNode?.id || edge.from,
              to: toNode?.id || edge.to,
              label: edge.label || "",
              val: edge.val || 0,
              tx_time: edge.tx_time || "",
              tx_hash_list: edge.tx_hash_list || [],
            };
          });

          setGraphData({ nodes: convertedNodes, links: convertedLinks });
          setLoading(false);
          return;
        }

        let response;

        if (
          selectedSnapshot.centerAddress &&
          selectedSnapshot.hops !== undefined
        ) {
          response = await transactionApi.getNhopGraph(
            selectedSnapshot.centerAddress,
            selectedSnapshot.hops,
          );

          const { node_list: nodes, edge_list: links } = response.data;

          const convertedNodes: NodeItem[] = nodes.map((node: any) => ({
            id: node.id || node.addr,
            label: node.label || node.addr,
            title: node.title || node.label || node.addr,
            addr: node.addr,
            layer: node.layer || 0,
            value: node.value || 0,
            malicious: node.malicious || undefined,
            shape: node.shape || undefined,
            image: node.image || undefined,
            expanded: node.expanded || false,
            track: node.track || "one",
            pid: node.pid || undefined,
            color: node.color || undefined,
            exg: node.exg || undefined,
            x: node.x,
            y: node.y,
            fx: node.fx,
            fy: node.fy,
          }));

          const convertedLinks: LinkItem[] = links.map((edge: any) => {
            const fromNode = convertedNodes.find(
              (n) => n.id === edge.from || n.addr === edge.from,
            );
            const toNode = convertedNodes.find(
              (n) => n.id === edge.to || n.addr === edge.to,
            );

            return {
              from: fromNode?.id || edge.from,
              to: toNode?.id || edge.to,
              label: edge.label || "",
              val: edge.val || 0,
              tx_time: edge.tx_time || "",
              tx_hash_list: edge.tx_hash_list || [],
            };
          });

          setGraphData({ nodes: convertedNodes, links: convertedLinks });
        }
      } catch (error) {
        setIsError(true);
        console.error("Failed to fetch graph data:", error);
      } finally {
        setLoading(false);
      }

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
    };

    fetchGraphData();
  }, [selectedSnapshot]);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { offsetWidth, offsetHeight } = containerRef.current;
        if (offsetWidth > 0 && offsetHeight > 0) {
          setDimensions({ width: offsetWidth, height: offsetHeight });
        }
      }
    };

    updateDimensions();

    const resizeObserver = new ResizeObserver(() => {
      if (dimensionsTimeoutRef.current) {
        clearTimeout(dimensionsTimeoutRef.current);
      }
      dimensionsTimeoutRef.current = setTimeout(updateDimensions, 100);
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
      if (dimensionsTimeoutRef.current) {
        clearTimeout(dimensionsTimeoutRef.current);
      }
    };
  }, []);

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

  const handleGraphUpdate = useCallback(
    (updatedNodes: NodeItem[], updatedLinks: LinkItem[]) => {
      setGraphData((prev) => ({
        ...prev,
        nodes: updatedNodes,
        links: updatedLinks,
      }));
    },
    [],
  );

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
        {(exportLoading.pdf ||
          exportLoading.csv ||
          exportLoading.png ||
          exportLoading.package) && (
          <div
            style={{
              padding: "10px 16px",
              marginBottom: "16px",
              borderRadius: "4px",
              backgroundColor: "#e6f7ff",
              border: "1px solid #91d5ff",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Spin size="small" style={{ marginRight: "8px" }} />
            <span>
              {exportLoading.pdf && "正在导出PDF"}
              {exportLoading.csv && "正在导出CSV"}
              {exportLoading.png && "正在导出PNG"}
              {exportLoading.package && "正在导出完整包"}
            </span>
          </div>
        )}
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
            ref={containerRef}
            style={{
              height: "500px",
              backgroundColor: "#ffffff",
              borderRadius: 8,
              position: "relative",
            }}
          >
            {dimensions && !isError ? (
              <TxGraph
                key={selectedSnapshot.id} // 添加key确保快照切换时重新渲染
                nodes={graphData.nodes}
                links={graphData.links}
                width={dimensions.width}
                height={dimensions.height}
                filter={{
                  ...filterConfig,
                  startDate:
                    parseDateSafely(filterConfig.startDate)?.toDate() || null,
                  endDate:
                    parseDateSafely(filterConfig.endDate)?.toDate() || null,
                }}
                onFilterChange={handleFilterChange}
                onGraphUpdate={handleGraphUpdate}
                initialTransform={graphTransform}
                onTransformChange={handleTransformChange}
              />
            ) : (
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  height: "500px",
                  backgroundColor: "#f5f5f5",
                  borderRadius: "8px",
                }}
              >
                {isError ? (
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
                ) : (
                  <span>暂无数据</span>
                )}
              </div>
            )}
            {loading && dimensions && (
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
