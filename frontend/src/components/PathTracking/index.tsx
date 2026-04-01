import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Row, Col, Form, Card, message, Spin } from "antd";
import {
  transactionApi,
  GraphAnalysisResponse,
} from "../../services/transaction/index";
import TxGraph from "../GraphCommon/TxGraph";
import PathTxAnalysis from "./PathTxAnalysis";
import TxGraphFilter from "../GraphCommon/TxGraphFilter";
import AddressInfo from "../GraphCommon/AddressInfo";
import GraphSnapshotButton from "../GraphCommon/GraphSnapshotButton";
import PathTrackingSearch from "./PathTrackingSearch";
import { NodeItem, LinkItem } from "../GraphCommon/types";
import dayjs from "dayjs";
import "dayjs/locale/zh-cn";
import { graphSnapshotApi } from "../../services/graph-snapshot/api";
import ResultSearchBar from "./ResultSearchBar";

dayjs.locale("zh-cn");

const PathTracking: React.FC = () => {
  const { crypto: routeCrypto } = useParams<{ crypto: string }>();
  const navigate = useNavigate();
  const urlParams = new URLSearchParams(window.location.search);
  const urlFromAddress = urlParams.get("fromAddress");
  const urlToAddress = urlParams.get("toAddress");

  const [form] = Form.useForm();
  const [loading, setLoading] = useState<boolean>(false);
  const [graphData, setGraphData] = useState<{
    nodes?: NodeItem[];
    links?: LinkItem[];
  }>({ nodes: [], links: [] });
  const [hasSearched, setHasSearched] = useState<boolean>(
    !!(urlFromAddress && urlToAddress),
  );
  const [currency, setCurrency] = useState<string>(routeCrypto || "eth");

  const [dimensions, setDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [filter, setFilter] = useState<{
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
    minAmount?: number;
    maxAmount?: number;
    startDate?: any;
    endDate?: any;
  }>({ txType: "all", addrType: "all", startDate: null, endDate: null });

  const [showSearchBoxOnly, setShowSearchBoxOnly] = useState<boolean>(
    !urlFromAddress || !urlToAddress,
  );

  useEffect(() => {
    setShowSearchBoxOnly(!urlFromAddress || !urlToAddress);
  }, [urlFromAddress, urlToAddress]);

  // 监听窗口大小变化，更新图表尺寸
  useEffect(() => {
    const currentContainer = containerRef.current;

    const updateDimensions = () => {
      if (currentContainer) {
        const containerWidth = currentContainer.clientWidth;
        if (containerWidth > 0) {
          const width = containerWidth - 480;
          const height = window.innerHeight - 120;
          setDimensions({ width, height: Math.max(height, 600) });
        }
      }
    };

    updateDimensions();

    const handleResize = () => {
      setTimeout(() => {
        updateDimensions();
      }, 300);
    };

    window.addEventListener("resize", handleResize);

    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined" && currentContainer) {
      resizeObserver = new ResizeObserver(() => {
        handleResize();
      });
      resizeObserver.observe(currentContainer);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      if (resizeObserver && currentContainer) {
        resizeObserver.unobserve(currentContainer);
        resizeObserver.disconnect();
      }
    };
  }, []);

  useEffect(() => {
    if (urlFromAddress && urlToAddress) {
      const currencyToUse = routeCrypto || "eth";
      form.setFieldsValue({
        fromAddress: urlFromAddress,
        toAddress: urlToAddress,
        currency: currencyToUse,
      });
      setCurrency(currencyToUse);
      handleSearch(urlFromAddress, urlToAddress, currencyToUse);
    }
  }, [form, urlFromAddress, urlToAddress, routeCrypto]);

  const handleSearch = async (
    fromAddr: string,
    toAddr: string,
    curr: string,
  ) => {
    if (!fromAddr || !toAddr) {
      message.error("请输入起始地址和目标地址");
      return;
    }

    setLoading(true);
    setHasSearched(true);

    try {
      const response: GraphAnalysisResponse = await transactionApi.getAllPath(
        fromAddr,
        toAddr,
      );

      if (response.success) {
        console.log(response.data);

        const { node_list: nodes, edge_list: edges } = response.data;

        const convertedNodes: NodeItem[] = nodes.map((node: any, index) => {
          const address = node.address || node.addr || "";
          return {
            id: node.id || address,
            label:
              node.label || (address ? address.substring(0, 10) + "..." : ""),
            title: address,
            addr: address,
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
          };
        });

        const convertedLinks: LinkItem[] = edges.map((edge: any) => {
          const fromNode = convertedNodes.find((n) => n.addr === edge.from);
          const toNode = convertedNodes.find((n) => n.addr === edge.to);

          let formattedTime = edge.tx_time || "";
          if (edge.tx_time) {
            try {
              if (
                typeof edge.tx_time === "string" &&
                edge.tx_time.startsWith("+")
              ) {
                const dateTimeStr = edge.tx_time.substring(1);
                formattedTime = dayjs(dateTimeStr).format("YYYY-MM-DD HH:mm");
              } else if (typeof edge.tx_time === "string") {
                formattedTime = dayjs(edge.tx_time).format("YYYY-MM-DD HH:mm");
              }
            } catch (error) {
              console.warn("时间格式转换失败:", edge.tx_time, error);
              formattedTime = edge.tx_time;
            }
          }

          return {
            from: fromNode?.id || edge.from,
            to: toNode?.id || edge.to,
            label: `${edge.val || edge.value} ${curr.toUpperCase()}`,
            val: edge.val || edge.value || 0,
            tx_time: formattedTime,
            tx_hash_list: Array.isArray(edge.tx_hash_list)
              ? edge.tx_hash_list
              : [edge.tx_hash],
          };
        });

        setGraphData({ nodes: convertedNodes, links: convertedLinks });

        // 数据更新时重置筛选条件
        setFilter({
          txType: "all",
          addrType: "all",
          minAmount: undefined,
          maxAmount: undefined,
          startDate: null,
          endDate: null,
        });

        message.success("路径数据加载成功");
      } else {
        message.error(`路径数据加载失败: ${response.msg}`);
      }
    } catch (error) {
      console.error("获取路径数据失败:", error);
      message.error("获取路径数据失败，请稍后重试");
    } finally {
      setLoading(false);
    }
  };

  const onFinish = (values: any) => {
    setCurrency(values.currency);

    const params = new URLSearchParams();
    params.set("fromAddress", values.fromAddress);
    params.set("toAddress", values.toAddress);

    navigate(`/path-tracking/${values.currency}?${params.toString()}`);

    handleSearch(values.fromAddress, values.toAddress, values.currency);
  };

  const mainNode = graphData.nodes?.find((node) => node.layer === 0);

  const handleCreateSnapshot = async (snapshotData: any) => {
    try {
      console.log("创建图谱快照", snapshotData);

      const centerAddress = mainNode?.addr || urlFromAddress || "";

      let riskLevel: "low" | "medium" | "high" = "low";
      const maliciousNodes =
        graphData.nodes?.filter((node) => node.malicious === 1) || [];
      if (maliciousNodes.length > 0) {
        riskLevel = maliciousNodes.length > 3 ? "high" : "medium";
      }

      const backendRiskLevel: "LOW" | "MEDIUM" | "HIGH" =
        riskLevel === "low"
          ? "LOW"
          : riskLevel === "medium"
            ? "MEDIUM"
            : "HIGH";

      // 转换时间为UTC+8
      const convertToUTC8 = (date: any) => {
        if (!date) return null;
        // 创建一个新的Date对象
        const d = new Date(date);
        // 转换为UTC+8时间（加上8小时）
        const utc8Time = d.getTime() + 8 * 60 * 60 * 1000;
        const utc8Date = new Date(utc8Time);
        // 转换为ISO字符串
        return utc8Date.toISOString();
      };

      const filterConfigWithUTC8 = {
        ...filter,
        startDate: convertToUTC8(filter.startDate),
        endDate: convertToUTC8(filter.endDate),
      };

      const snapshotRequest = {
        title: snapshotData.title,
        description: snapshotData.description,
        tags: snapshotData.tags,
        mainAddress: centerAddress,
        nodeCount: graphData.nodes?.length || 0,
        linkCount: graphData.links?.length || 0,
        riskLevel: snapshotData.riskLevel || backendRiskLevel,
        fromAddress: urlFromAddress || "",
        toAddress: urlToAddress || "",
        filterConfig: filterConfigWithUTC8,
      };

      const response = await graphSnapshotApi.createSnapshot(snapshotRequest);

      if (response.success) {
        message.success(response.msg || "图谱快照创建成功！");
        console.log("图谱快照创建成功:", response.data);
      } else {
        message.error(response.msg || "图谱快照创建失败");
        console.error("图谱快照创建失败:", response);
      }
    } catch (error) {
      console.error("创建图谱快照时发生错误:", error);
      message.error("创建图谱快照失败，请稍后重试");
    }
  };

  return (
    <div ref={containerRef}>
      {/* 搜索框部分 - 当没有查询字符串时显示完整界面，否则只显示搜索框 */}
      {showSearchBoxOnly ? (
        <PathTrackingSearch
          form={form}
          onFinish={onFinish}
          loading={loading}
          routeCrypto={routeCrypto}
          urlFromAddress={urlFromAddress}
          urlToAddress={urlToAddress}
        />
      ) : (
        <div style={{ padding: "0 16px" }}>
          <ResultSearchBar />
        </div>
      )}

      {!showSearchBoxOnly && (
        <div style={{ padding: "0 16px" }}>
          {/* 地址基本信息 - 显示起始地址信息 */}
          {hasSearched && mainNode && (
            <AddressInfo
              address={mainNode?.addr}
              txCount={graphData.links?.length || 0}
              firstTxTime={graphData.links?.[0]?.tx_time || ""}
              latestTxTime={
                graphData.links?.[graphData.links.length - 1]?.tx_time || ""
              }
              isMalicious={mainNode?.malicious === 1}
            />
          )}

          {/* 交易图谱主内容 */}
          <Card
            title={
              <Row style={{ width: "100%", alignItems: "center" }}>
                <Col flex={1} style={{ fontSize: 18 }}>
                  路径追踪结果
                </Col>
                <Col>
                  <Spin spinning={loading} size="small" />
                </Col>
                <Col>
                  <GraphSnapshotButton
                    onCreateSnapshot={handleCreateSnapshot}
                  />
                </Col>
              </Row>
            }
            style={{ margin: "16px 0", borderRadius: 8 }}
            bordered={false}
            bodyStyle={{ padding: 16 }}
          >
            {/* 图表内容 */}
            <div style={{ display: "flex", justifyContent: "center", gap: 20 }}>
              <div style={{ flex: 1, minWidth: "500px", position: "relative" }}>
                {dimensions ? (
                  <TxGraph
                    nodes={graphData.nodes}
                    links={graphData.links}
                    width={dimensions.width}
                    height={dimensions.height}
                    currencySymbol={currency.toUpperCase()}
                    filter={filter}
                    onFilterChange={setFilter}
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
                    <span>正在计算图表尺寸...</span>
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
                    <Spin size="large" tip="正在搜索路径..." />
                  </div>
                )}
              </div>

              {/* 交易分析（右侧） */}
              <div
                style={{
                  width: "400px",
                  minWidth: "300px",
                  maxWidth: "500px",
                  position: "relative",
                }}
              >
                <div style={{ marginBottom: 12 }}>
                  <TxGraphFilter
                    value={filter}
                    onChange={(v) => setFilter(v)}
                    links={graphData.links}
                  />
                </div>
                <PathTxAnalysis
                  nodes={graphData.nodes}
                  links={graphData.links}
                  currencySymbol={currency.toUpperCase()}
                />
                {loading && (
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      backgroundColor: "rgba(255, 255, 255, 0.7)",
                      zIndex: 1,
                      borderRadius: "8px",
                    }}
                  />
                )}
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default PathTracking;
