import React, { useEffect, useState, useRef } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import TxGraph from "../GraphCommon/TxGraph";
import TxAnalysis from "./TxAnalysis";
import TxGraphFilter from "../GraphCommon/TxGraphFilter";
import AddressInfo from "../GraphCommon/AddressInfo";
import GraphSnapshotButton from "../GraphCommon/GraphSnapshotButton";
import SearchBar from "./SearchBar";
import TransactionGraphSearch from "./TransactionGraphSearch";
import { NodeItem, LinkItem } from "../GraphCommon/types";
import { Row, Col, message, Card, Spin } from "antd";
import dayjs from "dayjs";
import "dayjs/locale/zh-cn";
import { transactionApi } from "../../services/transaction/index";
import { graphSnapshotApi } from "../../services/graph-snapshot/api";
import { formatEthValue } from "../../utils/ethUtils";

dayjs.locale("zh-cn");

const TransactionGraph: React.FC = () => {
  const { crypto, address: routeAddress } = useParams<{
    crypto: string;
    address: string;
  }>();
  const [searchParams] = useSearchParams();

  // Determine currency symbol and validate crypto parameter
  const currencySymbol = crypto?.toLowerCase() === "eth" ? "ETH" : "BTC";

  const [searchHops, setSearchHops] = useState<number>(
    parseInt(searchParams.get("hops") || "1"),
  );

  // 监听 searchParams 变化，更新 searchHops
  useEffect(() => {
    const hopsParam = searchParams.get("hops");
    const hops = hopsParam ? parseInt(hopsParam, 10) : 1;
    setSearchHops(hops);
  }, [searchParams]);

  const [graphData, setGraphData] = useState<{
    nodes?: NodeItem[];
    links?: LinkItem[];
  }>({ nodes: [], links: [] }); // 保持原始类型定义，初始化为空数组

  const [loading, setLoading] = useState<boolean>(true);
  const [isError, setIsError] = useState<boolean>(false); // 添加错误状态跟踪
  const [addressInfo, setAddressInfo] = useState<{
    txCount: number;
    firstTxTime: string;
    latestTxTime: string;
  }>({
    txCount: 0,
    firstTxTime: "",
    latestTxTime: "",
  });

  const [dimensions, setDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dimensionsTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [filter, setFilter] = useState<{
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
    minAmount?: number;
    maxAmount?: number;
    startDate?: any;
    endDate?: any;
  }>({ txType: "all", addrType: "all", startDate: null, endDate: null });

  const fetchGraphData = async () => {
    const targetAddress = routeAddress || "";
    if (!targetAddress) return;

    // 从查询参数获取hops，默认为1
    const hopsParam = searchParams.get("hops");
    const hops = hopsParam ? parseInt(hopsParam, 10) : 1;
    // 确保hops为有效数字，最小值为1
    const effectiveHops = isNaN(hops) || hops < 1 ? 1 : hops;

    try {
      setLoading(true);
      setIsError(false); // 重置错误状态

      // no hardcoded addresses anymore – data will come from API
      const response = await transactionApi.getNhopGraph(
        targetAddress,
        effectiveHops,
      );

      if (response.success) {
        console.log(response.data);
        // 转换API返回的数据为组件需要的格式
        const { node_list: nodes, edge_list: edges } = response.data;

        // 转换节点数据 - 根据实际API响应结构调整
        const convertedNodes: NodeItem[] = nodes.map((node: any, index) => ({
          id: node.id,
          label: node.label || "",
          title: node.title,
          addr: node.addr,
          layer: node.layer,
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

          // 处理交易值 - 如果是ETH则从wei转换为eth，如果是BTC则保持原样
          let processedVal = edge.val || edge.value || 0;
          let processedLabel = edge.label || "";

          if (currencySymbol === "ETH") {
            // 将wei转换为eth
            processedVal =
              typeof processedVal === "string" ||
              typeof processedVal === "number"
                ? parseFloat(formatEthValue(processedVal))
                : processedVal;
            processedLabel =
              processedLabel ||
              `${formatEthValue(edge.val || edge.value || 0)} ${currencySymbol}`;
          } else {
            // BTC或其他货币，保持原样
            processedLabel =
              processedLabel || `${processedVal} ${currencySymbol}`;
          }

          return {
            from: fromNode?.id || edge.from,
            to: toNode?.id || edge.to,
            label: processedLabel,
            val: processedVal,
            tx_time: (() => {
              const timeValue = edge.tx_time || edge.timestamp;
              const value = timeValue as unknown as string | number;
              return typeof value === "string" && value.includes("-")
                ? value.substring(0, 16)
                : dayjs(Number(value)).format("YYYY-MM-DD HH:mm");
            })(),
            tx_hash_list: edge.tx_hash_list || [edge.tx_hash],
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

        // 设置地址信息 - 根据实际API响应结构调整
        const firstTimeValue: any =
          response.data.address_first_tx_time ||
          response.data.first_tx_time ||
          Math.min(...edges.map((e: any) => e.timestamp));
        const latestTimeValue: any =
          response.data.address_latest_tx_time ||
          response.data.latest_tx_time ||
          Math.max(...edges.map((e: any) => e.timestamp));

        setAddressInfo({
          txCount: response.data.tx_count || edges.length, // 使用API返回的交易计数
          firstTxTime: (() => {
            const value = firstTimeValue as unknown as string | number;
            if (typeof value === "string" && value.includes("-")) {
              return value.substring(0, 16);
            } else {
              const timestamp =
                typeof value === "string" ? parseInt(value) : Number(value);
              return dayjs(timestamp).format("YYYY-MM-DD HH:mm");
            }
          })(),
          latestTxTime: (() => {
            const value = latestTimeValue as unknown as string | number;
            if (typeof value === "string" && value.includes("-")) {
              return value.substring(0, 16);
            } else {
              const timestamp =
                typeof value === "string" ? parseInt(value) : Number(value);
              return dayjs(timestamp).format("YYYY-MM-DD HH:mm");
            }
          })(),
        });

        message.success("图谱数据加载成功");
        setLoading(false);
      } else {
        message.error(`图谱数据加载失败: ${response.msg}`);
        setIsError(true);
        setLoading(false);
      }
    } catch (error) {
      console.error("获取图谱数据失败:", error);
      message.error("获取图谱数据失败，请稍后重试");
      setIsError(true);
      setLoading(false);
    } finally {
      //
    }
  };

  useEffect(() => {
    fetchGraphData();
  }, [routeAddress, crypto, searchHops]);

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

    // 在组件挂载时立即计算一次尺寸
    updateDimensions();

    // 添加防抖以避免频繁更新
    const handleResize = () => {
      if (dimensionsTimeoutRef.current) {
        clearTimeout(dimensionsTimeoutRef.current);
      }
      dimensionsTimeoutRef.current = setTimeout(() => {
        updateDimensions();
      }, 300);
    };

    // 添加窗口大小变化事件监听器
    window.addEventListener("resize", handleResize);

    // 使用ResizeObserver只监听宽度变化（不依赖高度）
    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined" && currentContainer) {
      resizeObserver = new ResizeObserver(() => {
        handleResize();
      });
      resizeObserver.observe(currentContainer);
    }

    // 清理事件监听器和ResizeObserver
    return () => {
      window.removeEventListener("resize", handleResize);
      if (dimensionsTimeoutRef.current) {
        clearTimeout(dimensionsTimeoutRef.current);
      }
      if (resizeObserver && currentContainer) {
        resizeObserver.unobserve(currentContainer);
        resizeObserver.disconnect();
      }
    };
  }, []);

  // 获取当前地址信息（主节点）
  const mainNode = graphData.nodes?.find((node) => node.layer === 0);

  const handleCreateSnapshot = async (snapshotData: any) => {
    try {
      // 创建图谱快照的逻辑
      console.log("创建图谱快照", snapshotData);

      // 获取中心地址（通常是第0层节点，即查询的目标地址）
      const centerAddress = mainNode?.addr || routeAddress || "";

      // 计算风险等级 - 基于恶意节点的数量
      let riskLevel: "low" | "medium" | "high" = "low";
      const maliciousNodes =
        graphData.nodes?.filter((node) => node.malicious === 1) || [];
      if (maliciousNodes.length > 0) {
        riskLevel = maliciousNodes.length > 3 ? "high" : "medium";
      }

      // 将前端风险等级转换为后端所需的大写格式
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

      // 构建快照请求数据
      const snapshotRequest = {
        title: snapshotData.title,
        description: snapshotData.description,
        tags: snapshotData.tags,
        nodeCount: graphData.nodes?.length || 0,
        linkCount: graphData.links?.length || 0,
        riskLevel: backendRiskLevel,
        centerAddress,
        hops: searchHops,
        filterConfig: filterConfigWithUTC8,
      };

      // 调用后端API创建快照
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

  // 检查是否有路由参数
  const hasRouteParams = !!routeAddress && !!crypto;

  return (
    <>
      {hasRouteParams ? (
        <div ref={containerRef} style={{ padding: 16 }}>
          {/* 搜索栏 */}
          <SearchBar
            defaultCrypto={currencySymbol}
            defaultAddress={routeAddress || ""}
            defaultHops={parseInt(searchParams.get("hops") || "1")}
          />

          {/* 地址基本信息 */}
          <AddressInfo
            address={mainNode?.addr}
            txCount={addressInfo.txCount}
            firstTxTime={addressInfo.firstTxTime}
            latestTxTime={addressInfo.latestTxTime}
            isMalicious={mainNode?.malicious === 1}
          />

          {/* 交易图谱主内容 */}
          <Card
            title={
              <Row style={{ width: "100%", alignItems: "center" }}>
                <Col flex={1} style={{ fontSize: 18 }}>
                  交易图谱
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
            <div style={{ display: "flex", justifyContent: "center" }}>
              <div style={{ flex: 1, position: "relative" }}>
                {dimensions && !isError ? (
                  <TxGraph
                    nodes={graphData.nodes}
                    links={graphData.links}
                    width={dimensions.width}
                    height={dimensions.height}
                    currencySymbol={currencySymbol}
                    filter={filter}
                    onFilterChange={setFilter}
                  />
                ) : (
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      height: dimensions ? dimensions.height : "500px",
                      backgroundColor: "#f5f5f5",
                      borderRadius: "8px",
                    }}
                  >
                    {isError ? (
                      <span style={{ color: "red" }}>
                        数据加载失败，请刷新重试
                      </span>
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

              {/* 交易分析（右侧） */}
              <div
                style={{
                  width: "400px",
                  minWidth: "400px",
                  marginLeft: 20,
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
                <TxAnalysis
                  nodes={graphData.nodes}
                  links={graphData.links}
                  currencySymbol={currencySymbol}
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
      ) : (
        <TransactionGraphSearch currencySymbol={currencySymbol} />
      )}
    </>
  );
};

export default TransactionGraph;
