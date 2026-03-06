import React, { useState, useEffect, useRef } from "react";
import { useParams, useSearchParams, useNavigate } from "react-router-dom";
import {
  Row,
  Col,
  Form,
  Input,
  Select,
  Button,
  Card,
  message,
  Spin,
} from "antd";
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
import SearchBar from "./SearchBar";
import ResultSearchBar from "./ResultSearchBar";

dayjs.locale("zh-cn");

const { Option } = Select;

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
  }>({ txType: "all", addrType: "all" });

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

    // 在组件挂载时立即计算一次尺寸
    updateDimensions();

    // 添加防抖以避免频繁更新
    const handleResize = () => {
      setTimeout(() => {
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

        // 将API响应数据转换为组件所需的格式
        const { node_list: nodes, edge_list: edges } = response.data;

        // 转换节点数据
        // 服务器返回字段可能是 `addr` 或 `address`，先统一处理
        const convertedNodes: NodeItem[] = nodes.map((node: any, index) => {
          const address = node.address || node.addr || ""; // 优先取 address，其次 addr
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
            expanded: index === 0, // 仅展开主节点
            track: node.track || "one",
            pid: node.pid || undefined,
            color: node.color || undefined,
            exg: node.exg || undefined,
          };
        });

        // 转换边数据
        const convertedLinks: LinkItem[] = edges.map((edge: any) => {
          // 查找对应的节点ID
          const fromNode = convertedNodes.find((n) => n.addr === edge.from);
          const toNode = convertedNodes.find((n) => n.addr === edge.to);

          // 处理时间字段 - 服务器返回的是日期时间字符串，需要转换
          let formattedTime = edge.tx_time || "";
          if (edge.tx_time) {
            try {
              // 检查是否为包含年份前缀的特殊格式（如 "+58073-07-30 22:16:22"）
              if (
                typeof edge.tx_time === "string" &&
                edge.tx_time.startsWith("+")
              ) {
                // 提取日期时间部分（去掉前面的 "+" 和年份）
                const dateTimeStr = edge.tx_time.substring(1); // 去掉开头的 "+"
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

    // 使用navigate更新URL，包含币种路径参数和地址查询参数
    const params = new URLSearchParams();
    params.set("fromAddress", values.fromAddress);
    params.set("toAddress", values.toAddress);

    navigate(`/path-tracking/${values.currency}?${params.toString()}`);

    handleSearch(values.fromAddress, values.toAddress, values.currency);
  };

  // 获取当前地址信息（主节点）
  const mainNode = graphData.nodes?.find((node) => node.layer === 0);

  const handleCreateSnapshot = (snapshotData: any) => {
    // 创建图谱快照的逻辑
    console.log("创建图谱快照", snapshotData);

    // 打印当前图谱的节点和边信息
    console.log("=== 图谱快照信息 ===");
    console.log("快照元数据:", {
      title: snapshotData.title,
      description: snapshotData.description,
      tags: snapshotData.tags,
    });

    console.log("节点信息 (Nodes):", graphData.nodes);
    console.log("边信息 (Links):", graphData.links);
    console.log("筛选条件 (Filter):", filter);

    message.success("图谱快照创建成功！");
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
            <div style={{ display: "flex", justifyContent: "center" }}>
              <div style={{ flex: 1, position: "relative" }}>
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
