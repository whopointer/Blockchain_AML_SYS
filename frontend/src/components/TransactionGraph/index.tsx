import React, { useEffect, useState, useRef } from "react";
import TxGraph from "./TxGraph";
import TxAnalysis from "./TxAnalysis";
import TxGraphFilter from "./TxGraphFilter";
import AddressInfo from "./AddressInfo";
import GraphSnapshotButton from "./GraphSnapshotButton";
import { NodeItem, LinkItem, sampleData } from "./types";
import graphAnalysisData from "./address_graph_analysis.json";
import { Row, Col, message, Card } from "antd";
import dayjs from "dayjs";
import "dayjs/locale/zh-cn";

dayjs.locale("zh-cn");

const TransactionGraph: React.FC = () => {
  const [graphData, setGraphData] = useState<{
    nodes?: NodeItem[];
    links?: LinkItem[];
  }>({});

  const [dimensions, setDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dimensionsTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [filter, setFilter] = useState<{
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
  }>({ txType: "all", addrType: "all" });

  useEffect(() => {
    const timer = setTimeout(() => {
      setGraphData(sampleData);
    }, 0);
    return () => clearTimeout(timer);
  }, []);

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
    <>
      <div ref={containerRef} style={{ padding: 16 }}>
        {/* 地址基本信息 */}
        <AddressInfo
          address={mainNode?.addr}
          txCount={graphAnalysisData.graph_dic.tx_count}
          firstTxTime={graphAnalysisData.graph_dic.first_tx_datetime}
          latestTxTime={graphAnalysisData.graph_dic.latest_tx_datetime}
          isMalicious={mainNode?.malicious === 1}
        />

        {/* 交易图谱主内容 */}
        <Card
          title={
            <Row style={{ width: "100%", alignItems: "center" }}>
              <Col flex={1} style={{fontSize: 18}}>交易图谱</Col>
              <Col>
                <GraphSnapshotButton onCreateSnapshot={handleCreateSnapshot} />
              </Col>
            </Row>
          }
          style={{ margin: "16px 0", borderRadius: 8, }}
          bordered={false}
          bodyStyle={{ padding: 16 }}
        >
          {/* 图表内容 */}
          <div style={{ display: "flex", justifyContent: "center" }}>
            <div style={{ flex: 1 }}>
              {dimensions && (
                <TxGraph
                  nodes={graphData.nodes}
                  links={graphData.links}
                  width={dimensions.width}
                  height={dimensions.height}
                  filter={filter}
                  onFilterChange={setFilter}
                />
              )}
            </div>

            {/* 交易分析（右侧） */}
            <div style={{ width: "400px", minWidth: "400px", marginLeft: 20 }}>
              <div style={{ marginBottom: 12 }}>
                <TxGraphFilter value={filter} onChange={(v) => setFilter(v)} />
              </div>
              <TxAnalysis nodes={graphData.nodes} links={graphData.links} />
            </div>
          </div>
        </Card>
      </div>
    </>
  );
};

export default TransactionGraph;