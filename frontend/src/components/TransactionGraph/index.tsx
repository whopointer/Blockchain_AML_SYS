import React, { useEffect, useState, useRef } from "react";
import TxGraph from "./TxGraph";
import TxAnalysis from "./TxAnalysis";
import TxGraphFilter from "./TxGraphFilter";
import AddressInfo from "./AddressInfo";
import { NodeItem, LinkItem, sampleData } from "./types";
import { ConfigProvider, Button, Row, Col } from "antd";
import { CameraOutlined } from "@ant-design/icons";
import dayjs from "dayjs";
import "dayjs/locale/zh-cn";
import zhCN from "antd/es/locale/zh_CN";

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
          const width = containerWidth - 450;
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

  const handleCreateSnapshot = () => {
    // 创建图谱快照的逻辑
    console.log("创建图谱快照");
    // TODO: 实现快照功能
  };

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorBgBase: "#244963",
          colorTextBase: "#ffffff",
          colorBorder: "#3a5f7f",
          colorPrimary: "#667eea",
        },
      }}
    >
      <div
        ref={containerRef}
        style={{ backgroundColor: "#1a3a52", borderRadius: 16 }}
      >
        {/* 地址基本信息 */}
        <AddressInfo
          address={mainNode?.addr}
          txCount={30} // 从 sampleData 获取的交易总次数
          firstTxTime="2025-07-03 04:06"
          latestTxTime="2025-07-05 01:57"
          isMalicious={mainNode?.malicious === 1}
        />

        {/* 标题栏 */}
        <Row
          style={{
            marginBottom: 16,
            padding: "12px 16px",
            backgroundColor: "#244963",
            borderRadius: 8,
            border: "1px solid #3a5f7f",
            alignItems: "center",
          }}
        >
          <Col flex={1}>
            <div
              style={{
                margin: 0,
                color: "#ffffff",
                fontSize: 18,
                lineHeight: "18px",
                fontWeight: 600,
              }}
            >
              交易图谱
            </div>
          </Col>
          <Col>
            <Button
              type="primary"
              icon={<CameraOutlined />}
              onClick={handleCreateSnapshot}
              // style={{ background: "#667eea" }}
            >
              创建图谱快照
            </Button>
          </Col>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              margin: "16px 0",
            }}
          >
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
        </Row>
      </div>
    </ConfigProvider>
  );
};

export default TransactionGraph;
