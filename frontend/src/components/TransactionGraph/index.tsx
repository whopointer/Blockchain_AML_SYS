import React, { useEffect, useState, useRef, useLayoutEffect } from "react";
import TxGraph from "./TxGraph";
import TxAnalysis from "./TxAnalysis";
import { NodeItem, LinkItem, sampleData } from "./types";

const TransactionGraph: React.FC = () => {
  const [graphData, setGraphData] = useState<{
    nodes?: NodeItem[];
    links?: LinkItem[];
  }>({});

  const [dimensions, setDimensions] = useState({ width: 600, height: 600 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setGraphData(sampleData);
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  // 监听窗口大小变化，更新图表尺寸
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth || 0;
        const width = containerWidth - 420;
        const height = Math.min(Math.max(window.innerHeight * 0.7, 400), 800);
        setDimensions({ width, height });
      }
    };

    // 在组件挂载时立即计算一次尺寸
    updateDimensions();

    // 添加窗口大小变化事件监听器
    window.addEventListener("resize", updateDimensions);

    // 使用ResizeObserver监听容器尺寸变化（如果浏览器支持）
    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined" && containerRef.current) {
      resizeObserver = new ResizeObserver(() => {
        requestAnimationFrame(updateDimensions);
      });
      resizeObserver.observe(containerRef.current);
    }

    // 清理事件监听器和ResizeObserver
    return () => {
      window.removeEventListener("resize", updateDimensions);
      if (resizeObserver && containerRef.current) {
        resizeObserver.unobserve(containerRef.current);
        resizeObserver.disconnect();
      }
    };
  }, []);

  return (
    <div className="dashboard">
      <div ref={containerRef} style={{ padding: 8 }}>
        <div style={{ textAlign: "center", marginBottom: 12 }}>
          <h3>📈 交易图谱</h3>
          <p className="text-secondary">输入区块链交易ID查看可视化图谱</p>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            marginBottom: 12,
          }}
        >
          <div style={{ flex: 1}}>
            <TxGraph
              nodes={graphData.nodes}
              links={graphData.links}
              width={dimensions.width}
              height={dimensions.height}
            />
          </div>

          {/* 交易分析 */}
          <div style={{ width: "400px", minWidth: "400px", marginLeft: 20 }}>
            <TxAnalysis nodes={graphData.nodes} links={graphData.links} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default TransactionGraph;
