import React, { useEffect, useState, useMemo } from "react";
import { Empty } from "antd";
import { NodeItem, LinkItem } from "../types";
import TxDetail from "../TxDetail";
import D3Renderer from "./D3Renderer";

interface TxGraphCoreProps {
  nodes?: NodeItem[];
  links?: LinkItem[];
  width: number;
  height: number;
  currencySymbol?: string;
  filter?: {
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
    minAmount?: number;
    maxAmount?: number;
    startDate?: Date | null;
    endDate?: Date | null;
  };
  onFilterChange?: (v: {
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
    minAmount?: number;
    maxAmount?: number;
    startDate?: Date | null;
    endDate?: Date | null;
  }) => void;
  onGraphUpdate?: (nodes: NodeItem[], links: LinkItem[]) => void;
}

const TxGraphCore: React.FC<TxGraphCoreProps> = ({
  nodes,
  links,
  width,
  height,
  currencySymbol,
  filter,
  onFilterChange,
  onGraphUpdate,
}) => {
  const [selectedLink, setSelectedLink] = useState<LinkItem | null>(null);
  const [showDetail, setShowDetail] = useState(false);
  const [internalFilter, setInternalFilter] = useState<{
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
    minAmount?: number;
    maxAmount?: number;
    startDate?: Date | null;
    endDate?: Date | null;
  }>({ txType: "all", addrType: "all", startDate: null, endDate: null });
  const [currentCenterNodeId, setCurrentCenterNodeId] = useState<string | null>(
    null,
  );

  const useFilter = useMemo(
    () => filter ?? internalFilter,
    [filter, internalFilter],
  );

  useEffect(() => {
    if (filter !== undefined) {
      setInternalFilter((prev) => {
        const hasChanged =
          prev.txType !== filter.txType ||
          prev.addrType !== filter.addrType ||
          prev.minAmount !== filter.minAmount ||
          prev.maxAmount !== filter.maxAmount ||
          (prev.startDate?.getTime?.() || null) !==
            (filter.startDate?.getTime?.() || null) ||
          (prev.endDate?.getTime?.() || null) !==
            (filter.endDate?.getTime?.() || null);

        if (hasChanged) {
          return { ...filter };
        }
        return prev;
      });
    }
  }, [filter]);

  // 根据节点类型和交易金额计算颜色
  // malicious > 0: 红色, 有标签: 灰色(根据金额深浅), 普通: 蓝色(根据金额深浅)
  const colorForNode = (
    node: NodeItem,
    amountRatio: number,
    isCenter: boolean,
  ) => {
    // 中心节点保持原色
    if (isCenter) {
      if (node.malicious && node.malicious > 0) return "#FF6B6B";
      if (node.image) return "#888888";
      return "#3A86FF";
    }

    // 恶意节点：红色
    if (node.malicious && node.malicious > 0) {
      return "#FF6B6B";
    }

    // 有标签的节点：灰色，根据金额从浅到深
    if (node.image) {
      // 灰色从浅到深：浅灰约 220，深灰约 60（增大对比度）
      const grayValue = Math.round(220 - amountRatio * 160);
      return `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
    }

    // 普通节点：蓝色，根据金额从浅到深
    // 浅蓝 -> 深蓝，增大对比度
    // 浅蓝: rgb(200, 220, 255) -> 深蓝: rgb(30, 80, 200)
    const blueR = Math.round(200 - amountRatio * 170); // 200 -> 30
    const blueG = Math.round(220 - amountRatio * 140); // 220 -> 80
    const blueB = Math.round(255 - amountRatio * 55); // 255 -> 200
    return `rgb(${blueR}, ${blueG}, ${blueB})`;
  };

  // 计算筛选后的节点与连线
  const { filteredNodes, filteredLinks } = useMemo(() => {
    if (!nodes || !links) return { filteredNodes: [], filteredLinks: [] };

    const root = nodes.find((n) => n.layer === 0) || nodes[0];

    let byTx = links.slice();
    if (useFilter.txType === "inflow") {
      // 查找所有已展开的节点
      const expandedNodeIds = nodes.filter((n) => n.expanded).map((n) => n.id);
      // 保留与根节点或已展开节点相关的入边
      byTx = byTx.filter(
        (l) => l.to === root.id || expandedNodeIds.includes(l.to),
      );
    } else if (useFilter.txType === "outflow") {
      // 查找所有已展开的节点
      const expandedNodeIds = nodes.filter((n) => n.expanded).map((n) => n.id);
      // 保留与根节点或已展开节点相关的出边
      byTx = byTx.filter(
        (l) => l.from === root.id || expandedNodeIds.includes(l.from),
      );
    }

    // 按金额范围过滤
    const minAmount = useFilter.minAmount ?? 0;
    const maxAmount = useFilter.maxAmount ?? Number.MAX_VALUE;
    byTx = byTx.filter((l) => l.val >= minAmount && l.val <= maxAmount);

    // 按时间范围过滤（精确到秒）
    if (useFilter.startDate || useFilter.endDate) {
      const startTimestamp = useFilter.startDate
        ? new Date(useFilter.startDate).getTime()
        : 0;
      const endTimestamp = useFilter.endDate
        ? new Date(useFilter.endDate).getTime()
        : Number.MAX_VALUE;
      byTx = byTx.filter((l) => {
        let txTimestamp: number;
        if (typeof l.tx_time === "string") {
          if (l.tx_time.includes("-")) {
            txTimestamp = new Date(l.tx_time).getTime();
          } else {
            txTimestamp = new Date(parseInt(l.tx_time)).getTime();
          }
        } else {
          const numTime = Number(l.tx_time);
          txTimestamp = numTime > 1e10 ? numTime : numTime * 1000;
        }
        return txTimestamp >= startTimestamp && txTimestamp <= endTimestamp;
      });
    }

    const matchAddr = (n: NodeItem) => {
      const isTagged = !!(n.label && !n.label.startsWith("0x"));
      const isMalicious = !!(n.malicious && n.malicious > 0);
      if (useFilter.addrType === "all") return true;
      if (useFilter.addrType === "tagged") return isTagged;
      if (useFilter.addrType === "malicious") return isMalicious;
      if (useFilter.addrType === "normal") return !isTagged && !isMalicious;
      if (useFilter.addrType === "tagged_malicious")
        return isTagged || isMalicious;
      return true;
    };

    const nodeSet = new Set<string>();
    byTx.forEach((l) => {
      nodeSet.add(l.from);
      nodeSet.add(l.to);
    });

    const filteredNodes = nodes.filter(
      (n) => n === root || (nodeSet.has(n.id) && matchAddr(n)),
    );

    const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredLinks = byTx.filter(
      (l) => filteredNodeIds.has(l.from) && filteredNodeIds.has(l.to),
    );

    return { filteredNodes, filteredLinks };
  }, [
    nodes,
    links,
    useFilter.txType,
    useFilter.addrType,
    useFilter.minAmount,
    useFilter.maxAmount,
    useFilter.startDate,
    useFilter.endDate,
  ]);

  // 当 nodes 或 links 发生变化时，只在初始加载时清除位置和缩放信息
  // 双击加载时保留现有位置
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  useEffect(() => {
    // 检测中心节点是否发生变化
    const newCenterNode = nodes?.find((n) => n.layer === 0);
    const newCenterNodeId = newCenterNode?.id || null;

    if (
      currentCenterNodeId &&
      newCenterNodeId &&
      currentCenterNodeId !== newCenterNodeId
    ) {
      // 中心节点发生变化，需要重新计算视图位置
      // Note: In the new architecture, we don't store positions in this component anymore
      // Positions will be handled in the D3Renderer component
    } else if (isInitialLoad) {
      // Initial load setup
      setIsInitialLoad(false);
    }

    // 更新当前中心节点ID
    if (newCenterNodeId) {
      setCurrentCenterNodeId(newCenterNodeId);
    }
  }, [nodes, links, isInitialLoad, currentCenterNodeId]);

  const handleLinkClick = (link: LinkItem) => {
    setSelectedLink(link);
    setShowDetail(true);
  };

  // 计算金额范围用于灰度映射
  const {
    minAmount,
    maxAmount,
    amountRange,
    getEdgeColor,
    isLinkToMalicious,
    nodeAmountMap,
    minNodeAmount,
    maxNodeAmount,
    nodeAmountRange,
  } = useMemo(() => {
    const amounts = filteredLinks.map((l) => l.val);
    const minAmt = amounts.length > 0 ? Math.min(...amounts) : 0;
    const maxAmt = amounts.length > 0 ? Math.max(...amounts) : 1;
    const amtRange = maxAmt - minAmt || 1;

    // 根据金额计算灰度颜色（金额越大颜色越深）
    const getEdgeClr = (val: number) => {
      const ratio = (val - minAmt) / amtRange;
      // 灰度从浅到深：浅色约 200，深色约 30
      const grayValue = Math.round(200 - ratio * 150);
      return `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
    };

    // 判断边是否连接到风险节点
    const isLinkToRisk = (link: LinkItem) => {
      const fromNode = filteredNodes.find((n) => n.id === link.from);
      const toNode = filteredNodes.find((n) => n.id === link.to);
      return (
        (fromNode?.malicious && fromNode.malicious > 0) ||
        (toNode?.malicious && toNode.malicious > 0) ||
        false
      );
    };

    // 计算每个节点的最大单笔交易金额（用于节点颜色深浅）
    // 使用最大单笔金额而不是总金额，避免大额节点垄断深色
    const nodeAmtMap = new Map<string, number>();
    filteredLinks.forEach((link) => {
      // 取该节点相关边的最大金额
      const fromMax = nodeAmtMap.get(link.from) || 0;
      const toMax = nodeAmtMap.get(link.to) || 0;
      nodeAmtMap.set(link.from, Math.max(fromMax, link.val));
      nodeAmtMap.set(link.to, Math.max(toMax, link.val));
    });

    // 使用与边相同的金额范围，确保节点和边的颜色映射一致
    const minNodeAmt = minAmt;
    const maxNodeAmt = maxAmt;
    const nodeAmtRng = amtRange;

    return {
      minAmount: minAmt,
      maxAmount: maxAmt,
      amountRange: amtRange,
      getEdgeColor: getEdgeClr,
      isLinkToMalicious: isLinkToRisk,
      nodeAmountMap: nodeAmtMap,
      minNodeAmount: minNodeAmt,
      maxNodeAmount: maxNodeAmt,
      nodeAmountRange: nodeAmtRng,
    };
  }, [filteredLinks, filteredNodes]);

  const handleCloseDetail = () => {
    setShowDetail(false);
    setSelectedLink(null);
  };

  // 空数据状态
  const isEmpty = !filteredNodes || filteredNodes.length === 0;

  if (isEmpty) {
    return (
      <div
        style={{
          width,
          height,
          border: "1px solid #e6e6e6",
          backgroundColor: "#fafafa",
          borderRadius: 8,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="暂无交易数据"
        />
      </div>
    );
  }

  return (
    <>
      <D3Renderer
        nodes={filteredNodes}
        links={filteredLinks}
        width={width}
        height={height}
        onLinkClick={handleLinkClick}
        onGraphUpdate={onGraphUpdate}
        nodeAmountMap={nodeAmountMap}
        minNodeAmount={minNodeAmount}
        maxNodeAmount={maxNodeAmount}
        nodeAmountRange={nodeAmountRange}
        colorForNode={colorForNode}
        isLinkToMalicious={isLinkToMalicious}
        getEdgeColor={getEdgeColor}
      />

      {/* 边详情弹窗 */}
      <TxDetail
        show={showDetail}
        onHide={handleCloseDetail}
        link={selectedLink}
        currencySymbol={currencySymbol}
      />
    </>
  );
};

export default TxGraphCore;
