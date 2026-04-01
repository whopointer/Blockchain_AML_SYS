import React, { useRef, useEffect, useState, useMemo } from "react";
import * as d3 from "d3";
import { NodeItem, LinkItem } from "./types";
import TxDetail from "./TxDetail";
import { formatEthValue } from "../../utils/ethUtils";

interface TxGraphProps {
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
}

const TxGraph: React.FC<TxGraphProps> = ({
  nodes,
  links,
  width,
  height,
  currencySymbol,
  filter,
  onFilterChange,
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

  // 计算简单的分层布局：中心节点在中间，右侧为 layer>=0，左侧为 layer<0
  function computePositions(
    nodes: NodeItem[],
    links: LinkItem[],
    width: number,
    height: number,
  ) {
    const centerX = width / 2;
    const centerY = height / 2;
    const layerSpacing = 200;

    const root = nodes.find((n) => n.layer === 0) || nodes[0];
    if (!root) return { nodes, links };
    root.x = centerX;
    root.y = centerY;

    nodes.forEach((n) => {
      if (n.layer === undefined || n.layer === null) n.layer = 0;
    });

    const layers: { [key: number]: NodeItem[] } = {};
    nodes.forEach((n) => {
      const l = n.layer as number;
      if (!layers[l]) layers[l] = [];
      layers[l].push(n);
    });

    Object.keys(layers)
      .map((k) => Number(k))
      .forEach((layer) => {
        const group = layers[layer];
        const x = centerX + layer * layerSpacing;

        const count = group.length;
        const maxSpacing = 100;
        const minSpacing = 40;
        const spacing = Math.min(
          maxSpacing,
          Math.max(minSpacing, height / (count + 1)),
        );
        const startY = centerY - (spacing * (count - 1)) / 2;

        group.forEach((node, i) => {
          if (node === root) {
            node.x = centerX;
            node.y = centerY;
          } else {
            node.x = x;
            node.y = startY + i * spacing;
          }
        });
      });

    return { nodes, links };
  }

  const svgRef = useRef<SVGSVGElement | null>(null);
  const gRef = useRef<SVGGElement | null>(null);
  const nodesWithPositionRef = useRef<Map<string, { x: number; y: number }>>(
    new Map(),
  );
  const transformRef = useRef<{
    x: number;
    y: number;
    k: number;
  }>({ x: 0, y: 0, k: 1 });

  // 计算筛选后的节点与连线
  const { filteredNodes, filteredLinks } = useMemo(() => {
    if (!nodes || !links) return { filteredNodes: [], filteredLinks: [] };

    const root = nodes.find((n) => n.layer === 0) || nodes[0];

    let byTx = links.slice();
    if (useFilter.txType === "inflow") {
      byTx = byTx.filter((l) => l.to === root.id);
    } else if (useFilter.txType === "outflow") {
      byTx = byTx.filter((l) => l.from === root.id);
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

  // 当 nodes 或 links 发生变化时，清除之前保存的位置和缩放信息
  useEffect(() => {
    // 清除保存的节点位置
    nodesWithPositionRef.current.clear();
    // 重置缩放和拖动信息
    transformRef.current = { x: 0, y: 0, k: 1 };
  }, [nodes, links]);

  // 当 props 中的 nodes/links 更新时执行绘图
  useEffect(() => {
    if (!filteredNodes || !filteredLinks) return;

    let layout: { nodes: NodeItem[]; links: LinkItem[] };
    const hasExistingPositions = filteredNodes.every((n) =>
      nodesWithPositionRef.current.has(n.id),
    );

    if (hasExistingPositions) {
      const layoutNodes = filteredNodes.map((n) => {
        const saved = nodesWithPositionRef.current.get(n.id);
        return saved ? { ...n, x: saved.x, y: saved.y } : n;
      });
      layout = { nodes: layoutNodes, links: filteredLinks };
    } else {
      layout = computePositions(
        [...filteredNodes],
        [...filteredLinks],
        width,
        height,
      );
      layout.nodes.forEach((n) => {
        nodesWithPositionRef.current.set(n.id, { x: n.x!, y: n.y! });
      });
    }

    const svg = d3.select(svgRef.current as unknown as SVGSVGElement);
    const g = d3.select(gRef.current as unknown as SVGGElement);

    // 添加箭头定义（始终指向右侧）
    svg.select("defs").remove();
    const defs = svg.append("defs");
    defs
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("markerWidth", 8)
      .attr("markerHeight", 8)
      .attr("markerUnits", "strokeWidth")
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "var(--text-muted)");

    g.selectAll("*").remove();

    // 计算金额范围用于灰度映射
    const amountValues = layout.links.map((l) => l.val);
    const minAmount = Math.min(...amountValues);
    const maxAmount = Math.max(...amountValues);
    const amountRange = maxAmount - minAmount || 1;

    // 根据金额计算灰度颜色（金额越大颜色越深）
    const getEdgeColor = (val: number) => {
      const ratio = (val - minAmount) / amountRange;
      // 灰度从浅到深：浅色约 200，深色约 30
      const grayValue = Math.round(200 - ratio * 150);
      return `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
    };

    // 判断边是否连接到风险节点
    const isLinkToMalicious = (link: LinkItem) => {
      const fromNode = layout.nodes.find((n) => n.id === link.from);
      const toNode = layout.nodes.find((n) => n.id === link.to);
      return (
        (fromNode && fromNode.malicious && fromNode.malicious > 0) ||
        (toNode && toNode.malicious && toNode.malicious > 0)
      );
    };

    // 计算每个节点的最大单笔交易金额（用于节点颜色深浅）
    // 使用最大单笔金额而不是总金额，避免大额节点垄断深色
    const nodeAmountMap = new Map<string, number>();
    layout.links.forEach((link) => {
      // 取该节点相关边的最大金额
      const fromMax = nodeAmountMap.get(link.from) || 0;
      const toMax = nodeAmountMap.get(link.to) || 0;
      nodeAmountMap.set(link.from, Math.max(fromMax, link.val));
      nodeAmountMap.set(link.to, Math.max(toMax, link.val));
    });

    // 使用与边相同的金额范围，确保节点和边的颜色映射一致
    const minNodeAmount = minAmount;
    const maxNodeAmount = maxAmount;
    const nodeAmountRange = amountRange;

    // 链接（直线）
    const linkLines = g
      .selectAll("line.link")
      .data(layout.links)
      .enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke-width", 1.5)
      .attr(
        "x1",
        (d: LinkItem) => (layout.nodes.find((n) => n.id === d.from) as any).x,
      )
      .attr(
        "y1",
        (d: LinkItem) => (layout.nodes.find((n) => n.id === d.from) as any).y,
      )
      .attr(
        "x2",
        (d: LinkItem) => (layout.nodes.find((n) => n.id === d.to) as any).x,
      )
      .attr(
        "y2",
        (d: LinkItem) => (layout.nodes.find((n) => n.id === d.to) as any).y,
      )
      .attr("stroke", (d: LinkItem) => {
        // 连接到风险节点的边用红色
        if (isLinkToMalicious(d)) {
          return "#E74C3C";
        }
        return getEdgeColor(d.val);
      })
      .attr("stroke-width", 1.5)
      .attr("cursor", "pointer")
      .on("click", function (event, d) {
        setSelectedLink(d);
        setShowDetail(true);
        event.stopPropagation();
      });

    // 定义高亮方法
    const highlightLink = (d: LinkItem) => {
      // 高亮对应的路径线
      g.selectAll("line.link")
        .filter((ld: any) => ld === d)
        .attr("stroke-width", 2.5)
        .attr("stroke", "var(--primary-color)");

      // 高亮对应的箭头
      g.selectAll("polygon.link-arrow")
        .filter((ad: any) => ad === d)
        .each(function (ad: any) {
          const poly = d3.select(this);
          poly.attr("data-scale", "1.2");
          poly.attr("fill", "var(--primary-color)");
          const sNode = layout.nodes.find((n) => n.id === ad.from) as any;
          const tNode = layout.nodes.find((n) => n.id === ad.to) as any;
          const mx = (sNode.x + tNode.x) / 2;
          const my = (sNode.y + tNode.y) / 2;
          const angle =
            (Math.atan2(tNode.y - sNode.y, tNode.x - sNode.x) * 180) / Math.PI;
          poly.attr(
            "transform",
            `translate(${mx},${my}) rotate(${angle}) scale(1.2)`,
          );
        });

      // 高亮连接线标签
      g.selectAll("text.link-label")
        .filter((ad: any) => ad === d)
        .attr("font-weight", "bold")
        .attr("font-size", 12)
        .attr("fill", "var(--primary-color)");
    };

    // 定义恢复原始状态方法
    const restoreLink = (d: LinkItem) => {
      // 恢复路径线
      g.selectAll("line.link")
        .filter((ld: any) => ld === d)
        .attr("stroke", isLinkToMalicious(d) ? "#E74C3C" : getEdgeColor(d.val))
        .attr("stroke-width", 1.5);

      // 恢复箭头
      g.selectAll("polygon.link-arrow")
        .filter((ad: any) => ad === d)
        .each(function (ad: any) {
          const poly = d3.select(this);
          poly
            .attr("fill", isLinkToMalicious(ad) ? "#E74C3C" : getEdgeColor(ad.val))
            .attr("data-scale", "1");
          const sNode = layout.nodes.find((n) => n.id === ad.from) as any;
          const tNode = layout.nodes.find((n) => n.id === ad.to) as any;
          const mx = (sNode.x + tNode.x) / 2;
          const my = (sNode.y + tNode.y) / 2;
          const angle =
            (Math.atan2(tNode.y - sNode.y, tNode.x - sNode.x) * 180) / Math.PI;
          poly.attr(
            "transform",
            `translate(${mx},${my}) rotate(${angle}) scale(1)`,
          );
        });

      // 恢复标签
      g.selectAll("text.link-label")
        .filter((ad: any) => ad === d)
        .attr("font-weight", "normal")
        .attr("font-size", 10)
        .attr("fill", "var(--text-secondary)");
    };

    // 在连线中点添加箭头（使用 polygon），并根据起点到终点角度旋转
    const linkArrows = g
      .selectAll("polygon.link-arrow")
      .data(layout.links)
      .enter()
      .append("polygon")
      .attr("class", "link-arrow")
      .attr("points", "5,0 -10,6 -10,-6")
      .attr("fill", (d: LinkItem) =>
        isLinkToMalicious(d) ? "#E74C3C" : getEdgeColor(d.val),
      )
      .attr("data-scale", "1")
      .attr("cursor", "pointer")
      .attr("transform", (d: any) => {
        const s = layout.nodes.find((n) => n.id === d.from) as any;
        const t = layout.nodes.find((n) => n.id === d.to) as any;
        const mx = (s.x + t.x) / 2;
        const my = (s.y + t.y) / 2;
        const angle = (Math.atan2(t.y - s.y, t.x - s.x) * 180) / Math.PI;
        return `translate(${mx},${my}) rotate(${angle}) scale(1)`;
      })
      .on("click", function (event, d) {
        setSelectedLink(d);
        setShowDetail(true);
        event.stopPropagation();
      })
      .on("mouseover", function (event, d) {
        highlightLink(d);
      })
      .on("mouseout", function (event, d) {
        restoreLink(d);
      });

    // 在连线中点上方添加标签文本
    const linkLabels = g
      .selectAll("text.link-label")
      .data(layout.links)
      .enter()
      .append("text")
      .attr("class", "link-label")
      .text((d: LinkItem) => {
        // Check if this is an ETH transaction and needs conversion
        if (d.label && d.label.includes("ETH")) {
          // Extract the numeric value from the label (e.g., "374708330000000000 ETH" -> "374708330000000000")
          const ethMatch = d.label.match(/^([\d.]+)\s*ETH$/);
          if (ethMatch) {
            const weiValue = ethMatch[1];
            return `${formatEthValue(weiValue)} ETH`;
          }
        }
        return d.label || "";
      })
      .attr("x", (d: any) => {
        const s = layout.nodes.find((n) => n.id === d.from) as any;
        const t = layout.nodes.find((n) => n.id === d.to) as any;
        return (s.x + t.x) / 2;
      })
      .attr("y", (d: any) => {
        const s = layout.nodes.find((n) => n.id === d.from) as any;
        const t = layout.nodes.find((n) => n.id === d.to) as any;
        const dx = t.x - s.x;
        const dy = t.y - s.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const offset = 8;
        return (s.y + t.y) / 2 - (dx / distance) * offset;
      })
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("fill", "var(--text-secondary)")
      .attr("pointer-events", "all")
      .attr("cursor", "pointer")
      .on("click", function (event, d) {
        setSelectedLink(d);
        setShowDetail(true);
        event.stopPropagation();
      });

    // 让鼠标悬停同时高亮线与对应的中点箭头和标签
    linkLines
      .on("mouseover", function (event, d) {
        highlightLink(d);
      })
      .on("mouseout", function (event, d) {
        restoreLink(d);
      });

    // 同样为标签文字添加悬停效果以保持一致性
    linkLabels
      .on("mouseover", function (event, d) {
        highlightLink(d);
      })
      .on("mouseout", function (event, d) {
        restoreLink(d);
      });

    // 节点
    const nodeGroup = g
      .selectAll("g.node")
      .data(layout.nodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("transform", (d: any) => `translate(${d.x},${d.y})`);

    // 创建 tooltip 元素
    const tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "node-tooltip")
      .style("position", "absolute")
      .style("visibility", "hidden")
      .style("background", "rgba(0, 0, 0, 0.85)")
      .style("color", "#fff")
      .style("padding", "8px 12px")
      .style("border-radius", "6px")
      .style("font-size", "12px")
      .style("font-family", 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace')
      .style("pointer-events", "none")
      .style("z-index", "1000")
      .style("box-shadow", "0 4px 12px rgba(0, 0, 0, 0.3)")
      .style("max-width", "500px")
      .style("white-space", "nowrap");

    nodeGroup
      .append("circle")
      .attr("r", (d: any) => (d.layer === 0 ? 14 : 8))
      .attr("fill", (d: any) => {
        const nodeAmount = nodeAmountMap.get(d.id) || 0;
        const amountRatio = (nodeAmount - minNodeAmount) / nodeAmountRange;
        const isCenter = d.layer === 0;
        return colorForNode(d, amountRatio, isCenter);
      })
      .attr("stroke", (d: any) => {
        // 风险节点用红色描边
        if (d.malicious && d.malicious > 0) {
          return "#E74C3C"; // 柔和的红色
        }
        // 普通节点描边根据金额设置灰度
        const nodeAmount = nodeAmountMap.get(d.id) || 0;
        const amountRatio = (nodeAmount - minNodeAmount) / nodeAmountRange;
        const grayValue = Math.round(150 - amountRatio * 120);
        return `rgb(${grayValue}, ${grayValue}, ${grayValue})`;
      })
      .attr("stroke-width", (d: any) => (d.layer === 0 ? 1.5 : 1))
      .style("cursor", "pointer")
      .on("mouseover", function (event, d) {
        // 如果正在拖拽，不显示 tooltip
        if (isDragging) return;
        d3.select(this).style("cursor", "pointer");
        // 显示 tooltip
        tooltip
          .style("visibility", "visible")
          .html(
            `<div style="margin-bottom: 4px; font-weight: 600;">${d.title || d.label || d.id}</div>
             <div style="font-size: 11px; opacity: 0.8;">点击复制地址</div>`,
          );
      })
      .on("mousemove", function (event) {
        tooltip
          .style("top", event.pageY - 10 + "px")
          .style("left", event.pageX + 10 + "px");
      })
      .on("mouseout", function () {
        d3.select(this).style("cursor", "default");
        // 隐藏 tooltip
        tooltip.style("visibility", "hidden");
      })
      .on("click", function (event, d) {
        // 复制地址到剪贴板
        const addressToCopy = d.addr || d.title || d.label || d.id;
        if (addressToCopy) {
          navigator.clipboard
            .writeText(addressToCopy)
            .then(() => {
              // 临时改变 tooltip 内容
              tooltip.html(
                `<div style="color: #52c41a; font-weight: 600;">✓ 已复制</div>`,
              );
              setTimeout(() => {
                tooltip.style("visibility", "hidden");
              }, 1000);
            })
            .catch(() => {
              tooltip.html(
                `<div style="color: #ff4d4f;">复制失败</div>`,
              );
            });
        }
        event.stopPropagation();
      });

    nodeGroup
      .append("text")
      .text((d: any) => d.label)
      .attr("x", 0)
      .attr("y", (d: any) => (d.layer === 0 ? -20 : -12))
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("fill", "var(--text-secondary)")
      .style("pointer-events", "none"); // 文字不响应鼠标事件，拖拽时不影响

    nodeGroup
      .append("text")
      .text((d: any) => {
        const t = d.title || "";
        if (t.length <= 14) return t;
        return `${t.slice(0, 7)}...${t.slice(-7)}`;
      })
      .attr("x", 0)
      .attr("y", (d: any) => (d.layer === 0 ? 25 : 16))
      .attr("text-anchor", "middle")
      .attr("font-size", 12)
      .attr("fill", "var(--text-color)")
      .style("pointer-events", "none"); // 文字不响应鼠标事件，拖拽时不影响

    // 允许拖动：只改变被拖动节点的位置，并及时更新连线
    const linksSel = g.selectAll("line.link");

    // 拖拽状态标记
    let isDragging = false;

    const dragHandler = d3
      .drag<SVGGElement, NodeItem>()
      .on("start", function (event, d) {
        isDragging = true;
        if (event.sourceEvent) (event.sourceEvent as any).stopPropagation();
        d3.select(this).raise();
        // 选择当前节点组下的圆形元素并设置光标
        d3.select(this).select("circle").style("cursor", "grabbing");
        // 拖拽开始时隐藏 tooltip
        tooltip.style("visibility", "hidden");
      })
      .on("drag", function (event, d) {
        d.x = event.x;
        d.y = event.y;
        nodesWithPositionRef.current.set(d.id, { x: d.x || 0, y: d.y || 0 });
        d3.select(this).attr("transform", `translate(${d.x},${d.y})`);
        linksSel
          .attr("x1", function (l: any) {
            return (layout.nodes.find((n) => n.id === l.from) as any).x;
          })
          .attr("y1", function (l: any) {
            return (layout.nodes.find((n) => n.id === l.from) as any).y;
          })
          .attr("x2", function (l: any) {
            return (layout.nodes.find((n) => n.id === l.to) as any).x;
          })
          .attr("y2", function (l: any) {
            return (layout.nodes.find((n) => n.id === l.to) as any).y;
          });
        g.selectAll("polygon.link-arrow").attr("transform", function (l: any) {
          const s = layout.nodes.find((n) => n.id === l.from) as any;
          const t = layout.nodes.find((n) => n.id === l.to) as any;
          const mx = (s.x + t.x) / 2;
          const my = (s.y + t.y) / 2;
          const angle = (Math.atan2(t.y - s.y, t.x - s.x) * 180) / Math.PI;
          const curScale = +(d3.select(this).attr("data-scale") || 1);
          return `translate(${mx},${my}) rotate(${angle}) scale(${curScale})`;
        });

        g.selectAll("text.link-label")
          .attr("x", function (l: any) {
            const s = layout.nodes.find((n) => n.id === l.from) as any;
            const t = layout.nodes.find((n) => n.id === l.to) as any;
            return (s.x + t.x) / 2;
          })
          .attr("y", function (l: any) {
            const s = layout.nodes.find((n) => n.id === l.from) as any;
            const t = layout.nodes.find((n) => n.id === l.to) as any;
            const dx = t.x - s.x;
            const dy = t.y - s.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const offset = 8;
            return (s.y + t.y) / 2 - (dx / distance) * offset;
          });
      })
      .on("end", function (event, d) {
        isDragging = false;
        d3.select(this).select("circle").style("cursor", "grab");
        // 拖拽结束时，如果鼠标还在节点上，重新显示 tooltip
        const nodeElement = d3.select(this).select("circle").node();
        if (nodeElement) {
          const rect = (nodeElement as Element).getBoundingClientRect();
          const mouseX = event.sourceEvent?.clientX || 0;
          const mouseY = event.sourceEvent?.clientY || 0;
          // 检查鼠标是否还在节点范围内
          const isMouseOverNode =
            mouseX >= rect.left &&
            mouseX <= rect.right &&
            mouseY >= rect.top &&
            mouseY <= rect.bottom;
          if (isMouseOverNode) {
            tooltip
              .style("visibility", "visible")
              .style("top", mouseY - 10 + "px")
              .style("left", mouseX + 10 + "px")
              .html(
                `<div style="margin-bottom: 4px; font-weight: 600;">${d.title || d.label || d.id}</div>
                 <div style="font-size: 11px; opacity: 0.8;">点击复制地址</div>`,
              );
          }
        }
      });

    nodeGroup.call(dragHandler as any);

    // 缩放与拖拽（对 g 生效）
    const zoomBehavior = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 2])
      .on("zoom", (event) => {
        transformRef.current = {
          x: event.transform.x,
          y: event.transform.y,
          k: event.transform.k,
        };
        g.attr("transform", event.transform as any);
      });

    svg.call(zoomBehavior as any);

    // 恢复之前保存的缩放和拖动位置
    const savedTransform = transformRef.current;
    if (
      savedTransform.k !== 1 ||
      savedTransform.x !== 0 ||
      savedTransform.y !== 0
    ) {
      // 使用 zoom 的 transform 方法来设置变换，这样 zoom 的内部状态会同步，后续交互不会重置位置
      svg.call(
        (zoomBehavior as any).transform,
        d3.zoomIdentity
          .translate(savedTransform.x, savedTransform.y)
          .scale(savedTransform.k),
      );
    } else {
      const root = nodes
        ? (nodes.find((n) => n.layer === 0) as any)
        : undefined;
      if (root && root.x && root.x > 0) {
        const tx = width / 2 - root.x;
        const ty = height / 2 - root.y;

        // 如果不存在负层级节点，左移150px，并保存为当前的变换（以便后续重绘保留该位置）
        const hasNegativeLayer = filteredNodes.some((n) => (n.layer ?? 0) < 0);
        const xOffset = !hasNegativeLayer ? -150 : 0;

        const newX = tx + xOffset;
        const newY = ty;
        // 保存当前的平移与缩放（缩放为1）并通过 zoomBehavior 设置 transform，保证内部状态同步
        transformRef.current = { x: newX, y: newY, k: 1 };
        svg.call(
          (zoomBehavior as any).transform,
          d3.zoomIdentity.translate(newX, newY).scale(1),
        );
      }
    }

    svg.on("click", function (event) {
      if (event.target === this) {
        setShowDetail(false);
      }
    });

    // cleanup 函数：移除 tooltip
    return () => {
      d3.selectAll(".node-tooltip").remove();
    };
  }, [filteredNodes, filteredLinks, nodes, width, height]);

  const handleCloseDetail = () => {
    setShowDetail(false);
    setSelectedLink(null);
  };

  return (
    <>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ border: "1px solid #e6e6e6", backgroundColor: "white" }}
      >
        <g ref={gRef} />
      </svg>

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

export default TxGraph;
