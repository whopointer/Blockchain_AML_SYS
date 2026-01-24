import React, { useRef, useEffect, useState, useMemo } from "react";
import * as d3 from "d3";
import { NodeItem, LinkItem } from "./types";
import TxDetail from "./TxDetail";

interface TxGraphProps {
  nodes?: NodeItem[];
  links?: LinkItem[];
  width: number;
  height: number;
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
    [filter, internalFilter]
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

  const colorForNode = (v: number, type: string) => {
    if (v > 0) return "#FF6B6B";
    if (type) return "#3A86FF";
    return "#888888";
  };

  // 计算简单的分层布局：中心节点在中间，右侧为 layer>=0，左侧为 layer<0
  function computePositions(
    nodes: NodeItem[],
    links: LinkItem[],
    width: number,
    height: number
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
          Math.max(minSpacing, height / (count + 1))
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
    new Map()
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
        const txTimestamp = new Date(l.tx_time).getTime();
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
      (n) => n === root || (nodeSet.has(n.id) && matchAddr(n))
    );

    const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredLinks = byTx.filter(
      (l) => filteredNodeIds.has(l.from) && filteredNodeIds.has(l.to)
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

  // 当 props 中的 nodes/links 更新时执行绘图
  useEffect(() => {
    if (!filteredNodes || !filteredLinks) return;

    let layout: { nodes: NodeItem[]; links: LinkItem[] };
    const hasExistingPositions = filteredNodes.every((n) =>
      nodesWithPositionRef.current.has(n.id)
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
        height
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
      .attr("fill", "#bbb");

    g.selectAll("*").remove();

    // 链接（直线）
    const linkLines = g
      .selectAll("line.link")
      .data(layout.links)
      .enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke-width", 2)
      .attr(
        "x1",
        (d: LinkItem) => (layout.nodes.find((n) => n.id === d.from) as any).x
      )
      .attr(
        "y1",
        (d: LinkItem) => (layout.nodes.find((n) => n.id === d.from) as any).y
      )
      .attr(
        "x2",
        (d: LinkItem) => (layout.nodes.find((n) => n.id === d.to) as any).x
      )
      .attr(
        "y2",
        (d: LinkItem) => (layout.nodes.find((n) => n.id === d.to) as any).y
      )
      .attr("stroke", "#bbb")
      .attr("stroke-width", 2)
      .on("click", function (event, d) {
        setSelectedLink(d);
        setShowDetail(true);
        event.stopPropagation();
      });

    // 在连线中点添加箭头（使用 polygon），并根据起点到终点角度旋转
    g.selectAll("polygon.link-arrow")
      .data(layout.links)
      .enter()
      .append("polygon")
      .attr("class", "link-arrow")
      .attr("points", "5,0 -10,6 -10,-6")
      .attr("fill", "#bbb")
      .attr("data-scale", "1")
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
      });

    // 在连线中点上方添加标签文本
    g.selectAll("text.link-label")
      .data(layout.links)
      .enter()
      .append("text")
      .attr("class", "link-label")
      .text((d: LinkItem) => d.label || "")
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
      .attr("fill", "#fff")
      .attr("pointer-events", "none")
      .on("click", function (event, d) {
        setSelectedLink(d);
        setShowDetail(true);
        event.stopPropagation();
      });

    // 让鼠标悬停同时高亮线与对应的中点箭头
    linkLines
      .on("mouseover", function (event, d) {
        d3.select(this).attr("stroke", "#eee").attr("stroke-width", 4);
        g.selectAll("polygon.link-arrow")
          .filter((ad: any) => ad === d)
          .each(function (ad: any) {
            const poly = d3.select(this);
            poly.attr("fill", "#eee").attr("data-scale", "1.4");
            const sNode = layout.nodes.find((n) => n.id === ad.from) as any;
            const tNode = layout.nodes.find((n) => n.id === ad.to) as any;
            const mx = (sNode.x + tNode.x) / 2;
            const my = (sNode.y + tNode.y) / 2;
            const angle =
              (Math.atan2(tNode.y - sNode.y, tNode.x - sNode.x) * 180) /
              Math.PI;
            poly.attr(
              "transform",
              `translate(${mx},${my}) rotate(${angle}) scale(1.4)`
            );
          });

        // 高亮连接线标签
        g.selectAll("text.link-label")
          .filter((ad: any) => ad === d)
          .attr("font-weight", "bold")
          .attr("font-size", 12)
          .attr("fill", "#fff");
      })
      .on("mouseout", function (event, d) {
        d3.select(this).attr("stroke", "#bbb").attr("stroke-width", 2);
        // 恢复箭头大小与颜色
        g.selectAll("polygon.link-arrow")
          .filter((ad: any) => ad === d)
          .each(function (ad: any) {
            const poly = d3.select(this);
            poly.attr("fill", "#bbb").attr("data-scale", "1");
            const sNode = layout.nodes.find((n) => n.id === ad.from) as any;
            const tNode = layout.nodes.find((n) => n.id === ad.to) as any;
            const mx = (sNode.x + tNode.x) / 2;
            const my = (sNode.y + tNode.y) / 2;
            const angle =
              (Math.atan2(tNode.y - sNode.y, tNode.x - sNode.x) * 180) /
              Math.PI;
            poly.attr(
              "transform",
              `translate(${mx},${my}) rotate(${angle}) scale(1)`
            );
          });

        g.selectAll("text.link-label")
          .filter((ad: any) => ad === d)
          .attr("font-weight", "normal")
          .attr("font-size", 10)
          .attr("fill", "#fff");
      });

    // 节点
    const nodeGroup = g
      .selectAll("g.node")
      .data(layout.nodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("transform", (d: any) => `translate(${d.x},${d.y})`);

    nodeGroup
      .append("circle")
      .attr("r", (d: any) => (d.layer === 0 ? 14 : 8))
      .attr("fill", (d: any) => colorForNode(d.malicious || 0, d.image || ""))
      .attr("stroke", "#222")
      .attr("stroke-width", (d: any) => (d.layer === 0 ? 1.5 : 1));

    nodeGroup
      .append("text")
      .text((d: any) => d.label)
      .attr("x", 0)
      .attr("y", (d: any) => (d.layer === 0 ? -20 : -12))
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("fill", "#eee");

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
      .attr("fill", "#eee");

    // 允许拖动：只改变被拖动节点的位置，并及时更新连线
    const linksSel = g.selectAll("line.link");

    const dragHandler = d3
      .drag<SVGGElement, NodeItem>()
      .on("start", function (event, d) {
        if (event.sourceEvent) (event.sourceEvent as any).stopPropagation();
        d3.select(this).raise();
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
      });

    nodeGroup.call(dragHandler as any);

    // 缩放与拖拽（对 g 生效）
    svg.call(
      d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.5, 2])
        .on("zoom", (event) => {
          transformRef.current = {
            x: event.transform.x,
            y: event.transform.y,
            k: event.transform.k,
          };
          g.attr("transform", event.transform as any);
        })
    );

    // 恢复之前保存的缩放和拖动位置
    const savedTransform = transformRef.current;
    if (
      savedTransform.k !== 1 ||
      savedTransform.x !== 0 ||
      savedTransform.y !== 0
    ) {
      g.attr(
        "transform",
        `translate(${savedTransform.x},${savedTransform.y}) scale(${savedTransform.k})`
      );
    } else {
      const root = nodes
        ? (nodes.find((n) => n.layer === 0) as any)
        : undefined;
      if (root && root.x && root.x > 0) {
        const tx = width / 2 - root.x;
        const ty = height / 2 - root.y;
        g.attr("transform", `translate(${tx},${ty})`);
      }
    }

    svg.on("click", function (event) {
      if (event.target === this) {
        setShowDetail(false);
      }
    });
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
        style={{ border: "1px solid #e6e6e6" }}
      >
        <g ref={gRef} />
      </svg>

      {/* 边详情弹窗 */}
      <TxDetail
        show={showDetail}
        onHide={handleCloseDetail}
        link={selectedLink}
      />
    </>
  );
};

export default TxGraph;
