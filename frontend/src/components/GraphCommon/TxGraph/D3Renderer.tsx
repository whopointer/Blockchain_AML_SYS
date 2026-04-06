import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { NodeItem, LinkItem } from "../types";
import { formatEthValue } from "@/utils/ethUtils";
import { message } from "antd";
import { transactionApi } from "@/services/transaction";
import { computeMissingPositions, mergeNodes, mergeLinks } from "./layoutUtils";

interface D3RendererProps {
  nodes: NodeItem[];
  links: LinkItem[];
  width: number;
  height: number;
  onLinkClick: (link: LinkItem) => void;
  onGraphUpdate?: (nodes: NodeItem[], links: LinkItem[]) => void;
  nodeAmountMap: Map<string, number>;
  minNodeAmount: number;
  maxNodeAmount: number;
  nodeAmountRange: number;
  colorForNode: (
    node: NodeItem,
    amountRatio: number,
    isCenter: boolean,
  ) => string;
  isLinkToMalicious: (link: LinkItem) => boolean;
  getEdgeColor: (val: number) => string;
  currencySymbol?: string;
  initialTransform?: { x: number; y: number; k: number };
  onTransformChange?: (transform: { x: number; y: number; k: number }) => void;
}

const D3Renderer: React.FC<D3RendererProps> = ({
  nodes,
  links,
  width,
  height,
  onLinkClick,
  onGraphUpdate,
  nodeAmountMap,
  minNodeAmount,
  maxNodeAmount,
  nodeAmountRange,
  colorForNode,
  isLinkToMalicious,
  getEdgeColor,
  currencySymbol,
  initialTransform,
  onTransformChange,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const gRef = useRef<SVGGElement | null>(null);
  const nodesWithPositionRef = useRef<Map<string, { x: number; y: number }>>(
    new Map(),
  );
  const transformRef = useRef<{
    x: number;
    y: number;
    k: number;
  }>(initialTransform || { x: 0, y: 0, k: 1 });

  // 标记是否跳过 onTransformChange 回调（避免程序设置变换时触发循环）
  const skipTransformChangeRef = useRef(false);

  // 存储上一次的 initialTransform 用于比较
  const prevInitialTransformRef = useRef(initialTransform);

  // 拖拽状态标记
  let isDragging = false;

  useEffect(() => {
    if (!nodes || !links) return;

    // 计算布局
    let layout: { nodes: NodeItem[]; links: LinkItem[] };

    // 逐节点判断：优先使用节点自带的 x,y，没有则从 ref 读取，都没有才需要计算
    const needsLayout = nodes.some((n) => {
      // 节点自带坐标
      if (n.x !== undefined && n.y !== undefined) {
        // 保存到 ref 供后续使用
        nodesWithPositionRef.current.set(n.id, { x: n.x, y: n.y });
        return false; // 有坐标，不需要计算
      }
      // ref 中有坐标
      if (nodesWithPositionRef.current.has(n.id)) {
        return false; // ref 有坐标，不需要计算
      }
      return true; // 没有坐标，需要计算布局
    });

    if (!needsLayout) {
      // 所有节点都有位置（要么自带，要么在 ref 中）
      const layoutNodes = nodes.map((n) => {
        // 优先使用节点自带的坐标
        if (n.x !== undefined && n.y !== undefined) {
          return n;
        }
        // 否则使用 ref 中保存的坐标
        const saved = nodesWithPositionRef.current.get(n.id);
        return saved ? { ...n, x: saved.x, y: saved.y } : n;
      });
      layout = { nodes: layoutNodes, links };
    } else {
      // 有节点没有位置，需要计算布局
      // 使用 computeMissingPositions 只计算缺失位置的节点，保留已有位置
      layout = computeMissingPositions([...nodes], [...links], width, height);
      // 保存计算后的位置到 ref
      layout.nodes.forEach((n) => {
        if (n.x !== undefined && n.y !== undefined) {
          nodesWithPositionRef.current.set(n.id, { x: n.x, y: n.y });
        }
      });
    }

    const svg = d3.select(svgRef.current as unknown as SVGSVGElement);
    const g = d3.select(gRef.current as unknown as SVGGElement);

    // 检查 initialTransform 是否变化，如果变化则更新 transformRef
    if (
      initialTransform &&
      (prevInitialTransformRef.current?.x !== initialTransform.x ||
        prevInitialTransformRef.current?.y !== initialTransform.y ||
        prevInitialTransformRef.current?.k !== initialTransform.k)
    ) {
      // 更新 transformRef
      transformRef.current = {
        x: initialTransform.x,
        y: initialTransform.y,
        k: initialTransform.k,
      };
      // 标记为跳过下一次 onTransformChange 回调
      skipTransformChangeRef.current = true;
    } else if (!initialTransform && prevInitialTransformRef.current) {
      // 如果 initialTransform 变为 undefined/null，重置为默认值
      transformRef.current = { x: 0, y: 0, k: 1 };
      skipTransformChangeRef.current = true;
    }

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

    // 链接（直线）
    const linkLines = g
      .selectAll("line.link")
      .data(links)
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
        onLinkClick(d);
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
            .attr(
              "fill",
              isLinkToMalicious(ad) ? "#E74C3C" : getEdgeColor(ad.val),
            )
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
      .data(links)
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
        onLinkClick(d);
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
      .data(links)
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
        onLinkClick(d);
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

    // 允许拖动：只改变被拖动节点的位置，并及时更新连线
    const linksSel = g.selectAll("line.link");

    // 创建节点ID到节点的映射，用于快速查找
    const nodeMap = new Map(layout.nodes.map((node) => [node.id, node]));

    // 存储待处理的动画帧ID
    let rafId: number | null = null;

    // 创建拖拽处理程序
    const dragHandler = d3
      .drag<SVGGElement, NodeItem>()
      .on("start", function (event, d) {
        isDragging = true;
        if (event.sourceEvent) (event.sourceEvent as any).stopPropagation();
        d3.select(this).raise();
        d3.select(this).select("circle").style("cursor", "grabbing");
        tooltip.style("visibility", "hidden");
      })
      .on("drag", function (event, d) {
        if (rafId) return;

        rafId = requestAnimationFrame(() => {
          d.x = event.x;
          d.y = event.y;
          nodesWithPositionRef.current.set(d.id, { x: d.x || 0, y: d.y || 0 });
          d3.select(this).attr("transform", `translate(${d.x},${d.y})`);

          const relatedLinks = links.filter(
            (link) => link.from === d.id || link.to === d.id,
          );

          linksSel
            .filter((l: any) => l.from === d.id || l.to === d.id)
            .attr("x1", function (l: any) {
              const node = nodeMap.get(l.from);
              return node && node.x !== undefined ? node.x : 0;
            })
            .attr("y1", function (l: any) {
              const node = nodeMap.get(l.from);
              return node && node.y !== undefined ? node.y : 0;
            })
            .attr("x2", function (l: any) {
              const node = nodeMap.get(l.to);
              return node && node.x !== undefined ? node.x : 0;
            })
            .attr("y2", function (l: any) {
              const node = nodeMap.get(l.to);
              return node && node.y !== undefined ? node.y : 0;
            });

          g.selectAll("polygon.link-arrow")
            .filter((l: any) => l.from === d.id || l.to === d.id)
            .attr("transform", function (l: any) {
              const s = nodeMap.get(l.from);
              const t = nodeMap.get(l.to);
              if (
                !s ||
                !t ||
                s.x === undefined ||
                s.y === undefined ||
                t.x === undefined ||
                t.y === undefined
              )
                return "";
              const mx = (s.x + t.x) / 2;
              const my = (s.y + t.y) / 2;
              const angle = (Math.atan2(t.y - s.y, t.x - s.x) * 180) / Math.PI;
              const curScale = +(d3.select(this).attr("data-scale") || 1);
              return `translate(${mx},${my}) rotate(${angle}) scale(${curScale})`;
            });

          g.selectAll("text.link-label")
            .filter((l: any) => l.from === d.id || l.to === d.id)
            .attr("x", function (l: any) {
              const s = nodeMap.get(l.from);
              const t = nodeMap.get(l.to);
              if (!s || !t || s.x === undefined || t.x === undefined) return 0;
              return (s.x + t.x) / 2;
            })
            .attr("y", function (l: any) {
              const s = nodeMap.get(l.from);
              const t = nodeMap.get(l.to);
              if (
                !s ||
                !t ||
                s.x === undefined ||
                s.y === undefined ||
                t.x === undefined ||
                t.y === undefined
              )
                return 0;
              const dx = t.x - s.x;
              const dy = t.y - s.y;
              const distance = Math.sqrt(dx * dx + dy * dy);
              const offset = 8;
              return (s.y + t.y) / 2 - (dx / distance) * offset;
            });

          rafId = null;
        });
      })
      .on("end", function (event, d) {
        if (rafId) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
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
                 <div style="font-size: 11px; opacity: 0.8;">点击复制地址, 双击展开节点</div>`,
              );
          }
        }
      });

    // 节点
    const nodeGroup = g
      .selectAll("g.node")
      .data(layout.nodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("transform", (d: any) => `translate(${d.x},${d.y})`)
      .style("will-change", "transform");

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
      .style(
        "font-family",
        'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
      )
      .style("pointer-events", "none")
      .style("z-index", "1000")
      .style("box-shadow", "0 4px 12px rgba(0, 0, 0, 0.3)")
      .style("max-width", "500px")
      .style("white-space", "nowrap");

    // 找到根节点（原始中心节点）
    const rootNode = nodes.find((n) => n.layer === 0) ||
      nodes[0] || { id: "", layer: 0 };

    // 绘制节点圆圈
    nodeGroup
      .append("circle")
      .attr("r", (d: any) => (d.id === rootNode.id ? 14 : 8))
      .attr("fill", (d: any) => {
        const nodeAmount = nodeAmountMap.get(d.id) || 0;
        const amountRatio = (nodeAmount - minNodeAmount) / nodeAmountRange;
        const isCenter = d.id === rootNode.id;
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
      .attr("stroke-width", (d: any) => (d.id === rootNode.id ? 1.5 : 1))
      .style("cursor", "pointer")
      .on("mouseover", function (event, d) {
        if (isDragging) return;
        d3.select(this).style("cursor", "pointer");
        tooltip.style("visibility", "visible").html(
          `<div style="margin-bottom: 4px; font-weight: 600;">${d.title || d.label || d.id}</div>
             <div style="font-size: 11px; opacity: 0.8;">点击复制地址, 双击展开节点</div>`,
        );
      })
      .on("mousemove", function (event) {
        if (isDragging) return;
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
              tooltip.html(`<div style="color: #ff4d4f;">复制失败</div>`);
            });
        }
        event.stopPropagation();
      });

    // 添加节点标签
    nodeGroup
      .append("text")
      .text((d: any) => d.label)
      .attr("x", 0)
      .attr("y", (d: any) => (d.id === rootNode.id ? -20 : -12))
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
      .attr("y", (d: any) => (d.id === rootNode.id ? 25 : 16))
      .attr("text-anchor", "middle")
      .attr("font-size", 12)
      .attr("fill", "var(--text-color)")
      .style("pointer-events", "none"); // 文字不响应鼠标事件，拖拽时不影响

    // 双击节点事件处理
    nodeGroup.on("dblclick", async function (event, d) {
      // 阻止事件冒泡
      event.stopPropagation();

      const loadingMsg = message.loading(`正在查询 ${d.addr} 的交易数据...`);

      try {
        // 调用 API 查询该节点的交易数据（使用 1 跳查询）
        const response = await transactionApi.getNhopGraph(d.addr, 1);

        if (response.success) {
          // 更新当前节点的 expanded 属性为 true
          const updatedNodes = (nodes || []).map((node) =>
            node.id === d.id ? { ...node, expanded: true } : node,
          );

          // 处理新的节点和边数据
          const newNodes = response.data.node_list || [];
          const newEdges = response.data.edge_list || [];

          // 转换新节点数据
          const convertedNewNodes = newNodes.map((node: any, index: number) => {
            // 查找现有节点中是否已有该地址的节点
            const existingNode = updatedNodes.find((n) => n.addr === node.addr);

            return {
              id: existingNode?.id || node.id || node.addr,
              label: node.label || node.addr,
              title: node.title || node.addr,
              addr: node.addr,
              layer: node.addr === d.addr ? d.layer : d.layer + node.layer,
              value: node.value || 0,
              pid: node.pid || undefined,
              color: node.color || undefined,
              shape: node.shape || undefined,
              image: node.image || undefined,
              track: node.track || "one",
              expanded: node.expanded || false,
              malicious: node.malicious || 0,
              exg: node.exg || undefined,
              x: existingNode?.x || node.x || undefined,
              y: existingNode?.y || node.y || undefined,
            };
          });

          // 转换新边数据
          const convertedNewLinks = newEdges.map((edge: any) => {
            // 查找对应的节点ID，优先使用现有节点
            let fromNodeId = edge.from;
            let toNodeId = edge.to;

            // 先在现有节点中查找
            const existingFromNode = updatedNodes.find(
              (n) => n.addr === edge.from,
            );
            const existingToNode = updatedNodes.find((n) => n.addr === edge.to);

            if (existingFromNode) {
              fromNodeId = existingFromNode.id;
            } else {
              // 在新节点中查找
              const newFromNode = convertedNewNodes.find(
                (n) => n.addr === edge.from,
              );
              if (newFromNode) {
                fromNodeId = newFromNode.id;
              }
            }

            if (existingToNode) {
              toNodeId = existingToNode.id;
            } else {
              // 在新节点中查找
              const newToNode = convertedNewNodes.find(
                (n) => n.addr === edge.to,
              );
              if (newToNode) {
                toNodeId = newToNode.id;
              }
            }

            return {
              from: fromNodeId,
              to: toNodeId,
              label:
                edge.label || `${edge.val || edge.value} ${currencySymbol}`,
              val: edge.val || edge.value || 0,
              tx_time:
                edge.tx_time || edge.timestamp || new Date().toISOString(),
              tx_hash_list: edge.tx_hash_list || edge.tx_hashes || [],
            };
          });

          // 合并节点数据（去重）
          const mergedNodes = mergeNodes(updatedNodes, convertedNewNodes);
          // 合并边数据（去重）
          const mergedLinks = mergeLinks(links || [], convertedNewLinks);

          // 通知父组件更新数据
          if (onGraphUpdate) {
            onGraphUpdate(mergedNodes, mergedLinks);
          }

          message.success(`成功查询到 ${d.addr} 的交易数据`);
          loadingMsg();
        } else {
          message.error(`查询交易数据失败: ${response.msg}`);
          loadingMsg();
        }
      } catch (error) {
        console.error("查询交易数据失败:", error);
        message.error("查询交易数据失败，请稍后重试");
        loadingMsg();
      }
    });

    nodeGroup.call(dragHandler as any);

    // 创建缩放行为
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
        // 注意：这里不再调用 onTransformChange，只在交互结束时调用
      })
      .on("end", (event) => {
        // 交互结束时（拖拽或缩放结束）保存变换状态
        if (
          onTransformChange &&
          !skipTransformChangeRef.current &&
          event.sourceEvent
        ) {
          // event.sourceEvent 存在表示是用户交互，而不是程序设置
          onTransformChange({
            x: event.transform.x,
            y: event.transform.y,
            k: event.transform.k,
          });
        }
      });

    // 应用缩放行为到SVG
    svg.call(zoomBehavior as any);

    // 应用初始变换（总是应用，确保视图状态正确重置）
    const savedTransform = transformRef.current;
    // 使用 zoom 的 transform 方法来设置变换，这样 zoom 的内部状态会同步
    // 设置跳过标志，避免程序设置变换时触发 onTransformChange 回调
    skipTransformChangeRef.current = true;
    try {
      svg.call(
        (zoomBehavior as any).transform,
        d3.zoomIdentity
          .translate(savedTransform.x, savedTransform.y)
          .scale(savedTransform.k),
      );
    } finally {
      // 确保在下一事件循环中重置跳过标志，让用户交互能正常触发回调
      setTimeout(() => {
        skipTransformChangeRef.current = false;
      }, 0);
    }

    svg.on("click", function (event) {
      if (event.target === this) {
        // 点击空白处，不做任何操作
      }
    });

    // 更新 prevInitialTransformRef
    prevInitialTransformRef.current = initialTransform;

    // 清理 tooltip
    return () => {
      d3.selectAll(".node-tooltip").remove();
    };
  }, [
    nodes,
    links,
    width,
    height,
    onLinkClick,
    onGraphUpdate,
    nodeAmountMap,
    minNodeAmount,
    maxNodeAmount,
    nodeAmountRange,
    colorForNode,
    isLinkToMalicious,
    getEdgeColor,
    currencySymbol,
    initialTransform,
    onTransformChange,
  ]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{
        border: "1px solid #e6e6e6",
        backgroundColor: "white",
        borderRadius: 8,
      }}
    >
      <g ref={gRef} />
    </svg>
  );
};

export default D3Renderer;
