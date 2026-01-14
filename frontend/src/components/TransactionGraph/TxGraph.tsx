import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import { NodeItem, LinkItem } from "./types";
import TxDetail from "./TxDetail";

interface TxGraphProps {
  nodes?: NodeItem[];
  links?: LinkItem[];
  width: number;
  height: number;
}

const TxGraph: React.FC<TxGraphProps> = ({ nodes, links, width, height }) => {
  const [selectedLink, setSelectedLink] = useState<LinkItem | null>(null);
  const [showDetail, setShowDetail] = useState(false);

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

    // 找到根（layer===0）
    const root = nodes.find((n) => n.layer === 0) || nodes[0];
    if (!root) return { nodes, links };
    root.x = centerX;
    root.y = centerY;

    // 确保所有节点都有 layer（默认 0）
    nodes.forEach((n) => {
      if (n.layer === undefined || n.layer === null) n.layer = 0;
    });

    // 按 layer 分组
    const layers: { [key: number]: NodeItem[] } = {};
    nodes.forEach((n) => {
      const l = n.layer as number;
      if (!layers[l]) layers[l] = [];
      layers[l].push(n);
    });

    // 对每一层设置 x,y：layer 0 在中间，正数层在右边，负数层在左边
    Object.keys(layers)
      .map((k) => Number(k))
      .forEach((layer) => {
        const group = layers[layer];
        const x = centerX + layer * layerSpacing;

        // 为该层计算垂直间距，使节点围绕 centerY 居中分布
        const count = group.length;
        const maxSpacing = 100;
        const minSpacing = 40;
        const spacing = Math.min(
          maxSpacing,
          Math.max(minSpacing, height / (count + 1))
        );
        const startY = centerY - (spacing * (count - 1)) / 2;

        group.forEach((node, i) => {
          // 中心根节点保持在 center
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

  // 当 props 中的 nodes/links 更新时执行绘图
  useEffect(() => {
    if (!nodes || !links) return;
    const layout = computePositions([...nodes], [...links], width, height);

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
      .attr("orient", "auto") // 随连线方向自动旋转
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#bbb");

    // 清空
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
        // 点击边时设置选中的边并显示详情弹窗
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
      // 三角形基于局部坐标 (0,0) 为箭头尖，其余两点在左侧
      .attr("points", "5,0 -10,6 -10,-6")
      .attr("fill", "#bbb")
      // 用 data-scale 保存当前缩放，默认 1（便于在拖拽时保持缩放）
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
        // 点击箭头时也触发边的点击事件
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
        // 将标签位置稍微偏移，使其显示在线段上方
        const dx = t.x - s.x;
        const dy = t.y - s.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        // 垂直偏移量，使文本在线段上方
        const offset = 8;
        return (s.y + t.y) / 2 - (dx / distance) * offset;
      })
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("fill", "#fff")
      .attr("pointer-events", "none") // 确保不会干扰鼠标的交互
      .on("click", function (event, d) {
        // 点击标签时也触发边的点击事件
        setSelectedLink(d);
        setShowDetail(true);
        event.stopPropagation();
      });

    // 让鼠标悬停同时高亮线与对应的中点箭头
    linkLines
      .on("mouseover", function (event, d) {
        d3.select(this).attr("stroke", "#eee").attr("stroke-width", 4);
        // 高亮并放大对应箭头
        g.selectAll("polygon.link-arrow")
          .filter((ad: any) => ad === d)
          .each(function (ad: any) {
            const poly = d3.select(this);
            poly.attr("fill", "#eee").attr("data-scale", "1.4");
            // 重新设置 transform（包含当前 scale）
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

        // 恢复连接线标签样式
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

    // 格式化 title：若长度 > 14，保留前7位和后7位，中间用 "..." 连接
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
        // 阻止缩放响应
        if (event.sourceEvent) (event.sourceEvent as any).stopPropagation();
        d3.select(this).raise();
      })
      .on("drag", function (event, d) {
        // 更新数据坐标
        d.x = event.x;
        d.y = event.y;
        // 更新节点位置
        d3.select(this).attr("transform", `translate(${d.x},${d.y})`);
        // 更新连线坐标（重新从 layout.nodes 中读取坐标）
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
        // 同步更新中点箭头位置与方向（保留每个箭头当前 data-scale）
        g.selectAll("polygon.link-arrow").attr("transform", function (l: any) {
          const s = layout.nodes.find((n) => n.id === l.from) as any;
          const t = layout.nodes.find((n) => n.id === l.to) as any;
          const mx = (s.x + t.x) / 2;
          const my = (s.y + t.y) / 2;
          const angle = (Math.atan2(t.y - s.y, t.x - s.x) * 180) / Math.PI;
          const curScale = +(d3.select(this).attr("data-scale") || 1);
          return `translate(${mx},${my}) rotate(${angle}) scale(${curScale})`;
        });

        // 同步更新连线标签位置
        g.selectAll("text.link-label")
          .attr("x", function (l: any) {
            const s = layout.nodes.find((n) => n.id === l.from) as any;
            const t = layout.nodes.find((n) => n.id === l.to) as any;
            return (s.x + t.x) / 2;
          })
          .attr("y", function (l: any) {
            const s = layout.nodes.find((n) => n.id === l.from) as any;
            const t = layout.nodes.find((n) => n.id === l.to) as any;
            // 将标签位置稍微偏移，使其显示在线段上方
            const dx = t.x - s.x;
            const dy = t.y - s.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            // 垂直偏移量，使文本在线段上方
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
          g.attr("transform", event.transform as any);
        })
    );

    // 初始将根节点居中（把根移到视图中心）
    const root = nodes.find((n) => n.layer === 0) as any;
    if (root && root.x <= 0) {
      const tx = width / 2 - root.x;
      const ty = height / 2 - root.y;
      g.attr("transform", `translate(${tx},${ty})`);
    }

    // 添加点击SVG背景关闭详情弹窗的功能
    svg.on("click", function (event) {
      // 只有当点击的是SVG背景而不是元素时才关闭
      if (event.target === this) {
        setShowDetail(false);
      }
    });
  }, [nodes, links, width, height]);

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
