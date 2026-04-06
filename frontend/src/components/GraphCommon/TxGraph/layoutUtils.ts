import { NodeItem, LinkItem } from "../types";

/**
 * 计算简单的分层布局：中心节点在中间，右侧为 layer>=0，左侧为 layer<0
 */
export function computePositions(
  nodes: NodeItem[],
  links: LinkItem[],
  width: number,
  height: number,
): { nodes: NodeItem[]; links: LinkItem[] } {
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

/**
 * 只计算缺失位置的节点，保留已有位置的节点
 * 用于双击展开新节点时，只计算新节点的位置而不影响旧节点
 */
export function computeMissingPositions(
  nodes: NodeItem[],
  links: LinkItem[],
  width: number,
  height: number,
): { nodes: NodeItem[]; links: LinkItem[] } {
  // 分离有位置和无位置的节点
  const nodesWithPosition = nodes.filter(
    (n) => n.x !== undefined && n.y !== undefined,
  );
  const nodesWithoutPosition = nodes.filter(
    (n) => n.x === undefined || n.y === undefined,
  );

  // 如果所有节点都有位置，直接返回
  if (nodesWithoutPosition.length === 0) {
    return { nodes, links };
  }

  // 如果所有节点都没有位置，使用完整布局计算
  if (nodesWithPosition.length === 0) {
    return computePositions(nodes, links, width, height);
  }

  // 有部分节点有位置，部分没有
  // 以有位置的节点为基础，为没有位置的节点计算位置
  const centerX = width / 2;
  const centerY = height / 2;
  const layerSpacing = 200;

  // 为没有位置的节点设置 layer（如果没有）
  nodesWithoutPosition.forEach((n) => {
    if (n.layer === undefined || n.layer === null) n.layer = 1;
  });

  // 按 layer 分组
  const layers: { [key: number]: NodeItem[] } = {};
  nodesWithoutPosition.forEach((n) => {
    const l = n.layer as number;
    if (!layers[l]) layers[l] = [];
    layers[l].push(n);
  });

  // 计算每个新节点的层中应该有多少个已有位置的节点
  const existingLayers: { [key: number]: number } = {};
  nodesWithPosition.forEach((n) => {
    const l = n.layer as number;
    existingLayers[l] = (existingLayers[l] || 0) + 1;
  });

  Object.keys(layers)
    .map((k) => Number(k))
    .forEach((layer) => {
      const group = layers[layer];
      const x = centerX + layer * layerSpacing;

      // 已有位置节点的数量作为偏移
      const existingCount = existingLayers[layer] || 0;
      const count = group.length;
      const maxSpacing = 100;
      const minSpacing = 40;
      const spacing = Math.min(
        maxSpacing,
        Math.max(minSpacing, height / (count + existingCount + 1)),
      );
      const startY = centerY - (spacing * (count - 1)) / 2;

      group.forEach((node, i) => {
        node.x = x;
        node.y = startY + i * spacing;
      });
    });

  return { nodes, links };
}

/**
 * 合并节点数据，去重
 */
export const mergeNodes = (
  existingNodes: NodeItem[],
  newNodes: NodeItem[],
): NodeItem[] => {
  const nodeMap = new Map<string, NodeItem>();

  // 先添加现有节点
  existingNodes.forEach((node) => {
    nodeMap.set(node.id, node);
  });

  // 添加新节点，如果已存在则更新
  newNodes.forEach((node) => {
    const existingNode = nodeMap.get(node.id);
    if (existingNode) {
      // 保留现有节点的属性，只更新必要的字段
      nodeMap.set(node.id, {
        ...existingNode,
        ...node,
        expanded: existingNode.expanded || node.expanded,
        x: existingNode.x, // 保留现有位置
        y: existingNode.y, // 保留现有位置
      });
    } else {
      nodeMap.set(node.id, node);
    }
  });

  return Array.from(nodeMap.values());
};

/**
 * 合并边数据，去重
 */
export const mergeLinks = (
  existingLinks: LinkItem[],
  newLinks: LinkItem[],
): LinkItem[] => {
  const linkMap = new Map<string, LinkItem>();

  // 生成边的唯一键
  const getLinkKey = (link: LinkItem) => `${link.from}-${link.to}`;

  // 先添加现有边
  existingLinks.forEach((link) => {
    linkMap.set(getLinkKey(link), link);
  });

  // 添加新边，如果已存在则更新
  newLinks.forEach((link) => {
    const key = getLinkKey(link);
    if (!linkMap.has(key)) {
      linkMap.set(key, link);
    }
  });

  return Array.from(linkMap.values());
};
