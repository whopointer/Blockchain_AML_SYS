package com.seecoder.DataProcessing.util;

import java.util.*;

/**
 * 图节点层级计算工具类
 */
public class GraphLayerCalculator {
    
    /**
     * 计算节点层级的方法，处理多节点多路径（包括循环路径）的情况
     * 保证起始节点层级为0，目标节点层级最大，其余节点合理分布
     */
    public static Map<String, Integer> calculateNodeLayers(List<List<Map<String, Object>>> allPaths, String fromAddress, String toAddress) {
        Map<String, Integer> nodeLayers = new HashMap<>();

        // 如果没有路径，只设置起始节点为第0层
        if (allPaths == null || allPaths.isEmpty()) {
            nodeLayers.put(fromAddress, 0);
            return nodeLayers;
        }

        // 1. 初始化图结构，建立邻接关系，并收集所有节点
        Map<String, Set<String>> forwardEdges = new HashMap<>(); // 前向边（正常流向）
        Set<String> allNodes = new HashSet<>(); // 收集所有路径中的所有节点

        for (List<Map<String, Object>> path : allPaths) {
            for (int i = 0; i < path.size(); i++) {
                String nodeAddr = getNodeAddress(path.get(i));
                if (nodeAddr != null) {
                    allNodes.add(nodeAddr);
                }
            }

            for (int i = 0; i < path.size() - 1; i++) {
                String fromNode = getNodeAddress(path.get(i));
                String toNode = getNodeAddress(path.get(i + 1));

                if (fromNode != null && toNode != null && !fromNode.equals(toNode)) {
                    forwardEdges.computeIfAbsent(fromNode, k -> new HashSet<>()).add(toNode);
                }
            }
        }

        // 2. 使用广度优先搜索(BFS)从起始节点计算最短跳数（层级），确保起始节点为0
        Map<String, Integer> distances = new HashMap<>();
        Queue<String> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

        queue.offer(fromAddress);
        visited.add(fromAddress);
        distances.put(fromAddress, 0);

        while (!queue.isEmpty()) {
            String current = queue.poll();

            // 获取当前节点的所有前向邻居
            Set<String> neighbors = forwardEdges.get(current);
            if (neighbors != null) {
                for (String neighbor : neighbors) {
                    if (!visited.contains(neighbor)) {
                        visited.add(neighbor);
                        distances.put(neighbor, distances.get(current) + 1);
                        queue.offer(neighbor);
                    }
                }
            }
        }

        // 3. 处理未访问到的节点（可能在反向路径或其他组件中）
        for (String node : allNodes) {
            if (!distances.containsKey(node)) {
                // 对于未访问的节点，尝试从目标节点反向BFS找到其层级
                // 或者简单地将其层级设为比目标节点更深一层
                distances.put(node, distances.getOrDefault(toAddress, allPaths.size()) + 1);
            }
        }

        return distances;
    }

    /**
     * 根据方向计算节点层级的方法，区分收入侧（负值）和支出侧（正值）
     * 起始节点为0，收入侧为负值，支出侧为正值
     */
    public static Map<String, Integer> calculateNodeLayersWithDirection(List<List<Map<String, Object>>> allPaths, List<String> directions, String startAddress) {
        Map<String, Integer> nodeLayers = new HashMap<>();
        nodeLayers.put(startAddress, 0); // 起始节点层级为0

        // 如果没有路径，只设置起始节点为第0层
        if (allPaths == null || allPaths.isEmpty()) {
            return nodeLayers;
        }

        // 遍历所有路径，根据方向确定每个节点相对于起始节点的位置
        for (int pathIdx = 0; pathIdx < allPaths.size(); pathIdx++) {
            List<Map<String, Object>> path = allPaths.get(pathIdx);
            String direction = directions.get(pathIdx);

            // 找到起始节点在路径中的位置
            int startIndex = -1;
            for (int i = 0; i < path.size(); i++) {
                String nodeAddr = getNodeAddress(path.get(i));
                if (nodeAddr != null && nodeAddr.equals(startAddress)) {
                    startIndex = i;
                    break;
                }
            }

            if (startIndex != -1) {
                if ("income".equals(direction)) {
                    // 收入侧路径，节点应该在起始节点的左侧（负值）
                    for (int i = 0; i < path.size(); i++) {
                        if (i != startIndex) {
                            String nodeAddr = getNodeAddress(path.get(i));
                            if (nodeAddr != null && !nodeAddr.equals(startAddress)) {
                                int distance = i - startIndex; // 在收入侧，所有节点都是负值
                                if (distance > 0) distance = -distance; // 确保收入侧为负值
                                if (!nodeLayers.containsKey(nodeAddr) || Math.abs(distance) < Math.abs(nodeLayers.get(nodeAddr))) {
                                    nodeLayers.put(nodeAddr, distance);
                                }
                            }
                        }
                    }
                } else if ("outcome".equals(direction)) {
                    // 支出侧路径，节点应该在起始节点的右侧（正值）
                    for (int i = 0; i < path.size(); i++) {
                        if (i != startIndex) {
                            String nodeAddr = getNodeAddress(path.get(i));
                            if (nodeAddr != null && !nodeAddr.equals(startAddress)) {
                                int distance = i - startIndex; // 在支出侧，所有节点都是正值
                                if (distance < 0) distance = -distance; // 确保支出侧为正值
                                if (!nodeLayers.containsKey(nodeAddr) || Math.abs(distance) < Math.abs(nodeLayers.get(nodeAddr))) {
                                    nodeLayers.put(nodeAddr, distance);
                                }
                            }
                        }
                    }
                }
            } else {
                // 如果起始节点不在路径中，根据方向标记
                for (int i = 0; i < path.size(); i++) {
                    String nodeAddr = getNodeAddress(path.get(i));
                    if (nodeAddr != null && !nodeAddr.equals(startAddress)) {
                        if ("income".equals(direction)) {
                            // 收入侧，设置为负值
                            if (!nodeLayers.containsKey(nodeAddr) || Math.abs(-1) < Math.abs(nodeLayers.get(nodeAddr))) {
                                nodeLayers.put(nodeAddr, -1);
                            }
                        } else if ("outcome".equals(direction)) {
                            // 支出侧，设置为正值
                            if (!nodeLayers.containsKey(nodeAddr) || Math.abs(1) < Math.abs(nodeLayers.get(nodeAddr))) {
                                nodeLayers.put(nodeAddr, 1);
                            }
                        }
                    }
                }
            }
        }

        return nodeLayers;
    }

    /**
     * 辅助方法：从节点数据中提取地址
     */
    public static String getNodeAddress(Map<String, Object> nodeData) {
        if (nodeData == null) {
            return null;
        }
        Object addrObj = nodeData.get("address");
        return addrObj != null ? addrObj.toString() : null;
    }
}