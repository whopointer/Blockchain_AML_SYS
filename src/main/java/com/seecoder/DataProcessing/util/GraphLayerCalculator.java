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

        // 1. 收集所有节点
        Set<String> allNodes = new HashSet<>();
        for (List<Map<String, Object>> path : allPaths) {
            for (Map<String, Object> node : path) {
                String nodeAddr = getNodeAddress(node);
                if (nodeAddr != null) {
                    allNodes.add(nodeAddr);
                }
            }
        }

        // 2. 计算每个节点的最大可能层级
        Map<String, Integer> maxDistances = new HashMap<>();
        
        // 初始化所有节点的层级为-1
        for (String node : allNodes) {
            maxDistances.put(node, -1);
        }
        
        // 设置起始节点层级为0
        maxDistances.put(fromAddress, 0);
        
        // 遍历所有路径，计算每个节点的最大层级
        for (List<Map<String, Object>> path : allPaths) {
            // 构建路径中节点的地址列表
            List<String> pathNodes = new ArrayList<>();
            for (Map<String, Object> node : path) {
                String nodeAddr = getNodeAddress(node);
                if (nodeAddr != null) {
                    pathNodes.add(nodeAddr);
                }
            }
            
            // 找到起始节点在路径中的位置
            int startIndex = -1;
            for (int i = 0; i < pathNodes.size(); i++) {
                if (pathNodes.get(i).equals(fromAddress)) {
                    startIndex = i;
                    break;
                }
            }
            
            if (startIndex != -1) {
                // 从起始节点开始，计算路径中每个节点的层级
                for (int i = 0; i < pathNodes.size(); i++) {
                    String currentNode = pathNodes.get(i);
                    int distance = Math.abs(i - startIndex);
                    
                    // 如果当前计算的层级大于已有的层级，则更新
                    if (distance > maxDistances.get(currentNode)) {
                        maxDistances.put(currentNode, distance);
                    }
                }
            }
        }
        
        // 3. 确保目标节点的层级是最大的
        int maxLayer = 0;
        for (String node : allNodes) {
            if (maxDistances.get(node) > maxLayer) {
                maxLayer = maxDistances.get(node);
            }
        }
        
        // 如果目标节点的层级不是最大的，将其设置为最大层级
        if (maxDistances.get(toAddress) < maxLayer) {
            maxDistances.put(toAddress, maxLayer);
        }
        
        // 4. 处理未访问到的节点（层级仍为-1的节点）
        for (String node : allNodes) {
            if (maxDistances.get(node) == -1) {
                // 对于未访问的节点，设置为目标节点的层级+1
                maxDistances.put(node, maxDistances.get(toAddress) + 1);
            }
        }

        // 5. 重新映射层级到从1开始的连续整数
        Set<Integer> uniqueLayers = new HashSet<>(maxDistances.values());
        List<Integer> sortedLayers = new ArrayList<>(uniqueLayers);
        Collections.sort(sortedLayers);
        Map<Integer, Integer> layerMap = new HashMap<>();
        for (int i = 0; i < sortedLayers.size(); i++) {
            layerMap.put(sortedLayers.get(i), i + 1);
        }
        Map<String, Integer> remappedLayers = new HashMap<>();
        for (Map.Entry<String, Integer> entry : maxDistances.entrySet()) {
            remappedLayers.put(entry.getKey(), layerMap.get(entry.getValue()));
        }

        return remappedLayers;
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

        // 重新映射层级到从1开始的连续整数
        Set<Integer> uniqueLayers = new HashSet<>(nodeLayers.values());
        List<Integer> sortedLayers = new ArrayList<>(uniqueLayers);
        Collections.sort(sortedLayers);
        Map<Integer, Integer> layerMap = new HashMap<>();
        for (int i = 0; i < sortedLayers.size(); i++) {
            layerMap.put(sortedLayers.get(i), i + 1);
        }
        Map<String, Integer> remappedLayers = new HashMap<>();
        for (Map.Entry<String, Integer> entry : nodeLayers.entrySet()) {
            remappedLayers.put(entry.getKey(), layerMap.get(entry.getValue()));
        }

        return remappedLayers;
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