// com/seecoder/DataProcessing/controller/Neo4jController.java
package com.seecoder.DataProcessing.controller;

import com.seecoder.DataProcessing.service.GraphService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/neo4j")
@Api(tags = "Neo4j图数据查询接口")
public class Neo4jController {

    @Autowired
    private GraphService graphService;

    @GetMapping("/test/neo4j")
    public ApiResponse<Map<String, Object>> testNeo4jConnection() {
        return graphService.testNeo4jConnection();
    }

    @GetMapping("/path")
    @ApiOperation("查找交易路径")
    public ApiResponse<Map<String, Object>> findTransactionPath(
            @RequestParam String fromAddress,
            @RequestParam String toAddress,
            @RequestParam(defaultValue = "5") Integer maxHops) {
        ApiResponse<Map<String, Object>> response = graphService.findNhopTransactionPath(fromAddress, toAddress, maxHops);
        
        if (response.getCode() != 200) {
            return ApiResponse.error(response.getCode(), response.getMessage());
        }
        
        // 直接返回graph_dic的内容
        Map<String, Object> result = response.getData();
        
        if (result == null) {
            // 如果没有找到路径，返回空的graph_dic结构
            result = new java.util.HashMap<>();
            result.put("node_list", new java.util.ArrayList<>());
            result.put("edge_list", new java.util.ArrayList<>());
            result.put("tx_count", 0);
            result.put("first_tx_datetime", "");
            result.put("latest_tx_datetime", "");
            result.put("address_first_tx_datetime", "");
            result.put("address_latest_tx_datetime", "");
        }
        
        return ApiResponse.success(result, response.getTotal());
    }

    @GetMapping("/address/hops")
    @ApiOperation("查找N跳内地址")
    public ApiResponse<List<Map<String, Object>>> findAddressesWithinHops(
            @RequestParam String address,
            @RequestParam(defaultValue = "3") Integer maxHops) {
        return graphService.findAddressesWithinNHops(address, maxHops);
    }

    @GetMapping("/address/stats")
    @ApiOperation("获取地址转账统计")
    public ApiResponse<Map<String, Object>> getAddressStats(
            @RequestParam String address) {
        return graphService.getAddressTransferStats(address);
    }

    @GetMapping("/address/connections")
    @ApiOperation("获取地址直接连接")
    public ApiResponse<Map<String, Object>> getAddressConnections(
            @RequestParam String address) {
        return graphService.getDirectConnections(address);
    }

    @GetMapping("/address/pattern")
    @ApiOperation("分析地址模式")
    public ApiResponse<Map<String, Object>> analyzeAddressPattern(
            @RequestParam String address,
            @RequestParam(defaultValue = "2") Integer depth) {
        return graphService.analyzeAddressPattern(address, depth);
    }

    @GetMapping("/transfer/stats")
    @ApiOperation("获取地址间转账统计")
    public ApiResponse<Map<String, Object>> getTransferStats(
            @RequestParam String fromAddress,
            @RequestParam String toAddress) {
        return graphService.getTransferStatsBetweenAddresses(fromAddress, toAddress);
    }

    @GetMapping("/transfer/large")
    @ApiOperation("查找大额转账")
    public ApiResponse<List<Map<String, Object>>> findLargeTransfers(
            @RequestParam(required = false) BigDecimal minAmount,
            @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime endTime,
            @RequestParam(defaultValue = "50") Integer limit) {
        return graphService.findLargeTransfers(minAmount, startTime, endTime, limit);
    }

    @PutMapping("/address/risk")
    @ApiOperation("更新地址风险等级")
    public ApiResponse<Void> updateRiskLevel(
            @RequestParam String address,
            @RequestParam Integer riskLevel) {
        return graphService.updateAddressRiskLevel(address, riskLevel);
    }

    @PutMapping("/address/tag")
    @ApiOperation("为地址打标签")
    public ApiResponse<Void> tagAddress(
            @RequestParam String address,
            @RequestParam String tag) {
        return graphService.tagAddress(address, tag);
    }

    @DeleteMapping("/clean")
    @ApiOperation("清理图数据")
    public ApiResponse<Void> cleanGraphData(
            @RequestParam(required = false, defaultValue = "ETH") String chain) {
        try {
            graphService.cleanGraphData(chain);
            return ApiResponse.success(null, null);
        } catch (Exception e) {
            log.error("清理图数据失败", e);
            return ApiResponse.error(500, "清理图数据失败: " + e.getMessage());
        }
    }
}