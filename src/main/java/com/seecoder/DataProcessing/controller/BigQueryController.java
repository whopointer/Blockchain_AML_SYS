package com.seecoder.DataProcessing.controller;

import com.seecoder.DataProcessing.po.TransactionData;
import com.seecoder.DataProcessing.po.BlockData;
import com.seecoder.DataProcessing.po.TokenTransferData;
import com.seecoder.DataProcessing.enums.NetworkType;
import com.seecoder.DataProcessing.service.BigQueryService;
import com.seecoder.DataProcessing.util.DateUtil;
import com.seecoder.DataProcessing.vo.ApiResponse;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

import java.util.Date;
import java.util.List;

/**
 * BigQuery 数据拉取控制器
 * 作用：提供 RESTful API 接口，处理前端对区块链数据的请求
 * 设计模式：MVC 模式中的 Controller 层，负责接收请求、调用服务、返回响应
 */
@Slf4j  // Lombok 注解，自动生成日志记录器（private static final Logger log）
@RestController  // 标识这是一个 REST 控制器，返回值会自动序列化为 JSON
@RequestMapping("/api/bigquery")  // 定义基础路径，所有这个控制器的方法都以 /api/bigquery 开头
@Api(tags = "BigQuery数据拉取接口")  // Swagger 注解，为 API 文档分组
public class BigQueryController {

    /**
     * 自动注入 BigQueryService
     * Spring 会自动查找匹配类型的 Bean 并注入到这里
     * 这是依赖注入（Dependency Injection）的典型用法
     */
    @Autowired
    private BigQueryService bigQueryService;

    /**
     * 获取历史交易数据的 REST 端点
     * HTTP 方法：GET
     * 路径：/api/bigquery/transactions
     *
     * @param network 区块链网络类型（枚举），必填
     * @param startTime 开始时间，格式 yyyy-MM-dd HH:mm:ss，必填
     * @param endTime 结束时间，格式 yyyy-MM-dd HH:mm:ss，必填
     * @param address 地址过滤（可选），可以是发送方或接收方地址
     * @param limit 返回记录数限制，默认100
     * @param offset 偏移量（分页用），默认0
     * @return 标准API响应，包含交易数据列表和总数
     */
    @GetMapping("/transactions")
    @ApiOperation("批量拉取历史交易数据")  // Swagger 注解，描述这个API的操作
    public ApiResponse<List<TransactionData>> getHistoricalTransactions(
            @RequestParam NetworkType network,  // 从请求参数获取，自动转换为枚举
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") Date startTime,  // 指定日期格式
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") Date endTime,
            @RequestParam(required = false) String address,  // required=false 表示可选参数
            @RequestParam(defaultValue = "100") int limit,  // 设置默认值
            @RequestParam(defaultValue = "0") int offset) {

        try {
            // 调用服务层获取数据
            List<TransactionData> transactions = bigQueryService.fetchHistoricalTransactions(
                    network, startTime, endTime, address, limit, offset);

            // 获取数据总数（用于分页）
            Long totalCount = bigQueryService.countTransactions(network, startTime, endTime);

            // 返回成功响应
            return ApiResponse.success(transactions, totalCount);

        } catch (Exception e) {
            // 记录错误日志
            log.error("Error fetching transactions", e);
            // 返回错误响应
            return ApiResponse.error(500, "拉取交易数据失败: " + e.getMessage());
        }
    }

    /**
     * 获取区块数据的 REST 端点
     * 参数结构与交易接口类似，增加了区块范围过滤
     */
    @GetMapping("/blocks")
    @ApiOperation("批量拉取区块数据")
    public ApiResponse<List<BlockData>> getHistoricalBlocks(
            @RequestParam NetworkType network,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") Date startTime,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") Date endTime,
            @RequestParam(required = false) Long startBlock,  // 起始区块号（可选）
            @RequestParam(required = false) Long endBlock,    // 结束区块号（可选）
            @RequestParam(defaultValue = "100") int limit,
            @RequestParam(defaultValue = "0") int offset) {

        try {
            List<BlockData> blocks = bigQueryService.fetchHistoricalBlocks(
                    network, startTime, endTime, startBlock, endBlock, limit, offset);

            Long totalCount = bigQueryService.countBlocks(network, startTime, endTime);

            return ApiResponse.success(blocks, totalCount);

        } catch (Exception e) {
            log.error("Error fetching blocks", e);
            return ApiResponse.error(500, "拉取区块数据失败: " + e.getMessage());
        }
    }

    /**
     * 获取 Token 转账数据的 REST 端点
     * 支持按 Token 地址和用户地址过滤
     */
    @GetMapping("/token-transfers")
    @ApiOperation("批量拉取Token转账数据")
    public ApiResponse<List<TokenTransferData>> getHistoricalTokenTransfers(
            @RequestParam NetworkType network,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") Date startTime,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") Date endTime,
            @RequestParam(required = false) String tokenAddress,  // Token 合约地址（可选）
            @RequestParam(required = false) String userAddress,   // 用户地址（可选）
            @RequestParam(defaultValue = "100") int limit,
            @RequestParam(defaultValue = "0") int offset) {

        try {
            List<TokenTransferData> transfers = bigQueryService.fetchHistoricalTokenTransfers(
                    network, startTime, endTime, tokenAddress, userAddress, limit, offset);

            Long totalCount = bigQueryService.countTokenTransfers(network, startTime, endTime);

            return ApiResponse.success(transfers, totalCount);

        } catch (Exception e) {
            log.error("Error fetching token transfers", e);
            return ApiResponse.error(500, "拉取Token转账数据失败: " + e.getMessage());
        }
    }

    /**
     * 获取数据统计信息的 REST 端点
     * 返回指定时间段内的交易、区块、Token转账数量统计
     */
    @GetMapping("/data-stats")
    @ApiOperation("获取数据统计信息")
    public ApiResponse<Object> getDataStats(
            @RequestParam NetworkType network,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") Date startTime,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") Date endTime) {

        try {
            // 分别获取三种数据的统计数量
            Long transactionCount = bigQueryService.countTransactions(network, startTime, endTime);
            Long blockCount = bigQueryService.countBlocks(network, startTime, endTime);
            Long tokenTransferCount = bigQueryService.countTokenTransfers(network, startTime, endTime);

            // 创建数据统计对象并返回
            return ApiResponse.success(
                    new DataStats(transactionCount, blockCount, tokenTransferCount),
                    null  // 统计接口不需要分页，所以total为null
            );

        } catch (Exception e) {
            log.error("Error getting data stats", e);
            return ApiResponse.error(500, "获取数据统计失败: " + e.getMessage());
        }
    }

    /**
     * 内部数据统计类
     * 作用：封装三种数据类型的统计结果
     * 设计模式：DTO（Data Transfer Object）模式，用于在不同层之间传输数据
     * 注意：这是一个静态内部类，只在当前控制器内部使用
     */
    static class DataStats {
        // 统计字段
        private Long transactionCount;    // 交易数量
        private Long blockCount;          // 区块数量
        private Long tokenTransferCount;  // Token转账数量

        /**
         * 构造函数
         * @param transactionCount 交易数量
         * @param blockCount 区块数量
         * @param tokenTransferCount Token转账数量
         */
        public DataStats(Long transactionCount, Long blockCount, Long tokenTransferCount) {
            this.transactionCount = transactionCount;
            this.blockCount = blockCount;
            this.tokenTransferCount = tokenTransferCount;
        }

        // 注意：原代码缺少getter和setter方法，会导致JSON序列化失败
        // 实际使用时需要添加getter和setter，或使用 @Data 注解

        // getters and setters (需要添加)
        public Long getTransactionCount() {
            return transactionCount;
        }

        public void setTransactionCount(Long transactionCount) {
            this.transactionCount = transactionCount;
        }

        public Long getBlockCount() {
            return blockCount;
        }

        public void setBlockCount(Long blockCount) {
            this.blockCount = blockCount;
        }

        public Long getTokenTransferCount() {
            return tokenTransferCount;
        }

        public void setTokenTransferCount(Long tokenTransferCount) {
            this.tokenTransferCount = tokenTransferCount;
        }
    }
}