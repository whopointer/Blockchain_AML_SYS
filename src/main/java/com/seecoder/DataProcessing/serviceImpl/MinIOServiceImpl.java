// com/seecoder/DataProcessing/serviceImpl/MinIOServiceImpl.java
package com.seecoder.DataProcessing.serviceImpl;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.seecoder.DataProcessing.service.MinIOService;
import com.seecoder.DataProcessing.util.MinIOUtil;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Slf4j
@Service
public class MinIOServiceImpl implements MinIOService {

    @Autowired
    private MinIOUtil minIOUtil;

    @Autowired
    private ObjectMapper objectMapper;

    @Value("${minio.archive.enabled:true}")
    private boolean archiveEnabled;

    private static final DateTimeFormatter DATE_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd");
    private static final DateTimeFormatter TIMESTAMP_FORMATTER =
            DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    @Override
    public ApiResponse<String> archiveBlockData(Long blockHeight, String jsonData) {
        if (!archiveEnabled) {
            return ApiResponse.success("归档功能已禁用", null);
        }

        try {
            // 添加元数据
            Map<String, Object> archiveData = new HashMap<>();
            archiveData.put("blockHeight", blockHeight);
            archiveData.put("archiveTime", LocalDateTime.now().toString());
            archiveData.put("dataType", "block");
            archiveData.put("data", objectMapper.readValue(jsonData, Object.class));

            String enhancedJson = objectMapper.writeValueAsString(archiveData);

            return minIOUtil.uploadBlockchainData("ETH", "blocks",
                    blockHeight, enhancedJson);

        } catch (JsonProcessingException e) {
            log.error("处理区块数据JSON失败", e);
            return ApiResponse.error(500, "处理区块数据失败");
        } catch (Exception e) {
            log.error("归档区块数据失败: height={}", blockHeight, e);
            return ApiResponse.error(500, "归档区块数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<String> archiveTransactionData(String txHash, String jsonData) {
        if (!archiveEnabled) {
            return ApiResponse.success("归档功能已禁用", null);
        }

        try {
            // 解析交易数据获取区块高度
            Map<String, Object> txData = objectMapper.readValue(jsonData, Map.class);
            Long blockHeight = txData.containsKey("block_number") ?
                    Long.parseLong(txData.get("block_number").toString()) : 0L;

            // 添加元数据
            Map<String, Object> archiveData = new HashMap<>();
            archiveData.put("txHash", txHash);
            archiveData.put("blockHeight", blockHeight);
            archiveData.put("archiveTime", LocalDateTime.now().toString());
            archiveData.put("dataType", "transaction");
            archiveData.put("data", txData);

            String enhancedJson = objectMapper.writeValueAsString(archiveData);

            return minIOUtil.uploadBlockchainData("ETH", "transactions",
                    blockHeight, enhancedJson);

        } catch (Exception e) {
            log.error("归档交易数据失败: txHash={}", txHash, e);
            return ApiResponse.error(500, "归档交易数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<String> archiveRawBigQueryResponse(String queryType, String jsonResponse) {
        if (!archiveEnabled) {
            return ApiResponse.success("归档功能已禁用", null);
        }

        try {
            String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMATTER);
            String fileName = String.format("bigquery_%s_%s", queryType, timestamp);

            // 添加查询元数据
            Map<String, Object> archiveData = new HashMap<>();
            archiveData.put("queryType", queryType);
            archiveData.put("executionTime", LocalDateTime.now().toString());
            archiveData.put("response", objectMapper.readValue(jsonResponse, Object.class));

            String enhancedJson = objectMapper.writeValueAsString(archiveData);

            return minIOUtil.uploadJsonData("bigquery", fileName, enhancedJson);

        } catch (Exception e) {
            log.error("归档BigQuery响应失败: type={}", queryType, e);
            return ApiResponse.error(500, "归档BigQuery响应失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<String> archiveApplicationLog(String logContent) {
        return minIOUtil.uploadLogFile("application", logContent);
    }

    @Override
    public ApiResponse<String> archiveSyncLog(String chain, String logContent) {
        return minIOUtil.uploadLogFile("sync/" + chain, logContent);
    }

    @Override
    public ApiResponse<String> archiveErrorLog(String errorContent) {
        return minIOUtil.uploadLogFile("error", errorContent);
    }

    @Override
    public ApiResponse<Map<String, Object>> listArchivedData(String dataType, String date) {
        try {
            String prefix = dataType + "/" + date.replace("-", "/");
            return minIOUtil.listFiles(prefix, true);

        } catch (Exception e) {
            log.error("列出归档数据失败: type={}, date={}", dataType, date, e);
            return ApiResponse.error(500, "列出归档数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<String> downloadArchivedFile(String objectName) {
        ApiResponse<byte[]> result = minIOUtil.downloadFile(objectName);
        if (result.isSuccess()) {
            return ApiResponse.success(new String(result.getData()), null);
        } else {
            return ApiResponse.error(result.getCode(), result.getMessage());
        }
    }

    @Override
    public ApiResponse<String> getFileUrl(String objectName, int expiryHours) {
        return minIOUtil.getPresignedUrl(objectName, expiryHours);
    }

    @Override
    public ApiResponse<String> cleanupOldData(int daysToKeep) {
        try {
            LocalDate cutoffDate = LocalDate.now().minusDays(daysToKeep);
            String cutoffPath = cutoffDate.format(DateTimeFormatter.ofPattern("yyyy/MM/dd"));

            // 这里需要实现更复杂的清理逻辑，根据您的需求
            // 目前返回提示信息
            return ApiResponse.success(
                    String.format("清理策略：保留 %d 天前的数据，截止日期：%s",
                            daysToKeep, cutoffDate.toString()),
                    null);

        } catch (Exception e) {
            log.error("清理旧数据失败", e);
            return ApiResponse.error(500, "清理旧数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getStorageStats() {
        try {
            Map<String, Object> stats = new HashMap<>();
            stats.put("archiveEnabled", archiveEnabled);
            stats.put("lastCheckTime", LocalDateTime.now().toString());

            // 可以添加更多统计信息
            return ApiResponse.success(stats, null);

        } catch (Exception e) {
            log.error("获取存储统计失败", e);
            return ApiResponse.error(500, "获取存储统计失败: " + e.getMessage());
        }
    }

    /**
     * 定时归档应用日志（每天凌晨2点执行）
     */
    @Scheduled(cron = "0 0 2 * * ?")
    public void scheduledLogArchive() {
        if (!archiveEnabled) {
            return;
        }

        log.info("开始定时归档应用日志...");
        try {
            // 这里可以实现读取日志文件并归档的逻辑
            String logContent = "应用日志归档 - " + LocalDateTime.now();
            archiveApplicationLog(logContent);
            log.info("定时归档应用日志完成");

        } catch (Exception e) {
            log.error("定时归档应用日志失败", e);
        }
    }

    /**
     * 每周清理一次旧数据（每周日凌晨3点执行）
     */
    @Scheduled(cron = "0 0 3 ? * SUN")
    public void scheduledDataCleanup() {
        if (!archiveEnabled) {
            return;
        }

        log.info("开始定时清理旧数据...");
        try {
            // 保留30天的数据
            cleanupOldData(30);
            log.info("定时清理旧数据完成");

        } catch (Exception e) {
            log.error("定时清理旧数据失败", e);
        }
    }
}