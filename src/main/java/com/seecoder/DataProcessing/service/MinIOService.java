// com/seecoder/DataProcessing/service/MinIOService.java
package com.seecoder.DataProcessing.service;

import com.seecoder.DataProcessing.vo.ApiResponse;

import java.time.LocalDateTime;
import java.util.Map;

public interface MinIOService {

    // 区块链数据归档
    ApiResponse<String> archiveBlockData(Long blockHeight, String jsonData);
    ApiResponse<String> archiveTransactionData(String txHash, String jsonData);
    ApiResponse<String> archiveRawBigQueryResponse(String queryType, String jsonResponse);

    // 日志归档
    ApiResponse<String> archiveApplicationLog(String logContent);
    ApiResponse<String> archiveSyncLog(String chain, String logContent);
    ApiResponse<String> archiveErrorLog(String errorContent);

    // 查询和下载
    ApiResponse<Map<String, Object>> listArchivedData(String dataType, String date);
    ApiResponse<String> downloadArchivedFile(String objectName);
    ApiResponse<String> getFileUrl(String objectName, int expiryHours);

    // 清理管理
    ApiResponse<String> cleanupOldData(int daysToKeep);
    ApiResponse<Map<String, Object>> getStorageStats();
}