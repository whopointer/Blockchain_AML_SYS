// com/seecoder/DataProcessing/service/FeatureExportService.java
package com.seecoder.DataProcessing.service;

import com.seecoder.DataProcessing.vo.ApiResponse;
import java.time.LocalDateTime;

public interface FeatureExportService {
    /**
     * 导出指定链、指定高度范围（或时间范围）的交易特征到CSV文件
     * @param chain        链标识（BTC/ETH）
     * @param startHeight  起始高度（含），与时间范围二选一
     * @param endHeight    结束高度（含），与时间范围二选一
     * @param startTime    起始时间，与高度范围二选一
     * @param endTime      结束时间，与高度范围二选一
     * @param filePath     输出文件路径
     * @return 操作结果
     */
    ApiResponse<String> exportFeatures(String chain,
                                       Long startHeight, Long endHeight,
                                       LocalDateTime startTime, LocalDateTime endTime,
                                       String filePath);
}