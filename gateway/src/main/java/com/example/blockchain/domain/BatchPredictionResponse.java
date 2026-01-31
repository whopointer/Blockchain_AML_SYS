package com.example.blockchain.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class BatchPredictionResponse {
    /** 统计信息 */
    private Map<String, Object> statistics;

    /** 响应时间戳 */
    private String timestamp;
}