package com.example.blockchain.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PredictionResponse {
    /** 预测结果列表 */
    private List<TransactionPrediction> results;

    /** 总交易数 */
    private int total_transactions;

    /** 可疑交易数 */
    private int suspicious_count;

    /** 响应时间戳 */
    private String timestamp;
}