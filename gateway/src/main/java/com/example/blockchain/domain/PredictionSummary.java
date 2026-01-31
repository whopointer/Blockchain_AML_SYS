package com.example.blockchain.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PredictionSummary {
    /** 总交易数 */
    private int total_transactions;

    /** 可疑交易数 */
    private int suspicious_count;

    /** 正常交易数 */
    private int legitimate_count;

    /** 可疑交易率 */
    private double suspicious_rate;

    /** 平均置信度 */
    private double average_confidence;

    /** 风险分布 */
    private RiskDistribution risk_distribution;

    /** 摘要时间戳 */
    private String timestamp;
}