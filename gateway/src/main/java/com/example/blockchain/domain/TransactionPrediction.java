package com.example.blockchain.domain;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TransactionPrediction {
    /** 交易ID */
    private String tx_id;

    /** 预测结果: 0-正常, 1-异常 */
    private int prediction;

    /** 异常概率 */
    private float probability;

    /** 是否可疑 */
    @JsonProperty("is_suspicious")
    private boolean is_suspicious;

    /** 预测置信度 */
    private float confidence;

    /** 是否出错 */
    private String error;

    /** 预测时间戳 */
    private String timestamp;

    /** 危险等级 */
    private String risk_level;
}
