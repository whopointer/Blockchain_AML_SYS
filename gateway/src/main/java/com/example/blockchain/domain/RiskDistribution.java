package com.example.blockchain.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class RiskDistribution {
    /** 高风险交易数 */
    private int high_risk;

    /** 中风险交易数 */
    private int medium_risk;

    /** 低风险交易数 */
    private int low_risk;
}