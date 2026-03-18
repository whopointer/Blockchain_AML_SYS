// com/seecoder/DataProcessing/po/FilterConfig.java
package com.seecoder.DataProcessing.po;

import lombok.Data;

import javax.persistence.Column;
import javax.persistence.Embeddable;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 快照的过滤配置，用于保存图谱生成时的筛选条件。
 * 以 Embeddable 形式嵌入到 {@link GraphSnapshot} 中。
 */
@Embeddable
@Data
public class FilterConfig {

    /** 交易类型：all/inflow/outflow */
    @Column(name = "tx_type", length = 20)
    private String txType;

    /** 地址类型：all/tagged/malicious/normal/tagged_malicious */
    @Column(name = "addr_type", length = 30)
    private String addrType;

    /** 最小金额 */
    @Column(name = "min_amount")
    private BigDecimal minAmount;

    /** 最大金额 */
    @Column(name = "max_amount")
    private BigDecimal maxAmount;

    /** 起始时间 */
    @Column(name = "start_date")
    private LocalDateTime startDate;

    /** 结束时间 */
    @Column(name = "end_date")
    private LocalDateTime endDate;
}