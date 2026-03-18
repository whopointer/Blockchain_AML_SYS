package com.seecoder.DataProcessing.po.clickhouse;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

public class DailyChainStats {
    private LocalDate date;
    private String chain;
    private Long blockCount;
    private Long transactionCount;
    private BigDecimal totalValueEth;
    private BigDecimal totalFeeEth;
    private BigDecimal avgValueEth;
    private BigDecimal avgFeeEth;
    private Long activeAddressCount;
    private LocalDateTime createdAt;

    // 必须提供 getter 和 setter
    public LocalDate getDate() { return date; }
    public void setDate(LocalDate date) { this.date = date; }

    public String getChain() { return chain; }
    public void setChain(String chain) { this.chain = chain; }

    public Long getBlockCount() { return blockCount; }
    public void setBlockCount(Long blockCount) { this.blockCount = blockCount; }

    public Long getTransactionCount() { return transactionCount; }
    public void setTransactionCount(Long transactionCount) { this.transactionCount = transactionCount; }

    public BigDecimal getTotalValueEth() { return totalValueEth; }
    public void setTotalValueEth(BigDecimal totalValueEth) { this.totalValueEth = totalValueEth; }

    public BigDecimal getTotalFeeEth() { return totalFeeEth; }
    public void setTotalFeeEth(BigDecimal totalFeeEth) { this.totalFeeEth = totalFeeEth; }

    public BigDecimal getAvgValueEth() { return avgValueEth; }
    public void setAvgValueEth(BigDecimal avgValueEth) { this.avgValueEth = avgValueEth; }

    public BigDecimal getAvgFeeEth() { return avgFeeEth; }
    public void setAvgFeeEth(BigDecimal avgFeeEth) { this.avgFeeEth = avgFeeEth; }

    public Long getActiveAddressCount() { return activeAddressCount; }
    public void setActiveAddressCount(Long activeAddressCount) { this.activeAddressCount = activeAddressCount; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
