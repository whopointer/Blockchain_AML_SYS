package com.seecoder.DataProcessing.po;

import lombok.Data;
import javax.persistence.*;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "chain_tx", uniqueConstraints = {
        @UniqueConstraint(name = "uk_chain_tx", columnNames = {"chain", "tx_hash"})
})
public class ChainTx {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "chain", nullable = false, length = 16)
    private String chain ;

    @Column(name = "tx_hash", nullable = false, length = 80)
    private String txHash;

    @Column(name = "block_height", nullable = false)
    private Long blockHeight;

    @Column(name = "block_time", nullable = false)
    private LocalDateTime blockTime;

    @Column(name = "total_input", precision = 38, scale = 8)
    private BigDecimal totalInput;

    @Column(name = "total_output", precision = 38, scale = 8)
    private BigDecimal totalOutput;

    @Column(name = "fee", precision = 38, scale = 8)
    private BigDecimal fee;

    @Column(name = "tx_index")
    private Integer txIndex;

    @Column(name = "status", length = 16)
    private String status = "confirmed";

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    // ============ 以太坊字段 ============
    @Column(name = "from_address", length = 128)
    private String fromAddress;

    @Column(name = "to_address", length = 128)
    private String toAddress;

    @Column(name = "size_bytes")
    private Long sizeBytes;

    @Column(name = "lock_time")
    private Long locktime;

    @Column(name = "gas_price", precision = 38, scale = 18)
    private BigDecimal gasPrice;

    @Column(name = "gas_used")
    private Long gasUsed;

    @Column(name = "input_data", columnDefinition = "TEXT")
    private String inputData;

    // ============ 新增字段：wei 值 ============
    @Column(name = "value_wei", precision = 38, scale = 0)
    private BigInteger valueWei;          // 交易金额（wei），用于CSV导出

    // ============ 辅助方法 ============
    public Double getTotalInputAsDouble() {
        return totalInput != null ? totalInput.doubleValue() : 0.0;
    }

    public Double getTotalOutputAsDouble() {
        return totalOutput != null ? totalOutput.doubleValue() : 0.0;
    }

    public Double getFeeAsDouble() {
        return fee != null ? fee.doubleValue() : 0.0;
    }

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
}