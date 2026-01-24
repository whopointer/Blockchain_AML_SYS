// com/seecoder/DataProcessing/po/ChainTx.java
package com.seecoder.DataProcessing.po;

import lombok.Data;
import javax.persistence.*;
import java.math.BigDecimal;
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
    private String chain = "BTC"; // 默认改为 ETH

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

    // ============ 新增的以太坊字段 ============
    @Column(name = "from_address", length = 128)
    private String fromAddress;

    @Column(name = "to_address", length = 128)
    private String toAddress;




    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
}