package com.seecoder.DataProcessing.po;

import lombok.Data;
import javax.persistence.*;
import java.math.BigInteger;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "chain_token_transfer", uniqueConstraints = {
        @UniqueConstraint(name = "uk_token_tx_log", columnNames = {"transaction_hash", "log_index"})
})
public class ChainTokenTransfer {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "chain", nullable = false, length = 16)
    private String chain = "ETH";

    @Column(name = "block_number", nullable = false)
    private Long blockNumber;

    @Column(name = "block_timestamp", nullable = false)
    private LocalDateTime blockTimestamp;

    @Column(name = "transaction_hash", nullable = false, length = 80)
    private String transactionHash;

    @Column(name = "log_index")
    private Integer logIndex;

    @Column(name = "token_address", length = 42, nullable = false)
    private String tokenAddress;

    @Column(name = "from_address", length = 42)
    private String fromAddress;

    @Column(name = "to_address", length = 42)
    private String toAddress;

    @Column(name = "value", length = 255)
    private String value;   // 改为 String 类型
    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
}