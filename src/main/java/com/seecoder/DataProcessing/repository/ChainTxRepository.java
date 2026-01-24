// com/seecoder/DataProcessing/repository/ChainTxRepository.java
package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.ChainTx;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface ChainTxRepository extends JpaRepository<ChainTx, Long> {

    // 基本查询方法
    Optional<ChainTx> findByChainAndTxHash(String chain, String txHash);

    List<ChainTx> findByChainAndBlockHeight(String chain, Long blockHeight, Pageable pageable);

    // 分页查询
    Page<ChainTx> findByChain(String chain, Pageable pageable);

    // 列表查询（用于非分页场景）
    List<ChainTx> findAllByChain(String chain, Pageable pageable);

    Long countByChain(String chain);

    // 地址相关查询
    @Query("SELECT tx FROM ChainTx tx WHERE tx.chain = :chain AND (tx.fromAddress = :address OR tx.toAddress = :address)")
    Page<ChainTx> findByFromAddressOrToAddress(@Param("chain") String chain,
                                               @Param("address") String address,
                                               Pageable pageable);

    @Query("SELECT COUNT(tx) FROM ChainTx tx WHERE tx.chain = :chain AND (tx.fromAddress = :address OR tx.toAddress = :address)")
    Long countByFromAddressOrToAddress(@Param("chain") String chain, @Param("address") String address);

    // 时间范围查询
    List<ChainTx> findByChainAndBlockTimeBetween(String chain, LocalDateTime startTime, LocalDateTime endTime, Pageable pageable);

    Long countByChainAndBlockTimeBetween(String chain, LocalDateTime startTime, LocalDateTime endTime);

    // 聚合查询 - 金额统计
    @Query("SELECT SUM(tx.totalOutput) FROM ChainTx tx WHERE tx.chain = :chain AND tx.toAddress = :address")
    BigDecimal sumTotalOutputByToAddress(@Param("chain") String chain, @Param("address") String address);

    @Query("SELECT SUM(tx.totalInput) FROM ChainTx tx WHERE tx.chain = :chain AND tx.fromAddress = :address")
    BigDecimal sumTotalInputByFromAddress(@Param("chain") String chain, @Param("address") String address);

    // 批量查询方法
    List<ChainTx> findByChainAndTxHashIn(String chain, List<String> txHashes);

    List<ChainTx> findByChainAndBlockHeightIn(String chain, List<Long> blockHeights);
}