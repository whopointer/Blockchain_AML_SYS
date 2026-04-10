// com/seecoder/DataProcessing/repository/ChainTxRepository.java
package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.ChainTx;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.Set;

@Repository
public interface ChainTxRepository extends JpaRepository<ChainTx, Long> {

    List<ChainTx> findByTxHashIn(List<String> txHashes);

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
    Page<ChainTx> findByChainAndBlockTimeBetween(String chain, LocalDateTime startTime, LocalDateTime endTime, Pageable pageable);

    Long countByChainAndBlockTimeBetween(String chain, LocalDateTime startTime, LocalDateTime endTime);

    // 聚合查询 - 金额统计
    @Query("SELECT SUM(tx.totalOutput) FROM ChainTx tx WHERE tx.chain = :chain AND tx.toAddress = :address")
    BigDecimal sumTotalOutputByToAddress(@Param("chain") String chain, @Param("address") String address);

    @Query("SELECT SUM(tx.totalInput) FROM ChainTx tx WHERE tx.chain = :chain AND tx.fromAddress = :address")
    BigDecimal sumTotalInputByFromAddress(@Param("chain") String chain, @Param("address") String address);

    Page<ChainTx> findByChainAndBlockHeightBetween(String chain, Long startHeight, Long endHeight, Pageable pageable);


    // 批量查询方法
    List<ChainTx> findByChainAndTxHashIn(String chain, List<String> txHashes);

    List<ChainTx> findByChainAndBlockHeightIn(String chain, List<Long> blockHeights);


    // ============ 新增方法 ============
    @Query("SELECT t FROM ChainTx t WHERE t.chain = :chain AND " +
            "(t.fromAddress = :address OR t.toAddress = :address) " +
            "AND t.blockTime BETWEEN :start AND :end")
    List<ChainTx> findByAddressAndTimeRange(@Param("chain") String chain,
                                            @Param("address") String address,
                                            @Param("start") LocalDateTime start,
                                            @Param("end") LocalDateTime end,
                                            Sort sort);

    @Query("SELECT COUNT(t) FROM ChainTx t WHERE t.chain = :chain AND " +
            "(t.fromAddress = :address OR t.toAddress = :address) " +
            "AND t.blockTime BETWEEN :start AND :end")
    long countByAddressAndTimeRange(@Param("chain") String chain,
                                    @Param("address") String address,
                                    @Param("start") LocalDateTime start,
                                    @Param("end") LocalDateTime end);



    @Query("SELECT t.fromAddress, COUNT(t) FROM ChainTx t WHERE t.chain = :chain GROUP BY t.fromAddress")
    List<Object[]> countGroupByFromAddress(@Param("chain") String chain);

    @Query("SELECT t.toAddress, COUNT(t) FROM ChainTx t WHERE t.chain = :chain AND t.toAddress IS NOT NULL GROUP BY t.toAddress")
    List<Object[]> countGroupByToAddress(@Param("chain") String chain);

    // 按链和时间范围统计总输出金额
    @Query("SELECT SUM(tx.totalOutput) FROM ChainTx tx WHERE tx.chain = :chain AND tx.blockTime BETWEEN :start AND :end")
    BigDecimal sumTotalOutputByChainAndTimeRange(@Param("chain") String chain,
                                                 @Param("start") LocalDateTime start,
                                                 @Param("end") LocalDateTime end);

    // 按链和时间范围统计总手续费
    @Query("SELECT SUM(tx.fee) FROM ChainTx tx WHERE tx.chain = :chain AND tx.blockTime BETWEEN :start AND :end")
    BigDecimal sumFeeByChainAndTimeRange(@Param("chain") String chain,
                                         @Param("start") LocalDateTime start,
                                         @Param("end") LocalDateTime end);

    // 按链和时间范围统计活跃地址数（from 和 to 的去重总数）

    @Query(value = "SELECT COUNT(DISTINCT addr) FROM (" +
            "SELECT from_address AS addr FROM chain_tx WHERE chain = ?1 AND block_time BETWEEN ?2 AND ?3 " +
            "UNION " +
            "SELECT to_address AS addr FROM chain_tx WHERE chain = ?1 AND block_time BETWEEN ?2 AND ?3" +
            ") AS u", nativeQuery = true)
    Long countDistinctAddressByChainAndTimeRange(String chain, LocalDateTime start, LocalDateTime end);


    // ChainTxRepository.java
    @Query("SELECT t.txHash FROM ChainTx t WHERE t.chain = :chain AND t.blockTime BETWEEN :start AND :end")
    Set<String> findTxHashesByTimeRange(@Param("chain") String chain,
                                        @Param("start") LocalDateTime start,
                                        @Param("end") LocalDateTime end);
}