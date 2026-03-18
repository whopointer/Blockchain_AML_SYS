// com/seecoder/DataProcessing/repository/ChainTxOutputRepository.java
package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.po.ChainTxOutput;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.util.List;

@Repository
public interface ChainTxOutputRepository extends JpaRepository<ChainTxOutput, Long> {
    // 根据地址查询所有相关的交易哈希（去重）
    @Query("SELECT DISTINCT t.txHash FROM ChainTxOutput o JOIN o.transaction t WHERE o.address = :address")
    List<String> findTxHashesByAddress(@Param("address") String address);

    // 计算某个地址所有输出的总金额
    @Query("SELECT SUM(o.value) FROM ChainTxOutput o WHERE o.address = :address")
    BigDecimal sumValueByAddress(@Param("address") String address);

    @Query("SELECT o.transaction.txHash, COUNT(o) FROM ChainTxOutput o WHERE o.transaction.txHash IN :txHashes GROUP BY o.transaction.txHash")
    List<Object[]> countOutputsByTxHashIn(@Param("txHashes") List<String> txHashes);
    // 根据交易对象查询所有输出
    List<ChainTxOutput> findByTransaction(ChainTx transaction);

    List<ChainTxOutput> findByChainAndTransaction_TxHash(String chain, String txHash);

    List<ChainTxOutput> findByTransactionIdIn(List<Long> txIds);

    @Query("SELECT o.address, COUNT(o) FROM ChainTxOutput o WHERE o.chain = :chain AND o.address IS NOT NULL GROUP BY o.address")
    List<Object[]> countGroupByAddress(@Param("chain") String chain);
}