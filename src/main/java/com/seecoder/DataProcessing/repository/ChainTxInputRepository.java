// com/seecoder/DataProcessing/repository/ChainTxInputRepository.java
package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.po.ChainTxInput;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.util.List;

@Repository
public interface ChainTxInputRepository extends JpaRepository<ChainTxInput, Long> {
    // 根据地址查询所有相关的交易哈希（去重？这里返回 List<String>）
    @Query("SELECT DISTINCT t.txHash FROM ChainTxInput i JOIN i.transaction t WHERE i.address = :address")
    List<String> findTxHashesByAddress(@Param("address") String address);

    // 计算某个地址所有输入的总金额
    @Query("SELECT SUM(i.value) FROM ChainTxInput i WHERE i.address = :address")
    BigDecimal sumValueByAddress(@Param("address") String address);

    @Query("SELECT i.transaction.txHash, COUNT(i) FROM ChainTxInput i WHERE i.transaction.txHash IN :txHashes GROUP BY i.transaction.txHash")
    List<Object[]> countInputsByTxHashIn(@Param("txHashes") List<String> txHashes);
    // 根据交易对象查询所有输入（用于获取交易详情）
    List<ChainTxInput> findByTransaction(ChainTx transaction);

    List<ChainTxInput> findByChainAndTransaction_TxHash(String chain, String txHash);

    // 根据交易ID列表查询所有输入
    List<ChainTxInput> findByTransactionIdIn(List<Long> txIds);

    @Query("SELECT i.address, COUNT(i) FROM ChainTxInput i WHERE i.chain = :chain AND i.address IS NOT NULL GROUP BY i.address")
    List<Object[]> countGroupByAddress(@Param("chain") String chain);

}