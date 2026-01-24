// com/seecoder/DataProcessing/service/BigQueryService.java
package com.seecoder.DataProcessing.service;

import com.seecoder.DataProcessing.po.TransactionData;
import com.seecoder.DataProcessing.po.BlockData;
import com.seecoder.DataProcessing.po.TokenTransferData;
import com.seecoder.DataProcessing.enums.NetworkType;

import java.util.Date;
import java.util.List;

public interface BigQueryService {

    // 批量拉取历史交易数据
    List<TransactionData> fetchHistoricalTransactions(
            NetworkType network,
            Date startTime,
            Date endTime,
            String address,
            int limit,
            int offset
    );

    // 批量拉取区块数据
    List<BlockData> fetchHistoricalBlocks(
            NetworkType network,
            Date startTime,
            Date endTime,
            Long startBlock,
            Long endBlock,
            int limit,
            int offset
    );

    // 批量拉取Token转账数据
    List<TokenTransferData> fetchHistoricalTokenTransfers(
            NetworkType network,
            Date startTime,
            Date endTime,
            String tokenAddress,
            String userAddress,
            int limit,
            int offset
    );

    // 获取交易数量统计
    Long countTransactions(NetworkType network, Date startTime, Date endTime);

    // 获取区块数量统计
    Long countBlocks(NetworkType network, Date startTime, Date endTime);

    // 获取Token转账数量统计
    Long countTokenTransfers(NetworkType network, Date startTime, Date endTime);
}