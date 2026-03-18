package com.seecoder.DataProcessing.serviceImpl;

import com.seecoder.DataProcessing.po.clickhouse.DailyChainStats;
import com.seecoder.DataProcessing.repository.clickhouse.ClickHouseStatsRepository;
import com.seecoder.DataProcessing.repository.ChainBlockRepository;
import com.seecoder.DataProcessing.repository.ChainTxRepository;
import com.seecoder.DataProcessing.service.ClickHouseAggregationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Slf4j
@Service
public class ClickHouseAggregationServiceImpl implements ClickHouseAggregationService {

    @Autowired
    private ChainBlockRepository chainBlockRepository;

    @Autowired
    private ChainTxRepository chainTxRepository;

    @Autowired
    private ClickHouseStatsRepository clickHouseStatsRepository;

    @Override
    public void aggregateDailyStats(LocalDate date, String chain) {
        log.info("开始聚合 {} 链 {} 的每日统计", chain, date);

        LocalDateTime startOfDay = date.atStartOfDay();
        LocalDateTime endOfDay = date.plusDays(1).atStartOfDay();

        // 区块数
        Long blockCount = chainBlockRepository.countByChainAndBlockTimeBetween(chain, startOfDay, endOfDay);
        // 交易数
        Long txCount = chainTxRepository.countByChainAndBlockTimeBetween(chain, startOfDay, endOfDay);
        // 总输出金额
        BigDecimal totalOutput = chainTxRepository.sumTotalOutputByChainAndTimeRange(chain, startOfDay, endOfDay);
        // 总手续费
        BigDecimal totalFee = chainTxRepository.sumFeeByChainAndTimeRange(chain, startOfDay, endOfDay);
        // 活跃地址数（去重）
        Long activeAddressCount = chainTxRepository.countDistinctAddressByChainAndTimeRange(chain, startOfDay, endOfDay);

        // 计算平均值
        BigDecimal avgValue = BigDecimal.ZERO;
        BigDecimal avgFee = BigDecimal.ZERO;
        if (txCount != null && txCount > 0) {
            avgValue = totalOutput.divide(BigDecimal.valueOf(txCount), 18, RoundingMode.HALF_UP);
            avgFee = totalFee.divide(BigDecimal.valueOf(txCount), 18, RoundingMode.HALF_UP);
        }

        DailyChainStats stats = new DailyChainStats();
        stats.setDate(date);
        stats.setChain(chain);
        stats.setBlockCount(blockCount != null ? blockCount : 0L);
        stats.setTransactionCount(txCount != null ? txCount : 0L);
        stats.setTotalValueEth(totalOutput != null ? totalOutput : BigDecimal.ZERO);
        stats.setTotalFeeEth(totalFee != null ? totalFee : BigDecimal.ZERO);
        stats.setAvgValueEth(avgValue);
        stats.setAvgFeeEth(avgFee);
        stats.setActiveAddressCount(activeAddressCount != null ? activeAddressCount : 0L);

        // 写入 ClickHouse
        clickHouseStatsRepository.insertDailyStats(stats);
        log.info("聚合完成并写入ClickHouse: {}", stats);
    }
}