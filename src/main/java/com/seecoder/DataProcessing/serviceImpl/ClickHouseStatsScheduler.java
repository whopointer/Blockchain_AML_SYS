package com.seecoder.DataProcessing.serviceImpl;

import com.seecoder.DataProcessing.service.ClickHouseAggregationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDate;

@Slf4j
@Component
public class ClickHouseStatsScheduler {

    @Autowired
    private ClickHouseAggregationService clickHouseAggregationService;

    private static final String CHAIN_BTC = "BTC";
    private static final String CHAIN_ETH = "ETH";

//    @Scheduled(cron = "0 0 2 * * ?") // 每天凌晨2点执行
    public void aggregateYesterdayStats() {
        LocalDate yesterday = LocalDate.now().minusDays(1);
        log.info("定时任务：开始聚合 {} 的统计数据", yesterday);

        try {
            clickHouseAggregationService.aggregateDailyStats(yesterday, CHAIN_BTC);
            clickHouseAggregationService.aggregateDailyStats(yesterday, CHAIN_ETH);
            log.info("定时任务完成");
        } catch (Exception e) {
            log.error("定时聚合失败", e);
        }
    }
}