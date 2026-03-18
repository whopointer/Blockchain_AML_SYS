package com.seecoder.DataProcessing.configure;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

@Slf4j
@Component
public class ClickHouseTableInitializer {

    @Autowired
    @Qualifier("clickHouseJdbcTemplate")
    private JdbcTemplate clickHouseJdbcTemplate;

    @PostConstruct
    public void initTables() {
        try {
            String createDailyStatsTable = "CREATE TABLE IF NOT EXISTS daily_chain_stats (" +
                    "date Date," +
                    "chain String," +
                    "block_count UInt64," +
                    "transaction_count UInt64," +
                    "total_value_eth Decimal(38,18)," +
                    "total_fee_eth Decimal(38,18)," +
                    "avg_value_eth Decimal(38,18)," +
                    "avg_fee_eth Decimal(38,18)," +
                    "active_address_count UInt64," +
                    "created_at DateTime DEFAULT now()" +
                    ") ENGINE = MergeTree() ORDER BY (date, chain)";
            clickHouseJdbcTemplate.execute(createDailyStatsTable);
            log.info("ClickHouse表初始化完成");
        } catch (Exception e) {
            log.error("初始化ClickHouse表失败", e);
        }
    }
}