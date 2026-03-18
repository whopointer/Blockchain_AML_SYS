package com.seecoder.DataProcessing.repository.clickhouse;


import com.seecoder.DataProcessing.po.clickhouse.DailyChainStats;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.LocalDate;
import java.util.List;

@Repository
public class ClickHouseStatsRepository {

    private final JdbcTemplate clickHouseJdbcTemplate;

    public ClickHouseStatsRepository(@Qualifier("clickHouseJdbcTemplate") JdbcTemplate clickHouseJdbcTemplate) {
        this.clickHouseJdbcTemplate = clickHouseJdbcTemplate;
    }

    public void insertDailyStats(DailyChainStats stats) {
        String sql = "INSERT INTO daily_chain_stats (date, chain, block_count, transaction_count, total_value_eth, total_fee_eth, avg_value_eth, avg_fee_eth, active_address_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";
        clickHouseJdbcTemplate.update(sql,
                stats.getDate(),
                stats.getChain(),
                stats.getBlockCount(),
                stats.getTransactionCount(),
                stats.getTotalValueEth(),
                stats.getTotalFeeEth(),
                stats.getAvgValueEth(),
                stats.getAvgFeeEth(),
                stats.getActiveAddressCount());
    }

    public List<DailyChainStats> findDailyStatsByChainAndDateRange(String chain, LocalDate startDate, LocalDate endDate) {
        String sql = "SELECT date, chain, block_count, transaction_count, total_value_eth, total_fee_eth, avg_value_eth, avg_fee_eth, active_address_count, created_at FROM daily_chain_stats WHERE chain = ? AND date BETWEEN ? AND ? ORDER BY date";
        return clickHouseJdbcTemplate.query(sql, new DailyChainStatsRowMapper(), chain, startDate, endDate);
    }

    private static class DailyChainStatsRowMapper implements RowMapper<DailyChainStats> {
        @Override
        public DailyChainStats mapRow(ResultSet rs, int rowNum) throws SQLException {
            DailyChainStats stats = new DailyChainStats();
            stats.setDate(rs.getObject("date", LocalDate.class));
            stats.setChain(rs.getString("chain"));
            stats.setBlockCount(rs.getLong("block_count"));
            stats.setTransactionCount(rs.getLong("transaction_count"));
            stats.setTotalValueEth(rs.getBigDecimal("total_value_eth"));
            stats.setTotalFeeEth(rs.getBigDecimal("total_fee_eth"));
            stats.setAvgValueEth(rs.getBigDecimal("avg_value_eth"));
            stats.setAvgFeeEth(rs.getBigDecimal("avg_fee_eth"));
            stats.setActiveAddressCount(rs.getLong("active_address_count"));
            stats.setCreatedAt(rs.getTimestamp("created_at").toLocalDateTime());
            return stats;
        }
    }
}