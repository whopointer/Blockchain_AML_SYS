package com.seecoder.DataProcessing.controller;

import com.seecoder.DataProcessing.po.clickhouse.DailyChainStats;
import com.seecoder.DataProcessing.repository.clickhouse.ClickHouseStatsRepository;
import com.seecoder.DataProcessing.service.ClickHouseAggregationService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;

@Api(tags = "ClickHouse 聚合管理")
@RestController
@RequestMapping("/api/clickhouse")
public class ClickHouseController {

    @Autowired
    private ClickHouseAggregationService clickHouseAggregationService;

    @Autowired
    private ClickHouseStatsRepository clickHouseStatsRepository;

    @ApiOperation("手动聚合指定日期的统计数据")
    @PostMapping("/aggregate/daily")
    public ApiResponse<String> aggregateDaily(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate date,
            @RequestParam String chain) {
        try {
            clickHouseAggregationService.aggregateDailyStats(date, chain);
            return ApiResponse.success("聚合任务已提交: " + date + " " + chain, null);
        } catch (Exception e) {
            return ApiResponse.error(500, "聚合失败: " + e.getMessage());
        }
    }

    @ApiOperation("手动聚合指定日期范围的统计数据")
    @PostMapping("/aggregate/range")
    public ApiResponse<String> aggregateRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate,
            @RequestParam String chain) {
        try {
            LocalDate current = startDate;
            while (!current.isAfter(endDate)) {
                clickHouseAggregationService.aggregateDailyStats(current, chain);
                current = current.plusDays(1);
            }
            return ApiResponse.success("聚合任务已提交: " + startDate + " 到 " + endDate + " " + chain, null);
        } catch (Exception e) {
            return ApiResponse.error(500, "聚合失败: " + e.getMessage());
        }
    }

    @ApiOperation("查询 ClickHouse 中的每日统计数据")
    @GetMapping("/stats")
    public ApiResponse<List<DailyChainStats>> getStats(
            @RequestParam String chain,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate) {
        try {
            List<DailyChainStats> stats = clickHouseStatsRepository.findDailyStatsByChainAndDateRange(chain, startDate, endDate);
            return ApiResponse.success(stats, (long) stats.size());
        } catch (Exception e) {
            return ApiResponse.error(500, "查询失败: " + e.getMessage());
        }
    }
}