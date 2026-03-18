package com.seecoder.DataProcessing.service;

import java.time.LocalDate;

public interface ClickHouseAggregationService {
    void aggregateDailyStats(LocalDate date, String chain);
}