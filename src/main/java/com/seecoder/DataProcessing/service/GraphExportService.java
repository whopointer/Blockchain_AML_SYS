package com.seecoder.DataProcessing.service;

import com.seecoder.DataProcessing.vo.ApiResponse;

import java.time.LocalDateTime;

public interface GraphExportService {
    ApiResponse<String> exportGraph(String chain,
                                    Long startHeight, Long endHeight,
                                    LocalDateTime startTime, LocalDateTime endTime,
                                    String edgesFilePath, String classesFilePath);
    ApiResponse<Void> refreshBlackWhitelist();
}