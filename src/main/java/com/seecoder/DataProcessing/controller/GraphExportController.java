package com.seecoder.DataProcessing.controller;

import com.seecoder.DataProcessing.service.GraphExportService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;

@RestController
@RequestMapping("/api/graph")
public class GraphExportController {

    @Autowired
    private GraphExportService graphExportService;

    @GetMapping("/export")
    public ApiResponse<String> exportGraph(
            @RequestParam String chain,
            @RequestParam(required = false) Long startHeight,
            @RequestParam(required = false) Long endHeight,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(required = false) String edgesFilePath,
            @RequestParam(required = false) String classesFilePath) {
        return graphExportService.exportGraph(chain, startHeight, endHeight,
                startTime, endTime, edgesFilePath, classesFilePath);
    }

    @PostMapping("/refresh-blacklist")
    public ApiResponse<Void> refreshBlackWhitelist() {
        return graphExportService.refreshBlackWhitelist();
    }
}