package com.seecoder.DataProcessing.controller;

import com.seecoder.DataProcessing.service.FeatureExportService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/features")
public class FeatureExportController {

    @Autowired
    @Qualifier("bitcoinFeatureExportService")
    private FeatureExportService bitcoinService;

    @Autowired
    @Qualifier("ethereumFeatureExportService")
    private FeatureExportService ethereumService;

    private final Map<String, FeatureExportService> serviceMap = new HashMap<>();

    @Autowired
    public void initServiceMap() {
        serviceMap.put("BTC", bitcoinService);
        serviceMap.put("ETH", ethereumService);
    }

    @PostMapping("/export")
    public ApiResponse<String> exportFeatures(
            @RequestParam String chain,
            @RequestParam(required = false) Long startHeight,
            @RequestParam(required = false) Long endHeight,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(required = false) String filePath) {   // 改为可选
        FeatureExportService service = serviceMap.get(chain.toUpperCase());
        if (service == null) {
            return ApiResponse.error(400, "不支持的链: " + chain);
        }
        return service.exportFeatures(chain, startHeight, endHeight, startTime, endTime, filePath);
    }
}