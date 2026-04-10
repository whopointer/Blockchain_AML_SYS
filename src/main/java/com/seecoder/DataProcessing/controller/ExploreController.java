package com.seecoder.DataProcessing.controller;

import com.seecoder.DataProcessing.service.EthereumDataService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import com.seecoder.DataProcessing.vo.ExploreRequest;
import com.seecoder.DataProcessing.vo.ExploreTaskStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;

@RestController
@RequestMapping("/api/explore")
public class ExploreController {

    private static final Logger log = LoggerFactory.getLogger(ExploreController.class);

    @Autowired
    private EthereumDataService ethereumDataService;

    private final Map<String, ExploreTaskStatus> taskStatusMap = new ConcurrentHashMap<>();

    @PostMapping("/start")
    public ApiResponse<Map<String, Object>> startExplore(@RequestBody ExploreRequest request) {
        String taskId = UUID.randomUUID().toString();
        // 可选：记录任务状态
        ExploreTaskStatus status = new ExploreTaskStatus();
        status.setTaskId(taskId);
        status.setStatus("PENDING");
        taskStatusMap.put(taskId, status);

        try {
            // 同步等待异步任务完成
            ApiResponse<Map<String, Object>> response = ethereumDataService.exploreAndExport(
                    taskId,
                    request.getSources(),
                    request.getAllowed(),
                    request.getForbidden(),
                    request.getStartTime(),
                    request.getEndTime()
            ).get();

            return response;
        } catch (InterruptedException | ExecutionException e) {
            log.error("探索任务执行异常", e);
            return ApiResponse.error(500, "探索任务执行失败: " + e.getMessage());
        }
    }
}