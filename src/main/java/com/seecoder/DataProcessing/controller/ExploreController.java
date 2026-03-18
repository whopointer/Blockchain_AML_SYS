package com.seecoder.DataProcessing.controller;

import com.seecoder.DataProcessing.service.EthereumDataService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import com.seecoder.DataProcessing.vo.ExploreRequest;
import com.seecoder.DataProcessing.vo.ExploreTaskStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequestMapping("/api/explore")
public class ExploreController {

    @Autowired
    private EthereumDataService ethereumDataService;

    // 简单内存存储任务状态（生产环境建议使用Redis）
    private final Map<String, ExploreTaskStatus> taskStatusMap = new ConcurrentHashMap<>();

    @PostMapping("/start")
    public ApiResponse<String> startExplore(@RequestBody ExploreRequest request) {
        String taskId = UUID.randomUUID().toString();
        // 初始化状态
        ExploreTaskStatus status = new ExploreTaskStatus();
        status.setTaskId(taskId);
        status.setStatus("PENDING");
        taskStatusMap.put(taskId, status);

        // 异步执行
        ethereumDataService.exploreAndExport(taskId, request.getSources(),
                        request.getAllowed(), request.getForbidden(),
                        request.getStartTime(), request.getEndTime())
                .thenAccept(response -> {
                    // 可选：更新任务状态（已在 service 中完成）
                });

        // 修正：只传递 taskId，或传递 null 作为 total
        return ApiResponse.success("任务已提交，任务ID：" + taskId, null);
    }
}