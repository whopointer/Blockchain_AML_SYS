package com.seecoder.DataProcessing.vo;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class ExploreTaskStatus {
    private String taskId;
    private String status;          // PENDING, RUNNING, COMPLETED, FAILED
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private int processedAddresses;
    private int progress;
    private String message;
    private List<String> result;
}