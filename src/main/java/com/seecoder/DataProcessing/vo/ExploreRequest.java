package com.seecoder.DataProcessing.vo;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class ExploreRequest {
    private List<String> sources;
    private List<String> allowed;
    private List<String> forbidden;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
}