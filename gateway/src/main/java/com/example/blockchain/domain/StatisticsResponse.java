package com.example.blockchain.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class StatisticsResponse {
    /** 系统状态 */
    private String system_status;

    /** 模型是否已加载 */
    private boolean model_loaded;

    /** 统计时间戳 */
    private String timestamp;

    /** 系统版本 */
    private String version;
}