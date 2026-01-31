package com.example.blockchain.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class HealthResponse {
    /** 服务状态 */
    private String status;

    /** 检查时间戳 */
    private String timestamp;

    /** 模型是否已加载 */
    private boolean model_loaded;

    /** 系统版本 */
    private String version;
}