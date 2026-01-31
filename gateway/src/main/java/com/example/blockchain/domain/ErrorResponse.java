package com.example.blockchain.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ErrorResponse {
    /** 错误信息 */
    private String error;

    /** 错误时间戳 */
    private String timestamp;
}