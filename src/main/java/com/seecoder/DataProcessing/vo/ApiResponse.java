// com/seecoder/DataProcessing/vo/ApiResponse.java
package com.seecoder.DataProcessing.vo;

import lombok.Data;

@Data
public class ApiResponse<T> {
    private boolean success;
    private String message;
    private T data;
    private Long total;
    private Integer code;

    // 添加这个getter方法
    public T getData() {
        return data;
    }

    // 静态工厂方法
    public static <T> ApiResponse<T> success(T data, Long total) {
        ApiResponse<T> response = new ApiResponse<>();
        response.setSuccess(true);
        response.setCode(200);
        response.setMessage("success");
        response.setData(data);
        response.setTotal(total);
        return response;
    }

    public static <T> ApiResponse<T> error(Integer code, String message) {
        ApiResponse<T> response = new ApiResponse<>();
        response.setSuccess(false);
        response.setCode(code);
        response.setMessage(message);
        return response;
    }

    // 添加isSuccess方法
    public boolean isSuccess() {
        return success;
    }
}