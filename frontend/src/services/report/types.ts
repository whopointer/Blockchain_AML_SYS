// src/api/types.ts
/**
 * API类型定义
 */

// 生成报告请求
export interface GenerateReportRequest {
    address: string;
}

// 生成报告响应
export interface ReportResponse {
    id: number;
    created_at: string;
    file_path: string;
    target_address: string;
    title: string;
}

// 下载报告响应
export interface DownloadReportResponse {
    id: number;
    created_at: string;
    target_address: string;
    title: string;
    download_url: string;
}

// 删除报告响应
export interface DeleteReportResponse {
    id: number;
    message: string;
    deleted_at: string;
}

// 报告列表项
export interface ReportListItem {
    id: number;
    created_at: string;
    target_address: string;
    title: string;
}

// API响应包装
export interface ApiResponse<T> {
    data: T;
    success: boolean;
    message?: string;
    timestamp: string;
}

// 错误响应
export interface ApiError {
    detail: string;
    message: string;
    timestamp?: string;
}