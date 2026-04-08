// src/api/reportApi.ts
/**
 * 报告管理API接口
 */

import {apiClient} from './client';
import {
    ApiError,
    DeleteReportResponse,
    DownloadReportResponse,
    GenerateReportRequest,
    ReportListItem,
    ReportResponse,
} from './types';

// API端点
const ENDPOINTS = {
    REPORTS: '/api/reports',
    DOWNLOAD: (id: number) => `/api/reports/${id}/download`,
    DELETE: (id: number) => `/api/reports/${id}`,
} as const;

/**
 * 生成报告
 * @param address 区块链地址
 * @returns 报告信息
 */
export const generateReport = async (
    address: string
): Promise<ReportResponse> => {
    try {
        const request: GenerateReportRequest = { address };

        const response = await apiClient.post<ReportResponse>(
            ENDPOINTS.REPORTS,
            request
        );

        return response.data;
    } catch (error) {
        if (error.response?.data) {
            const apiError = error.response.data as ApiError;
            throw new Error(apiError.detail || apiError.message || '生成报告失败');
        }
        throw error;
    }
};

/**
 * 获取报告下载URL
 * @param reportId 报告ID
 * @returns 下载报告响应
 */
export const getReportDownloadUrl = async (
    reportId: number
): Promise<DownloadReportResponse> => {
    try {
        if (!reportId || reportId <= 0) {
            throw new Error('报告ID无效');
        }

        const response = await apiClient.get<DownloadReportResponse>(
            ENDPOINTS.DOWNLOAD(reportId)
        );

        return response.data;
    } catch (error) {
        if (error.response?.data) {
            const apiError = error.response.data as ApiError;
            throw new Error(apiError.detail || apiError.message || '获取下载链接失败');
        }
        throw error;
    }
};

/**
 * 下载报告文件
 * @param reportId 报告ID
 * @returns 下载的Blob对象
 */
export const downloadReportFile = async (
    reportId: number
): Promise<Blob> => {
    try {
        // 先获取预签名URL
        const { download_url } = await getReportDownloadUrl(reportId);

        // 使用预签名URL下载文件
        const response = await apiClient.get(download_url, {
            responseType: 'blob',
        });

        return response.data;
    } catch (error) {
        if (error.response?.data) {
            const apiError = (error).data as ApiError;
            throw new Error(apiError.detail || apiError.message || '下载报告失败');
        }
        throw error;
    }
};

/**
 * 在线预览报告
 * @param reportId 报告ID
 */
export const previewReport = async (reportId: number): Promise<void> => {
    try {
        const { download_url } = await getReportDownloadUrl(reportId);

        // 在新标签页打开
        window.open(download_url, '_blank');
    } catch (error) {
        if (error.response?.data) {
            const apiError = error.response.data as ApiError;
            throw new Error(apiError.detail || apiError.message || '预览报告失败');
        }
        throw error;
    }
};

/**
 * 删除报告
 * @param reportId 报告ID
 * @returns 删除响应
 */
export const deleteReport = async (
    reportId: number
): Promise<DeleteReportResponse> => {
    try {
        if (!reportId || reportId <= 0) {
            throw new Error('报告ID无效');
        }

        const response = await apiClient.delete<DeleteReportResponse>(
            ENDPOINTS.DELETE(reportId)
        );

        return response.data;
    } catch (error) {
        if (error.response?.data) {
            const apiError = error.response.data as ApiError;
            throw new Error(apiError.detail || apiError.message || '删除报告失败');
        }
        throw error;
    }
};

/**
 * 获取所有报告列表
 * @returns 报告列表
 */
export const getReportList = async (): Promise<ReportListItem[]> => {
    try {
        const response = await apiClient.get<ReportListItem[]>(
            ENDPOINTS.REPORTS
        );

        return response.data;
    } catch (error) {
        if (error.response?.data) {
            const apiError = error.response.data as ApiError;
            throw new Error(apiError.detail || apiError.message || '获取报告列表失败');
        }
        throw error;
    }
};

/**
 * 分页获取报告列表
 * @param page 页码（从1开始）
 * @param pageSize 每页大小
 * @returns 分页报告列表
 */
export const getReportListPaginated = async (
    page: number = 1,
    pageSize: number = 10
): Promise<{
    data: ReportListItem[];
    page: number;
    pageSize: number;
    total: number;
}> => {
    try {
        const allReports = await getReportList();

        // 前端分页（如果后端支持分页，应修改为后端分页）
        const start = (page - 1) * pageSize;
        const end = start + pageSize;
        const paginatedData = allReports.slice(start, end);

        return {
            data: paginatedData,
            page,
            pageSize,
            total: allReports.length,
        };
    } catch (error) {
        if (error.response?.data) {
            const apiError = error.response.data as ApiError;
            throw new Error(apiError.detail || apiError.message || '获取报告列表失败');
        }
        throw error;
    }
};