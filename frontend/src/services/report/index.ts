// services/report/index.ts
import { ReportAPI } from './api';
import { mapBackendToFrontend, FrontendReport, ReportResponse } from './types';

// 创建API实例
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:7999/api';
console.log('API Base URL:', API_BASE_URL);

export const reportApi = new ReportAPI(API_BASE_URL);

// 获取报告列表
export const getReportList = async (): Promise<FrontendReport[]> => {
    try {
        console.log('开始获取报告列表...');
        const response = await reportApi.listReports();
        console.log('API响应:', response);

        if (response.success) {
            const reports = response.data?.reports || [];
            console.log('处理后的报告数据:', reports);

            // 映射后端数据到前端格式
            return reports.map(mapBackendToFrontend);
        } else {
            console.warn('API返回失败:', response.message);
            return [];
        }
    } catch (error) {
        console.error('获取报告列表失败:', error);
        throw error;
    }
};

// 预览报告
export const previewReport = async (
    reportId: number,
    options?: {
        method?: 'redirect' | 'embed' | 'direct' | 'blob';
        openInNewTab?: boolean;
        filename?: string;
    }
): Promise<any> => {
    try {
        console.log(`预览报告: ID=${reportId}, 选项:`, options);

        const {
            method = 'redirect',
            openInNewTab = true,
            filename
        } = options || {};

        // 构建代理端点URL
        const proxyUrl = `http://localhost:7999/api/reports/view/${reportId}`;
        console.log('预览请求URL:', proxyUrl);

        if (method === 'redirect') {
            // 重定向模式：直接在新窗口打开
            if (openInNewTab) {
                window.open(proxyUrl, '_blank');
                return {
                    success: true,
                    message: '在新标签页打开预览',
                    url: proxyUrl
                };
            } else {
                window.location.href = proxyUrl;
                return {
                    success: true,
                    message: '在当前窗口打开预览',
                    url: proxyUrl
                };
            }
        } else if (method === 'blob') {
            // Blob模式：获取PDF并创建Blob URL
            const response = await fetch(proxyUrl);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('预览响应状态:', response.status, '错误信息:', errorText);
                throw new Error(`预览失败: ${response.status} ${response.statusText}`);
            }

            const blob = await response.blob();
            const blobUrl = URL.createObjectURL(blob);

            console.log('创建Blob URL成功:', blobUrl);

            if (openInNewTab) {
                window.open(blobUrl, '_blank');
            }

            return {
                success: true,
                blobUrl: blobUrl,
                filename: filename || `report_${reportId}.pdf`,
                originalUrl: proxyUrl
            };
        } else if (method === 'embed' || method === 'direct') {
            // 嵌入或直接模式：返回代理URL
            if (openInNewTab) {
                window.open(proxyUrl, '_blank');
            }

            return {
                success: true,
                url: proxyUrl,
                filename: filename || `report_${reportId}.pdf`,
                method: method
            };
        } else {
            throw new Error(`不支持的预览方法: ${method}`);
        }

    } catch (error: any) {
        console.error('预览报告异常:', error);
        throw error;
    }
};

// 下载报告文件
export const downloadReportFile = async (reportId: number): Promise<Blob> => {
    try {
        const { url } = await reportApi.downloadReport(reportId);

        // 通过URL获取Blob
        const response = await fetch(url);
        const blob = await response.blob();

        // 清理URL
        URL.revokeObjectURL(url);

        return blob;
    } catch (error) {
        console.error('下载报告失败:', error);
        throw error;
    }
};

// 删除报告
export const deleteReport = async (reportId: number): Promise<{ success: boolean; message: string }> => {
    try {
        return await reportApi.deleteReport(reportId);
    } catch (error) {
        console.error('删除报告失败:', error);
        throw error;
    }
};

// 生成报告
export const generateReport = async (
    address: string,
    options?: {
        type?: 'basic' | 'enhanced';
        includePredictions?: boolean;
    }
): Promise<ReportResponse> => {
    return await reportApi.generateReport(address, options);
};

// 获取报告状态
export const getReportStatus = async (reportId: number): Promise<any> => {
    return await reportApi.getReportStatus(reportId);
};

// 工具函数
export const isValidEthereumAddress = (address: string): boolean => {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
};

export const formatReportDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    }).replace(/\//g, '-');
};