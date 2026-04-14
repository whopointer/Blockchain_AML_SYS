import { useState, useEffect, useCallback } from 'react';
import {ReportAPI} from './api'
import {ReportResponse, ReportStatus} from "@/services/report/types";

export function useReportGenerator(api: ReportAPI) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [report, setReport] = useState<ReportResponse | null>(null);
    const [status, setStatus] = useState<ReportStatus | null>(null);

    const generate = useCallback(async (
        address: string,
        options?: { type?: 'basic' | 'enhanced' }
    ) => {
        setLoading(true);
        setError(null);

        try {
            const result = await api.generateReport(address, options);
            setReport(result);
            return result;
        } catch (err) {
            setError(err instanceof Error ? err.message : '未知错误');
            throw err;
        } finally {
            setLoading(false);
        }
    }, [api]);

    const pollStatus = useCallback(async (reportId: number) => {
        try {
            const status = await api.getReportStatus(reportId);
            setStatus(status);
            return status;
        } catch (err) {
            setError(err instanceof Error ? err.message : '获取状态失败');
            throw err;
        }
    }, [api]);

    return {
        loading,
        error,
        report,
        status,
        generate,
        pollStatus,
    };
}

// utils.ts - 工具函数
export function validateAddress(address: string): boolean {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
}

export function formatReportDate(dateString: string): string {
    return new Date(dateString).toLocaleString();
}

export function getReportStatusColor(status: string): string {
    const colors: Record<string, string> = {
        pending: 'orange',
        processing: 'blue',
        completed: 'green',
        failed: 'red',
    };
    return colors[status] || 'gray';
}

// 使用示例
async function exampleUsage() {
    const api = new ReportAPI();

    try {
        // 1. 生成报告
        const report = await api.generateReport('0x1234...', {
            type: 'enhanced',
            includePredictions: true,
        });

        console.log('报告已创建:', report.report_id);

        // 2. 轮询状态
        const status = await api.pollReportStatus(report.report_id);

        if (status.status === 'completed') {
            // 3. 下载报告
            const { url, filename } = await api.downloadReport(report.report_id);

            // 创建下载链接
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();

            // 清理URL
            URL.revokeObjectURL(url);
        }

    } catch (error) {
        console.error('操作失败:', error);
    }
}