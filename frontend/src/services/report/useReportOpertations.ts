// src/api/useReportOperations.ts
/**
 * 报告操作组合hook
 */

import { useState, useCallback } from 'react';
import { generateReport, deleteReport, previewReport } from './reportApi';
import { ReportResponse, DeleteReportResponse, ReportListItem } from './types';

/**
 * 报告操作组合hook
 */
export const useReportOperations = () => {
    const [generating, setGenerating] = useState(false);
    const [deleting, setDeleting] = useState<number | null>(null);
    const [previewing, setPreviewing] = useState<number | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    // 清空状态
    const clearStatus = useCallback(() => {
        setError(null);
        setSuccess(null);
    }, []);

    // 生成报告
    const handleGenerateReport = useCallback(async (
        address: string,
        onSuccess?: (report: ReportResponse) => void
    ) => {
        setGenerating(true);
        clearStatus();

        try {
            const report = await generateReport(address);
            setSuccess(`报告生成成功: ${report.title}`);

            if (onSuccess) {
                onSuccess(report);
            }

            return report;
        } catch (err) {
            const message = err instanceof Error ? err.message : '生成报告失败';
            setError(message);
            throw err;
        } finally {
            setGenerating(false);
        }
    }, [clearStatus]);

    // 删除报告
    const handleDeleteReport = useCallback(async (
        reportId: number,
        onSuccess?: (result: DeleteReportResponse) => void
    ) => {
        setDeleting(reportId);
        clearStatus();

        try {
            const result = await deleteReport(reportId);
            setSuccess(`报告删除成功: ID ${reportId}`);

            if (onSuccess) {
                onSuccess(result);
            }

            return result;
        } catch (err) {
            const message = err instanceof Error ? err.message : '删除报告失败';
            setError(message);
            throw err;
        } finally {
            setDeleting(null);
        }
    }, [clearStatus]);

    // 批量删除报告
    const handleBatchDeleteReports = useCallback(async (
        reportIds: number[],
        onSuccess?: () => void
    ) => {
        clearStatus();

        const results = [];
        const errors = [];

        for (const reportId of reportIds) {
            try {
                setDeleting(reportId);
                const result = await deleteReport(reportId);
                results.push(result);
            } catch (err) {
                errors.push({
                    id: reportId,
                    error: err instanceof Error ? err.message : '删除失败',
                });
            } finally {
                setDeleting(null);
            }
        }

        if (errors.length > 0) {
            const errorMessage = `部分删除失败: ${errors.map(e => `ID ${e.id}`).join(', ')}`;
            setError(errorMessage);
        }

        if (results.length > 0) {
            setSuccess(`成功删除 ${results.length} 个报告`);
        }

        if (onSuccess && results.length > 0) {
            onSuccess();
        }

        return { results, errors };
    }, [clearStatus]);

    // 预览报告
    const handlePreviewReport = useCallback(async (
        reportId: number,
        onSuccess?: () => void
    ) => {
        setPreviewing(reportId);
        clearStatus();

        try {
            await previewReport(reportId);

            if (onSuccess) {
                onSuccess();
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : '预览报告失败';
            setError(message);
            throw err;
        } finally {
            setPreviewing(null);
        }
    }, [clearStatus]);

    return {
        // 状态
        generating,
        deleting,
        previewing,
        error,
        success,

        // 操作
        generateReport: handleGenerateReport,
        deleteReport: handleDeleteReport,
        batchDeleteReports: handleBatchDeleteReports,
        previewReport: handlePreviewReport,
        clearStatus,

        // 工具状态检查
        isDeleting: (reportId: number) => deleting === reportId,
        isPreviewing: (reportId: number) => previewing === reportId,
    };
};