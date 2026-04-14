// services/report/useReportOperations.ts
import { useState, useCallback } from 'react';
import { deleteReport as deleteReportApi } from './index';
import { message } from 'antd';

export const useReportOperations = () => {
    const [deletingIds, setDeletingIds] = useState<Set<number>>(new Set());
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const deleteReport = useCallback(async (reportId: number) => {
        setDeletingIds(prev => {
            const newSet = new Set(prev);
            newSet.add(reportId);
            return newSet;
        });

        setError(null);
        setSuccess(null);

        try {
            const result = await deleteReportApi(reportId);

            if (result.success) {
                setSuccess('报告删除成功');
                message.success('报告删除成功');
            } else {
                setError(result.message || '删除失败');
                message.error(result.message || '删除失败');
            }

            return result;
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : '删除报告失败';
            setError(errorMessage);
            message.error(errorMessage);
            throw err;
        } finally {
            setDeletingIds(prev => {
                const newSet = new Set(prev);
                newSet.delete(reportId);
                return newSet;
            });
        }
    }, []);

    const isDeleting = useCallback((reportId: number) => {
        return deletingIds.has(reportId);
    }, [deletingIds]);

    const clearStatus = useCallback(() => {
        setError(null);
        setSuccess(null);
    }, []);

    return {
        delete: deleteReport,
        isDeleting,
        error,
        success,
        clearStatus,
    };
};