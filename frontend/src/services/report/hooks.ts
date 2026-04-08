// src/api/hooks.ts
/**
 * React Hooks for API operations
 */

import { useState, useCallback } from 'react';
import {
    generateReport,
    getReportDownloadUrl,
    downloadReportFile,
    previewReport,
    deleteReport,
    getReportList,
    getReportListPaginated,
} from './api';
import {
    ReportResponse,
    DownloadReportResponse,
    DeleteReportResponse,
    ReportListItem,
} from './types';

// 状态接口
interface ApiState<T> {
    data: T | null;
    loading: boolean;
    error: string | null;
}

// 初始状态
const initialApiState = <T>(): ApiState<T> => ({
    data: null,
    loading: false,
    error: null,
});

/**
 * 生成报告hook
 */
export const useGenerateReport = () => {
    const [state, setState] = useState<ApiState<ReportResponse>>(initialApiState());

    const generate = useCallback(async (address: string) => {
        setState(prev => ({ ...prev, loading: true, error: null }));

        try {
            const data = await generateReport(address);
            setState({ data, loading: false, error: null });
            return data;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '生成报告失败';
            setState({ data: null, loading: false, error: errorMessage });
            throw error;
        }
    }, []);

    const reset = useCallback(() => {
        setState(initialApiState());
    }, []);

    return {
        ...state,
        generate,
        reset,
    };
};

/**
 * 下载报告hook
 */
export const useDownloadReport = () => {
    const [state, setState] = useState<ApiState<DownloadReportResponse>>(initialApiState());

    const download = useCallback(async (reportId: number) => {
        setState(prev => ({ ...prev, loading: true, error: null }));

        try {
            const data = await getReportDownloadUrl(reportId);
            setState({ data, loading: false, error: null });
            return data;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '获取下载链接失败';
            setState({ data: null, loading: false, error: errorMessage });
            throw error;
        }
    }, []);

    const reset = useCallback(() => {
        setState(initialApiState());
    }, []);

    return {
        ...state,
        download,
        reset,
    };
};

/**
 * 删除报告hook
 */
export const useDeleteReport = () => {
    const [state, setState] = useState<ApiState<DeleteReportResponse>>(initialApiState());

    const remove = useCallback(async (reportId: number) => {
        setState(prev => ({ ...prev, loading: true, error: null }));

        try {
            const data = await deleteReport(reportId);
            setState({ data, loading: false, error: null });
            return data;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '删除报告失败';
            setState({ data: null, loading: false, error: errorMessage });
            throw error;
        }
    }, []);

    const reset = useCallback(() => {
        setState(initialApiState());
    }, []);

    return {
        ...state,
        delete: remove,
        reset,
    };
};

/**
 * 获取报告列表hook
 */
export const useReportList = () => {
    const [state, setState] = useState<ApiState<ReportListItem[]>>(initialApiState());

    const fetch = useCallback(async () => {
        setState(prev => ({ ...prev, loading: true, error: null }));

        try {
            const data = await getReportList();
            setState({ data, loading: false, error: null });
            return data;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '获取报告列表失败';
            setState({ data: null, loading: false, error: errorMessage });
            throw error;
        }
    }, []);

    const reset = useCallback(() => {
        setState(initialApiState());
    }, []);

    return {
        ...state,
        fetch,
        reset,
    };
};

/**
 * 分页获取报告列表hook
 */
export const usePaginatedReportList = (initialPage: number = 1, initialPageSize: number = 10) => {
    const [state, setState] = useState<{
        data: ReportListItem[];
        loading: boolean;
        error: string | null;
        page: number;
        pageSize: number;
        total: number;
    }>({
        data: [],
        loading: false,
        error: null,
        page: initialPage,
        pageSize: initialPageSize,
        total: 0,
    });

    const fetch = useCallback(async (page?: number, pageSize?: number) => {
        const targetPage = page ?? state.page;
        const targetPageSize = pageSize ?? state.pageSize;

        setState(prev => ({ ...prev, loading: true, error: null }));

        try {
            const result = await getReportListPaginated(targetPage, targetPageSize);
            setState({
                data: result.data,
                loading: false,
                error: null,
                page: targetPage,
                pageSize: targetPageSize,
                total: result.total,
            });
            return result;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '获取报告列表失败';
            setState(prev => ({
                ...prev,
                loading: false,
                error: errorMessage,
            }));
            throw error;
        }
    }, [state.page, state.pageSize]);

    const goToPage = useCallback((page: number) => {
        fetch(page, state.pageSize);
    }, [fetch, state.pageSize]);

    const changePageSize = useCallback((pageSize: number) => {
        fetch(1, pageSize);
    }, [fetch]);

    const reset = useCallback(() => {
        setState({
            data: [],
            loading: false,
            error: null,
            page: initialPage,
            pageSize: initialPageSize,
            total: 0,
        });
    }, [initialPage, initialPageSize]);

    return {
        ...state,
        fetch,
        goToPage,
        changePageSize,
        reset,
    };
};