/**
 * 告警记录相关React Hooks
 */

import { useState, useCallback, useEffect } from 'react';
import { alertsApi } from './api';
import {
    Alert,
    AlertListResponse,
    AlertStatistics,
    MonitorAddressRequest,
    BatchMonitorRequest,
    MonitorResponse,
    BatchMonitorResponse,
    UpdateAlertStatusRequest,
    AlertQueryParams
} from './types';

interface UseAlertsOptions {
    autoFetch?: boolean;
    initialParams?: AlertQueryParams;
}

/**
 * 获取告警列表的Hook
 */
export function useAlerts(options: UseAlertsOptions = {}) {
    const { autoFetch = false, initialParams } = options;

    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [total, setTotal] = useState(0);
    const [pendingCount, setPendingCount] = useState(0);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [params, setParams] = useState<AlertQueryParams | undefined>(initialParams);

    const fetchAlerts = useCallback(async (queryParams?: AlertQueryParams) => {
        setLoading(true);
        setError(null);

        try {
            const response = await alertsApi.getAlerts(queryParams || params);
            setAlerts(response.alerts);
            setTotal(response.total);
            setPendingCount(response.pending_count);
            return response;
        } catch (err: any) {
            const errorMessage = err.response?.data?.message || err.message || '获取告警列表失败';
            setError(errorMessage);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [params]);

    // 自动获取
    useEffect(() => {
        if (autoFetch) {
            fetchAlerts();
        }
    }, [autoFetch, fetchAlerts]);

    return {
        alerts,
        total,
        pendingCount,
        loading,
        error,
        params,
        setParams,
        fetchAlerts,
        refresh: () => fetchAlerts(params),
    };
}

/**
 * 监控地址的Hook
 */
export function useAddressMonitor() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<MonitorResponse | null>(null);

    const monitorAddress = useCallback(async (request: MonitorAddressRequest) => {
        setLoading(true);
        setError(null);

        try {
            const response = await alertsApi.monitorAddress(request);
            setResult(response);
            return response;
        } catch (err: any) {
            const errorMessage = err.response?.data?.message || err.message || '监控地址失败';
            setError(errorMessage);
            throw err;
        } finally {
            setLoading(false);
        }
    }, []);

    const reset = useCallback(() => {
        setLoading(false);
        setError(null);
        setResult(null);
    }, []);

    return {
        monitorAddress,
        result,
        loading,
        error,
        reset,
    };
}

/**
 * 批量监控地址的Hook
 */
export function useBatchMonitor() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<BatchMonitorResponse | null>(null);

    const batchMonitor = useCallback(async (request: BatchMonitorRequest) => {
        setLoading(true);
        setError(null);

        try {
            const response = await alertsApi.batchMonitorAddresses(request);
            setResult(response);
            return response;
        } catch (err: any) {
            const errorMessage = err.response?.data?.message || err.message || '批量监控失败';
            setError(errorMessage);
            throw err;
        } finally {
            setLoading(false);
        }
    }, []);

    const reset = useCallback(() => {
        setLoading(false);
        setError(null);
        setResult(null);
    }, []);

    return {
        batchMonitor,
        result,
        loading,
        error,
        reset,
    };
}

/**
 * 更新告警状态的Hook
 */
export function useUpdateAlertStatus(alertId?: string) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [alert, setAlert] = useState<Alert | null>(null);

    const updateStatus = useCallback(async (
        request: UpdateAlertStatusRequest,
        id?: string
    ) => {
        setLoading(true);
        setError(null);

        const targetAlertId = id || alertId;
        if (!targetAlertId) {
            throw new Error('告警ID不能为空');
        }

        try {
            const response = await alertsApi.updateAlertStatus(targetAlertId, request);
            setAlert(response);
            return response;
        } catch (err: any) {
            const errorMessage = err.response?.data?.message || err.message || '更新告警状态失败';
            setError(errorMessage);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [alertId]);

    const reset = useCallback(() => {
        setLoading(false);
        setError(null);
        setAlert(null);
    }, []);

    return {
        updateStatus,
        alert,
        loading,
        error,
        reset,
    };
}

/**
 * 获取告警详情的Hook
 */
export function useAlertDetail(alertId?: string) {
    const [alert, setAlert] = useState<Alert | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchAlert = useCallback(async (id?: string) => {
        setLoading(true);
        setError(null);

        const targetAlertId = id || alertId;
        if (!targetAlertId) {
            setError('告警ID不能为空');
            setLoading(false);
            return null;
        }

        try {
            const response = await alertsApi.getAlertById(targetAlertId);
            setAlert(response);
            return response;
        } catch (err: any) {
            const errorMessage = err.response?.data?.message || err.message || '获取告警详情失败';
            setError(errorMessage);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [alertId]);

    // 自动获取
    useEffect(() => {
        if (alertId) {
            fetchAlert();
        }
    }, [alertId, fetchAlert]);

    return {
        alert,
        loading,
        error,
        fetchAlert,
        refresh: () => fetchAlert(alertId),
    };
}

/**
 * 获取告警统计的Hook
 */
export function useAlertStatistics(params?: { address?: string; hours?: number }) {
    const [statistics, setStatistics] = useState<AlertStatistics | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchStatistics = useCallback(async (queryParams?: { address?: string; hours?: number }) => {
        setLoading(true);
        setError(null);

        try {
            const response = await alertsApi.getAlertStatistics(queryParams || params);
            setStatistics(response);
            return response;
        } catch (err: any) {
            const errorMessage = err.response?.data?.message || err.message || '获取告警统计失败';
            setError(errorMessage);
            throw err;
        } finally {
            setLoading(false);
        }
    }, [params]);

    // 自动获取
    useEffect(() => {
        fetchStatistics();
    }, [fetchStatistics]);

    return {
        statistics,
        loading,
        error,
        fetchStatistics,
        refresh: () => fetchStatistics(params),
    };
}