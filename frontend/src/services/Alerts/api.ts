/**
 * 告警记录API接口
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
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

// 创建axios实例
const createApiInstance = (): AxiosInstance => {
    const instance = axios.create({
        baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:7999',
        timeout: 30000,
        headers: {
            'Content-Type': 'application/json',
        },
    });

    // 请求拦截器
    instance.interceptors.request.use(
        (config) => {
            // 可以在这里添加认证token
            const token = localStorage.getItem('token');
            if (token) {
                config.headers.Authorization = `Bearer ${token}`;
            }
            return config;
        },
        (error) => {
            return Promise.reject(error);
        }
    );

    // 响应拦截器
    instance.interceptors.response.use(
        (response) => response,
        (error) => {
            console.error('API Error:', error.response || error.message);

            // 处理常见HTTP错误
            if (error.response) {
                switch (error.response.status) {
                    case 401:
                        // 未授权，跳转到登录
                        window.location.href = '/login';
                        break;
                    case 403:
                        // 权限不足
                        console.error('权限不足');
                        break;
                    case 404:
                        // 资源不存在
                        console.error('资源不存在');
                        break;
                    case 500:
                        // 服务器错误
                        console.error('服务器内部错误');
                        break;
                    default:
                        console.error(`请求失败: ${error.response.status}`);
                }
            } else if (error.request) {
                // 请求已发送但没有响应
                console.error('服务器无响应，请检查网络连接');
            } else {
                // 请求配置错误
                console.error('请求配置错误:', error.message);
            }

            return Promise.reject(error);
        }
    );

    return instance;
};

// API实例
const api = createApiInstance();

/**
 * 告警记录API
 */
export const alertsApi = {
    /**
     * 监控地址并生成告警
     */
    async monitorAddress(request: MonitorAddressRequest): Promise<MonitorResponse> {
        const response: AxiosResponse<MonitorResponse> = await api.post(
            '/api/alerts/monitor',
            request
        );
        return response.data;
    },

    /**
     * 批量监控地址
     */
    async batchMonitorAddresses(request: BatchMonitorRequest): Promise<BatchMonitorResponse> {
        const response: AxiosResponse<BatchMonitorResponse> = await api.post(
            '/api/alerts/monitor/batch',
            request
        );
        return response.data;
    },

    /**
     * 获取告警列表
     */
    async getAlerts(params?: AlertQueryParams): Promise<AlertListResponse> {
        const queryParams = new URLSearchParams();

        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined && value !== null) {
                    queryParams.append(key, value.toString());
                }
            });
        }

        const response: AxiosResponse<AlertListResponse> = await api.get(
            `/api/alerts?${queryParams.toString()}`
        );
        return response.data;
    },

    /**
     * 获取告警详情
     */
    async getAlertById(alertId: string): Promise<Alert> {
        const response: AxiosResponse<Alert> = await api.get(`/api/alerts/${alertId}`);
        return response.data;
    },

    /**
     * 更新告警状态
     */
    async updateAlertStatus(
        alertId: string,
        request: UpdateAlertStatusRequest
    ): Promise<Alert> {
        const response: AxiosResponse<Alert> = await api.put(
            `/api/alerts/${alertId}/status`,
            request
        );
        return response.data;
    },

    /**
     * 删除告警
     */
    async deleteAlert(alertId: string): Promise<void> {
        await api.delete(`/api/alerts/${alertId}`);
    },

    /**
     * 获取告警统计
     */
    async getAlertStatistics(params?: { address?: string; hours?: number }): Promise<AlertStatistics> {
        const queryParams = new URLSearchParams();

        if (params?.address) {
            queryParams.append('address', params.address);
        }

        if (params?.hours) {
            queryParams.append('hours', params.hours.toString());
        }

        const response: AxiosResponse<AlertStatistics> = await api.get(
            `/api/alerts/statistics/summary?${queryParams.toString()}`
        );
        return response.data;
    },

    /**
     * 获取地址的告警列表
     */
    async getAlertsByAddress(
        address: string,
        params?: Omit<AlertQueryParams, 'address'>
    ): Promise<AlertListResponse> {
        return this.getAlerts({ ...params, address });
    },

    /**
     * 获取待处理告警
     */
    async getPendingAlerts(params?: Omit<AlertQueryParams, 'status'>): Promise<AlertListResponse> {
        return this.getAlerts({ ...params, status: 'PENDING' });
    },
};