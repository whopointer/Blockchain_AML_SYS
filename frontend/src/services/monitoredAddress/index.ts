import axios, { AxiosResponse } from "axios";
import {
    MonitoredAddressDTO,
    MonitoredAddress,
    CreateMonitoredAddressRequest,
    ApiResponse
} from '../../components/MonitoredAddresses/types';

// 创建 axios 实例
export const apiClient = axios.create({
    baseURL: 'http://localhost:8079/api',
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// 请求拦截器（可用于添加认证token）
apiClient.interceptors.request.use(
    (config) => {
        // 可以从 localStorage 获取 token
        const token = localStorage.getItem('access_token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// 响应拦截器（统一处理错误）
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        console.error('API请求错误:', error);

        if (error.response) {
            // 服务器返回错误状态码
            switch (error.response.status) {
                case 403:
                    console.error('权限不足');
                    break;
                case 500:
                    console.error('服务器内部错误');
                    break;
                default:
                    console.error(`请求失败: ${error.response.status}`);
            }
        } else if (error.request) {
            // 请求发送但无响应
            console.error('网络错误，请检查网络连接');
        } else {
            // 请求配置错误
            console.error('请求配置错误:', error.message);
        }

        return Promise.reject(error);
    }
);

// 类型转换函数
export const convertDTOToAddress = (dto: MonitoredAddressDTO): MonitoredAddress => ({
    ...dto,
    createdAt: new Date(dto.createdAt),
    updatedAt: new Date(dto.updatedAt),
});

export const convertAddressToDTO = (address: MonitoredAddress): MonitoredAddressDTO => ({
    ...address,
    createdAt: address.createdAt.toISOString(),
    updatedAt: address.updatedAt.toISOString(),
});

// 地址相关 API
export const monitoredAddressApi = {
    /**
     * 获取所有监控地址
     */
    async getAll(): Promise<MonitoredAddress[]> {
        try {
            const response: AxiosResponse<ApiResponse<MonitoredAddressDTO[]>> =
                await apiClient.get('/monitored-addresses');

            if (response.data.success) {
                return response.data.data.map(convertDTOToAddress);
            } else {
                throw new Error(response.data.message);
            }
        } catch (error) {
            console.error('获取地址列表失败:', error);
            throw error;
        }
    },

    /**
     * 获取单个监控地址
     */
    async getById(id: string): Promise<MonitoredAddress> {
        try {
            const response: AxiosResponse<ApiResponse<MonitoredAddressDTO>> =
                await apiClient.get(`/monitored-addresses/${id}`);

            if (response.data.success) {
                return convertDTOToAddress(response.data.data);
            } else {
                throw new Error(response.data.message);
            }
        } catch (error) {
            console.error(`获取地址 ${id} 失败:`, error);
            throw error;
        }
    },

    /**
     * 添加监控地址
     */
    async create(addressData: CreateMonitoredAddressRequest): Promise<MonitoredAddress> {
        try {
            const response: AxiosResponse<ApiResponse<MonitoredAddressDTO>> =
                await apiClient.post('/monitored-addresses', addressData);

            if (response.data.success) {
                return convertDTOToAddress(response.data.data);
            } else {
                throw new Error(response.data.message);
            }
        } catch (error) {
            console.error('添加地址失败:', error);
            throw error;
        }
    },

    /**
     * 更新监控地址
     */
    async update(id: string, addressData: Partial<CreateMonitoredAddressRequest>): Promise<MonitoredAddress> {
        try {
            const response: AxiosResponse<ApiResponse<MonitoredAddressDTO>> =
                await apiClient.put(`/monitored-addresses/${id}`, addressData);

            if (response.data.success) {
                return convertDTOToAddress(response.data.data);
            } else {
                throw new Error(response.data.message);
            }
        } catch (error) {
            console.error(`更新地址 ${id} 失败:`, error);
            throw error;
        }
    },

    /**
     * 删除监控地址
     */
    async delete(id: string): Promise<boolean> {
        try {
            const response: AxiosResponse<ApiResponse<void>> =
                await apiClient.delete(`/monitored-addresses/${id}`);

            return response.data.success;
        } catch (error) {
            console.error(`删除地址 ${id} 失败:`, error);
            throw error;
        }
    },

    /**
     * 批量删除监控地址
     */
    async deleteBatch(ids: string[]): Promise<boolean> {
        try {
            const response: AxiosResponse<ApiResponse<void>> =
                await apiClient.delete('/monitored-addresses/batch', {
                    data: { ids },
                });

            return response.data.success;
        } catch (error) {
            console.error('批量删除地址失败:', error);
            throw error;
        }
    },
};

export default monitoredAddressApi;