// src/api/client.ts
/**
 * API客户端配置
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';

// 基础配置
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30秒超时

// 创建axios实例
const createAxiosInstance = (): AxiosInstance => {
    const instance = axios.create({
        baseURL: API_BASE_URL,
        timeout: API_TIMEOUT,
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
    });

    // 请求拦截器
    instance.interceptors.request.use(
        (config) => {
            // 这里可以添加认证token等
            // const token = localStorage.getItem('token');
            // if (token) {
            //   config.headers.Authorization = `Bearer ${token}`;
            // }
            return config;
        },
        (error) => {
            return Promise.reject(error);
        }
    );

    // 响应拦截器
    instance.interceptors.response.use(
        (response) => {
            return response;
        },
        (error: AxiosError) => {
            // 统一错误处理
            if (error.response) {
                const { status, data } = error.response;

                switch (status) {
                    case 400:
                        console.error('请求参数错误:', data);
                        break;
                    case 401:
                        console.error('未授权，请重新登录');
                        // 可以在这里触发登出逻辑
                        break;
                    case 403:
                        console.error('拒绝访问');
                        break;
                    case 404:
                        console.error('资源不存在');
                        break;
                    case 500:
                        console.error('服务器内部错误');
                        break;
                    default:
                        console.error('请求失败:', error.message);
                }
            } else if (error.request) {
                console.error('网络错误，请检查网络连接');
            } else {
                console.error('请求配置错误:', error.message);
            }

            return Promise.reject(error);
        }
    );

    return instance;
};

export const apiClient = createAxiosInstance();