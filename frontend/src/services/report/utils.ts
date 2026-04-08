// src/api/utils.ts
/**
 * API工具函数
 */

import { ReportListItem } from './types';

/**
 * 验证以太坊地址格式
 * @param address 地址字符串
 * @returns 是否有效
 */
export const isValidEthereumAddress = (address: string): boolean => {
    if (!address) return false;

    // 基本格式验证
    const ethAddressRegex = /^0x[a-fA-F0-9]{40}$/;
    return ethAddressRegex.test(address);
};

/**
 * 格式化时间显示
 * @param dateString ISO日期字符串
 * @returns 格式化后的时间
 */
export const formatReportDate = (dateString: string): string => {
    try {
        const date = new Date(dateString);
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
        });
    } catch (error) {
        return dateString;
    }
};

/**
 * 从报告标题提取信息
 * @param title 报告标题
 * @returns 解析后的信息
 */
export const parseReportTitle = (title: string) => {
    const parts = title.split(' - ');
    return {
        address: parts[1] || '',
        timestamp: parts[2] || '',
    };
};

/**
 * 生成报告文件下载
 * @param blob 文件Blob
 * @param filename 文件名
 */
export const downloadBlob = (blob: Blob, filename: string): void => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

/**
 * 对报告列表进行排序
 * @param reports 报告列表
 * @param field 排序字段
 * @param direction 排序方向
 * @returns 排序后的列表
 */
export const sortReports = (
    reports: ReportListItem[],
    field: keyof ReportListItem = 'created_at',
    direction: 'asc' | 'desc' = 'desc'
): ReportListItem[] => {
    return [...reports].sort((a, b) => {
        let aValue = a[field];
        let bValue = b[field];

        // 日期字段特殊处理
        if (field === 'created_at') {
            aValue = new Date(aValue as string).getTime();
            bValue = new Date(bValue as string).getTime();
        }

        if (direction === 'asc') {
            return aValue > bValue ? 1 : -1;
        } else {
            return aValue < bValue ? 1 : -1;
        }
    });
};

/**
 * 搜索报告列表
 * @param reports 报告列表
 * @param searchTerm 搜索词
 * @returns 搜索后的列表
 */
export const searchReports = (
    reports: ReportListItem[],
    searchTerm: string
): ReportListItem[] => {
    if (!searchTerm.trim()) return reports;

    const term = searchTerm.toLowerCase();
    return reports.filter(report =>
        report.target_address.toLowerCase().includes(term) ||
        report.title.toLowerCase().includes(term) ||
        report.id.toString().includes(term)
    );
};