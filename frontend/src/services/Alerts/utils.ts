/**
 * 告警记录工具函数
 */

import { Alert, AlertStatus, AlertLevel } from './types';

/**
 * 格式化风险评分
 */
export function formatRiskScore(score: number): string {
    return (score * 100).toFixed(1) + '%';
}

/**
 * 获取风险等级
 */
export function getRiskLevel(score: number): AlertLevel {
    if (score >= 0.8) return 'CRITICAL';
    if (score >= 0.6) return 'HIGH';
    if (score >= 0.4) return 'MEDIUM';
    return 'LOW';
}

/**
 * 获取风险等级颜色
 */
export function getRiskColor(score: number | AlertLevel): string {
    let level: AlertLevel;

    if (typeof score === 'number') {
        level = getRiskLevel(score);
    } else {
        level = score;
    }

    switch (level) {
        case 'CRITICAL':
            return '#f5222d'; // 红色
        case 'HIGH':
            return '#fa8c16'; // 橙色
        case 'MEDIUM':
            return '#faad14'; // 黄色
        case 'LOW':
            return '#52c41a'; // 绿色
        default:
            return '#8c8c8c'; // 灰色
    }
}

/**
 * 格式化时间
 */
export function formatDateTime(dateString: string): string {
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
    });
}

/**
 * 格式化时间相对
 */
export function formatTimeAgo(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diffInSeconds < 60) {
        return '刚刚';
    } else if (diffInSeconds < 3600) {
        return `${Math.floor(diffInSeconds / 60)}分钟前`;
    } else if (diffInSeconds < 86400) {
        return `${Math.floor(diffInSeconds / 3600)}小时前`;
    } else {
        return `${Math.floor(diffInSeconds / 86400)}天前`;
    }
}

/**
 * 截取地址
 */
export function truncateAddress(address: string, startLength = 6, endLength = 4): string {
    if (!address) return '';
    if (address.length <= startLength + endLength) return address;
    return `${address.substring(0, startLength)}...${address.substring(address.length - endLength)}`;
}

/**
 * 获取状态标签
 */
export function getStatusLabel(status: AlertStatus): string {
    const labels: Record<AlertStatus, string> = {
        PENDING: '待处理',
        CHECKED: '已检查',
        ACKNOWLEDGED: '已确认',
    };
    return labels[status] || status;
}

/**
 * 获取状态颜色
 */
export function getStatusColor(status: AlertStatus): string {
    const colors: Record<AlertStatus, string> = {
        PENDING: '#fa8c16', // 橙色
        CHECKED: '#52c41a', // 绿色
        ACKNOWLEDGED: '#8c8c8c', // 灰色
    };
    return colors[status] || '#8c8c8c';
}

export function formatReportDate(dateString: string): string {
    const date = new Date(dateString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
}

/**
 * 验证以太坊地址格式
 */
export function isValidEthereumAddress(address: string): boolean {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
}

/**
 * 检查地址格式
 */
export function isValidAddress(address: string): boolean {
    // 基本的以太坊地址格式检查
    return /^0x[a-fA-F0-9]{40}$/.test(address);
}

/**
 * 过滤告警列表
 */
export function filterAlerts(
    alerts: Alert[],
    filters: {
        status?: AlertStatus;
        minRisk?: number;
        maxRisk?: number;
        address?: string;
    }
): Alert[] {
    return alerts.filter(alert => {
        if (filters.status && alert.status !== filters.status) return false;
        if (filters.minRisk !== undefined && alert.risk_score < filters.minRisk) return false;
        if (filters.maxRisk !== undefined && alert.risk_score > filters.maxRisk) return false;
        if (filters.address && !alert.monitored_address_id.includes(filters.address)) return false;
        return true;
    });
}

/**
 * 排序告警列表
 */
export function sortAlerts(
    alerts: Alert[],
    sortBy: 'created_at' | 'risk_score',
    sortOrder: 'asc' | 'desc' = 'desc'
): Alert[] {
    return [...alerts].sort((a, b) => {
        let comparison = 0;

        if (sortBy === 'created_at') {
            const timeA = new Date(a.created_at).getTime();
            const timeB = new Date(b.created_at).getTime();
            comparison = timeA - timeB;
        } else if (sortBy === 'risk_score') {
            comparison = a.risk_score - b.risk_score;
        }

        return sortOrder === 'asc' ? comparison : -comparison;
    });
}

