/**
 * 告警记录相关类型定义
 */

// 告警状态枚举
export type AlertStatus = 'PENDING' | 'CHECKED' | 'ACKNOWLEDGED';

// 告警级别枚举
export type AlertLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

// 告警记录接口
export interface Alert {
    id: string;
    monitored_address_id: string;
    risk_score: number;
    status: AlertStatus;
    metadata: AlertMetadata;
    created_at: string;
}

// 告警元数据接口
export interface AlertMetadata {
    alert_level?: AlertLevel;
    transaction_count?: number;
    highest_risk_transaction?: {
        hash: string;
        risk_score: number;
        risk_factors: string[];
        prediction_time?: string;
    };
    prediction_statistics?: {
        total_transactions: number;
        avg_risk_score: number;
        max_risk_score: number;
    };
    analysis?: string;
    generated_at?: string;
    [key: string]: any; // 允许其他元数据字段
}

// 告警列表响应
export interface AlertListResponse {
    alerts: Alert[];
    total: number;
    pending_count: number;
    page?: number;
    pageSize?: number;
}

// 告警统计信息
export interface AlertStatistics {
    total_alerts: number;
    pending_alerts: number;
    checked_alerts: number;
    acknowledged_alerts: number;
    avg_risk_score: number;
    max_risk_score: number;
    time_range_hours: number;
}

// 监控地址请求
export interface MonitorAddressRequest {
    address: string;
    hours?: number;
}

// 批量监控请求
export interface BatchMonitorRequest {
    addresses: string[];
    hours?: number;
}

// 监控响应
export interface MonitorResponse {
    success: boolean;
    message: string;
    address: string;
    hours: number;
    alert_generated: boolean;
    alert_info?: AlertInfo;
    alert_record?: Alert;
    statistics?: Record<string, any>;
}

// 告警信息
export interface AlertInfo {
    address: string;
    risk_score: number;
    alert_level: AlertLevel;
    metadata: AlertMetadata;
    highest_risk_transaction?: string;
    transaction_count: number;
}

// 批量监控响应
export interface BatchMonitorResponse {
    success: boolean;
    message: string;
    total_addresses: number;
    successful_monitors: number;
    failed_monitors: number;
    total_alerts_generated: number;
    results: MonitorResponse[];
}

// 更新告警状态请求
export interface UpdateAlertStatusRequest {
    status: AlertStatus;
    additional_metadata?: Record<string, any>;
}

// 查询参数
export interface AlertQueryParams {
    address?: string;
    status?: AlertStatus;
    hours?: number;
    page?: number;
    pageSize?: number;
    risk_min?: number;
    risk_max?: number;
    sortBy?: 'created_at' | 'risk_score';
    sortOrder?: 'asc' | 'desc';
}