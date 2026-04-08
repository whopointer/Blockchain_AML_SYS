export interface MonitoredAddressDTO {
    id: string;
    address: string;
    addressType: 'BITCOIN'|'ETHEREUM';
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    monitoringEnabled: boolean;
    createdAt: string;
    updatedAt: string;
    description: string;
}

export interface MonitoredAddress {
    id: string;
    address: string;
    addressType: 'BITCOIN'|'ETHEREUM';
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    monitoringEnabled: boolean;
    createdAt: Date;
    updatedAt: Date;
    description: string;
}

export interface CreateMonitoredAddressRequest {
    address: string;
    addressType: 'BITCOIN'|'ETHEREUM';
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    monitoringEnabled?: boolean;
    description?: string;
}

export interface UpdateMonitoredAddressRequest extends Partial<CreateMonitoredAddressRequest> {
    id: string;
}

export interface ApiResponse<T = any> {
    success: boolean;
    data: T;
    message: string;
}

export const ADDRESS_TYPE_OPTIONS = [
    { value: 'BITCOIN', label: '比特币' },
    { value: 'ETHEREUM', label: '以太坊' }
] as const;

export const RISK_LEVEL_OPTIONS = [
    { value: 'LOW', label: '低风险' },
    { value: 'MEDIUM', label: '中风险' },
    { value: 'HIGH', label: '高风险' },
    { value: 'CRITICAL', label: '严重风险' },
] as const;

export interface MonitoredAddressDetailResponse{
    success:boolean;
    message:string;
    data:{
        id: string;
        address: string;
        address_type: string;
        risk_level: string;
        monitoring_enabled: boolean;
        created_at: string;
        updated_at: string;
        description: string;
    }
}

export interface ReportDetailResponse{
    data:{
        id:number;
        target_address:string;
        file_path:string;
        created_at:string;
    }
    success: boolean;
    message: string;
}
