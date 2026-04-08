// 通用响应结构
export interface ApiResponse<T = any> {
    code: number;
    message: string;
    data: T;
    timestamp: number;
}

// 分页参数
export interface PaginationParams {
    page: number;
    pageSize: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
}

// 分页响应
export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    pageSize: number;
    totalPages: number;
}

// 错误类型
export interface ApiError {
    code: number;
    message: string;
    details?: Record<string, any>;
}