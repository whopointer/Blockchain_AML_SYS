// types.ts - 推荐方案
export interface ReportRequest {
    address: string;
}

export interface ReportResponse {
    report_id: number;
    message: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    estimated_time?: number;
}

// 后端返回的原始数据类型
export interface BackendReportRaw {
    id: number;
    target_address: string;
    created_at: string;
    title: string;
}

// 前端使用的数据类型
export interface FrontendReport {
    report_id: number;
    address: string;
    created_at: string;
    title: string;
    filename?: string;
}

// API响应类型
export interface ApiResponse<T> {
    success: boolean;
    data: T;
    message: string;
}

export interface PaginatedReports {
    reports: BackendReportRaw[];  // 或者 FrontendReport[] 根据映射时机决定
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
}

// 错误响应
export interface ErrorResponse {
    detail: string;
    code?: number;
    timestamp?: string;
}

// 工具函数：映射后端数据到前端格式
export function mapBackendToFrontend(backend: BackendReportRaw): FrontendReport {
    return {
        report_id: backend.id,
        address: backend.target_address,
        created_at: backend.created_at,
        title: backend.title,
        filename: backend.title ? `${backend.title}.pdf` : undefined,
    };
}

// 工具函数：映射前端数据到后端格式
export function mapFrontendToBackend(frontend: FrontendReport): BackendReportRaw {
    return {
        id: frontend.report_id,
        target_address: frontend.address,
        created_at: frontend.created_at,
        title: frontend.title,
    };
}

export interface PreviewOptions {
    // 预览方式
    method?: 'redirect' | 'embed' | 'direct' | 'auto';
    // 嵌入类型
    embedType?: 'inline' | 'viewer';
    // 是否直接在新窗口打开
    openInNewTab?: boolean;
    // 自定义文件名
    filename?: string;
    // 预览窗口尺寸
    windowSize?: {
        width?: number;
        height?: number;
    };
}

export interface PreviewOptions {
    // 预览方式
    method?: 'redirect' | 'embed' | 'direct' | 'auto';
    // 嵌入类型
    embedType?: 'inline' | 'viewer';
    // 是否直接在新窗口打开
    openInNewTab?: boolean;
    // 自定义文件名
    filename?: string;
    // 预览窗口尺寸
    windowSize?: {
        width?: number;
        height?: number;
    };
}

export interface PreviewInfo {
    // 预览URL
    preview_url: string;
    // 嵌入URL
    embed_url?: string;
    // 预览方式
    preview_method: string;
    // 嵌入类型
    embed_type?: string;
    // 文件名
    filename: string;
    // HTML嵌入代码
    html_embed?: string;
    // 内容类型
    content_type?: string;
    // 是否支持直接预览
    direct_preview?: boolean;
    // 过期时间（秒）
    expires_in?: number;
}

export interface PreviewResponse {
    success: boolean;
    data: PreviewInfo;
    message?: string;
}

// 嵌入式预览的HTML内容
export interface EmbedPreviewHTML {
    html: string;
    url: string;
    title: string;
}