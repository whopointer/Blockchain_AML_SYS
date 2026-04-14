// services/report/api.ts
// 更新导入的类型
import {
    ReportRequest,
    ReportResponse,
    BackendReportRaw,
    ApiResponse,
    PaginatedReports,
    ErrorResponse,
    PreviewOptions,
    PreviewInfo,
    PreviewResponse,
    EmbedPreviewHTML
} from './types';

export class ReportAPI {
    private baseURL: string;

    constructor(baseURL: string) {
        this.baseURL = baseURL;
    }

    // 生成报告 - 保持不变
    async generateReport(
        address: string,
        options: {
            type?: 'basic' | 'enhanced';
            includePredictions?: boolean;
        } = {}
    ): Promise<ReportResponse> {
        const {
            type = 'enhanced',
            includePredictions = true
        } = options;

        const endpoint = type === 'basic'
            ? '/reports/generate/basic'
            : '/reports/generate';

        console.log('生成报告请求URL:', `${this.baseURL}${endpoint}`);

        const response = await fetch(`${this.baseURL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ address }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('响应状态:', response.status, '错误信息:', errorText);
            throw new Error(`生成报告失败: ${response.status} ${response.statusText}`);
        }

        const data: ReportResponse = await response.json();
        return data;
    }

    // 获取报告列表 - 保持不变
    async listReports(filters?: {
        address?: string;
        limit?: number;
        offset?: number;
    }): Promise<ApiResponse<PaginatedReports>> {
        const params = new URLSearchParams();
        if (filters?.address) params.append('address', filters.address);
        params.append('limit', (filters?.limit || 50).toString());
        params.append('offset', (filters?.offset || 0).toString());

        const url = `${this.baseURL}/reports/list?${params.toString()}`;
        console.log('获取报告列表URL:', url);

        const response = await fetch(url);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('响应状态:', response.status, '错误信息:', errorText);
            throw new Error(`获取报告列表失败: ${response.status} ${response.statusText}`);
        }

        const data: ApiResponse<PaginatedReports> = await response.json();
        return data;
    }

    // 获取报告状态 - 保持不变
    async getReportStatus(reportId: number): Promise<any> {
        const url = `${this.baseURL}/reports/status/${reportId}`;
        console.log('获取状态请求URL:', url);

        const response = await fetch(url);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('响应状态:', response.status, '错误信息:', errorText);
            throw new Error(`获取状态失败: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    }

    // 下载报告 - 保持不变
    async downloadReport(reportId: number, format: 'blob' | 'base64' = 'blob'): Promise<any> {
        const endpoint = format === 'base64'
            ? `/reports/download-base64/${reportId}`
            : `/reports/download/${reportId}`;

        const url = `${this.baseURL}${endpoint}`;
        console.log('下载报告请求URL:', url);

        const response = await fetch(url);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('响应状态:', response.status, '错误信息:', errorText);
            throw new Error(`下载失败: ${response.status} ${response.statusText}`);
        }

        if (format === 'base64') {
            const data = await response.json();
            return {
                data: data.report_data,
                filename: data.filename,
                type: 'base64'
            };
        } else {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const filename = `compliance_report_${reportId}.pdf`;

            return { url, filename, type: 'blob' };
        }
    }

    // 删除报告 - 保持不变
    async deleteReport(reportId: number): Promise<{ success: boolean; message: string }> {
        const url = `${this.baseURL}/reports/${reportId}`;
        console.log('删除报告请求URL:', url);

        const response = await fetch(url, {
            method: 'DELETE',
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('响应状态:', response.status, '错误信息:', errorText);
            throw new Error(`删除报告失败: ${response.status} ${response.statusText}`);
        }

        return response.json();
    }

    // 预览报告 - 修改为使用代理方式
    async previewReport(
        reportId: number,
        options: PreviewOptions = {}
    ): Promise<PreviewInfo> {
        const {
            method = 'auto',
            embedType = 'inline',
            filename,
            openInNewTab = true
        } = options;

        console.log('预览报告，报告ID:', reportId, '选项:', options);

        try {
            // 如果方法为auto，根据环境决定使用哪种预览方式
            const previewMethod = method === 'auto' ?
                (this.isMobileDevice() ? 'direct' : 'redirect') :
                method;

            // 构建查询参数
            const params = new URLSearchParams();

            if (previewMethod === 'redirect') {
                // 重定向模式，使用原来的预览接口
                params.append('preview_method', 'redirect');
            } else if (previewMethod === 'embed') {
                // 嵌入模式，使用代理视图
                // 这里我们使用代理端点，不设置特殊参数
                const proxyUrl = `${this.baseURL}/reports/view/${reportId}`;

                if (openInNewTab) {
                    this.openPreviewInNewTab(proxyUrl, reportId);
                }

                return {
                    preview_url: proxyUrl,
                    preview_method: 'embed',
                    embed_url: proxyUrl,
                    embed_type: embedType,
                    filename: filename || `report_${reportId}.pdf`,
                    content_type: 'application/pdf',
                    html_embed: `<iframe src="${proxyUrl}" width="100%" height="600px" style="border: none;"></iframe>`
                };
            } else {
                // 直接模式，返回预览信息
                params.append('preview_method', 'direct');
            }

            if (filename) {
                params.append('filename', filename);
            }

            const url = `${this.baseURL}/reports/preview/${reportId}?${params.toString()}`;
            console.log('预览报告请求URL:', url);

            if (previewMethod === 'redirect' && openInNewTab) {
                // 重定向模式，直接在新窗口打开
                this.openPreviewInNewTab(url, reportId);
                return {
                    preview_url: url,
                    preview_method: 'redirect',
                    filename: filename || `report_${reportId}.pdf`
                };
            }

            // 发送请求
            const response = await fetch(url, {
                headers: {
                    'Accept': 'application/json',
                },
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('预览响应状态:', response.status, '错误信息:', errorText);

                // 如果失败，尝试代理方式
                if (response.status === 403 || response.status === 404) {
                    console.log('原预览方式失败，尝试代理方式');
                    return this.previewViaProxy(reportId, options);
                }

                throw new Error(`预览失败: ${response.status} ${response.statusText}`);
            }

            // 处理响应
            if (previewMethod === 'redirect') {
                // 如果是重定向，不应该走到这里
                throw new Error('重定向预览模式异常');
            } else {
                // 返回JSON数据
                const data: PreviewResponse = await response.json();

                if (data.success && data.data) {
                    const previewInfo = data.data;

                    // 如果指定了在新标签页打开，自动打开
                    if (openInNewTab && previewInfo.preview_url) {
                        this.openPreviewInNewTab(previewInfo.preview_url, reportId, options.windowSize);
                    }

                    return previewInfo;
                } else {
                    throw new Error(data.message || '预览失败');
                }
            }

        } catch (error) {
            console.error('预览报告异常:', error);

            // 如果出错，尝试代理方式
            return this.previewViaProxy(reportId, options);
        }
    }

    // 代理方式预览
    private async previewViaProxy(
        reportId: number,
        options: PreviewOptions
    ): Promise<PreviewInfo> {
        const {
            embedType = 'inline',
            filename,
            openInNewTab = true
        } = options;

        const proxyUrl = `${this.baseURL}/reports/view/${reportId}`;

        console.log('使用代理预览:', proxyUrl);

        if (openInNewTab) {
            this.openPreviewInNewTab(proxyUrl, reportId);
        }

        return {
            preview_url: proxyUrl,
            preview_method: 'embed',
            embed_url: proxyUrl,
            embed_type: embedType,
            filename: filename || `report_${reportId}.pdf`,
            content_type: 'application/pdf',
            html_embed: `<iframe src="${proxyUrl}" width="100%" height="600px" style="border: none;"></iframe>`
        };
    }

    // 获取嵌入式预览HTML - 修改为使用代理端点
    async getEmbeddedPreviewHtml(reportId: number): Promise<EmbedPreviewHTML> {
        const proxyUrl = `${this.baseURL}/reports/view/${reportId}`;

        const html = `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>报告预览 - ${reportId}</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
        }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
    </style>
</head>
<body>
    <iframe src="${proxyUrl}" title="报告预览"></iframe>
</body>
</html>`;

        console.log('生成的嵌入式预览HTML');

        return {
            html,
            url: proxyUrl,
            title: `报告预览_${reportId}`
        };
    }

    // 在新窗口中打开预览 - 修改为优先使用代理
    async openPreviewInNewWindow(
        reportId: number,
        windowOptions?: {
            width?: number;
            height?: number;
            title?: string;
        }
    ): Promise<Window | null> {
        const {
            width = 1200,
            height = 800,
            title = `报告预览_${reportId}`
        } = windowOptions || {};

        try {
            // 使用代理端点
            const proxyUrl = `${this.baseURL}/reports/view/${reportId}`;

            // 计算窗口位置
            const left = (window.screen.width - width) / 2;
            const top = (window.screen.height - height) / 2;

            const windowFeatures = `
                width=${width},
                height=${height},
                left=${left},
                top=${top},
                scrollbars=yes,
                resizable=yes,
                toolbar=no,
                menubar=no,
                location=no,
                status=no
            `;

            // 打开新窗口
            const previewWindow = window.open(proxyUrl, title, windowFeatures);

            if (!previewWindow) {
                console.warn('弹出窗口被阻止，回退到新标签页');
                window.open(proxyUrl, '_blank');
            }

            return previewWindow;

        } catch (error) {
            console.error('打开预览窗口失败:', error);

            // 回退方案：使用原始预览
            const fallbackUrl = `${this.baseURL}/reports/preview/${reportId}?preview_method=redirect`;
            return window.open(fallbackUrl, '_blank');
        }
    }

    // 快速预览（简化版，无参数输入时直接在新标签页打开）
    async quickPreview(reportId: number): Promise<void> {
        console.log('快速预览报告，报告ID:', reportId);

        // 使用代理端点
        const proxyUrl = `${this.baseURL}/reports/view/${reportId}`;
        this.openPreviewInNewTab(proxyUrl, reportId);
    }

    // 在新标签页中打开预览
    private openPreviewInNewTab(
        url: string,
        reportId: number,
        windowSize?: { width?: number; height?: number }
    ): void {
        const width = windowSize?.width || 1200;
        const height = windowSize?.height || 800;

        const left = (window.screen.width - width) / 2;
        const top = (window.screen.height - height) / 2;

        const windowFeatures = `
            width=${width},
            height=${height},
            left=${left},
            top=${top},
            scrollbars=yes,
            resizable=yes
        `;

        const previewWindow = window.open(
            url,
            `report_preview_${reportId}`,
            windowFeatures
        );

        if (!previewWindow) {
            // 如果弹窗被阻止，回退到新标签页
            window.open(url, '_blank');
        }
    }

    // 检查是否为移动设备
    private isMobileDevice(): boolean {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }

    // 获取预览信息（不自动打开）
    async getPreviewInfo(
        reportId: number,
        options: Omit<PreviewOptions, 'openInNewTab'> = {}
    ): Promise<PreviewInfo> {
        return this.previewReport(reportId, {
            ...options,
            openInNewTab: false
        });
    }

    // 创建嵌入预览iframe - 修改为使用代理端点
    async createPreviewIframe(
        reportId: number,
        containerId?: string
    ): Promise<HTMLIFrameElement> {
        try {
            // 使用代理端点
            const proxyUrl = `${this.baseURL}/reports/view/${reportId}`;

            console.log('创建iframe，使用代理URL:', proxyUrl);

            // 创建iframe
            const iframe = document.createElement('iframe');
            iframe.src = proxyUrl;
            iframe.width = '100%';
            iframe.height = '600px';
            iframe.style.border = 'none';
            iframe.title = `报告预览 - ${reportId}`;
            iframe.name = `report_preview_iframe_${reportId}`;

            // 添加必要的属性来避免安全限制
            iframe.setAttribute('sandbox', 'allow-same-origin allow-scripts');
            iframe.setAttribute('allow', 'autoplay; encrypted-media; picture-in-picture');

            // 如果有容器ID，添加到容器中
            if (containerId) {
                const container = document.getElementById(containerId);
                if (container) {
                    container.innerHTML = ''; // 清空容器
                    container.appendChild(iframe);
                }
            }

            return iframe;

        } catch (error) {
            console.error('创建预览iframe失败:', error);
            throw error;
        }
    }

    // 预览并下载报告（一体化操作）
    async previewAndDownload(
        reportId: number,
        downloadAfterPreview: boolean = true
    ): Promise<void> {
        try {
            // 1. 先预览
            await this.quickPreview(reportId);

            // 2. 询问是否下载
            if (downloadAfterPreview) {
                setTimeout(async () => {
                    const shouldDownload = window.confirm('是否要下载此报告？');

                    if (shouldDownload) {
                        try {
                            await this.downloadReport(reportId);
                        } catch (downloadError) {
                            console.error('下载失败:', downloadError);
                        }
                    }
                }, 2000); // 2秒后询问
            }
        } catch (error) {
            console.error('预览并下载失败:', error);
            throw error;
        }
    }

    // 批量预览报告
    async previewMultipleReports(
        reportIds: number[],
        options: PreviewOptions = {}
    ): Promise<void> {
        if (reportIds.length === 0) {
            return;
        }

        // 限制同时打开的预览窗口数量
        const maxConcurrent = 3;
        const idsToPreview = [...reportIds];

        // 打开前几个预览
        const firstBatch = idsToPreview.splice(0, Math.min(maxConcurrent, idsToPreview.length));

        for (const reportId of firstBatch) {
            this.previewReport(reportId, options);

            // 添加延迟，避免浏览器阻止多个弹出窗口
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        // 如果还有更多报告，询问用户
        if (idsToPreview.length > 0) {
            setTimeout(() => {
                const shouldContinue = window.confirm(
                    `还有 ${idsToPreview.length} 个报告未预览。是否继续？`
                );

                if (shouldContinue) {
                    // 继续预览剩余报告
                    for (const reportId of idsToPreview) {
                        this.previewReport(reportId, options);
                    }
                }
            }, 1000);
        }
    }

    // 新增：通过代理端点获取PDF Blob
    async getReportPdfBlob(reportId: number): Promise<Blob> {
        const proxyUrl = `${this.baseURL}/reports/view/${reportId}`;
        console.log('获取PDF Blob:', proxyUrl);

        const response = await fetch(proxyUrl);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`获取PDF失败: ${response.status} ${errorText}`);
        }

        return await response.blob();
    }

    // 新增：创建Blob URL用于预览
    async createBlobPreviewUrl(reportId: number): Promise<string> {
        try {
            const blob = await this.getReportPdfBlob(reportId);
            const blobUrl = URL.createObjectURL(blob);

            // 返回blob URL
            return blobUrl;
        } catch (error) {
            console.error('创建Blob URL失败:', error);
            throw error;
        }
    }
}