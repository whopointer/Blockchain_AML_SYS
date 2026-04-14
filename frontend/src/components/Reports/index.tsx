// components/Reports/index.tsx
import React, { useState, useEffect, useCallback, useRef } from "react";
import {
    Table, Card, Tag, Space, Button, Pagination, Tooltip,
    Popconfirm, message, Spin, Layout, Row, Col, Input,
    Typography, Empty, Badge, Divider, Alert
} from "antd";
import {
    EyeOutlined, DeleteOutlined, DownloadOutlined,
    FileTextOutlined, ReloadOutlined, SearchOutlined,
    CloseOutlined, LoadingOutlined, CheckOutlined
} from "@ant-design/icons";

// 导入新API
import {
    getReportList,
    deleteReport,
    downloadReportFile,
    formatReportDate
} from "../../services/report/index";
import { useReportOperations } from "../../services/report/useReportOperations";
import { FrontendReport } from "../../services/report/types";

const { Title, Text, Paragraph } = Typography;
const { Search } = Input;
const { Content, Sider } = Layout;

// 扩展的表格数据接口
interface TableReport extends FrontendReport {
    key: number;
    short_address: string;
    formatted_date: string;
    address_type: 'ethereum' | 'bitcoin' | 'other';
}

const ReportList: React.FC = () => {
    const [reports, setReports] = useState<FrontendReport[]>([]);
    const [filteredReports, setFilteredReports] = useState<FrontendReport[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [searchLoading, setSearchLoading] = useState<boolean>(false);
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [pageSize, setPageSize] = useState<number>(5);
    const [totalReports, setTotalReports] = useState<number>(0);
    const [searchTerm, setSearchTerm] = useState<string>("");
    const [selectedReport, setSelectedReport] = useState<FrontendReport | null>(null);
    const [previewLoading, setPreviewLoading] = useState<boolean>(false);
    const [pdfBlobUrl, setPdfBlobUrl] = useState<string | null>(null);
    const [previewError, setPreviewError] = useState<string | null>(null);
    const [previewMethod, setPreviewMethod] = useState<'embed' | 'window' | 'tab' | null>(null);

    const previewWindowRef = useRef<Window | null>(null);
    const previousBlobUrlRef = useRef<string | null>(null);

    const {
        delete: deleteReportAction,
        isDeleting,
        error,
        success,
        clearStatus
    } = useReportOperations();

    // 清理Blob URL
    const cleanupBlobUrl = () => {
        if (previousBlobUrlRef.current) {
            URL.revokeObjectURL(previousBlobUrlRef.current);
            previousBlobUrlRef.current = null;
        }
    };

    // 组件卸载时清理
    useEffect(() => {
        return () => {
            cleanupBlobUrl();
        };
    }, []);

    // 加载报告列表
    const loadReports = useCallback(async () => {
        setLoading(true);
        try {
            const data = await getReportList();
            console.log('加载的报告数据:', data);
            setReports(data || []);
            setFilteredReports(data || []);
            setTotalReports(Array.isArray(data) ? data.length : 0);

            // 如果有数据，默认选中第一个报告
            if (data && data.length > 0) {
                setSelectedReport(data[0]);
                loadPdfEmbed(data[0].report_id);
            }
        } catch (err) {
            console.error('加载报告失败:', err);
            message.error('加载报告列表失败，请稍后重试');
        } finally {
            setLoading(false);
        }
    }, []);

    // 搜索报告
    const handleSearch = (value: string) => {
        setSearchTerm(value);
        setSearchLoading(true);

        setTimeout(() => {
            if (!value.trim()) {
                setFilteredReports(reports);
            } else {
                const filtered = reports.filter(report =>
                    report.address?.toLowerCase().includes(value.toLowerCase()) ||
                    report.title?.toLowerCase().includes(value.toLowerCase()) ||
                    String(report.report_id).includes(value)
                );
                setFilteredReports(filtered);
            }
            setSearchLoading(false);
        }, 300);
    };

    // 清除搜索
    const handleClearSearch = () => {
        setSearchTerm("");
        setFilteredReports(reports);
    };

    // 组件挂载时加载报告
    useEffect(() => {
        loadReports();
    }, [loadReports]);

    // 监听成功/错误消息
    useEffect(() => {
        if (success) {
            message.success(success);
            clearStatus();
        }
        if (error) {
            message.error(error);
            clearStatus();
        }
    }, [success, error, clearStatus]);

    // 删除报告
    const handleDeleteReport = async (reportId: number) => {
        try {
            // 如果要删除的是当前选中的报告，先清除预览
            if (selectedReport?.report_id === reportId) {
                setSelectedReport(null);
                cleanupBlobUrl();
                setPdfBlobUrl(null);
                setPreviewMethod(null);
            }

            await deleteReportAction(reportId);
            loadReports();
        } catch (err) {
            console.error('删除报告失败:', err);
        }
    };

    // 加载PDF并创建Blob URL（嵌入模式）
    const loadPdfEmbed = async (reportId: number) => {
        setPreviewLoading(true);
        setPreviewError(null);
        setPreviewMethod('embed');

        try {
            // 清理之前的Blob URL
            cleanupBlobUrl();

            // 通过代理端点获取PDF
            const proxyUrl = `http://localhost:7999/api/reports/view/${reportId}`;
            console.log('加载PDF嵌入预览:', proxyUrl);

            const response = await fetch(proxyUrl, {
                headers: {
                    'Accept': 'application/pdf',
                },
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('PDF响应状态:', response.status, '错误信息:', errorText);
                throw new Error(`加载PDF失败: ${response.status} ${response.statusText}`);
            }

            // 创建Blob URL
            const blob = await response.blob();
            const blobUrl = URL.createObjectURL(blob);

            // 保存Blob URL引用
            previousBlobUrlRef.current = blobUrl;
            setPdfBlobUrl(blobUrl);

            console.log('PDF Blob URL创建成功');

        } catch (err: any) {
            console.error('加载PDF失败:', err);
            setPreviewError(err.message || '加载PDF失败');
            message.error('加载PDF失败: ' + (err.message || '未知错误'));
        } finally {
            setPreviewLoading(false);
        }
    };

    // 在新窗口/标签页打开PDF
    const openPdfInNewWindow = async (reportId: number, method: 'window' | 'tab') => {
        setPreviewMethod(method);
        setPreviewError(null);

        try {
            // 通过代理端点获取PDF
            const proxyUrl = `http://localhost:7999/api/reports/view/${reportId}`;
            console.log(`在${method}中打开PDF:`, proxyUrl);

            if (method === 'window') {
                const width = 1200;
                const height = 800;
                const left = (window.screen.width - width) / 2;
                const top = (window.screen.height - height) / 2;

                previewWindowRef.current = window.open(
                    proxyUrl,
                    'report_preview',
                    `width=${width},height=${height},left=${left},top=${top},scrollbars=yes,resizable=yes`
                );

                if (!previewWindowRef.current) {
                    throw new Error('无法打开预览窗口，可能被浏览器阻止');
                }
            } else {
                window.open(proxyUrl, '_blank');
            }
        } catch (err: any) {
            console.error(`在${method}中打开PDF失败:`, err);
            message.error(`打开预览失败: ${err.message || '未知错误'}`);
        }
    };

    // 预览报告
    const handlePreviewReport = async (reportId: number, method: 'embed' | 'window' | 'tab' = 'embed') => {
        if (method === 'embed') {
            await loadPdfEmbed(reportId);
        } else {
            await openPdfInNewWindow(reportId, method);
        }
    };

    // 下载报告
    const handleDownloadReport = async (reportId: number, title?: string) => {
        try {
            setLoading(true);
            const blob = await downloadReportFile(reportId);

            // 创建下载链接
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${title || `compliance_report_${reportId}`}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            message.success('报告下载开始');
        } catch (err: any) {
            message.error(err.message || '下载报告失败');
        } finally {
            setLoading(false);
        }
    };

    // 获取地址类型
    const getAddressType = (address: string): 'ethereum' | 'bitcoin' | 'other' => {
        if (!address) return 'other';
        if (address.startsWith("0x")) {
            return 'ethereum';
        } else if (address.startsWith("1") || address.startsWith("3") || address.startsWith("bc1")) {
            return 'bitcoin';
        }
        return 'other';
    };

    // 获取地址类型标签
    const getAddressTypeTag = (address: string) => {
        const type = getAddressType(address);
        const config = {
            ethereum: { color: 'blue', text: 'ETH' },
            bitcoin: { color: 'orange', text: 'BTC' },
            other: { color: 'gray', text: 'OTHER' }
        };
        return <Badge color={config[type].color} text={config[type].text} />;
    };

    // 格式化地址显示
    const formatAddress = (address: string): string => {
        if (!address || address.length <= 20) return address || '';
        return `${address.substring(0, 10)}...${address.substring(address.length - 8)}`;
    };

    // 准备表格数据
    const prepareTableData = (): TableReport[] => {
        if (!filteredReports || !Array.isArray(filteredReports)) {
            return [];
        }

        return filteredReports.map(report => ({
            ...report,
            key: report.report_id,
            short_address: formatAddress(report.address),
            formatted_date: formatReportDate(report.created_at),
            address_type: getAddressType(report.address)
        }));
    };

    // 表格列定义
    const columns = [
        {
            title: "ID",
            dataIndex: "report_id",
            key: "report_id",
            width: 60,
            sorter: (a: TableReport, b: TableReport) => a.report_id - b.report_id,
        },
        {
            title: "时间",
            dataIndex: "formatted_date",
            key: "formatted_date",
            width: 100,
            render: (text: string) => (
                <Tooltip title={text}>
                    <Text style={{ fontSize: '12px' }}>{text}</Text>
                </Tooltip>
            )
        },
        {
            title: "地址",
            dataIndex: "address",
            key: "address",
            width: 200,
            render: (text: string, record: TableReport) => (
                <div style={{ lineHeight: 1.2 }}>
                    {getAddressTypeTag(text)}
                    <Tooltip title={text}>
                        <div style={{
                            fontFamily: "monospace",
                            fontSize: "11px",
                            marginTop: 2,
                            cursor: 'pointer',
                            backgroundColor: '#f8f9fa',
                            padding: '2px 4px',
                            borderRadius: '3px',
                            wordBreak: 'break-all',
                            lineHeight: 1.2
                        }}>
                            {record.short_address}
                        </div>
                    </Tooltip>
                </div>
            )
        },
        {
            title: "操作",
            key: "action",
            width: 100,
            render: (_: any, record: TableReport) => (
                <Space size="small" direction="vertical">
                    <Button
                        type="link"
                        size="small"
                        icon={<EyeOutlined />}
                        onClick={() => {
                            setSelectedReport(record);
                            handlePreviewReport(record.report_id, 'embed');
                        }}
                        loading={loading}
                        disabled={isDeleting(record.report_id)}
                        style={{ padding: 0 }}
                    >
                        预览
                    </Button>
                    <Button
                        type="link"
                        size="small"
                        icon={<DownloadOutlined />}
                        onClick={() => handleDownloadReport(record.report_id, record.title)}
                        loading={loading}
                        disabled={isDeleting(record.report_id)}
                        style={{ padding: 0 }}
                    >
                        下载
                    </Button>
                    <Popconfirm
                        title="确认删除报告"
                        description="此操作将永久删除该报告，是否继续？"
                        onConfirm={() => handleDeleteReport(record.report_id)}
                        okText="确定"
                        cancelText="取消"
                        okButtonProps={{ danger: true }}
                    >
                        <Button
                            type="link"
                            size="small"
                            danger
                            icon={<DeleteOutlined />}
                            loading={isDeleting(record.report_id)}
                            disabled={loading}
                            style={{ padding: 0 }}
                        >
                            删除
                        </Button>
                    </Popconfirm>
                </Space>
            )
        }
    ];

    // 处理行点击
    const handleRowClick = (record: TableReport) => {
        setSelectedReport(record);
        handlePreviewReport(record.report_id, 'embed');
    };

    // 行样式
    const rowClassName = (record: TableReport) => {
        return selectedReport?.report_id === record.report_id ? 'selected-row' : '';
    };

    // 分页数据
    const tableData = prepareTableData();
    const paginatedReports = tableData.slice(
        (currentPage - 1) * pageSize,
        currentPage * pageSize
    );

    // 渲染空状态
    const renderEmpty = () => (
        <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={
                <span>
          {searchTerm ? "未找到匹配的报告" : "暂无报告数据"}
        </span>
            }
        >
            {!searchTerm && (
                <Button
                    type="primary"
                    onClick={loadReports}
                    icon={<ReloadOutlined />}
                >
                    刷新
                </Button>
            )}
        </Empty>
    );

    // 渲染PDF预览
    const renderPdfPreview = () => {
        if (!pdfBlobUrl) {
            return (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    height: '100%',
                    flexDirection: 'column',
                    backgroundColor: '#f5f5f5'
                }}>
                    <FileTextOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
                    <Text type="secondary">点击预览按钮加载PDF</Text>
                </div>
            );
        }

        return (
            <div style={{
                width: '100%',
                height: '100%',
                display: 'flex',
                flexDirection: 'column'
            }}>
                {/* 使用embed标签显示PDF */}
                <embed
                    src={pdfBlobUrl}
                    type="application/pdf"
                    width="100%"
                    height="100%"
                    style={{
                        border: 'none',
                        flex: 1
                    }}
                />

                {/* 备用方案：如果embed不工作，显示下载链接 */}
                <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    display: 'none',
                    justifyContent: 'center',
                    alignItems: 'center',
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    zIndex: 10
                }}>
                    <Alert
                        message="PDF预览失败"
                        description={
                            <div>
                                <p>您的浏览器可能不支持PDF预览。</p>
                                <Button
                                    onClick={() => {
                                        const link = document.createElement('a');
                                        link.href = pdfBlobUrl;
                                        link.download = selectedReport?.title || `report_${selectedReport?.report_id}.pdf`;
                                        link.click();
                                    }}
                                    type="primary"
                                    size="small"
                                >
                                    下载PDF
                                </Button>
                            </div>
                        }
                        type="warning"
                        showIcon
                    />
                </div>
            </div>
        );
    };

    // 渲染报告详情
    const renderReportDetail = () => {
        if (!selectedReport) {
            return (
                <Card style={{ height: '100%', borderRadius: 8 }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        height: '100%',
                        flexDirection: 'column'
                    }}>
                        <FileTextOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
                        <Title level={4} style={{ color: '#bfbfbf' }}>选择报告进行预览</Title>
                        <Text type="secondary">点击左侧报告列表中的任意报告开始预览</Text>
                    </div>
                </Card>
            );
        }

        return (
            <Card
                title={
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <div style={{ display: "flex", alignItems: "center" }}>
                            <FileTextOutlined style={{ marginRight: 8 }} />
                            <span>报告预览</span>
                            {previewLoading && <Spin size="small" style={{ marginLeft: 12 }} />}
                        </div>
                        <div>
                            <Space>
                                <Button
                                    type={previewMethod === 'embed' ? 'primary' : 'default'}
                                    size="small"
                                    icon={<EyeOutlined />}
                                    onClick={() => handlePreviewReport(selectedReport.report_id, 'embed')}
                                    disabled={previewLoading}
                                >
                                    页面内预览
                                </Button>
                                <Button
                                    type={previewMethod === 'window' ? 'primary' : 'default'}
                                    size="small"
                                    icon={<EyeOutlined />}
                                    onClick={() => handlePreviewReport(selectedReport.report_id, 'window')}
                                    disabled={previewLoading}
                                >
                                    新窗口预览
                                </Button>
                                <Button
                                    type={previewMethod === 'tab' ? 'primary' : 'default'}
                                    size="small"
                                    icon={<EyeOutlined />}
                                    onClick={() => handlePreviewReport(selectedReport.report_id, 'tab')}
                                    disabled={previewLoading}
                                >
                                    新标签页
                                </Button>
                                <Button
                                    type="primary"
                                    size="small"
                                    icon={<DownloadOutlined />}
                                    onClick={() => handleDownloadReport(selectedReport.report_id, selectedReport.title)}
                                    loading={loading}
                                >
                                    下载
                                </Button>
                            </Space>
                        </div>
                    </div>
                }
                bordered={false}
                style={{
                    height: '100%',
                    borderRadius: 8,
                    display: 'flex',
                    flexDirection: 'column'
                }}
                bodyStyle={{ flex: 1, padding: 0, overflow: 'hidden' }}
            >
                {/* 报告头部信息 */}
                <div style={{ padding: 16, borderBottom: '1px solid #f0f0f0' }}>
                    <Row gutter={16}>
                        <Col span={8}>
                            <Text strong>报告ID:</Text>
                            <div>{selectedReport.report_id}</div>
                        </Col>
                        <Col span={8}>
                            <Text strong>生成时间:</Text>
                            <div>{formatReportDate(selectedReport.created_at)}</div>
                        </Col>
                        <Col span={8}>
                            <Text strong>地址类型:</Text>
                            <div>{getAddressTypeTag(selectedReport.address)}</div>
                        </Col>
                    </Row>
                    <Row gutter={16} style={{ marginTop: 8 }}>
                        <Col span={24}>
                            <Text strong>目标地址:</Text>
                            <div style={{
                                fontFamily: 'monospace',
                                backgroundColor: '#f5f5f5',
                                padding: '4px 8px',
                                borderRadius: 4,
                                marginTop: 4
                            }}>
                                {selectedReport.address}
                            </div>
                        </Col>
                    </Row>
                </div>

                {/* 预览区域 */}
                <div style={{
                    flex: 1,
                    position: 'relative',
                    overflow: 'hidden',
                    height: '90%'
                }}>
                    {previewLoading && (
                        <div style={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0,
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            backgroundColor: 'rgba(255, 255, 255, 0.8)',
                            zIndex: 10
                        }}>
                            <Spin
                                indicator={<LoadingOutlined style={{ fontSize: 48 }} spin />}
                                tip="正在加载PDF..."
                            />
                        </div>
                    )}

                    {previewError && (
                        <Alert
                            message="预览加载失败"
                            description={previewError}
                            type="error"
                            showIcon
                            action={
                                <Button
                                    size="small"
                                    onClick={() => handlePreviewReport(selectedReport.report_id, 'embed')}
                                >
                                    重试
                                </Button>
                            }
                            style={{ margin: 16 }}
                        />
                    )}

                    {!previewLoading && !previewError && renderPdfPreview()}
                </div>
            </Card>
        );
    };

    return (
        <Layout style={{
            height: 'calc(100vh - 64px)',
            backgroundColor: '#fff',
            padding: 0
        }}>
            <Sider
                width="25%"
                style={{
                    backgroundColor: '#fff',
                    borderRight: '1px solid #f0f0f0',
                    overflow: 'auto',
                    padding: 16
                }}
                breakpoint="lg"
                collapsedWidth={0}
                zeroWidthTriggerStyle={{ top: 12 }}
            >
                <Card
                    title={
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                            <div style={{ display: "flex", alignItems: "center" }}>
                                <FileTextOutlined style={{ marginRight: 8 }} />
                                <span>报告列表</span>
                                {loading && <Spin size="small" style={{ marginLeft: 12 }} />}
                            </div>
                            <Button
                                icon={<ReloadOutlined />}
                                onClick={loadReports}
                                loading={loading}
                                size="small"
                            >
                                刷新
                            </Button>
                        </div>
                    }
                    bordered={false}
                    style={{
                        borderRadius: 8,
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column'
                    }}
                    bodyStyle={{
                        flex: 1,
                        padding: 0,
                        display: 'flex',
                        flexDirection: 'column'
                    }}
                    extra={
                        <div style={{ fontSize: 12, color: '#999' }}>
                            共 {totalReports} 个报告
                        </div>
                    }
                >
                    {/* 搜索框 */}
                    <div style={{ padding: 16, borderBottom: '1px solid #f0f0f0' }}>
                        <Search
                            placeholder="搜索地址、标题或ID"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            onSearch={handleSearch}
                            loading={searchLoading}
                            enterButton={<SearchOutlined />}
                            addonAfter={
                                searchTerm ? (
                                    <CloseOutlined
                                        onClick={handleClearSearch}
                                        style={{ cursor: 'pointer' }}
                                    />
                                ) : null
                            }
                            allowClear
                        />
                    </div>

                    {/* 报告列表 */}
                    <div style={{
                        flex: 1,
                        overflow: 'auto',
                        padding: 0
                    }}>
                        {tableData.length === 0 ? (
                            <div style={{ padding: 24 }}>
                                {renderEmpty()}
                            </div>
                        ) : (
                            <>
                                <Table
                                    columns={columns}
                                    dataSource={paginatedReports}
                                    rowKey="report_id"
                                    pagination={false}
                                    size="small"
                                    bordered={false}
                                    showHeader={false}
                                    loading={loading}
                                    locale={{ emptyText: '暂无数据' }}
                                    onRow={(record) => ({
                                        onClick: () => handleRowClick(record),
                                    })}
                                    rowClassName={rowClassName}
                                    style={{ cursor: 'pointer' }}
                                    scroll={{ y: 'calc(100vh - 300px)' }}
                                />

                                <div style={{
                                    padding: 16,
                                    borderTop: '1px solid #f0f0f0',
                                    textAlign: 'center'
                                }}>
                                    <Pagination
                                        current={currentPage}
                                        pageSize={pageSize}
                                        total={tableData.length}
                                        onChange={setCurrentPage}
                                        onShowSizeChange={(current, size) => {
                                            setPageSize(size);
                                            setCurrentPage(1);
                                        }}
                                        showSizeChanger
                                        showQuickJumper={false}
                                        showTotal={(total) => `共 ${total} 条`}
                                        pageSizeOptions={['5', '10', '20']}
                                        size="small"
                                        simple
                                    />
                                </div>
                            </>
                        )}
                    </div>
                </Card>
            </Sider>

            <Content style={{
                padding: 16,
                backgroundColor: '#f5f5f5',
                overflow: 'auto'
            }}>
                {renderReportDetail()}
            </Content>
        </Layout>
    );
};

export default ReportList;