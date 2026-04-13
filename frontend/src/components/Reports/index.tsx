// ReportList.tsx
import React, { useState, useEffect, useCallback } from "react";
import { Table, Card, Tag, Space, Button, Pagination, Tooltip, Popconfirm, message, Spin, Alert } from "antd";
import { EyeOutlined, DeleteOutlined, DownloadOutlined, FileTextOutlined, ReloadOutlined } from "@ant-design/icons";
import dayjs from "dayjs";

// 导入API
import {
    getReportList,
    previewReport,
    deleteReport,
    downloadReportFile,
    isValidEthereumAddress,
    formatReportDate
} from "../../services/report/index";

// 导入hooks
import { useReportOperations } from "../../services/report/useReportOpertations";

// 定义报告类型
interface Report {
    id: number;
    target_address: string;
    file_path: string;
    created_at: string;
    title: string;
}

// 扩展的表格数据接口
interface TableReport extends Report {
    key: number;
    address_type: 'ethereum' | 'bitcoin' | 'other';
    short_address: string;
    formatted_date: string;
}

const ReportList: React.FC = () => {
    const [reports, setReports] = useState<Report[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [pageSize, setPageSize] = useState<number>(5);
    const [totalReports, setTotalReports] = useState<number>(0);

    // 使用自定义hook
    const {
        delete: deleteReportAction,
        isDeleting,
        error,
        success,
        clearStatus
    } = useReportOperations();

    // 加载报告列表
    const loadReports = useCallback(async () => {
        setLoading(true);
        try {
            const data = await getReportList();
            setReports(data);
            setTotalReports(data.length);
        } catch (err) {
            console.error('加载报告失败:', err);
            message.error('加载报告列表失败，请稍后重试');
        } finally {
            setLoading(false);
        }
    }, []);

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
    const handleDeleteReport = async (id: number) => {
        try {
            await deleteReportAction(id);
            // 删除成功后重新加载列表
            loadReports();
        } catch (err) {
            // 错误信息已经在hook中处理
            console.error('删除报告失败:', err);
        }
    };

    // 预览报告
    const handlePreviewReport = async (id: number) => {
        try {
            await previewReport(id);
        } catch (err: any) {
            message.error(err.message || '预览报告失败');
        }
    };

    // 下载报告
    const handleDownloadReport = async (id: number, title: string) => {
        try {
            setLoading(true);
            const blob = await downloadReportFile(id);

            // 创建下载链接
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${title.replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '_')}.pdf`;
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
            ethereum: { color: 'blue', text: '以太坊' },
            bitcoin: { color: 'orange', text: '比特币' },
            other: { color: 'gray', text: '其他' }
        };
        return <Tag color={config[type].color}>{config[type].text}</Tag>;
    };

    // 格式化地址显示
    const formatAddress = (address: string): string => {
        if (address.length <= 20) return address;
        return `${address.substring(0, 10)}...${address.substring(address.length - 8)}`;
    };

    // 准备表格数据
    const prepareTableData = (): TableReport[] => {
        return reports.map(report => ({
            ...report,
            key: report.id,
            address_type: getAddressType(report.target_address),
            short_address: formatAddress(report.target_address),
            formatted_date: formatReportDate(report.created_at)
        }));
    };

    // 表格列定义
    const columns = [
        {
            title: "ID",
            dataIndex: "id",
            key: "id",
            width: 80,
            sorter: (a: TableReport, b: TableReport) => a.id - b.id,
        },
        {
            title: "创建时间",
            dataIndex: "formatted_date",
            key: "formatted_date",
            width: 150,
            sorter: (a: TableReport, b: TableReport) =>
                new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
        },
        {
            title: "目标地址",
            dataIndex: "target_address",
            key: "target_address",
            width: 250,
            render: (text: string, record: TableReport) => (
                <div>
                    {getAddressTypeTag(text)}
                    <Tooltip title={text}>
                        <div style={{
                            fontFamily: "monospace",
                            fontSize: "12px",
                            marginTop: 4,
                            cursor: 'pointer',
                            backgroundColor: '#f5f5f5',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            wordBreak: 'break-all'
                        }}>
                            {record.short_address}
                        </div>
                    </Tooltip>
                </div>
            )
        },
        {
            title: "报告标题",
            dataIndex: "title",
            key: "title",
            render: (text: string) => (
                <Tooltip title={text}>
                    <div style={{
                        maxWidth: 200,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap"
                    }}>
                        {text}
                    </div>
                </Tooltip>
            )
        },
        {
            title: "操作",
            key: "action",
            width: 140,
            render: (_: any, record: TableReport) => (
                <Space size="small">
                    <Tooltip title="预览报告">
                        <Button
                            type="text"
                            size="small"
                            icon={<EyeOutlined />}
                            onClick={() => handlePreviewReport(record.id)}
                            loading={loading}
                            disabled={isDeleting(record.id)}
                        />
                    </Tooltip>
                    <Tooltip title="下载报告">
                        <Button
                            type="text"
                            size="small"
                            icon={<DownloadOutlined />}
                            onClick={() => handleDownloadReport(record.id, record.title)}
                            loading={loading}
                            disabled={isDeleting(record.id)}
                        />
                    </Tooltip>
                    <Popconfirm
                        title="确认删除报告"
                        description="此操作将永久删除该报告，是否继续？"
                        onConfirm={() => handleDeleteReport(record.id)}
                        okText="确定"
                        cancelText="取消"
                        okButtonProps={{ danger: true }}
                    >
                        <Tooltip title="删除报告">
                            <Button
                                type="text"
                                size="small"
                                danger
                                icon={<DeleteOutlined />}
                                loading={isDeleting(record.id)}
                                disabled={loading}
                            />
                        </Tooltip>
                    </Popconfirm>
                </Space>
            )
        }
    ];

    // 分页数据
    const paginatedReports = prepareTableData().slice(
        (currentPage - 1) * pageSize,
        currentPage * pageSize
    );

    // 渲染空状态
    const renderEmpty = () => (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <FileTextOutlined style={{ fontSize: 48, color: '#ccc', marginBottom: 16 }} />
            <p>暂无报告数据</p>
            <Button
                type="primary"
                onClick={loadReports}
                icon={<ReloadOutlined />}
            >
                刷新
            </Button>
        </div>
    );

    return (
        <div style={{ padding: 24 }}>
            <Card
                title={
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <div style={{ display: "flex", alignItems: "center" }}>
                            <FileTextOutlined style={{ marginRight: 8 }} />
                            <span>合规报告列表</span>
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
                style={{ borderRadius: 8 }}
                extra={
                    <div style={{ fontSize: 12, color: '#999' }}>
                        共 {totalReports} 个报告
                    </div>
                }
            >
                {reports.length === 0 && !loading ? (
                    renderEmpty()
                ) : (
                    <>
                        <Table
                            columns={columns}
                            dataSource={paginatedReports}
                            rowKey="id"
                            pagination={false}
                            size="middle"
                            bordered
                            scroll={{ x: 700 }}
                            loading={loading}
                            locale={{ emptyText: '暂无数据' }}
                        />

                        <div style={{ marginTop: 16, textAlign: "right" }}>
                            <Pagination
                                current={currentPage}
                                pageSize={pageSize}
                                total={totalReports}
                                onChange={setCurrentPage}
                                onShowSizeChange={(current, size) => {
                                    setPageSize(size);
                                    setCurrentPage(1);
                                }}
                                showSizeChanger
                                showQuickJumper
                                showTotal={(total, range) =>
                                    `第 ${range[0]}-${range[1]} 条，共 ${total} 条`
                                }
                                pageSizeOptions={['5', '10', '20', '50']}
                            />
                        </div>
                    </>
                )}
            </Card>
        </div>
    );
};

export default ReportList;