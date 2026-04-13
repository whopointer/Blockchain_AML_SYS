// src/components/alerts/AlertList.tsx
import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
    Table, Card, Tag, Space, Button, Pagination, Tooltip, Popconfirm,
    message, Spin, Modal, Input, Select, Slider, Row, Col, Statistic,
    Progress, Badge, Descriptions, Divider
} from "antd";
import {
    EyeOutlined, DeleteOutlined, FilterOutlined, CheckCircleOutlined,
    ReloadOutlined, ExclamationCircleOutlined, ClockCircleOutlined,
    SortAscendingOutlined, SortDescendingOutlined, SearchOutlined
} from "@ant-design/icons";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/zh-cn";

// 导入API和hooks
import { alertsApi } from "../../services/Alerts/api";
import { useAlertStatistics } from "../../services/Alerts/useAlerts";
import {
    Alert,
    AlertStatus,
    AlertLevel,
    AlertQueryParams,
    UpdateAlertStatusRequest
} from "../../services/Alerts/types";

// 导入工具函数
import {
    formatRiskScore,
    getRiskColor,
    formatDateTime,
    formatTimeAgo,
    truncateAddress,
    getStatusLabel,
    getStatusColor,
    sortAlerts,
    filterAlerts
} from "../../services/Alerts/utils";

// 扩展dayjs
dayjs.extend(relativeTime);
dayjs.locale('zh-cn');

// 定义扩展的表格数据接口
interface TableAlert extends Alert {
    key: string;
    short_address: string;
    formatted_date: string;
    time_ago: string;
    risk_level: AlertLevel;
    risk_percent: number;
}

// 告警详情接口
interface AlertDetailProps {
    alert: Alert;
    visible: boolean;
    onClose: () => void;
    onStatusChange: (alertId: string, status: AlertStatus) => Promise<void>;
    loading: boolean;
}

// 状态筛选选项
const statusOptions = [
    { value: undefined, label: '全部状态' },
    { value: 'PENDING', label: '未查看' },
    { value: 'CHECKED', label: '已查看' },
    { value: 'ACKNOWLEDGED', label: '已确认' }
];

// 风险等级筛选选项
const riskLevelOptions = [
    { value: undefined, label: '全部等级' },
    { value: 'LOW', label: '低风险' },
    { value: 'MEDIUM', label: '中风险' },
    { value: 'HIGH', label: '高风险' },
    { value: 'CRITICAL', label: '极高风险' }
];

const AlertList: React.FC = () => {
    // 状态管理
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [pageSize, setPageSize] = useState<number>(10);
    const [totalAlerts, setTotalAlerts] = useState<number>(0);

    // 筛选状态
    const [filters, setFilters] = useState<AlertQueryParams>({
        page: 1,
        pageSize: 10,
        sortBy: 'created_at',
        sortOrder: 'desc'
    });

    // 详情模态框状态
    const [detailVisible, setDetailVisible] = useState<boolean>(false);
    const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
    const [updatingStatus, setUpdatingStatus] = useState<boolean>(false);

    // 高级筛选面板状态
    const [showAdvancedFilter, setShowAdvancedFilter] = useState<boolean>(false);

    // 使用统计hook
    const { statistics, loading: statsLoading } = useAlertStatistics();

    // 加载告警列表
    const loadAlerts = useCallback(async (queryParams?: AlertQueryParams) => {
        setLoading(true);
        try {
            const response = await alertsApi.getAlerts(queryParams || filters);
            setAlerts(response.alerts);
            setTotalAlerts(response.total);

            // 更新当前分页信息
            if (queryParams?.page) {
                setCurrentPage(queryParams.page);
            }
            if (queryParams?.pageSize) {
                setPageSize(queryParams.pageSize);
            }
        } catch (err: any) {
            console.error('加载告警失败:', err);
            message.error(err.message || '加载告警列表失败，请稍后重试');
        } finally {
            setLoading(false);
        }
    }, [filters]);

    // 组件挂载时加载告警
    useEffect(() => {
        loadAlerts();
    }, [loadAlerts]);

    // 更新告警状态
    const updateAlertStatus = async (alertId: string, status: AlertStatus, additional_metadata?: any) => {
        setUpdatingStatus(true);
        try {
            const request: UpdateAlertStatusRequest = { status };
            if (additional_metadata) {
                request.additional_metadata = additional_metadata;
            }

            await alertsApi.updateAlertStatus(alertId, request);

            // 更新本地状态
            setAlerts(prev => prev.map(alert =>
                alert.id === alertId ? { ...alert, status } : alert
            ));

            message.success(`告警状态已更新为${getStatusLabel(status)}`);
            return true;
        } catch (err: any) {
            console.error('更新告警状态失败:', err);
            message.error(err.message || '更新告警状态失败');
            return false;
        } finally {
            setUpdatingStatus(false);
        }
    };

    // 删除告警
    const handleDeleteAlert = async (alertId: string) => {
        try {
            await alertsApi.deleteAlert(alertId);

            // 更新本地状态
            setAlerts(prev => prev.filter(alert => alert.id !== alertId));
            setTotalAlerts(prev => prev - 1);

            message.success('告警删除成功');
        } catch (err: any) {
            console.error('删除告警失败:', err);
            message.error(err.message || '删除告警失败');
        }
    };

    // 处理查看告警详情
    const handleViewAlertDetail = async (alert: Alert) => {
        setSelectedAlert(alert);
        setDetailVisible(true);

        // 如果状态是PENDING，自动更新为CHECKED
        if (alert.status === 'PENDING') {
            await updateAlertStatus(alert.id, 'CHECKED', {
                first_viewed_at: new Date().toISOString(),
                viewed_by: 'user' // 这里可以替换为实际用户信息
            });
            alert.status = 'CHECKED';
        }
    };

    // 处理确认告警
    const handleAcknowledgeAlert = async () => {
        if (!selectedAlert) return;

        const success = await updateAlertStatus(selectedAlert.id, 'ACKNOWLEDGED', {
            acknowledged_at: new Date().toISOString(),
            acknowledged_by: 'user' // 这里可以替换为实际用户信息
        });

        if (success) {
            setDetailVisible(false);
        }
    };

    // 处理筛选变化
    const handleFilterChange = (key: keyof AlertQueryParams, value: any) => {
        const newFilters = { ...filters, [key]: value, page: 1 }; // 重置到第一页
        setFilters(newFilters);
        loadAlerts(newFilters);
    };

    // 处理地址搜索
    const handleAddressSearch = (value: string) => {
        handleFilterChange('address', value || undefined);
    };

    // 处理状态筛选
    const handleStatusFilter = (status: AlertStatus | undefined) => {
        handleFilterChange('status', status);
    };

    // 处理风险评分范围筛选
    const handleRiskFilter = (value: [number, number]) => {
        handleFilterChange('risk_min', value[0] / 100);
        handleFilterChange('risk_max', value[1] / 100);
    };

    // 处理排序
    const handleSort = (sortBy: 'created_at' | 'risk_score') => {
        const newSortOrder = filters.sortBy === sortBy && filters.sortOrder === 'desc' ? 'asc' : 'desc';
        handleFilterChange('sortBy', sortBy);
        handleFilterChange('sortOrder', newSortOrder);
    };

    // 重置筛选
    const handleResetFilters = () => {
        const defaultFilters: AlertQueryParams = {
            page: 1,
            pageSize: 10,
            sortBy: 'created_at',
            sortOrder: 'desc'
        };
        setFilters(defaultFilters);
        loadAlerts(defaultFilters);
    };

    // 获取状态标签
    const getStatusTag = (status: AlertStatus) => {
        const config = {
            PENDING: { color: '#f5222d', text: '未查看' },
            CHECKED: { color: '#fa8c16', text: '已查看' },
            ACKNOWLEDGED: { color: '#52c41a', text: '已确认' }
        };

        return (
            <Tag
                color={config[status].color}
                style={{ fontWeight: 'bold' }}
            >
                {config[status].text}
            </Tag>
        );
    };

    // 获取风险等级标签
    const getRiskLevelTag = (score: number) => {
        let level: AlertLevel = 'LOW';
        if (score >= 0.8) level = 'CRITICAL';
        else if (score >= 0.6) level = 'HIGH';
        else if (score >= 0.4) level = 'MEDIUM';

        const config = {
            LOW: { color: '#52c41a', text: '低风险' },
            MEDIUM: { color: '#faad14', text: '中风险' },
            HIGH: { color: '#fa8c16', text: '高风险' },
            CRITICAL: { color: '#f5222d', text: '极高风险' }
        };

        return (
            <Tag color={config[level].color}>
                {config[level].text}
            </Tag>
        );
    };

    // 准备表格数据（包含排序逻辑）
    const prepareTableData = (): TableAlert[] => {
        // 首先排序：PENDING和CHECKED置顶，然后按时间排序
        const sortedAlerts = [...alerts].sort((a, b) => {
            // 状态优先级：PENDING > CHECKED > ACKNOWLEDGED
            const statusPriority = { PENDING: 0, CHECKED: 1, ACKNOWLEDGED: 2 };

            if (statusPriority[a.status] !== statusPriority[b.status]) {
                return statusPriority[a.status] - statusPriority[b.status];
            }

            // 状态相同，按创建时间降序
            return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        });

        return sortedAlerts.map(alert => {
            const riskPercent = Math.round(alert.risk_score * 100);
            let riskLevel: AlertLevel = 'LOW';
            if (alert.risk_score >= 0.8) riskLevel = 'CRITICAL';
            else if (alert.risk_score >= 0.6) riskLevel = 'HIGH';
            else if (alert.risk_score >= 0.4) riskLevel = 'MEDIUM';

            return {
                ...alert,
                key: alert.id,
                short_address: truncateAddress(alert.monitored_address_id),
                formatted_date: formatDateTime(alert.created_at),
                time_ago: formatTimeAgo(alert.created_at),
                risk_level: riskLevel,
                risk_percent: riskPercent
            };
        });
    };

    // 表格列定义
    const columns = [
        {
            title: "监控地址",
            dataIndex: "monitored_address_id",
            key: "address",
            width: 200,
            render: (text: string, record: TableAlert) => (
                <div>
                    <div style={{ fontFamily: "monospace", fontSize: "12px" }}>
                        {record.short_address}
                    </div>
                    <div style={{ fontSize: "11px", color: "#999", marginTop: 2 }}>
                        创建于: {record.time_ago}
                    </div>
                </div>
            )
        },
        {
            title: "风险评分",
            dataIndex: "risk_score",
            key: "risk_score",
            width: 150,
            sorter: true,
            render: (score: number, record: TableAlert) => (
                <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <Progress
                            percent={record.risk_percent}
                            size="small"
                            strokeColor={getRiskColor(score)}
                            format={() => formatRiskScore(score)}
                        />
                        {getRiskLevelTag(score)}
                    </div>
                </div>
            )
        },
        {
            title: "状态",
            dataIndex: "status",
            key: "status",
            width: 100,
            filters: [
                { text: '未查看', value: 'PENDING' },
                { text: '已查看', value: 'CHECKED' },
                { text: '已确认', value: 'ACKNOWLEDGED' }
            ],
            onFilter: (value: any, record: TableAlert) => record.status === value,
            render: (status: AlertStatus) => getStatusTag(status)
        },
        {
            title: "创建时间",
            dataIndex: "created_at",
            key: "created_at",
            width: 150,
            sorter: true,
            render: (date: string, record: TableAlert) => (
                <Tooltip title={record.formatted_date}>
                    <span>{record.time_ago}</span>
                </Tooltip>
            )
        },
        {
            title: "操作",
            key: "action",
            width: 120,
            render: (_: any, record: TableAlert) => (
                <Space size="small">
                    <Tooltip title="查看详情">
                        <Button
                            type="text"
                            size="small"
                            icon={<EyeOutlined />}
                            onClick={() => handleViewAlertDetail(record)}
                        />
                    </Tooltip>
                    <Popconfirm
                        title="确认删除告警"
                        description="此操作将永久删除该告警，是否继续？"
                        onConfirm={() => handleDeleteAlert(record.id)}
                        okText="确定"
                        cancelText="取消"
                        okButtonProps={{ danger: true }}
                    >
                        <Tooltip title="删除告警">
                            <Button
                                type="text"
                                size="small"
                                danger
                                icon={<DeleteOutlined />}
                            />
                        </Tooltip>
                    </Popconfirm>
                </Space>
            )
        }
    ];

    // 渲染统计面板
    const renderStatistics = () => {
        if (statsLoading || !statistics) return null;

        return (
            <Row gutter={16} style={{ marginBottom: 16 }}>
                <Col span={6}>
                    <Card size="small">
                        <Statistic
                            title="总告警数"
                            value={statistics.total_alerts}
                            valueStyle={{ color: '#1890ff' }}
                            prefix={<ExclamationCircleOutlined />}
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card size="small">
                        <Statistic
                            title="待处理告警"
                            value={statistics.pending_alerts}
                            valueStyle={{ color: '#f5222d' }}
                            prefix={<ClockCircleOutlined />}
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card size="small">
                        <Statistic
                            title="平均风险"
                            value={statistics.avg_risk_score.toFixed(2)}
                            valueStyle={{ color: '#fa8c16' }}
                            suffix="/1.0"
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card size="small">
                        <Statistic
                            title="最高风险"
                            value={statistics.max_risk_score.toFixed(2)}
                            valueStyle={{ color: '#f5222d' }}
                            suffix="/1.0"
                        />
                    </Card>
                </Col>
            </Row>
        );
    };

    // 渲染筛选面板
    const renderFilterPanel = () => (
        <Card
            size="small"
            style={{ marginBottom: 16 }}
            title={
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <FilterOutlined />
                    <span>筛选条件</span>
                </div>
            }
            extra={
                <Button
                    size="small"
                    type="link"
                    onClick={() => setShowAdvancedFilter(!showAdvancedFilter)}
                >
                    {showAdvancedFilter ? '隐藏高级筛选' : '显示高级筛选'}
                </Button>
            }
        >
            <Row gutter={16} align="middle">
                <Col span={6}>
                    <Input
                        placeholder="搜索地址..."
                        prefix={<SearchOutlined />}
                        allowClear
                        onChange={(e) => handleAddressSearch(e.target.value)}
                        style={{ width: '100%' }}
                    />
                </Col>
                <Col span={4}>
                    <Select
                        style={{ width: '100%' }}
                        placeholder="状态筛选"
                        value={filters.status}
                        onChange={handleStatusFilter}
                        options={statusOptions}
                    />
                </Col>
                <Col span={4}>
                    <Button
                        icon={filters.sortOrder === 'desc' ? <SortDescendingOutlined /> : <SortAscendingOutlined />}
                        onClick={() => handleSort('created_at')}
                    >
                        按时间排序
                    </Button>
                </Col>
                <Col span={4}>
                    <Button
                        icon={filters.sortOrder === 'desc' ? <SortDescendingOutlined /> : <SortAscendingOutlined />}
                        onClick={() => handleSort('risk_score')}
                    >
                        按风险排序
                    </Button>
                </Col>
                <Col span={4}>
                    <Button onClick={handleResetFilters}>
                        重置筛选
                    </Button>
                </Col>
            </Row>

            {showAdvancedFilter && (
                <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #f0f0f0' }}>
                    <Row gutter={16}>
                        <Col span={12}>
                            <div style={{ marginBottom: 8 }}>
                                <span>风险评分范围: </span>
                                <span style={{ fontWeight: 'bold' }}>
                  {((filters.risk_min || 0) * 100).toFixed(0)}% - {((filters.risk_max || 1) * 100).toFixed(0)}%
                </span>
                            </div>
                            <Slider
                                range
                                min={0}
                                max={100}
                                value={[
                                    (filters.risk_min || 0) * 100,
                                    (filters.risk_max || 1) * 100
                                ]}
                                onChange={handleRiskFilter}
                                tooltip={{ formatter: (value) => `${value}%` }}
                            />
                        </Col>
                    </Row>
                </div>
            )}
        </Card>
    );

    // 渲染空状态
    const renderEmpty = () => (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <ExclamationCircleOutlined style={{ fontSize: 48, color: '#ccc', marginBottom: 16 }} />
            <p>暂无告警数据</p>
            <Button
                type="primary"
                onClick={() => loadAlerts()}
                icon={<ReloadOutlined />}
            >
                刷新
            </Button>
        </div>
    );

    // 告警详情组件
    const AlertDetail: React.FC<AlertDetailProps> = ({ alert, visible, onClose, onStatusChange, loading }) => {
        if (!alert) return null;

        const riskPercent = Math.round(alert.risk_score * 100);
        const isPending = alert.status === 'PENDING';
        const isChecked = alert.status === 'CHECKED';

        return (
            <Modal
                title="告警详情"
                open={visible}
                onCancel={onClose}
                width={800}
                footer={[
                    <Button key="close" onClick={onClose}>
                        关闭
                    </Button>,
                    isChecked && (
                        <Button
                            key="acknowledge"
                            type="primary"
                            icon={<CheckCircleOutlined />}
                            loading={loading}
                            onClick={() => onStatusChange(alert.id, 'ACKNOWLEDGED')}
                        >
                            确认完毕
                        </Button>
                    )
                ]}
            >
                <Spin spinning={loading}>
                    <div style={{ padding: 16 }}>
                        {/* 基本信息 */}
                        <Card size="small" style={{ marginBottom: 16 }}>
                            <Descriptions column={2} bordered>
                                <Descriptions.Item label="告警ID" span={2}>
                                    <code>{alert.id}</code>
                                </Descriptions.Item>
                                <Descriptions.Item label="监控地址">
                                    <div style={{ fontFamily: 'monospace' }}>
                                        {alert.monitored_address_id}
                                    </div>
                                </Descriptions.Item>
                                <Descriptions.Item label="创建时间">
                                    {formatDateTime(alert.created_at)}
                                </Descriptions.Item>
                                <Descriptions.Item label="风险评分">
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                        <Progress
                                            percent={riskPercent}
                                            size="small"
                                            strokeColor={getRiskColor(alert.risk_score)}
                                            format={() => formatRiskScore(alert.risk_score)}
                                        />
                                        {getRiskLevelTag(alert.risk_score)}
                                    </div>
                                </Descriptions.Item>
                                <Descriptions.Item label="当前状态">
                                    {getStatusTag(alert.status)}
                                </Descriptions.Item>
                            </Descriptions>
                        </Card>

                        {/* 风险分析 */}
                        {alert.metadata && (
                            <Card
                                size="small"
                                title="风险分析详情"
                                style={{ marginBottom: 16 }}
                            >
                                {alert.metadata.alert_level && (
                                    <div style={{ marginBottom: 8 }}>
                                        <strong>告警级别: </strong>
                                        {getRiskLevelTag(alert.risk_score)}
                                    </div>
                                )}

                                {alert.metadata.transaction_count && (
                                    <div style={{ marginBottom: 8 }}>
                                        <strong>交易数量: </strong>
                                        {alert.metadata.transaction_count} 笔
                                    </div>
                                )}

                                {alert.metadata.highest_risk_transaction && (
                                    <>
                                        <Divider orientation="left" plain>最高风险交易</Divider>
                                        <Descriptions column={1} size="small">
                                            <Descriptions.Item label="交易哈希">
                                                <code>{alert.metadata.highest_risk_transaction.hash}</code>
                                            </Descriptions.Item>
                                            <Descriptions.Item label="风险评分">
                                                {formatRiskScore(alert.metadata.highest_risk_transaction.risk_score)}
                                            </Descriptions.Item>
                                            <Descriptions.Item label="风险因素">
                                                {alert.metadata.highest_risk_transaction.risk_factors?.join(', ') || '无'}
                                            </Descriptions.Item>
                                        </Descriptions>
                                    </>
                                )}

                                {alert.metadata.analysis && (
                                    <>
                                        <Divider orientation="left" plain>分析说明</Divider>
                                        <p style={{ whiteSpace: 'pre-wrap' }}>
                                            {alert.metadata.analysis}
                                        </p>
                                    </>
                                )}

                                {alert.metadata.prediction_statistics && (
                                    <>
                                        <Divider orientation="left" plain>预测统计</Divider>
                                        <Row gutter={16}>
                                            <Col span={8}>
                                                <div>总交易数: {alert.metadata.prediction_statistics.total_transactions}</div>
                                            </Col>
                                            <Col span={8}>
                                                <div>平均风险: {alert.metadata.prediction_statistics.avg_risk_score.toFixed(3)}</div>
                                            </Col>
                                            <Col span={8}>
                                                <div>最高风险: {alert.metadata.prediction_statistics.max_risk_score.toFixed(3)}</div>
                                            </Col>
                                        </Row>
                                    </>
                                )}
                            </Card>
                        )}

                        {/* 状态变更记录 */}
                        {alert.metadata?.generated_at && (
                            <Card size="small" title="时间戳">
                                <div>生成时间: {formatDateTime(alert.metadata.generated_at)}</div>
                            </Card>
                        )}
                    </div>
                </Spin>
            </Modal>
        );
    };

    return (
        <div style={{ padding: 24 }}>
            {renderStatistics()}
            {renderFilterPanel()}

            <Card
                title={
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <div style={{ display: "flex", alignItems: "center" }}>
                            <ExclamationCircleOutlined style={{ marginRight: 8, color: '#f5222d' }} />
                            <span>告警管理</span>
                            {loading && <Spin size="small" style={{ marginLeft: 12 }} />}
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <Badge
                                count={statistics?.pending_alerts || 0}
                                style={{ backgroundColor: '#f5222d' }}
                            />
                            <Button
                                icon={<ReloadOutlined />}
                                onClick={() => loadAlerts()}
                                loading={loading}
                                size="small"
                            >
                                刷新
                            </Button>
                        </div>
                    </div>
                }
                bordered={false}
                style={{ borderRadius: 8 }}
                extra={
                    <div style={{ fontSize: 12, color: '#999' }}>
                        共 {totalAlerts} 个告警
                        {filters.status && `，${getStatusLabel(filters.status)}: ${alerts.length}`}
                    </div>
                }
            >
                {alerts.length === 0 && !loading ? (
                    renderEmpty()
                ) : (
                    <>
                        <Table
                            columns={columns}
                            dataSource={prepareTableData()}
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
                                total={totalAlerts}
                                onChange={(page) => handleFilterChange('page', page)}
                                onShowSizeChange={(current, size) => {
                                    handleFilterChange('pageSize', size);
                                    handleFilterChange('page', 1);
                                }}
                                showSizeChanger
                                showQuickJumper
                                showTotal={(total, range) =>
                                    `第 ${range[0]}-${range[1]} 条，共 ${total} 条`
                                }
                                pageSizeOptions={['10', '20', '50', '100']}
                            />
                        </div>
                    </>
                )}
            </Card>

            {/* 告警详情模态框 */}
            {selectedAlert && (
                <AlertDetail
                    alert={selectedAlert}
                    visible={detailVisible}
                    onClose={() => setDetailVisible(false)}
                    onStatusChange={updateAlertStatus}
                    loading={updatingStatus}
                />
            )}
        </div>
    );
};

export default AlertList;