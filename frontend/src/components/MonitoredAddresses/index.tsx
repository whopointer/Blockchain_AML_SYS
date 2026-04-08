import React, { useState, useEffect, useCallback } from "react";
import {
    Table,
    Card,
    Row,
    Col,
    Tag,
    Space,
    Button,
    Input,
    Select,
    DatePicker,
    Pagination,
    Spin,
    message,
    Popconfirm,
    Tooltip,
    Modal,
    Form,
    Switch
} from "antd";
import {
    SearchOutlined,
    ReloadOutlined,
    PlusOutlined,
    EditOutlined,
    DeleteOutlined,
    EyeOutlined,
    FilterOutlined
} from "@ant-design/icons";
import dayjs from "dayjs";
import "dayjs/locale/zh-cn";
import {
    MonitoredAddress,
    CreateMonitoredAddressRequest
} from "./types";
import {
    monitoredAddressApi,
    convertDTOToAddress,
    apiClient
} from "../../services/monitoredAddress";
import api from "../../services/api";

dayjs.locale("zh-cn");

// 定义搜索参数类型
interface SearchParams {
    keyword?: string;
    addressType?: string;
    riskLevel?: string;
    monitoringEnabled?: boolean;
    startDate?: string;
    endDate?: string;
}

// 地址类型选项
const addressTypeOptions = [
    { value: "WALLET", label: "钱包" },
    { value: "CONTRACT", label: "合约" },
    { value: "EXCHANGE", label: "交易所" }
];

// 风险等级选项
const riskLevelOptions = [
    { value: "LOW", label: "低风险" },
    { value: "MEDIUM", label: "中风险" },
    { value: "HIGH", label: "高风险" },
    { value: "CRITICAL", label: "严重风险" }
];

// 监控状态选项
const monitoringStatusOptions = [
    { value: "all", label: "全部状态" },
    { value: "enabled", label: "监控中" },
    { value: "disabled", label: "已停用" }
];

const AddressListPage: React.FC = () => {
    // 状态管理
    const [addresses, setAddresses] = useState<MonitoredAddress[]>([]);
    const [filteredAddresses, setFilteredAddresses] = useState<MonitoredAddress[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [creating, setCreating] = useState<boolean>(false);
    const [updating, setUpdating] = useState<boolean>(false);
    const [deleting, setDeleting] = useState<boolean>(false);
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [pageSize, setPageSize] = useState<number>(10);
    const [totalItems, setTotalItems] = useState<number>(0);

    // 筛选条件
    const [searchText, setSearchText] = useState<string>("");
    const [selectedAddressType, setSelectedAddressType] = useState<string>("all");
    const [selectedRiskLevel, setSelectedRiskLevel] = useState<string>("all");
    const [selectedMonitoringStatus, setSelectedMonitoringStatus] = useState<string>("all");
    const [selectedDateRange, setSelectedDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null);

    // 模态框状态
    const [isModalVisible, setIsModalVisible] = useState<boolean>(false);
    const [isEditMode, setIsEditMode] = useState<boolean>(false);
    const [currentAddress, setCurrentAddress] = useState<MonitoredAddress | null>(null);
    const [form] = Form.useForm();

    // 获取所有地址
    const fetchAddresses = useCallback(async () => {
        setLoading(true);
        try {
            console.log("=== 开始获取地址 ===");
            console.log("当前 baseURL:", apiClient.defaults.baseURL);

            const addresses = await monitoredAddressApi.getAll();
            console.log("成功获取地址:", addresses);

            setAddresses(addresses);
            setFilteredAddresses(addresses);
            setTotalItems(addresses.length);
            message.success("地址列表加载成功");
        } catch (error) {
            console.error("获取地址列表失败:", error);
            console.error("错误详情:", {
                message: (error as Error).message
            });
            message.error("获取地址列表失败，请稍后重试");
        } finally {
            setLoading(false);
        }
    }, []);

    // 初始化加载数据
    useEffect(() => {
        fetchAddresses();
    }, [fetchAddresses]);

    // 应用前端筛选条件
    useEffect(() => {
        let result = [...addresses];

        // 按搜索文本筛选
        if (searchText) {
            result = result.filter(address =>
                address.address.toLowerCase().includes(searchText.toLowerCase()) ||
                (address.description && address.description.toLowerCase().includes(searchText.toLowerCase()))
            );
        }

        // 按地址类型筛选
        if (selectedAddressType !== "all") {
            result = result.filter(address => address.addressType === selectedAddressType);
        }

        // 按风险等级筛选
        if (selectedRiskLevel !== "all") {
            result = result.filter(address => address.riskLevel === selectedRiskLevel);
        }

        // 按监控状态筛选
        if (selectedMonitoringStatus !== "all") {
            const enabled = selectedMonitoringStatus === "enabled";
            result = result.filter(address => address.monitoringEnabled === enabled);
        }

        // 按创建时间范围筛选
        if (selectedDateRange && selectedDateRange[0] && selectedDateRange[1]) {
            const [startDate, endDate] = selectedDateRange;
            result = result.filter(address => {
                const createdAt = dayjs(address.createdAt);
                return createdAt.isAfter(startDate) && createdAt.isBefore(endDate);
            });
        }

        setFilteredAddresses(result);
        setTotalItems(result.length);
        setCurrentPage(1); // 重置到第一页
    }, [searchText, selectedAddressType, selectedRiskLevel, selectedMonitoringStatus, selectedDateRange, addresses]);

    // 分页数据
    const paginatedAddresses = filteredAddresses.slice(
        (currentPage - 1) * pageSize,
        currentPage * pageSize
    );

    // 获取风险等级标签颜色
    const getRiskLevelColor = (riskLevel: string) => {
        switch (riskLevel) {
            case "LOW":
                return "green";
            case "MEDIUM":
                return "yellow";
            case "HIGH":
                return "orange";
            case "CRITICAL":
                return "red";
            default:
                return "default";
        }
    };

    // 获取地址类型标签
    const getAddressTypeTag = (addressType: string) => {
        const typeMap: Record<string, { color: string; label: string }> = {
            "WALLET": { color: "blue", label: "钱包" },
            "CONTRACT": { color: "purple", label: "合约" },
            "EXCHANGE": { color: "cyan", label: "交易所" }
        };

        const typeInfo = typeMap[addressType] || { color: "default", label: addressType };
        return <Tag color={typeInfo.color}>{typeInfo.label}</Tag>;
    };

    // 获取监控状态标签
    const getMonitoringStatusTag = (enabled: boolean) => {
        if (enabled) {
            return <Tag color="green">监控中</Tag>;
        } else {
            return <Tag color="red">已停用</Tag>;
        }
    };

    // 查看详情
    const handleViewDetails = async (address: MonitoredAddress) => {
        try {
            const detailedAddress = await monitoredAddressApi.getById(address.id);
            Modal.info({
                title: "地址详情",
                width: 600,
                content: (
                    <div style={{ marginTop: 20 }}>
                        <Row gutter={[16, 16]}>
                            <Col span={12}>
                                <strong>地址ID:</strong> {detailedAddress.id}
                            </Col>
                            <Col span={12}>
                                <strong>类型:</strong> {getAddressTypeTag(detailedAddress.addressType)}
                            </Col>
                            <Col span={12}>
                                <strong>地址:</strong>
                                <div style={{ fontFamily: "monospace", marginTop: 4 }}>
                                    {detailedAddress.address}
                                </div>
                            </Col>
                            <Col span={12}>
                                <strong>风险等级:</strong>
                                <div style={{ marginTop: 4 }}>
                                    <Tag color={getRiskLevelColor(detailedAddress.riskLevel)}>
                                        {detailedAddress.riskLevel}
                                    </Tag>
                                </div>
                            </Col>
                            <Col span={12}>
                                <strong>监控状态:</strong>
                                <div style={{ marginTop: 4 }}>
                                    {getMonitoringStatusTag(detailedAddress.monitoringEnabled)}
                                </div>
                            </Col>
                            <Col span={12}>
                                <strong>创建时间:</strong>
                                <div style={{ marginTop: 4 }}>
                                    {dayjs(detailedAddress.createdAt).format("YYYY-MM-DD HH:mm:ss")}
                                </div>
                            </Col>
                            <Col span={12}>
                                <strong>更新时间:</strong>
                                <div style={{ marginTop: 4 }}>
                                    {dayjs(detailedAddress.updatedAt).format("YYYY-MM-DD HH:mm:ss")}
                                </div>
                            </Col>
                            <Col span={24}>
                                <strong>备注:</strong>
                                <div style={{ marginTop: 4, padding: 8, background: "#f5f5f5", borderRadius: 4 }}>
                                    {detailedAddress.description || "无备注"}
                                </div>
                            </Col>
                        </Row>
                    </div>
                ),
                okText: "关闭"
            });
        } catch (error) {
            message.error("获取地址详情失败");
        }
    };

    // 打开编辑/添加模态框
    const handleOpenModal = (address?: MonitoredAddress) => {
        if (address) {
            // 编辑模式
            setIsEditMode(true);
            setCurrentAddress(address);
            form.setFieldsValue({
                ...address,
                createdAt: dayjs(address.createdAt),
                updatedAt: dayjs(address.updatedAt)
            });
        } else {
            // 添加模式
            setIsEditMode(false);
            setCurrentAddress(null);
            form.resetFields();
            form.setFieldsValue({
                monitoringEnabled: true
            });
        }
        setIsModalVisible(true);
    };

    // 关闭模态框
    const handleCloseModal = () => {
        setIsModalVisible(false);
        form.resetFields();
    };

    // 提交表单（添加/编辑）
    const handleSubmit = async () => {
        try {
            const values = await form.validateFields();

            if (isEditMode && currentAddress) {
                // 编辑地址
                setUpdating(true);
                try {
                    await monitoredAddressApi.update(currentAddress.id, {
                        address: values.address,
                        addressType: values.addressType,
                        riskLevel: values.riskLevel,
                        monitoringEnabled: values.monitoringEnabled,
                        description: values.description
                    });
                    message.success("地址更新成功");
                    fetchAddresses(); // 刷新列表
                    handleCloseModal();
                } finally {
                    setUpdating(false);
                }
            } else {
                // 添加地址
                setCreating(true);
                try {
                    const newAddress: CreateMonitoredAddressRequest = {
                        address: values.address,
                        addressType: values.addressType,
                        riskLevel: values.riskLevel,
                        monitoringEnabled: values.monitoringEnabled !== undefined ? values.monitoringEnabled : true,
                        description: values.description
                    };

                    await monitoredAddressApi.create(newAddress);
                    message.success("地址添加成功");
                    fetchAddresses(); // 刷新列表
                    handleCloseModal();
                } finally {
                    setCreating(false);
                }
            }
        } catch (error) {
            console.error("表单提交失败:", error);
            message.error("表单验证失败，请检查输入");
        }
    };

    // 删除地址
    const handleDeleteAddress = async (id: string) => {
        setDeleting(true);
        try {
            await monitoredAddressApi.delete(id);
            message.success("地址删除成功");
            fetchAddresses(); // 刷新列表
        } catch (error) {
            console.error("删除地址失败:", error);
            message.error("删除地址失败");
        } finally {
            setDeleting(false);
        }
    };

    // 刷新数据
    const handleRefresh = () => {
        fetchAddresses();
    };

    // 清空筛选条件
    const handleClearFilters = () => {
        setSearchText("");
        setSelectedAddressType("all");
        setSelectedRiskLevel("all");
        setSelectedMonitoringStatus("all");
        setSelectedDateRange(null);
    };

    // 表格列定义
    const columns = [
        {
            title: "创建时间",
            dataIndex: "createdAt",
            key: "createdAt",
            width: 170,
            sorter: (a: MonitoredAddress, b: MonitoredAddress) =>
                dayjs(a.createdAt).unix() - dayjs(b.createdAt).unix(),
            render: (text: string) => (
                <Tooltip title={dayjs(text).format("YYYY-MM-DD HH:mm:ss")}>
                    <span>{dayjs(text).format("YYYY-MM-DD HH:mm")}</span>
                </Tooltip>
            )
        },
        {
            title: "类型",
            dataIndex: "addressType",
            key: "addressType",
            width: 100,
            render: (text: string) => getAddressTypeTag(text)
        },
        {
            title: "风险等级",
            dataIndex: "riskLevel",
            key: "riskLevel",
            width: 120,
            render: (text: string) => (
                <Tag color={getRiskLevelColor(text)}>
                    {riskLevelOptions.find(opt => opt.value === text)?.label || text}
                </Tag>
            )
        },
        {
            title: "地址",
            dataIndex: "address",
            key: "address",
            width: 300,
            render: (text: string) => (
                <div style={{ fontFamily: "monospace", fontSize: "12px" }}>
                    {text}
                </div>
            )
        },
        {
            title: "备注信息",
            dataIndex: "description",
            key: "description",
            render: (text: string) => (
                <div style={{
                    maxWidth: 300,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap"
                }}>
                    {text || "无备注"}
                </div>
            )
        },
        {
            title: "监控状态",
            dataIndex: "monitoringEnabled",
            key: "monitoringEnabled",
            width: 100,
            render: (enabled: boolean) => getMonitoringStatusTag(enabled)
        },
        {
            title: "操作",
            key: "action",
            width: 150,
            render: (_: any, record: MonitoredAddress) => (
                <Space size="small">
                    <Tooltip title="查看详情">
                        <Button
                            type="text"
                            size="small"
                            icon={<EyeOutlined />}
                            onClick={() => handleViewDetails(record)}
                        />
                    </Tooltip>
                    <Tooltip title="编辑">
                        <Button
                            type="text"
                            size="small"
                            icon={<EditOutlined />}
                            onClick={() => handleOpenModal(record)}
                        />
                    </Tooltip>
                    <Popconfirm
                        title="确认删除"
                        description="确定要删除这个监控地址吗？"
                        onConfirm={() => handleDeleteAddress(record.id)}
                        okText="确定"
                        cancelText="取消"
                        disabled={deleting}
                    >
                        <Tooltip title="删除">
                            <Button
                                type="text"
                                size="small"
                                danger
                                icon={<DeleteOutlined />}
                                loading={deleting}
                            />
                        </Tooltip>
                    </Popconfirm>
                </Space>
            )
        }
    ];

    return (
        <div style={{ padding: 24 }}>
            <Card
                title="监控地址管理"
                extra={
                    <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => handleOpenModal()}
                        loading={creating}
                    >
                        添加地址
                    </Button>
                }
                bordered={false}
                style={{ borderRadius: 8, boxShadow: "0 1px 3px rgba(0,0,0,0.1)" }}
            >
                {/* 筛选工具栏 */}
                <div style={{ marginBottom: 16, paddingBottom: 16, borderBottom: "1px solid #f0f0f0" }}>
                    <Row gutter={[16, 16]} align="middle">
                        <Col xs={24} sm={12} md={8} lg={6}>
                            <Input
                                placeholder="搜索地址或备注"
                                prefix={<SearchOutlined />}
                                value={searchText}
                                onChange={(e) => setSearchText(e.target.value)}
                                allowClear
                            />
                        </Col>

                        <Col xs={12} sm={8} md={6} lg={4}>
                            <Select
                                style={{ width: "100%" }}
                                placeholder="地址类型"
                                value={selectedAddressType}
                                onChange={setSelectedAddressType}
                                options={[
                                    { value: "all", label: "全部类型" },
                                    ...addressTypeOptions
                                ]}
                            />
                        </Col>

                        <Col xs={12} sm={8} md={6} lg={4}>
                            <Select
                                style={{ width: "100%" }}
                                placeholder="风险等级"
                                value={selectedRiskLevel}
                                onChange={setSelectedRiskLevel}
                                options={[
                                    { value: "all", label: "全部等级" },
                                    ...riskLevelOptions
                                ]}
                            />
                        </Col>

                        <Col xs={12} sm={8} md={6} lg={4}>
                            <Select
                                style={{ width: "100%" }}
                                placeholder="监控状态"
                                value={selectedMonitoringStatus}
                                onChange={setSelectedMonitoringStatus}
                                options={monitoringStatusOptions}
                            />
                        </Col>

                        <Col xs={12} sm={8} md={6} lg={6}>
                            <DatePicker.RangePicker
                                style={{ width: "100%" }}
                                placeholder={["开始日期", "结束日期"]}
                                value={selectedDateRange}
                                onChange={(dates) => setSelectedDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
                                format="YYYY-MM-DD"
                            />
                        </Col>

                        <Col xs={24} sm={24} md={24} lg={4}>
                            <Space>
                                <Button
                                    type="primary"
                                    icon={<FilterOutlined />}
                                    onClick={() => message.info("已应用筛选条件")}
                                >
                                    筛选
                                </Button>
                                <Button
                                    icon={<ReloadOutlined />}
                                    onClick={handleClearFilters}
                                >
                                    重置
                                </Button>
                            </Space>
                        </Col>
                    </Row>

                    {/* 统计信息 */}
                    <div style={{ marginTop: 16 }}>
                        <Space size="large">
                            <span>总数: <strong>{filteredAddresses.length}</strong> 个</span>
                            <span>监控中: <strong>{filteredAddresses.filter(a => a.monitoringEnabled).length}</strong> 个</span>
                            <span>高风险: <strong>{filteredAddresses.filter(a => a.riskLevel === "HIGH" || a.riskLevel === "CRITICAL").length}</strong> 个</span>
                        </Space>
                    </div>
                </div>

                {/* 数据表格 */}
                <div style={{ position: "relative" }}>
                    <Spin spinning={loading} tip="加载中...">
                        <Table
                            columns={columns}
                            dataSource={paginatedAddresses}
                            rowKey="id"
                            pagination={false}
                            size="middle"
                            bordered
                            scroll={{ x: 1200 }}
                            rowClassName={(record) =>
                                record.riskLevel === "CRITICAL" ? "critical-row" :
                                    record.riskLevel === "HIGH" ? "high-risk-row" : ""
                            }
                        />
                    </Spin>

                    {/* 分页控件 */}
                    <div style={{ marginTop: 16, textAlign: "right" }}>
                        <Space>
                            <span>每页显示:</span>
                            <Select
                                value={pageSize}
                                style={{ width: 100 }}
                                onChange={(value) => {
                                    setPageSize(value);
                                    setCurrentPage(1);
                                }}
                                options={[
                                    { value: 5, label: "5 条" },
                                    { value: 10, label: "10 条" },
                                    { value: 20, label: "20 条" },
                                    { value: 50, label: "50 条" }
                                ]}
                            />
                            <Pagination
                                current={currentPage}
                                pageSize={pageSize}
                                total={totalItems}
                                onChange={setCurrentPage}
                                showSizeChanger={false}
                                showQuickJumper
                                showTotal={(total) => `共 ${total} 条`}
                            />
                            <Button
                                type="text"
                                icon={<ReloadOutlined />}
                                onClick={handleRefresh}
                                loading={loading}
                            />
                        </Space>
                    </div>
                </div>
            </Card>

            {/* 添加/编辑地址模态框 */}
            <Modal
                title={isEditMode ? "编辑监控地址" : "添加监控地址"}
                open={isModalVisible}
                onCancel={handleCloseModal}
                onOk={handleSubmit}
                confirmLoading={creating || updating}
                width={600}
            >
                <Form
                    form={form}
                    layout="vertical"
                    style={{ marginTop: 20 }}
                >
                    <Row gutter={16}>
                        <Col span={12}>
                            <Form.Item
                                name="address"
                                label="地址"
                                rules={[
                                    { required: true, message: "请输入地址" },
                                    { min: 1, max: 255, message: "地址长度在1-255个字符之间" }
                                ]}
                            >
                                <Input placeholder="请输入区块链地址" />
                            </Form.Item>
                        </Col>

                        <Col span={12}>
                            <Form.Item
                                name="addressType"
                                label="地址类型"
                                rules={[{ required: true, message: "请选择地址类型" }]}
                            >
                                <Select placeholder="请选择地址类型">
                                    {addressTypeOptions.map(option => (
                                        <Select.Option key={option.value} value={option.value}>
                                            {option.label}
                                        </Select.Option>
                                    ))}
                                </Select>
                            </Form.Item>
                        </Col>
                    </Row>

                    <Row gutter={16}>
                        <Col span={12}>
                            <Form.Item
                                name="riskLevel"
                                label="风险等级"
                                rules={[{ required: true, message: "请选择风险等级" }]}
                            >
                                <Select placeholder="请选择风险等级">
                                    {riskLevelOptions.map(option => (
                                        <Select.Option key={option.value} value={option.value}>
                                            {option.label}
                                        </Select.Option>
                                    ))}
                                </Select>
                            </Form.Item>
                        </Col>

                        <Col span={12}>
                            <Form.Item
                                name="monitoringEnabled"
                                label="监控状态"
                                valuePropName="checked"
                            >
                                <Switch
                                    checkedChildren="启用"
                                    unCheckedChildren="停用"
                                />
                            </Form.Item>
                        </Col>
                    </Row>

                    <Form.Item
                        name="description"
                        label="备注"
                        rules={[{ max: 1000, message: "备注不能超过1000个字符" }]}
                    >
                        <Input.TextArea
                            placeholder="请输入备注信息（可选）"
                            rows={4}
                            maxLength={1000}
                            showCount
                        />
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
};

// 自定义样式
const styles = `
  .critical-row {
    background-color: #fff2f0;
  }
  
  .critical-row:hover > td {
    background-color: #ffccc7 !important;
  }
  
  .high-risk-row {
    background-color: #fff7e6;
  }
  
  .high-risk-row:hover > td {
    background-color: #ffe7ba !important;
  }
  
  .ant-table-thead > tr > th {
    background-color: #fafafa;
    font-weight: 600;
  }
  
  .ant-tag {
    border-radius: 12px;
    font-size: 12px;
    padding: 0 8px;
  }
`;

// 在组件中注入样式
const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet);

export default AddressListPage;