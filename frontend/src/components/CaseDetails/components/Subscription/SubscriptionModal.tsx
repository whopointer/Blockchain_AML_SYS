import React, { useEffect } from "react";
import {
  Modal,
  Form,
  Input,
  Select,
  Button,
  Space,
  Row,
  Col,
  Switch,
} from "antd";
import { SubscribedNode, SubscribedTransaction } from "../../types";

const { TextArea } = Input;

interface SubscriptionModalProps {
  visible: boolean;
  type: "node" | "transaction";
  isEdit?: boolean;
  initialValues?: SubscribedNode | SubscribedTransaction | null;
  onCancel: () => void;
  onSubmit: (values: any) => void;
}

const SubscriptionModal: React.FC<SubscriptionModalProps> = ({
  visible,
  type,
  isEdit = false,
  initialValues,
  onCancel,
  onSubmit,
}) => {
  const [form] = Form.useForm();
  const isNode = type === "node";

  useEffect(() => {
    if (visible) {
      if (initialValues) {
        if (isNode) {
          const node = initialValues as SubscribedNode;
          form.setFieldsValue({
            address: node.address,
            label: node.label,
            riskLevel: node.riskLevel,
            tags: node.tags,
            remark: node.remark,
            alertEnabled: node.alertEnabled,
          });
        } else {
          const tx = initialValues as SubscribedTransaction;
          form.setFieldsValue({
            txHash: tx.txHash,
            fromAddress: tx.fromAddress,
            toAddress: tx.toAddress,
            amount: tx.amount,
            token: tx.token,
            riskLevel: tx.riskLevel,
            tags: tx.tags,
            remark: tx.remark,
            alertEnabled: tx.alertEnabled,
          });
        }
      } else {
        form.resetFields();
        form.setFieldsValue({
          riskLevel: "MEDIUM",
          tags: [],
          alertEnabled: true,
          token: "ETH",
        });
      }
    }
  }, [visible, initialValues, isNode, form]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      onSubmit(values);
      form.resetFields();
    } catch (error) {
      console.error("表单验证失败:", error);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  const tagOptions = [
    "可疑交易",
    "大额转账",
    "多地址关联",
    "地址监控",
    "风险标记",
    "常规审查",
    "交易所",
    "混币器",
    "暗网",
    "钓鱼",
    "稳定币",
    "监控",
  ];

  const riskLevelOptions = [
    { value: "HIGH", label: "高风险", color: "red" },
    { value: "MEDIUM", label: "中风险", color: "blue" },
    { value: "LOW", label: "低风险", color: "green" },
  ];

  return (
    <Modal
      title={`${isEdit ? "编辑" : "添加"}${isNode ? "节点" : "交易"}订阅`}
      open={visible}
      onCancel={handleCancel}
      width={700}
      footer={
        <Space>
          <Button onClick={handleCancel}>取消</Button>
          <Button type="primary" onClick={handleSubmit}>
            {isEdit ? "保存修改" : "添加订阅"}
          </Button>
        </Space>
      }
    >
      <Form
        form={form}
        layout="vertical"
        className="subscription-modal-form"
        style={{ marginTop: 16 }}
      >
        {isNode ? (
          // 节点订阅表单
          <>
            <Row gutter={16}>
              <Col span={24}>
                <Form.Item
                  label="钱包地址"
                  name="address"
                  rules={[
                    { required: true, message: "请输入钱包地址" },
                    { min: 20, message: "请输入有效的钱包地址" },
                  ]}
                >
                  <Input placeholder="请输入要监控的钱包地址" />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="标签" name="label">
                  <Input placeholder="给地址添加一个标签" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="风险等级"
                  name="riskLevel"
                  rules={[{ required: true, message: "请选择风险等级" }]}
                >
                  <Select placeholder="选择风险等级">
                    {riskLevelOptions.map((opt) => (
                      <Select.Option key={opt.value} value={opt.value}>
                        {opt.label}
                      </Select.Option>
                    ))}
                  </Select>
                </Form.Item>
              </Col>
            </Row>
          </>
        ) : (
          // 交易订阅表单
          <>
            <Row gutter={16}>
              <Col span={24}>
                <Form.Item
                  label="交易哈希"
                  name="txHash"
                  rules={[
                    { required: true, message: "请输入交易哈希" },
                    { min: 20, message: "请输入有效的交易哈希" },
                  ]}
                >
                  <Input placeholder="请输入交易哈希" />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="发送方地址"
                  name="fromAddress"
                  rules={[{ required: true, message: "请输入发送方地址" }]}
                >
                  <Input placeholder="发送方地址" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="接收方地址"
                  name="toAddress"
                  rules={[{ required: true, message: "请输入接收方地址" }]}
                >
                  <Input placeholder="接收方地址" />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="金额"
                  name="amount"
                  rules={[{ required: true, message: "请输入金额" }]}
                >
                  <Input type="number" placeholder="交易金额" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="代币"
                  name="token"
                  rules={[{ required: true, message: "请输入代币类型" }]}
                >
                  <Select placeholder="选择代币类型">
                    <Select.Option value="ETH">ETH</Select.Option>
                    <Select.Option value="BTC">BTC</Select.Option>
                    <Select.Option value="USDT">USDT</Select.Option>
                    <Select.Option value="USDC">USDC</Select.Option>
                    <Select.Option value="BNB">BNB</Select.Option>
                    <Select.Option value="其他">其他</Select.Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="风险等级"
                  name="riskLevel"
                  rules={[{ required: true, message: "请选择风险等级" }]}
                >
                  <Select placeholder="选择风险等级">
                    {riskLevelOptions.map((opt) => (
                      <Select.Option key={opt.value} value={opt.value}>
                        {opt.label}
                      </Select.Option>
                    ))}
                  </Select>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="标签" name="tags">
                  <Select
                    mode="tags"
                    placeholder="选择或输入标签"
                    options={tagOptions.map((tag) => ({
                      label: tag,
                      value: tag,
                    }))}
                  />
                </Form.Item>
              </Col>
            </Row>
          </>
        )}

        {/* 公共字段 */}
        <Row gutter={16}>
          {!isNode && (
            <Col span={12}>
              <Form.Item label="标签" name="tags">
                <Select
                  mode="tags"
                  placeholder="选择或输入标签"
                  options={tagOptions.map((tag) => ({
                    label: tag,
                    value: tag,
                  }))}
                />
              </Form.Item>
            </Col>
          )}
          <Col span={isNode ? 12 : 12}>
            <Form.Item
              label="开启告警"
              name="alertEnabled"
              valuePropName="checked"
            >
              <Switch checkedChildren="开启" unCheckedChildren="关闭" />
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={24}>
            <Form.Item
              label="备注"
              name="remark"
              rules={[{ required: true, message: "请输入备注信息" }]}
            >
              <TextArea
                rows={3}
                placeholder="添加备注信息，说明订阅原因或监控重点"
                maxLength={500}
                showCount
              />
            </Form.Item>
          </Col>
        </Row>
      </Form>
    </Modal>
  );
};

export default SubscriptionModal;
