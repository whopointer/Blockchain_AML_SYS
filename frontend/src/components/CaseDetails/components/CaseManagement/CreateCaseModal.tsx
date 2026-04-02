import React, { useEffect } from "react";
import { Modal, Form, Input, Select, Button, Space, Row, Col } from "antd";
import { Case } from "../../types";

const { TextArea } = Input;

interface CreateCaseModalProps {
  visible: boolean;
  onCancel: () => void;
  onSubmit: (values: any) => void;
  initialValues?: Case | null;
  isEdit?: boolean;
}

const CreateCaseModal: React.FC<CreateCaseModalProps> = ({
  visible,
  onCancel,
  onSubmit,
  initialValues,
  isEdit = false,
}) => {
  const [form] = Form.useForm();

  useEffect(() => {
    if (visible) {
      if (initialValues) {
        form.setFieldsValue({
          title: initialValues.title,
          description: initialValues.description,
          riskLevel: initialValues.riskLevel,
          priority: initialValues.priority,
          tags: initialValues.tags,
          assignedTo: initialValues.assignedTo,
        });
      } else {
        form.resetFields();
        form.setFieldsValue({
          riskLevel: "MEDIUM",
          priority: "MEDIUM",
          tags: [],
        });
      }
    }
  }, [visible, initialValues, form]);

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
    "诈骗",
    "洗钱",
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
    "勒索软件",
    "恶意合约",
    "非法集资",
    "恐怖主义融资",
    "网络犯罪",
    "身份盗用",
    "虚假平台",
    "庞氏骗局",
  ];

  return (
    <Modal
      title={isEdit ? "编辑案件" : "创建案件"}
      open={visible}
      onCancel={handleCancel}
      width={700}
      footer={
        <Space>
          <Button onClick={handleCancel}>取消</Button>
          <Button type="primary" onClick={handleSubmit}>
            {isEdit ? "保存修改" : "创建案件"}
          </Button>
        </Space>
      }
    >
      <Form
        form={form}
        layout="vertical"
        className="case-modal-form"
        style={{ marginTop: 16 }}
      >
        <Row gutter={16}>
          <Col span={24}>
            <Form.Item
              label="案件标题"
              name="title"
              rules={[{ required: true, message: "请输入案件标题" }]}
            >
              <Input placeholder="请输入案件标题" maxLength={100} showCount />
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
                <Select.Option value="HIGH">高风险</Select.Option>
                <Select.Option value="MEDIUM">中风险</Select.Option>
                <Select.Option value="LOW">低风险</Select.Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="优先级"
              name="priority"
              rules={[{ required: true, message: "请选择优先级" }]}
            >
              <Select placeholder="选择优先级">
                <Select.Option value="URGENT">紧急</Select.Option>
                <Select.Option value="HIGH">高</Select.Option>
                <Select.Option value="MEDIUM">中</Select.Option>
                <Select.Option value="LOW">低</Select.Option>
              </Select>
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item label="负责人" name="assignedTo">
              <Input placeholder="输入负责人姓名" />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="标签" name="tags">
              <Select
                mode="tags"
                placeholder="选择或输入标签"
                options={tagOptions.map((tag) => ({ label: tag, value: tag }))}
              />
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={24}>
            <Form.Item
              label="案件描述"
              name="description"
              rules={[{ required: true, message: "请输入案件描述" }]}
            >
              <TextArea
                rows={4}
                placeholder="详细描述案件情况、可疑点、调查方向等"
                maxLength={500}
                showCount
              />
            </Form.Item>
          </Col>
        </Row>

        {isEdit && initialValues && (
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="案件编号">
                <Input value={initialValues.id} disabled />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="当前状态">
                <Input
                  value={
                    initialValues.status === "ACTIVE"
                      ? "进行中"
                      : initialValues.status === "ARCHIVED"
                        ? "已归档"
                        : "已关闭"
                  }
                  disabled
                />
              </Form.Item>
            </Col>
          </Row>
        )}
      </Form>
    </Modal>
  );
};

export default CreateCaseModal;
