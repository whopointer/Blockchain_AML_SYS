import React, { useState } from "react";
import {
  Button,
  Modal,
  Form,
  Input,
  Select,
  message,
} from "antd";
import { CameraOutlined } from "@ant-design/icons";
import dayjs from "dayjs";

const { Option } = Select;
const { TextArea } = Input;

interface GraphSnapshotButtonProps {
  onCreateSnapshot?: (snapshotData: SnapshotData) => void;
}

interface SnapshotData {
  title: string;
  description: string;
  date: dayjs.Dayjs;
  tags: string[];
}

const GraphSnapshotButton: React.FC<GraphSnapshotButtonProps> = ({
  onCreateSnapshot,
}) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [formValid, setFormValid] = useState(false);

  const showModal = () => {
    setIsModalVisible(true);
    form.setFieldsValue({
      tags: [],
    });
    // 初始状态下表单无效，因为必填项未填写
    setFormValid(false);
  };

  // 处理表单字段变化，验证必填项
  const handleFieldsChange = () => {
    const titleValue = form.getFieldValue("title");
    // 只有当标题不为空时，表单才有效
    setFormValid(!!titleValue && titleValue.trim() !== "");
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
    setFormValid(false); // 重置表单验证状态
  };

  const handleCreateSnapshot = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();

      const snapshotData: SnapshotData = {
        title: values.title,
        description: values.description,
        date: dayjs(),
        tags: values.tags || [],
      };

      // 调用父组件传递的回调函数
      if (onCreateSnapshot) {
        onCreateSnapshot(snapshotData);
      }

      // TODO: 添加实际的快照创建逻辑

      message.success("图谱快照创建成功！");
      setIsModalVisible(false);
      form.resetFields();
      setFormValid(false); // 重置表单验证状态
    } catch (error) {
      console.error("创建快照失败:", error);
      message.error("创建快照失败，请重试");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Button type="primary" icon={<CameraOutlined />} onClick={showModal}>
        创建图谱快照
      </Button>

      <Modal
        title="创建图谱快照"
        open={isModalVisible}
        onOk={handleCreateSnapshot}
        onCancel={handleCancel}
        confirmLoading={loading}
        width={600}
        okText="创建快照"
        cancelText="取消"
        okButtonProps={{
          disabled: !formValid,
        }}
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            tags: [],
          }}
          onFieldsChange={handleFieldsChange}
        >
          <Form.Item
            name="title"
            label="快照标题"
            rules={[{ required: true, message: "请输入快照标题" }]}
          >
            <Input placeholder="请输入快照标题" />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
            rules={[{ message: "请输入快照描述" }]}
          >
            <TextArea rows={4} placeholder="请输入快照描述信息" />
          </Form.Item>

          <Form.Item name="tags" label="标签">
            <Select
              mode="tags"
              style={{ width: "100%" }}
              placeholder="添加标签，按回车确认"
            >
              <Option value="可疑交易">可疑交易</Option>
              <Option value="高风险">高风险</Option>
              <Option value="资金追踪">资金追踪</Option>
              <Option value="重要节点">重要节点</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
};

export default GraphSnapshotButton;
