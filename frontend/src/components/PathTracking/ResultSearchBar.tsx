import React from "react";
import { useNavigate } from "react-router-dom";
import { Select, Input, Button, Form, Row, Col, Card } from "antd";
import { useSearchParams } from "react-router-dom";

const { Option } = Select; // 从Select中解构Option组件

interface ResultSearchBarProps {
  defaultCrypto?: string;
  defaultFromAddress?: string;
  defaultToAddress?: string;
}

const ResultSearchBar: React.FC<ResultSearchBarProps> = ({
  defaultCrypto = "eth",
  defaultFromAddress = "",
  defaultToAddress = "",
}) => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams(); // 获取URL参数
  const [form] = Form.useForm(); // 创建表单实例

  // 从URL参数或props中获取默认值
  const routeCrypto = searchParams.get("crypto");
  const urlFromAddress = searchParams.get("fromAddress");
  const urlToAddress = searchParams.get("toAddress");

  // 定义表单提交处理函数
  const onFinish = (values: any) => {
    const { currency, fromAddress, toAddress } = values;
    navigate(
      `/path-tracking/${currency.toLowerCase()}?fromAddress=${encodeURIComponent(fromAddress)}&toAddress=${encodeURIComponent(toAddress)}`,
    );
  };

  return (
    <div style={{ margin: "0 auto", maxWidth: 1200 }}>
      <Card title="🔍 路径追踪搜索" style={{ borderRadius: "8px" }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={onFinish}
          initialValues={{
            currency: routeCrypto || defaultCrypto,
            fromAddress: urlFromAddress || defaultFromAddress,
            toAddress: urlToAddress || defaultToAddress,
          }}
        >
          <Row gutter={16}>
            <Col xs={24} md={4}>
              <Form.Item
                label="币种筛选"
                name="currency"
                rules={[{ required: true, message: "请选择币种" }]}
              >
                <Select placeholder="选择币种" size="large">
                  <Option value="eth">ETH (以太坊)</Option>
                  <Option value="btc">BTC (比特币)</Option>
                </Select>
              </Form.Item>
            </Col>

            <Col xs={24} md={8}>
              <Form.Item
                label="起始地址"
                name="fromAddress"
                rules={[
                  { required: true, message: "请输入起始地址" },
                  {
                    pattern:
                      /^(0x)?[0-9a-fA-F]{40}$|^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}$/,
                    message: "请输入有效的地址",
                  },
                ]}
              >
                <Input placeholder="请输入起始地址" size="large" />
              </Form.Item>
            </Col>

            <Col xs={24} md={8}>
              <Form.Item
                label="目标地址"
                name="toAddress"
                rules={[
                  { required: true, message: "请输入目标地址" },
                  {
                    pattern:
                      /^(0x)?[0-9a-fA-F]{40}$|^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}$/,
                    message: "请输入有效的地址",
                  },
                ]}
              >
                <Input placeholder="请输入目标地址" size="large" />
              </Form.Item>
            </Col>
            <Col xs={24} md={3}>
              <Form.Item label=" ">
                <Button
                  type="primary"
                  size="large"
                  htmlType="submit"
                  style={{ width: "100%" }}
                >
                  搜索路径
                </Button>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Card>
    </div>
  );
};

export default ResultSearchBar;
