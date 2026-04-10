import React from "react";
import { useNavigate } from "react-router-dom";
import { Select, Input, Button, Form, Row, Col, Card } from "antd";

const { Option } = Select;

interface ResultSearchBarProps {
  defaultCrypto?: string;
  defaultFromAddress?: string;
  defaultToAddress?: string;
}

// 以太坊地址校验
const isValidEthAddress = (address: string): boolean => {
  return /^0x[a-fA-F0-9]{40}$/.test(address);
};

// ENS 域名校验
const isValidEns = (address: string): boolean => {
  return /^[a-zA-Z0-9-]+\.eth$/i.test(address);
};

// 比特币地址校验
const isValidBtcAddress = (address: string): boolean => {
  return (
    /^1[a-zA-Z0-9]{25,34}$/.test(address) ||
    /^3[a-zA-Z0-9]{25,34}$/.test(address) ||
    /^bc1[a-zA-Z0-9]{6,87}$/i.test(address)
  );
};

// 根据币种校验地址
const validateAddressByCrypto = (
  crypto: string,
  address: string,
): string | null => {
  if (!address || address.trim() === "") {
    return "请输入地址";
  }

  const trimmedAddress = address.trim();

  if (crypto === "eth" || crypto === "ETH") {
    if (isValidEns(trimmedAddress)) {
      return null;
    }
    if (!isValidEthAddress(trimmedAddress)) {
      return "请输入有效的以太坊地址（0x 开头，40位十六进制）或 ENS 域名";
    }
  } else if (crypto === "btc" || crypto === "BTC") {
    if (!isValidBtcAddress(trimmedAddress)) {
      return "请输入有效的比特币地址（以 1、3 或 bc1 开头）";
    }
  }

  return null;
};

const ResultSearchBar: React.FC<ResultSearchBarProps> = ({
  defaultCrypto = "eth",
  defaultFromAddress = "",
  defaultToAddress = "",
}) => {
  const navigate = useNavigate();
  const [form] = Form.useForm();
  const [currency, setCurrency] = React.useState<string>(defaultCrypto);

  // 根据币种获取 placeholder
  const getPlaceholder = (crypto: string): string => {
    if (crypto === "eth" || crypto === "ETH") {
      return "输入以太坊地址 / ENS";
    }
    return "输入比特币地址";
  };

  const onFinish = (values: any) => {
    const { currency, fromAddress, toAddress } = values;

    // 校验起始地址
    const fromError = validateAddressByCrypto(currency, fromAddress);
    if (fromError) {
      form.setFields([{ name: "fromAddress", errors: [fromError] }]);
      return;
    }

    // 校验目标地址
    const toError = validateAddressByCrypto(currency, toAddress);
    if (toError) {
      form.setFields([{ name: "toAddress", errors: [toError] }]);
      return;
    }

    navigate(
      `/path-tracking/${currency.toLowerCase()}?fromAddress=${encodeURIComponent(fromAddress.trim())}&toAddress=${encodeURIComponent(toAddress.trim())}`,
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
            currency: defaultCrypto,
            fromAddress: defaultFromAddress,
            toAddress: defaultToAddress,
          }}
        >
          <Row gutter={16}>
            <Col xs={24} md={4}>
              <Form.Item
                label="币种筛选"
                name="currency"
                rules={[{ required: true, message: "请选择币种" }]}
              >
                <Select
                  placeholder="选择币种"
                  size="large"
                  onChange={(value) => {
                    setCurrency(value);
                    // 清空之前的校验错误
                    form.setFields([
                      { name: "fromAddress", errors: [] },
                      { name: "toAddress", errors: [] },
                    ]);
                  }}
                >
                  <Option value="eth">ETH (以太坊)</Option>
                  <Option value="btc">BTC (比特币)</Option>
                </Select>
              </Form.Item>
            </Col>

            <Col xs={24} md={8}>
              <Form.Item
                label="起始地址"
                name="fromAddress"
                rules={[{ required: true, message: "请输入起始地址" }]}
              >
                <Input
                  placeholder={getPlaceholder(currency)}
                  size="large"
                  allowClear
                />
              </Form.Item>
            </Col>

            <Col xs={24} md={8}>
              <Form.Item
                label="目标地址"
                name="toAddress"
                rules={[{ required: true, message: "请输入目标地址" }]}
              >
                <Input
                  placeholder={getPlaceholder(currency)}
                  size="large"
                  allowClear
                />
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
