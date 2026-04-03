import React, { useState } from "react";
import { Card, Row, Col, Statistic, Tag, Tooltip, message, Button } from "antd";
import {
  EnvironmentOutlined,
  SwapOutlined,
  ClockCircleOutlined,
  CopyOutlined,
  WarningOutlined,
  StarOutlined,
} from "@ant-design/icons";
import SubscriptionModal from "../CaseDetails/components/Subscription/SubscriptionModal";

interface AddressInfoProps {
  address?: string;
  txCount?: number;
  firstTxTime?: string;
  latestTxTime?: string;
  isMalicious?: boolean;
}

// 格式化地址显示：前8位 + ... + 后8位
const formatAddress = (address?: string): string => {
  if (!address) return "";
  if (address.length <= 20) return address;
  return `${address.slice(0, 12)}...${address.slice(-12)}`;
};

// 复制到剪贴板
const copyToClipboard = async (text?: string) => {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    message.success("地址已复制到剪贴板");
  } catch (err) {
    message.error("复制失败，请手动复制");
  }
};

const AddressInfo: React.FC<AddressInfoProps> = ({
  address,
  txCount = 0,
  firstTxTime,
  latestTxTime,
  isMalicious = false,
}) => {
  const [subscriptionModalVisible, setSubscriptionModalVisible] =
    useState(false);

  const handleSubscribe = () => {
    setSubscriptionModalVisible(true);
  };

  const handleCancelSubscription = () => {
    setSubscriptionModalVisible(false);
  };

  const handleSubmitSubscription = (values: any) => {
    console.log("Subscription submitted:", values);
    message.success("订阅成功");
    setSubscriptionModalVisible(false);
  };

  return (
    <Card
      style={{
        borderRadius: 8,
        boxShadow: "0 2px 8px rgba(0, 0, 0, 0.06)",
      }}
      bodyStyle={{ padding: "20px 24px" }}
    >
      <Row gutter={[24, 16]} align="middle">
        {/* 地址信息 - 占据更多空间 */}
        <Col xs={24} sm={24} md={10}>
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 12,
            }}
          >
            {/* 地址图标 */}
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: "50%",
                background: isMalicious
                  ? "linear-gradient(135deg, #ff4d4f 0%, #cf1322 100%)"
                  : "linear-gradient(135deg, #1890ff 0%, #096dd9 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <EnvironmentOutlined style={{ color: "#fff", fontSize: 18 }} />
            </div>

            {/* 地址内容 */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  fontSize: 12,
                  color: "var(--text-muted)",
                  marginBottom: 4,
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                当前地址
                {isMalicious && (
                  <Tag color="error" icon={<WarningOutlined />}>
                    高风险
                  </Tag>
                )}
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <Tooltip title={address} placement="bottomLeft">
                  <span
                    style={{
                      fontSize: 15,
                      fontWeight: 600,
                      fontFamily:
                        'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
                      color: "var(--text-color)",
                      letterSpacing: "0.3px",
                    }}
                  >
                    {formatAddress(address)}
                  </span>
                </Tooltip>
                <Tooltip title="复制地址">
                  <CopyOutlined
                    onClick={() => copyToClipboard(address)}
                    style={{
                      color: "var(--text-muted)",
                      cursor: "pointer",
                      fontSize: 14,
                      transition: "color 0.2s",
                    }}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.color = "#1890ff")
                    }
                    onMouseLeave={(e) =>
                      (e.currentTarget.style.color = "var(--text-muted)")
                    }
                  />
                </Tooltip>
                <Button
                  type="link"
                  icon={<StarOutlined />}
                  onClick={handleSubscribe}
                  style={{
                    padding: 0,
                    marginLeft: 8,
                  }}
                >
                  订阅
                </Button>
              </div>
            </div>
          </div>
        </Col>

        {/* 分隔线 - 桌面端显示 */}
        <Col md={1} style={{ display: "flex", justifyContent: "center" }}>
          <div
            style={{
              width: 1,
              height: 50,
              background: "var(--border-color, #e8e8e8)",
              display: "none",
            }}
            className="address-info-divider"
          />
        </Col>

        {/* 交易总次数 */}
        <Col xs={8} sm={8} md={4}>
          <Statistic
            title={
              <span
                style={{
                  fontSize: 12,
                  color: "var(--text-muted)",
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                <SwapOutlined /> 交易次数
              </span>
            }
            value={txCount}
            valueStyle={{
              color: "var(--text-color)",
              fontSize: 20,
              fontWeight: 600,
            }}
          />
        </Col>

        {/* 首次交易时间 */}
        <Col xs={8} sm={8} md={5}>
          <div>
            <div
              style={{
                fontSize: 12,
                color: "var(--text-muted)",
                marginBottom: 4,
                display: "flex",
                alignItems: "center",
                gap: 4,
              }}
            >
              <ClockCircleOutlined /> 首次交易
            </div>
            <div
              style={{
                fontSize: 14,
                fontWeight: 500,
                color: "var(--text-color)",
                fontFamily:
                  'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
              }}
            >
              {firstTxTime || "-"}
            </div>
          </div>
        </Col>

        {/* 最近交易时间 */}
        <Col xs={8} sm={8} md={4}>
          <div>
            <div
              style={{
                fontSize: 12,
                color: "var(--text-muted)",
                marginBottom: 4,
                display: "flex",
                alignItems: "center",
                gap: 4,
              }}
            >
              <ClockCircleOutlined /> 最近交易
            </div>
            <div
              style={{
                fontSize: 14,
                fontWeight: 500,
                color: "var(--text-color)",
                fontFamily:
                  'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
              }}
            >
              {latestTxTime || "-"}
            </div>
          </div>
        </Col>
      </Row>

      <style>{`
        @media (min-width: 992px) {
          .address-info-divider {
            display: block !important;
          }
        }
      `}</style>

      <SubscriptionModal
        visible={subscriptionModalVisible}
        type="node"
        isEdit={false}
        initialValues={null}
        address={address}
        onCancel={handleCancelSubscription}
        onSubmit={handleSubmitSubscription}
      />
    </Card>
  );
};

export default AddressInfo;
