import React from "react";
import { Card, Row, Col, Statistic, Tag } from "antd";
import {
  EnvironmentOutlined,
  SwapOutlined,
  ClockCircleOutlined,
} from "@ant-design/icons";

interface AddressInfoProps {
  address?: string;
  txCount?: number;
  firstTxTime?: string;
  latestTxTime?: string;
  isMalicious?: boolean;
}

const AddressInfo: React.FC<AddressInfoProps> = ({
  address,
  txCount = 0,
  firstTxTime,
  latestTxTime,
  isMalicious = false,
}) => {
  return (
    <Card>
      <Row gutter={[16, 16]} align="middle">
        {/* 地址信息 */}
        <Col xs={24} sm={12} md={6}>
          <div style={{ textAlign: "center" }}>
            <div
              style={{
                fontSize: 12,
                color: "var(--text-muted)",
                marginBottom: 8,
              }}
            >
              <EnvironmentOutlined /> 当前地址
            </div>
            <div
              style={{
                fontSize: 14,
                fontWeight: "bold",
                fontFamily: "monospace",
                wordBreak: "break-all",
                marginBottom: 8,
                color: "var(--text-color)",
              }}
              title={address}
            >
              {address || ""}
            </div>
            {isMalicious && <Tag color="red">高风险</Tag>}
          </div>
        </Col>

        {/* 交易总次数 */}
        <Col xs={24} sm={12} md={6}>
          <Statistic
            title={
              <span style={{ color: "var(--text-secondary)" }}>
                <SwapOutlined /> 交易总次数
              </span>
            }
            value={txCount}
            valueStyle={{
              color: "var(--text-color)",
              fontSize: 18,
              fontWeight: "bold",
            }}
          />
        </Col>

        {/* 首次交易时间 */}
        <Col xs={24} sm={12} md={6}>
          <div style={{ textAlign: "center" }}>
            <div
              style={{
                fontSize: 12,
                color: "var(--text-muted)",
                marginBottom: 8,
              }}
            >
              <ClockCircleOutlined /> 首次交易时间
            </div>
            <div
              style={{
                fontSize: 14,
                fontWeight: "bold",
                color: "var(--text-color)",
              }}
            >
              {firstTxTime || "-"}
            </div>
          </div>
        </Col>

        {/* 最近交易时间 */}
        <Col xs={24} sm={12} md={6}>
          <div style={{ textAlign: "center" }}>
            <div
              style={{
                fontSize: 12,
                color: "var(--text-muted)",
                marginBottom: 8,
              }}
            >
              <ClockCircleOutlined /> 最近交易时间
            </div>
            <div
              style={{
                fontSize: 14,
                fontWeight: "bold",
                color: "var(--text-color)",
              }}
            >
              {latestTxTime || "-"}
            </div>
          </div>
        </Col>
      </Row>
    </Card>
  );
};

export default AddressInfo;
