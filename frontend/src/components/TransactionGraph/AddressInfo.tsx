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
    <Card
      style={{
        marginBottom: 16,
        backgroundColor: "#244963",
        borderColor: "#3a5f7f",
        color: "#ffffff",
      }}
    >
      <Row gutter={[16, 16]} align="middle">
        {/* 地址信息 */}
        <Col xs={24} sm={12} md={6}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 12, color: "#9bb3c8", marginBottom: 8 }}>
              <EnvironmentOutlined /> 当前地址
            </div>
            <div
              style={{
                fontSize: 14,
                fontWeight: "bold",
                fontFamily: "monospace",
                wordBreak: "break-all",
                marginBottom: 8,
                color: "#ffffff",
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
              <span style={{ color: "#d8e3f0" }}>
                <SwapOutlined /> 交易总次数
              </span>
            }
            value={txCount}
            valueStyle={{ color: "#667eea", fontSize: 18, fontWeight: "bold" }}
          />
        </Col>

        {/* 首次交易时间 */}
        <Col xs={24} sm={12} md={6}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 12, color: "#9bb3c8", marginBottom: 8 }}>
              <ClockCircleOutlined /> 首次交易时间
            </div>
            <div style={{ fontSize: 14, fontWeight: "bold", color: "#ffffff" }}>
              {firstTxTime || "-"}
            </div>
          </div>
        </Col>

        {/* 最近交易时间 */}
        <Col xs={24} sm={12} md={6}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 12, color: "#9bb3c8", marginBottom: 8 }}>
              <ClockCircleOutlined /> 最近交易时间
            </div>
            <div style={{ fontSize: 14, fontWeight: "bold", color: "#ffffff" }}>
              {latestTxTime || "-"}
            </div>
          </div>
        </Col>
      </Row>
    </Card>
  );
};

export default AddressInfo;
