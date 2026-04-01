import React from "react";
import { Spin } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import "./LoadingOverlay.css";

interface LoadingOverlayProps {
  loading: boolean;
  text?: string;
  fullScreen?: boolean;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  loading,
  text = "加载中...",
  fullScreen = false,
}) => {
  if (!loading) return null;

  const antIcon = <LoadingOutlined style={{ fontSize: 48 }} spin />;

  return (
    <div className={`loading-overlay ${fullScreen ? "full-screen" : ""}`}>
      <div className="loading-content">
        <Spin indicator={antIcon} />
        <p className="loading-text">{text}</p>
      </div>
    </div>
  );
};

export default LoadingOverlay;
