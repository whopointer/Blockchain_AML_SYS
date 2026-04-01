import React from "react";
import { Spin } from "antd";
import "./LoadingOverlay.css";

interface LoadingOverlayProps {
  loading: boolean;
  text?: string;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  loading,
  text = "加载中...",
}) => {
  if (!loading) return null;

  return (
    <div className="loading-overlay">
      <div className="loading-content">
        <Spin size="large" />
        <div className="loading-text">{text}</div>
      </div>
    </div>
  );
};

export default LoadingOverlay;
