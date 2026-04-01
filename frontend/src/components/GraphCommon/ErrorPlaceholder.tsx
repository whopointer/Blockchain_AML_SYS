import React from "react";
import { Button, Empty } from "antd";
import {
  ReloadOutlined,
  WarningOutlined,
  DisconnectOutlined,
} from "@ant-design/icons";
import "./ErrorPlaceholder.css";

interface ErrorPlaceholderProps {
  title?: string;
  description?: string;
  onRetry?: () => void;
  type?: "error" | "network" | "empty";
}

const ErrorPlaceholder: React.FC<ErrorPlaceholderProps> = ({
  title = "数据加载失败",
  description = "请检查网络连接后重试",
  onRetry,
  type = "error",
}) => {
  const getIcon = () => {
    switch (type) {
      case "network":
        return <DisconnectOutlined className="error-icon network" />;
      case "empty":
        return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />;
      default:
        return <WarningOutlined className="error-icon error" />;
    }
  };

  const getTitle = () => {
    switch (type) {
      case "network":
        return "网络连接失败";
      case "empty":
        return "暂无数据";
      default:
        return title;
    }
  };

  if (type === "empty") {
    return (
      <div className="error-placeholder empty">
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={description} />
      </div>
    );
  }

  return (
    <div className="error-placeholder">
      <div className="error-content">
        <div className="error-icon-wrapper">{getIcon()}</div>
        <h3 className="error-title">{getTitle()}</h3>
        <p className="error-description">{description}</p>
        {onRetry && (
          <Button
            type="primary"
            icon={<ReloadOutlined />}
            onClick={onRetry}
            className="error-retry-btn"
          >
            重新加载
          </Button>
        )}
      </div>
    </div>
  );
};

export default ErrorPlaceholder;
