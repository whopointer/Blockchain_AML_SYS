import React, { useState } from "react";
import { Layout, ConfigProvider } from "antd";
import zhCN from "antd/locale/zh_CN";
import dayjs from "dayjs";
import "dayjs/locale/zh-cn";
import Sidebar from "./components/Sidebar";
import CaseManagement from "./components/CaseManagement";
import GraphSnapshot from "./components/GraphSnapshot";
import Subscription from "./components/Subscription";
import { MenuKey } from "./types";
import "./CaseDetails.css";

dayjs.locale("zh-cn");

const { Content } = Layout;

const CaseDetails: React.FC = () => {
  const [activeMenu, setActiveMenu] = useState<MenuKey>("case-management");

  const renderContent = () => {
    switch (activeMenu) {
      case "case-management":
        return <CaseManagement />;
      case "graph-snapshot":
        return <GraphSnapshot />;
      case "subscription":
        return <Subscription />;
      default:
        return <CaseManagement />;
    }
  };

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: "#667eea",
          colorSuccess: "#13b497",
          colorWarning: "#faad14",
          colorError: "#ff4d4f",
          colorInfo: "#1890ff",
        },
        components: {
          Table: {
            cellPaddingBlock: 12,
            cellPaddingInline: 16,
          },
        },
      }}
    >
      <Layout style={{ minHeight: "100vh" }}>
        <Sidebar activeKey={activeMenu} onMenuSelect={setActiveMenu} />
        <Content style={{ margin: 0, background: "#f5f7fa", flex: 1 }}>
          {renderContent()}
        </Content>
      </Layout>
    </ConfigProvider>
  );
};

export default CaseDetails;
