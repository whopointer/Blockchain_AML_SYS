import React, { useEffect, useState } from "react";
import { Layout, ConfigProvider } from "antd";
import { Helmet } from "react-helmet-async";
import { useLocation, useNavigate } from "react-router-dom";
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

const menuKeys: MenuKey[] = [
  "case-management",
  "graph-snapshot",
  "subscription",
];

const getMenuKeyFromPath = (pathname: string): MenuKey => {
  const segments = pathname.split("/").filter(Boolean);
  const lastSegment = segments[segments.length - 1] as MenuKey;
  return menuKeys.includes(lastSegment) ? lastSegment : "case-management";
};

const CaseDetails: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [activeMenu, setActiveMenu] = useState<MenuKey>(
    getMenuKeyFromPath(location.pathname),
  );

  useEffect(() => {
    setActiveMenu(getMenuKeyFromPath(location.pathname));
  }, [location.pathname]);

  const handleMenuSelect = (key: MenuKey) => {
    navigate(`/case-details/${key}`);
  };

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
    <>
      <Helmet>
        <title>案件中心 - 区块链AML反洗钱系统</title>
      </Helmet>
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
          <Sidebar activeKey={activeMenu} onMenuSelect={handleMenuSelect} />
          <Content style={{ margin: 0, background: "#f5f7fa", flex: 1 }}>
            {renderContent()}
          </Content>
        </Layout>
      </ConfigProvider>
    </>
  );
};

export default CaseDetails;
