import React from "react";
import { Layout, Menu, Button } from "antd";
import {
  FolderOutlined,
  ApartmentOutlined,
  BellOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from "@ant-design/icons";
import { SidebarProps, MenuKey } from "../types";

const { Sider } = Layout;

interface MenuItem {
  key: MenuKey;
  icon: React.ReactNode;
  label: string;
}

const menuItems: MenuItem[] = [
  {
    key: "case-management",
    icon: <FolderOutlined />,
    label: "案件管理",
  },
  {
    key: "graph-snapshot",
    icon: <ApartmentOutlined />,
    label: "图谱快照",
  },
  {
    key: "subscription",
    icon: <BellOutlined />,
    label: "订阅节点/交易",
  },
];

const Sidebar: React.FC<SidebarProps> = ({
  activeKey,
  onMenuSelect,
  collapsed = false,
}) => {
  const [isCollapsed, setIsCollapsed] = React.useState(collapsed);

  return (
    <Sider
      trigger={null}
      collapsible
      collapsed={isCollapsed}
      theme="light"
      style={{
        boxShadow: "2px 0 8px rgba(0,0,0,0.06)",
        zIndex: 10,
      }}
      width={200}
    >
      <div
        style={{
          height: 64,
          display: "flex",
          alignItems: "center",
          justifyContent: isCollapsed ? "center" : "space-between",
          padding: isCollapsed ? 0 : "0 16px",
          borderBottom: "1px solid #f0f0f0",
        }}
      >
        {!isCollapsed && (
          <span
            style={{
              fontSize: 16,
              fontWeight: 600,
              color: "#667eea",
            }}
          >
            案件中心
          </span>
        )}
        <Button
          type="text"
          icon={isCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          onClick={() => setIsCollapsed(!isCollapsed)}
          style={{ fontSize: 16 }}
        />
      </div>
      <Menu
        mode="inline"
        selectedKeys={[activeKey]}
        onClick={({ key }) => onMenuSelect(key as MenuKey)}
        style={{ borderRight: 0 }}
        items={menuItems.map((item) => ({
          key: item.key,
          icon: item.icon,
          label: item.label,
        }))}
      />
    </Sider>
  );
};

export default Sidebar;
