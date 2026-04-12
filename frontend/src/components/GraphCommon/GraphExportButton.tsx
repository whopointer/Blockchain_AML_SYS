import React, { useState } from "react";
import { Button, message } from "antd";
import { DownloadOutlined, ExportOutlined } from "@ant-design/icons";
import {
  convertGraphToCSV,
  downloadCSV,
  exportFullGraphToPNG,
} from "../../utils/exportUtils";
import { NodeItem, LinkItem } from "./types";

interface GraphExportButtonProps {
  nodes: NodeItem[];
  links: LinkItem[];
  graphElementId?: string;
  snapshot?: {
    title: string;
    description?: string;
    riskLevel: string;
    createTime: string;
    centerAddress?: string;
    fromAddress?: string;
    toAddress?: string;
    nodeCount: number;
    linkCount: number;
    tags: string[];
    filterConfig?: any;
  };
  disabled?: boolean;
}

const GraphExportButton: React.FC<GraphExportButtonProps> = ({
  nodes,
  links,
  graphElementId,
  snapshot,
  disabled = false,
}) => {
  const [loading, setLoading] = useState<string | null>(null);

  const handleExportCSV = async () => {
    if (nodes.length === 0 && links.length === 0) {
      message.warning("暂无数据可导出");
      return;
    }

    setLoading("csv");
    try {
      const snapshotData = snapshot || {
        title: "图谱导出",
        description: "",
        riskLevel: "LOW",
        createTime: new Date().toISOString(),
        nodeCount: nodes.length,
        linkCount: links.length,
        tags: [],
      };

      const csvContent = convertGraphToCSV(nodes, links, snapshotData as any);
      const filename = `graph_export_${Date.now()}.csv`;
      downloadCSV(csvContent, filename);
      message.success("CSV导出成功");
    } catch (error) {
      console.error("导出CSV失败:", error);
      message.error("导出CSV失败");
    } finally {
      setLoading(null);
    }
  };

  const handleExportPNG = async () => {
    if (nodes.length === 0 && links.length === 0) {
      message.warning("暂无数据可导出");
      return;
    }

    setLoading("png");
    try {
      const svgElement = document.querySelector(
        `#${graphElementId} svg`,
      ) as SVGSVGElement;

      if (!svgElement) {
        message.error("未找到图谱元素");
        setLoading(null);
        return;
      }

      const filename = `graph_export_${Date.now()}.png`;
      const success = await exportFullGraphToPNG(svgElement, filename);

      if (success) {
        message.success("PNG导出成功");
      } else {
        message.error("PNG导出失败");
      }
    } catch (error) {
      console.error("导出PNG失败:", error);
      message.error("导出PNG失败");
    } finally {
      setLoading(null);
    }
  };

  const items = [
    {
      key: "png",
      label: "导出PNG",
      icon: <ExportOutlined />,
      onClick: handleExportPNG,
      disabled: loading !== null,
    },
    {
      key: "csv",
      label: "导出CSV",
      icon: <DownloadOutlined />,
      onClick: handleExportCSV,
      disabled: loading !== null,
    },
  ];

  return (
    <div style={{ display: "flex", gap: 8 }}>
      <Button
        type="primary"
        icon={<ExportOutlined />}
        loading={loading === "png"}
        onClick={handleExportPNG}
        disabled={disabled || loading !== null}
      >
        导出PNG
      </Button>
      <Button
        icon={<DownloadOutlined />}
        loading={loading === "csv"}
        onClick={handleExportCSV}
        disabled={disabled || loading !== null}
      >
        导出CSV
      </Button>
    </div>
  );
};

export default GraphExportButton;
