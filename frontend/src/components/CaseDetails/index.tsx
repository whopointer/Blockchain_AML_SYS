import React, { useState, useEffect, useCallback } from "react";
import { Card, Modal, Drawer, message, ConfigProvider } from "antd";
import dayjs, { Dayjs } from "dayjs";
import zhCN from "antd/locale/zh_CN";
import "./CaseDetails.css";

import SnapshotTable from "./SnapshotTable";
import GraphDisplay from "./GraphDisplay";

import { GraphSnapshot, FilterConfig } from "./types";

dayjs.locale("zh-cn");

const CaseDetails: React.FC = () => {
  const [snapshots, setSnapshots] = useState<GraphSnapshot[]>([]);
  const [filteredSnapshots, setFilteredSnapshots] = useState<GraphSnapshot[]>(
    []
  );
  const [loading, setLoading] = useState(false);
  const [selectedSnapshot, setSelectedSnapshot] =
    useState<GraphSnapshot | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [snapshotToDelete, setSnapshotToDelete] =
    useState<GraphSnapshot | null>(null);
  const [filterConfig, setFilterConfig] = useState<FilterConfig>({
    title: "",
    riskLevel: "",
    tags: [],
    dateRange: [null, null],
  });
  const [editingField, setEditingField] = useState<string | null>(null);
  const [tempValue, setTempValue] = useState<any>(null);

  const filterSnapshots = useCallback(() => {
    let filtered = snapshots;

    // 按标题过滤
    if (filterConfig.title) {
      filtered = filtered.filter((snapshot) =>
        snapshot.title.toLowerCase().includes(filterConfig.title.toLowerCase())
      );
    }

    // 按风险等级过滤
    if (filterConfig.riskLevel) {
      filtered = filtered.filter(
        (snapshot) => snapshot.riskLevel === filterConfig.riskLevel
      );
    }

    // 按标签过滤
    if (filterConfig.tags.length > 0) {
      filtered = filtered.filter((snapshot) =>
        filterConfig.tags.some((tag) => snapshot.tags.includes(tag))
      );
    }

    // 按日期范围过滤
    if (filterConfig.dateRange[0] && filterConfig.dateRange[1]) {
      filtered = filtered.filter((snapshot) => {
        const snapshotDate =
          typeof snapshot.createTime === "string"
            ? dayjs(snapshot.createTime)
            : snapshot.createTime;
        return (
          snapshotDate.isAfter(filterConfig.dateRange[0]) &&
          snapshotDate.isBefore(
            (filterConfig.dateRange[1] as Dayjs).add(1, "day")
          )
        );
      });
    }

    setFilteredSnapshots(filtered);
  }, [snapshots, filterConfig]);

  // 初始化数据 - 从 localStorage 加载快照
  useEffect(() => {
    loadSnapshots();
  }, []);

  // 当过滤条件改变时，更新过滤后的快照列表
  useEffect(() => {
    filterSnapshots();
  }, [filterSnapshots]);

  const loadSnapshots = () => {
    setLoading(true);
    try {
      // 模拟从 localStorage 或 API 加载快照数据
      const savedSnapshots = localStorage.getItem("graphSnapshots");
      const defaultSnapshots: GraphSnapshot[] = [
        {
          id: "snapshot-001",
          title: "高风险交易链路分析",
          description: "针对地址 0x1234 的深度分析，发现多个可疑交易节点",
          tags: ["可疑交易", "高风险"],
          createTime: dayjs().subtract(5, "days"),
          mainAddress: "0x1234567890abcdef1234567890abcdef12345678",
          nodeCount: 45,
          linkCount: 120,
          riskLevel: "high",
          filterConfig: {
            txType: "all",
            addrType: "all",
            minAmount: 0,
            maxAmount: 100000,
            startDate: dayjs().subtract(10, "years"),
            endDate: dayjs(),
          },
          graphData: {
            nodes: [
              {
                id: "node-001",
                label: "0x1234",
                title: "中心节点",
                addr: "0x1234567890abcdef1234567890abcdef12345678",
                layer: 0,
                pid: 0,
                color: "#4fae7b",
                shape: "star",
                track: "one",
                expanded: true,
                malicious: 1,
              },
              {
                id: "node-002",
                label: "0x5678",
                title: "关联账户1",
                addr: "0x5678abcdef1234567890abcdef1234567890abcd",
                layer: 1,
                pid: "node-001",
                track: "one",
                expanded: false,
                malicious: 0,
              },
              {
                id: "node-003",
                label: "0x9abc",
                title: "关联账户2",
                addr: "0x9abcdef1234567890abcdef1234567890ef1234",
                layer: 1,
                pid: "node-001",
                track: "one",
                expanded: false,
                malicious: 1,
                shape: "circularImage",
                image: "/malicious.png",
              },
              {
                id: "node-004",
                label: "0xdef0",
                title: "下游节点",
                addr: "0xdef01234567890abcdef1234567890abcdef123",
                layer: 2,
                pid: "node-003",
                track: "one",
                expanded: false,
                malicious: 0,
              },
            ],
            links: [
              {
                from: "node-001",
                to: "node-002",
                label: "1000 ETH",
                val: 1000,
                tx_time: "2023-01-01",
                tx_hash_list: ["0x1234abcd", "0x5678efgh"],
              },
              {
                from: "node-001",
                to: "node-003",
                label: "2500 ETH",
                val: 2500,
                tx_time: "2023-01-02",
                tx_hash_list: ["0x9abc1234", "0xdef56789"],
              },
              {
                from: "node-002",
                to: "node-004",
                label: "500 ETH",
                val: 500,
                tx_time: "2023-01-03",
                tx_hash_list: ["0x4567ijkl"],
              },
            ],
          },
        },
        {
          id: "snapshot-002",
          title: "资金流向追踪 - 第一阶段",
          description: "追踪可疑资金的初始来源和流向",
          tags: ["资金追踪"],
          createTime: dayjs().subtract(3, "days"),
          mainAddress: "0xabcdef1234567890abcdef1234567890abcdef12",
          nodeCount: 32,
          linkCount: 85,
          riskLevel: "medium",
          filterConfig: {
            txType: "inflow",
            addrType: "tagged",
            minAmount: 100,
            maxAmount: 50000,
            startDate: dayjs().subtract(7, "years"),
            endDate: dayjs().subtract(1, "day"),
          },
          graphData: {
            nodes: [
              {
                id: "node-005",
                label: "0xabc",
                title: "源地址",
                addr: "0xabcdef1234567890abcdef1234567890abcdef12",
                layer: 0,
                pid: 0,
                color: "#4fae7b",
                shape: "star",
                track: "one",
                expanded: true,
                malicious: 0,
              },
              {
                id: "node-006",
                label: "0xdef",
                title: "中间节点1",
                addr: "0xdef1234567890abcdef1234567890abcdef1234",
                layer: -1,
                pid: "node-005",
                track: "one",
                expanded: false,
                malicious: 1,
                shape: "circularImage",
                image: "/malicious.png",
              },
              {
                id: "node-007",
                label: "0x123",
                title: "中间节点2",
                addr: "0x1234567890abcdef1234567890abcdef12345678",
                layer: -1,
                pid: "node-005",
                track: "one",
                expanded: false,
                malicious: 0,
              },
              {
                id: "node-008",
                label: "0x456",
                title: "目标地址",
                addr: "0x4567890abcdef1234567890abcdef1234567890a",
                layer: -2,
                pid: "node-006",
                track: "one",
                expanded: false,
                malicious: 1,
                shape: "circularImage",
                image: "/malicious.png",
              },
            ],
            links: [
              {
                from: "node-005",
                to: "node-006",
                label: "2000 ETH",
                val: 2000,
                tx_time: "2023-01-05",
                tx_hash_list: ["0x5678mnop", "0x9012qrst"],
              },
              {
                from: "node-005",
                to: "node-007",
                label: "1500 ETH",
                val: 1500,
                tx_time: "2023-01-06",
                tx_hash_list: ["0x3456uvwx"],
              },
              {
                from: "node-006",
                to: "node-008",
                label: "1200 ETH",
                val: 1200,
                tx_time: "2023-01-07",
                tx_hash_list: ["0x7890yzab", "0xcdef1234"],
              },
            ],
          },
        },
        {
          id: "snapshot-003",
          title: "重要节点分析",
          description: "系统内中心化程度高的关键交易节点",
          tags: ["重要节点"],
          createTime: dayjs().subtract(1, "days"),
          mainAddress: "0x5678abcdef1234567890abcdef1234567890abcd",
          nodeCount: 28,
          linkCount: 62,
          riskLevel: "medium",
          filterConfig: {
            txType: "outflow",
            addrType: "malicious",
            minAmount: 500,
            maxAmount: 20000,
            startDate: dayjs().subtract(5, "years"),
            endDate: dayjs(),
          },
          graphData: {
            nodes: [
              {
                id: "node-009",
                label: "0x567",
                title: "关键节点",
                addr: "0x5678abcdef1234567890abcdef1234567890abcd",
                layer: 0,
                pid: 0,
                color: "#4fae7b",
                shape: "circularImage",
                track: "one",
                expanded: true,
                malicious: 1,
                image: "/malicious.png",
              },
              {
                id: "node-010",
                label: "0x890",
                title: "高风险地址1",
                addr: "0x890abcdef1234567890abcdef1234567890abcde",
                layer: -1,
                pid: "node-009",
                track: "one",
                expanded: false,
                malicious: 1,
                shape: "circularImage",
                image: "/malicious.png",
              },
              {
                id: "node-011",
                label: "0xabc",
                title: "高风险地址2",
                addr: "0xabcdef1234567890abcdef1234567890abcdef12",
                layer: -1,
                pid: "node-009",
                track: "one",
                expanded: false,
                malicious: 1,
                shape: "circularImage",
                image: "/malicious.png",
              },
              {
                id: "node-012",
                label: "0xdef",
                title: "普通地址",
                addr: "0xdef1234567890abcdef1234567890abcdef1234",
                layer: 1,
                pid: "node-009",
                track: "one",
                expanded: false,
                malicious: 0,
              },
            ],
            links: [
              {
                from: "node-010",
                to: "node-009",
                label: "3000 ETH",
                val: 3000,
                tx_time: "2023-01-08",
                tx_hash_list: ["0x2345ijkl", "0x6789mnop"],
              },
              {
                from: "node-011",
                to: "node-009",
                label: "4500 ETH",
                val: 4500,
                tx_time: "2023-01-09",
                tx_hash_list: ["0x0123qrst", "0x4567uvwx", "0x8901xyz"],
              },
              {
                from: "node-009",
                to: "node-012",
                label: "2000 ETH",
                val: 2000,
                tx_time: "2023-01-10",
                tx_hash_list: ["0x2345abcd"],
              },
            ],
          },
        },
        {
          id: "snapshot-004",
          title: "洗钱模式识别",
          description: "识别出的典型洗钱交易模式和参与者",
          tags: ["可疑交易", "资金追踪"],
          createTime: dayjs(),
          mainAddress: "0x9012efab1234567890abcdef1234567890abcdef",
          nodeCount: 56,
          linkCount: 156,
          riskLevel: "high",
          filterConfig: {
            txType: "all",
            addrType: "all",
            minAmount: 0,
            maxAmount: 100000,
            startDate: dayjs().subtract(3, "years"),
            endDate: dayjs(),
          },
          graphData: {
            nodes: [
              {
                id: "node-013",
                label: "0x901",
                title: "源头",
                addr: "0x9012efab1234567890abcdef1234567890abcdef",
                layer: 0,
                pid: 0,
                color: "#4fae7b",
                shape: "star",
                track: "one",
                expanded: true,
                malicious: 1,
              },
              {
                id: "node-014",
                label: "0x2ef",
                title: "第一层拆分",
                addr: "0x2efab1234567890abcdef1234567890abcdef123",
                layer: 1,
                pid: "node-013",
                track: "one",
                expanded: false,
                malicious: 1,
                shape: "circularImage",
                image: "/malicious.png",
              },
              {
                id: "node-015",
                label: "0xab1",
                title: "第二层混币",
                addr: "0xab1234567890abcdef1234567890abcdef123456",
                layer: 1,
                pid: "node-013",
                track: "one",
                expanded: false,
                malicious: 1,
                shape: "circularImage",
                image: "/malicious.png",
              },
              {
                id: "node-016",
                label: "0x234",
                title: "最终接收",
                addr: "0x234567890abcdef1234567890abcdef12345678",
                layer: 2,
                pid: "node-014",
                track: "one",
                expanded: false,
                malicious: 1,
                shape: "circularImage",
                image: "/malicious.png",
              },
            ],
            links: [
              {
                from: "node-013",
                to: "node-014",
                label: "10000 ETH",
                val: 10000,
                tx_time: "2023-01-11",
                tx_hash_list: ["0xbcde1234", "0x5678fghi", "0x9012jklm"],
              },
              {
                from: "node-013",
                to: "node-015",
                label: "8000 ETH",
                val: 8000,
                tx_time: "2023-01-11",
                tx_hash_list: ["0x3456nopq", "0x7890rstu"],
              },
              {
                from: "node-014",
                to: "node-016",
                label: "5000 ETH",
                val: 5000,
                tx_time: "2023-01-12",
                tx_hash_list: ["0x1234vwxy"],
              },
              {
                from: "node-015",
                to: "node-016",
                label: "7000 ETH",
                val: 7000,
                tx_time: "2023-01-12",
                tx_hash_list: ["0x5678zabc", "0xdef12345"],
              },
            ],
          },
        },
      ];

      const snapshotsToUse = savedSnapshots
        ? JSON.parse(savedSnapshots)
        : defaultSnapshots;

      // 转换日期字符串为 dayjs 对象
      const parsedSnapshots = snapshotsToUse.map((snapshot: any) => {
        const convertedSnapshot = {
          ...snapshot,
          createTime:
            typeof snapshot.createTime === "string"
              ? dayjs(snapshot.createTime)
              : snapshot.createTime,
        };

        // 如果存在filterConfig，则转换其中的日期
        if (convertedSnapshot.filterConfig) {
          convertedSnapshot.filterConfig = {
            ...convertedSnapshot.filterConfig,
            startDate: snapshot.filterConfig.startDate
              ? typeof snapshot.filterConfig.startDate === "string"
                ? dayjs(snapshot.filterConfig.startDate)
                : snapshot.filterConfig.startDate
              : null,
            endDate: snapshot.filterConfig.endDate
              ? typeof snapshot.filterConfig.endDate === "string"
                ? dayjs(snapshot.filterConfig.endDate)
                : snapshot.filterConfig.endDate
              : null,
          };
        }

        return convertedSnapshot;
      });

      setSnapshots(parsedSnapshots);
      setFilteredSnapshots(parsedSnapshots);
    } catch (error) {
      console.error("加载快照失败:", error);
      message.error("加载快照失败");
    } finally {
      setLoading(false);
    }
  };

  const handleViewSnapshot = (snapshot: GraphSnapshot) => {
    setSelectedSnapshot(snapshot);
    setDrawerVisible(true);
  };

  const handleDeleteSnapshot = (snapshot: GraphSnapshot) => {
    setSnapshotToDelete(snapshot);
    setDeleteModalVisible(true);
  };

  const confirmDeleteSnapshot = () => {
    if (snapshotToDelete) {
      const newSnapshots = snapshots.filter(
        (s) => s.id !== snapshotToDelete.id
      );
      setSnapshots(newSnapshots);
      localStorage.setItem("graphSnapshots", JSON.stringify(newSnapshots));
      message.success("快照已删除");
      setDeleteModalVisible(false);
      setSnapshotToDelete(null);
      // 如果删除的是当前打开的快照，则关闭抽屉
      if (selectedSnapshot && selectedSnapshot.id === snapshotToDelete.id) {
        setDrawerVisible(false);
        setSelectedSnapshot(null);
      }
    }
  };

  const handleDownloadSnapshot = (snapshot: GraphSnapshot) => {
    try {
      const dataStr = JSON.stringify(snapshot, null, 2);
      const dataBlob = new Blob([dataStr], { type: "application/json" });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${snapshot.title}-${snapshot.id}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      message.success("快照已下载");
    } catch (error) {
      console.error("下载快照失败:", error);
      message.error("下载快照失败");
    }
  };

  const handleClearFilters = () => {
    setFilterConfig({
      title: "",
      riskLevel: "",
      tags: [],
      dateRange: [null, null],
    });
  };

  const allTags = Array.from(
    new Set(snapshots.flatMap((snapshot) => snapshot.tags))
  );

  const handleEditSnapshot = (
    snapshot: GraphSnapshot,
    field: string,
    value: any
  ) => {
    const updatedSnapshots = snapshots.map((s) => {
      if (s.id === snapshot.id) {
        return {
          ...s,
          [field]: value,
        };
      }
      return s;
    });

    setSnapshots(updatedSnapshots);
    setFilteredSnapshots(updatedSnapshots);

    // 更新本地存储
    localStorage.setItem("graphSnapshots", JSON.stringify(updatedSnapshots));

    // 如果编辑的是当前选中的快照，也更新它
    if (selectedSnapshot && selectedSnapshot.id === snapshot.id) {
      setSelectedSnapshot(
        updatedSnapshots.find((s) => s.id === snapshot.id) || null
      );
    }
  };

  const startEditing = (field: string, currentValue: any) => {
    setEditingField(field);
    setTempValue(currentValue);
  };

  const saveEdit = (snapshotId: string, field: string) => {
    if (editingField && tempValue !== null) {
      const snapshot = snapshots.find((s) => s.id === snapshotId);
      if (snapshot) {
        handleEditSnapshot(snapshot, field, tempValue);
        setEditingField(null);
        setTempValue(null);
      }
    }
  };

  const cancelEdit = () => {
    setEditingField(null);
    setTempValue(null);
  };

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: "#667eea", // 主要强调色
          colorSuccess: "#13b497", // 成功色
          colorWarning: "#faad14", // 警告色
          colorError: "#ff4d4f", // 错误色
          colorInfo: "#1890ff", // 信息色
        },
        components: {
          Table: {
            cellPaddingBlock: 12,
            cellPaddingInline: 16,
          },
        },
      }}
    >
      <div className="case-details-container">
        <Card className="case-details-card">


          <SnapshotTable
            filteredSnapshots={filteredSnapshots}
            snapshots={snapshots}
            loading={loading}
            filterConfig={filterConfig}
            allTags={allTags}
            onFilterChange={setFilterConfig}
            onViewSnapshot={handleViewSnapshot}
            onDownloadSnapshot={handleDownloadSnapshot}
            onDeleteSnapshot={handleDeleteSnapshot}
            onClearFilters={handleClearFilters}
            editingField={editingField}
            tempValue={tempValue}
            startEditing={startEditing}
            saveEdit={saveEdit}
            cancelEdit={cancelEdit}
          />
        </Card>

        {/* 快照详情 Drawer */}
        <Drawer
          title={selectedSnapshot?.title}
          placement="right"
          width={800}
          onClose={() => setDrawerVisible(false)}
          open={drawerVisible}
          className="snapshot-drawer"
        >
          {selectedSnapshot && (
            <GraphDisplay
              selectedSnapshot={selectedSnapshot}
              setDrawerVisible={setDrawerVisible}
              handleDownloadSnapshot={handleDownloadSnapshot}
              handleDeleteSnapshot={handleDeleteSnapshot}
              editingField={editingField}
              tempValue={tempValue}
              setTempValue={setTempValue}
              startEditing={startEditing}
              saveEdit={saveEdit}
              cancelEdit={cancelEdit}
            />
          )}
        </Drawer>

        {/* 删除确认对话框 */}
        <Modal
          title="删除快照"
          open={deleteModalVisible}
          onOk={confirmDeleteSnapshot}
          onCancel={() => setDeleteModalVisible(false)}
          okText="删除"
          cancelText="取消"
          okButtonProps={{ danger: true }}
        >
          <p>确定要删除快照 "{snapshotToDelete?.title}" 吗？</p>
          <p style={{ color: "#999", fontSize: 12 }}>
            此操作无法撤销，请谨慎操作。
          </p>
        </Modal>
      </div>
    </ConfigProvider>
  );
};

export default CaseDetails;
