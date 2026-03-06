import React, { useState, useEffect, useCallback } from "react";
import { Card, Modal, Drawer, message, ConfigProvider } from "antd";
import dayjs, { Dayjs } from "dayjs";
import zhCN from "antd/locale/zh_CN";
import "./CaseDetails.css";

import SnapshotTable from "./SnapshotTable";
import GraphDisplay from "./GraphDisplay";

import { GraphSnapshot, FilterConfig } from "./types";
import { graphSnapshotApi } from "../../services/graph-snapshot/api";

dayjs.locale("zh-cn");

const CaseDetails: React.FC = () => {
  const [snapshots, setSnapshots] = useState<GraphSnapshot[]>([]);
  const [filteredSnapshots, setFilteredSnapshots] = useState<GraphSnapshot[]>(
    [],
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
        snapshot.title.toLowerCase().includes(filterConfig.title.toLowerCase()),
      );
    }

    // 按风险等级过滤
    if (filterConfig.riskLevel) {
      filtered = filtered.filter(
        (snapshot) => snapshot.riskLevel === filterConfig.riskLevel,
      );
    }

    // 按标签过滤
    if (filterConfig.tags.length > 0) {
      filtered = filtered.filter((snapshot) =>
        filterConfig.tags.some((tag) => snapshot.tags.includes(tag)),
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
            (filterConfig.dateRange[1] as Dayjs).add(1, "day"),
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
      // 调用API获取快照数据
      graphSnapshotApi.getSnapshots().then((response) => {
        if (response.success) {
          // 转换日期字符串为 dayjs 对象
          const parsedSnapshots = response.data.map((snapshot: any) => {
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
        } else {
          message.error(response.msg || "加载快照失败");
        }
      });
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
      graphSnapshotApi
        .deleteSnapshot(snapshotToDelete.id)
        .then((response) => {
          if (response.success) {
            const newSnapshots = snapshots.filter(
              (s) => s.id !== snapshotToDelete.id,
            );
            setSnapshots(newSnapshots);
            setFilteredSnapshots(newSnapshots);
            message.success("快照已删除");
            setDeleteModalVisible(false);
            setSnapshotToDelete(null);
            // 如果删除的是当前打开的快照，则关闭抽屉
            if (
              selectedSnapshot &&
              selectedSnapshot.id === snapshotToDelete.id
            ) {
              setDrawerVisible(false);
              setSelectedSnapshot(null);
            }
          } else {
            message.error(response.msg || "删除快照失败");
          }
        })
        .catch((error) => {
          console.error("删除快照失败:", error);
          message.error("删除快照失败");
        });
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
    new Set(snapshots.flatMap((snapshot) => snapshot.tags)),
  );

  const handleEditSnapshot = (
    snapshot: GraphSnapshot,
    field: string,
    value: any,
  ) => {
    // 准备更新数据，确保包含所有必要字段
    const updateData: any = {
      title: snapshot.title,
      description: snapshot.description,
      tags: snapshot.tags,
      riskLevel: snapshot.riskLevel,
      filterConfig: snapshot.filterConfig,
    };
    // 更新要修改的字段
    updateData[field] = value;

    // 调用API更新快照
    graphSnapshotApi
      .updateSnapshot(snapshot.id, updateData)
      .then((response) => {
        if (response.success) {
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

          // 如果编辑的是当前选中的快照，也更新它
          if (selectedSnapshot && selectedSnapshot.id === snapshot.id) {
            setSelectedSnapshot(
              updatedSnapshots.find((s) => s.id === snapshot.id) || null,
            );
          }

          message.success("快照已更新");
        } else {
          message.error(response.msg || "更新快照失败");
        }
      })
      .catch((error) => {
        console.error("更新快照失败:", error);
        message.error("更新快照失败");
      });
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
