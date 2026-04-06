import React, { useState, useEffect, useCallback } from "react";
import { Card, Modal, Drawer, message, ConfigProvider } from "antd";
import dayjs from "dayjs";
import zhCN from "antd/locale/zh_CN";
import "./CaseDetails.css";

import SnapshotTable from "./SnapshotTable";
import GraphDisplay from "./GraphDisplay";

import type {
  GraphSnapshot,
  FilterConfig,
  CaseComment,
  CaseStatusFilter,
} from "../../types";
import { graphSnapshotApi } from "@/services/graph-snapshot/api";

dayjs.locale("zh-cn");

const GraphSnapshotPage: React.FC = () => {
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
  const [statusFilter, setStatusFilter] = useState<CaseStatusFilter>("ALL");
  const [graphDataLoading, setGraphDataLoading] = useState(false);
  const [editingField, setEditingField] = useState<string | null>(null);
  const [tempValue, setTempValue] = useState<any>(null);

  const SNAPSHOT_META_STORAGE_KEY = "aml_case_snapshot_meta";

  const loadSnapshotMetaMap = (): Record<
    string,
    {
      archived: boolean;
      comments: CaseComment[];
      transformConfig?: {
        x: number;
        y: number;
        k: number;
      };
    }
  > => {
    try {
      const raw = localStorage.getItem(SNAPSHOT_META_STORAGE_KEY);
      if (!raw) {
        return {};
      }
      return JSON.parse(raw) as Record<
        string,
        {
          archived: boolean;
          comments: CaseComment[];
          transformConfig?: {
            x: number;
            y: number;
            k: number;
          };
        }
      >;
    } catch (error) {
      console.warn("读取案件元数据失败：", error);
      return {};
    }
  };

  const saveSnapshotMetaMap = (
    metaMap: Record<
      string,
      {
        archived: boolean;
        comments: CaseComment[];
        transformConfig?: { x: number; y: number; k: number };
      }
    >,
  ) => {
    localStorage.setItem(SNAPSHOT_META_STORAGE_KEY, JSON.stringify(metaMap));
  };

  const normalizeSnapshotMeta = useCallback(
    (snapshot: GraphSnapshot): GraphSnapshot => {
      const metaMap = loadSnapshotMetaMap();
      const meta = metaMap[snapshot.id] || { archived: false, comments: [] };
      return {
        ...snapshot,
        archived: meta.archived,
        comments: meta.comments || [],
        transformConfig: meta.transformConfig,
      };
    },
    [],
  );

  const getSnapshotMainAddress = (snapshot: GraphSnapshot) => {
    if (snapshot.centerAddress) {
      return snapshot.centerAddress;
    }
    if (snapshot.fromAddress && snapshot.toAddress) {
      return `${snapshot.fromAddress} → ${snapshot.toAddress}`;
    }
    return "";
  };

  const updateSnapshotTransform = useCallback(
    (snapshotId: string, transform: { x: number; y: number; k: number }) => {
      const metaMap = loadSnapshotMetaMap();
      if (!metaMap[snapshotId]) {
        metaMap[snapshotId] = { archived: false, comments: [] };
      }
      metaMap[snapshotId].transformConfig = transform;
      saveSnapshotMetaMap(metaMap);

      setSnapshots((prev) =>
        prev.map((s) =>
          s.id === snapshotId ? { ...s, transformConfig: transform } : s,
        ),
      );
      setFilteredSnapshots((prev) =>
        prev.map((s) =>
          s.id === snapshotId ? { ...s, transformConfig: transform } : s,
        ),
      );
      if (selectedSnapshot && selectedSnapshot.id === snapshotId) {
        setSelectedSnapshot({
          ...selectedSnapshot,
          transformConfig: transform,
        });
      }
    },
    [selectedSnapshot],
  );

  const filterSnapshots = useCallback(() => {
    let filtered = snapshots;

    // 按标题或主地址过滤
    if (filterConfig.title) {
      const keyword = filterConfig.title.toLowerCase();
      filtered = filtered.filter((snapshot) => {
        const mainAddress = getSnapshotMainAddress(snapshot).toLowerCase();
        return (
          snapshot.title.toLowerCase().includes(keyword) ||
          mainAddress.includes(keyword)
        );
      });
    }

    // 按风险等级过滤
    if (filterConfig.riskLevel) {
      const normalizedRisk = filterConfig.riskLevel.toUpperCase();
      filtered = filtered.filter(
        (snapshot) => snapshot.riskLevel === normalizedRisk,
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
      const rangeStart = filterConfig.dateRange[0].startOf("day").valueOf();
      const rangeEnd = filterConfig.dateRange[1].endOf("day").valueOf();
      filtered = filtered.filter((snapshot) => {
        const snapshotDate =
          typeof snapshot.createTime === "string"
            ? dayjs(snapshot.createTime)
            : snapshot.createTime;
        const snapshotTime = snapshotDate.valueOf();
        return snapshotTime >= rangeStart && snapshotTime <= rangeEnd;
      });
    }

    // 按归档状态过滤
    if (statusFilter === "ACTIVE") {
      filtered = filtered.filter((snapshot) => !snapshot.archived);
    } else if (statusFilter === "ARCHIVED") {
      filtered = filtered.filter((snapshot) => snapshot.archived);
    }

    setFilteredSnapshots(filtered);
  }, [snapshots, filterConfig, statusFilter]);

  const loadSnapshots = useCallback(async () => {
    setLoading(true);
    try {
      // 调用API获取快照数据
      const response = await graphSnapshotApi.getSnapshots();
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

          return normalizeSnapshotMeta(convertedSnapshot);
        });

        setSnapshots(parsedSnapshots);
        setFilteredSnapshots(parsedSnapshots);
      } else {
        message.error(response.msg || "加载快照失败");
      }
    } catch (error) {
      console.error("加载快照失败:", error);
      message.error("加载快照失败");
    } finally {
      setLoading(false);
    }
  }, [normalizeSnapshotMeta]);

  // 初始化数据 - 从 localStorage 加载快照
  useEffect(() => {
    loadSnapshots();
  }, [loadSnapshots]);

  // 当过滤条件改变时，更新过滤后的快照列表
  useEffect(() => {
    filterSnapshots();
  }, [filterSnapshots]);

  const handleViewSnapshot = async (snapshot: GraphSnapshot) => {
    setDrawerVisible(true);
    setGraphDataLoading(true);

    try {
      const response = await graphSnapshotApi.getSnapshotDetail(snapshot.id);
      if (response.success && response.data) {
        setSelectedSnapshot(response.data);
      } else {
        message.error(response.msg || "获取快照详情失败");
        setSelectedSnapshot(snapshot);
      }
    } catch (error) {
      console.error("获取快照详情失败:", error);
      message.error("获取快照详情失败");
      setSelectedSnapshot(snapshot);
    } finally {
      setGraphDataLoading(false);
    }
  };

  const handleExportPDF = (snapshot: GraphSnapshot) => {
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

  const toggleArchiveSnapshot = (snapshot: GraphSnapshot) => {
    const updatedSnapshots = snapshots.map((s) =>
      s.id === snapshot.id ? { ...s, archived: !s.archived } : s,
    );
    const metaMap = loadSnapshotMetaMap();
    metaMap[snapshot.id] = {
      archived: !snapshot.archived,
      comments: snapshot.comments || [],
    };
    saveSnapshotMetaMap(metaMap);
    setSnapshots(updatedSnapshots);
    setFilteredSnapshots(updatedSnapshots);
    if (selectedSnapshot?.id === snapshot.id) {
      setSelectedSnapshot({
        ...selectedSnapshot,
        archived: !selectedSnapshot.archived,
      });
    }
    message.success(snapshot.archived ? "已取消归档" : "已将案件归档");
  };

  const addCommentToSnapshot = (
    snapshotId: string,
    content: string,
    author = "分析员 A",
  ) => {
    const newComment: CaseComment = {
      id: `${snapshotId}-${Date.now()}`,
      author,
      content,
      createdAt: dayjs().format("YYYY-MM-DD HH:mm:ss"),
    };
    const updatedSnapshots = snapshots.map((snapshot) => {
      if (snapshot.id !== snapshotId) {
        return snapshot;
      }
      const comments = [...(snapshot.comments || []), newComment];
      return {
        ...snapshot,
        comments,
      };
    });
    const updatedSnapshot = updatedSnapshots.find(
      (snapshot) => snapshot.id === snapshotId,
    );
    const metaMap = loadSnapshotMetaMap();
    metaMap[snapshotId] = {
      archived: updatedSnapshot?.archived ?? false,
      comments: updatedSnapshot?.comments || [newComment],
    };
    saveSnapshotMetaMap(metaMap);
    setSnapshots(updatedSnapshots);
    setFilteredSnapshots(updatedSnapshots);
    if (selectedSnapshot?.id === snapshotId) {
      setSelectedSnapshot(
        updatedSnapshots.find((snapshot) => snapshot.id === snapshotId) || null,
      );
    }
    message.success("评论已保存");
  };

  const handleClearFilters = () => {
    setFilterConfig({
      title: "",
      riskLevel: "",
      tags: [],
      dateRange: [null, null],
    });
    setStatusFilter("ALL");
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
      <div className="case-details-container">
        <Card className="case-details-card">
          <SnapshotTable
            filteredSnapshots={filteredSnapshots}
            snapshots={snapshots}
            loading={loading}
            filterConfig={filterConfig}
            statusFilter={statusFilter}
            allTags={allTags}
            onFilterChange={setFilterConfig}
            onStatusFilterChange={setStatusFilter}
            onViewSnapshot={handleViewSnapshot}
            onDownloadSnapshot={handleDownloadSnapshot}
            onDeleteSnapshot={handleDeleteSnapshot}
            onToggleArchive={toggleArchiveSnapshot}
            onClearFilters={handleClearFilters}
            onExportPDF={handleExportPDF}
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
              loading={graphDataLoading}
              onClose={() => setDrawerVisible(false)}
              onDownloadSnapshot={handleDownloadSnapshot}
              onDeleteSnapshot={handleDeleteSnapshot}
              onToggleArchiveSnapshot={toggleArchiveSnapshot}
              onAddComment={addCommentToSnapshot}
              onTransformChange={updateSnapshotTransform}
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

export default GraphSnapshotPage;
