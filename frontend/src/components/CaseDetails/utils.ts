/**
 * Utility functions and mappings for case management
 */

// Priority mappings
const PRIORITY_COLORS: Record<string, string> = {
  URGENT: "red",
  HIGH: "orange",
  MEDIUM: "blue",
  LOW: "default",
};

const PRIORITY_LABELS: Record<string, string> = {
  URGENT: "紧急",
  HIGH: "高",
  MEDIUM: "中",
  LOW: "低",
};

// Status mappings
const STATUS_COLORS: Record<string, string> = {
  NEW: "#52c41a",
  IN_PROGRESS: "#52c41a",
  ARCHIVED: "#8c8c8c",
  CLOSED: "#ff4d4f",
};

const STATUS_LABELS: Record<string, string> = {
  NEW: "进行中",
  IN_PROGRESS: "进行中",
  ARCHIVED: "已归档",
  CLOSED: "已关闭",
};

// Risk level mappings
const RISK_LEVEL_COLORS: Record<string, string> = {
  LOW: "green",
  MEDIUM: "orange",
  HIGH: "red",
};

const RISK_LEVEL_LABELS: Record<string, string> = {
  LOW: "低风险",
  MEDIUM: "中风险",
  HIGH: "高风险",
};

// Utility functions
export const getPriorityColor = (priority: string): string => {
  if (!priority) return PRIORITY_COLORS.LOW;
  return PRIORITY_COLORS[priority] || PRIORITY_COLORS.LOW;
};

export const getPriorityLabel = (priority: string): string => {
  if (!priority) return "未知";
  return PRIORITY_LABELS[priority] || "未知";
};

export const getStatusColor = (status: string): string => {
  return STATUS_COLORS[status] || STATUS_COLORS.NEW;
};

export const getStatusLabel = (status: string): string => {
  return STATUS_LABELS[status] || "未知状态";
};

export const getStatusDotColor = (status: string): string => {
  return STATUS_COLORS[status] || "#bfbfbf";
};

export const getRiskLevelColor = (riskLevel: string): string => {
  return RISK_LEVEL_COLORS[riskLevel] || RISK_LEVEL_COLORS.LOW;
};

export const getRiskLevelLabel = (riskLevel: string): string => {
  return RISK_LEVEL_LABELS[riskLevel] || RISK_LEVEL_LABELS.LOW;
};
