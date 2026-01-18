import React from "react";
import { Modal, Button } from "antd";
import { GraphSnapshot } from "./types";

interface DeleteModalProps {
  visible: boolean;
  snapshot: GraphSnapshot | null;
  onCancel: () => void;
  onConfirm: () => void;
}

const DeleteModal: React.FC<DeleteModalProps> = ({
  visible,
  snapshot,
  onCancel,
  onConfirm,
}) => {
  return (
    <Modal
      title="确认删除"
      open={visible}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          取消
        </Button>,
        <Button key="confirm" type="primary" danger onClick={onConfirm}>
          确认删除
        </Button>,
      ]}
    >
      <p>您确定要删除以下快照吗？</p>
      {snapshot && (
        <p>
          <strong>{snapshot.title}</strong>
        </p>
      )}
      <p>此操作不可撤销。</p>
    </Modal>
  );
};

export default DeleteModal;
