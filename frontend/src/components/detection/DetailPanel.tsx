import React from 'react';
import { Modal, Badge, Row, Col, Card, Button } from 'react-bootstrap';
import { DetectResponse, getRiskLevelColor } from '../../services/api';

interface DetailPanelProps {
  show: boolean;
  data: DetectResponse | null;
  onHide: () => void;
}

const DetailPanel: React.FC<DetailPanelProps> = ({ show, data, onHide }) => {
  if (!data) return null;

  return (
    <Modal show={show} onHide={onHide} size="lg" centered>
      <Modal.Header closeButton>
        <Modal.Title>🔍 地址详细信息</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {/* 地址信息 */}
        <Card className="mb-3">
          <Card.Header className="bg-light">
            <h6 className="mb-0">📍 地址信息</h6>
          </Card.Header>
          <Card.Body>
            <div className="mb-2">
              <strong>地址:</strong>
              <div className="text-muted small" style={{ wordBreak: 'break-all' }}>
                {data.address}
              </div>
            </div>
            <div>
              <strong>地址类型:</strong>
              <div className="text-muted">{data.address_type}</div>
            </div>
          </Card.Body>
        </Card>

        {/* 风险评估结果 */}
        <Card className="mb-3" style={{
          borderLeft: `4px solid ${getRiskLevelColor(data.risk_label)}`
        }}>
          <Card.Header className="bg-light">
            <h6 className="mb-0">⚠️ 风险评估</h6>
          </Card.Header>
          <Card.Body>
            <Row>
              <Col md={4} className="text-center">
                <div style={{ fontSize: '48px' }}>
                  {data.is_suspicious ? '⚠️' : '✅'}
                </div>
                <Badge bg={data.is_suspicious ? 'danger' : 'success'} className="mt-2">
                  {data.is_suspicious ? '可疑' : '正常'}
                </Badge>
              </Col>
              <Col md={4} className="text-center">
                <div className="mb-2">
                  <small className="text-muted">风险等级</small>
                  <div style={{
                    fontSize: '24px',
                    color: getRiskLevelColor(data.risk_label),
                    fontWeight: 'bold'
                  }}>
                    {data.risk_label === 'high' ? '高风险' :
                     data.risk_label === 'medium' ? '中风险' :
                     data.risk_label === 'low' ? '低风险' : '未知'}
                  </div>
                </div>
              </Col>
              <Col md={4} className="text-center">
                <div className="mb-2">
                  <small className="text-muted">异常概率</small>
                  <div className="text-info fw-bold" style={{ fontSize: '20px' }}>
                    {(data.probability * 100).toFixed(2)}%
                  </div>
                </div>
              </Col>
            </Row>
          </Card.Body>
        </Card>

        {/* 交易网络信息 */}
        {data.subgraph_info && (
          <Card className="mb-3">
            <Card.Header className="bg-light">
              <h6 className="mb-0">📊 交易网络信息</h6>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={4} className="text-center">
                  <div>
                    <strong style={{ fontSize: '18px', color: '#667eea' }}>
                      {data.subgraph_info.total_nodes}
                    </strong>
                    <div className="small text-muted">节点数</div>
                  </div>
                </Col>
                <Col md={4} className="text-center">
                  <div>
                    <strong style={{ fontSize: '18px', color: '#667eea' }}>
                      {data.subgraph_info.total_edges}
                    </strong>
                    <div className="small text-muted">交易数</div>
                  </div>
                </Col>
                <Col md={4} className="text-center">
                  <div>
                    <strong style={{ fontSize: '18px', color: '#667eea' }}>
                      {data.subgraph_info.neighbor_depth}
                    </strong>
                    <div className="small text-muted">邻居深度</div>
                  </div>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        )}

        {/* 原始标签 */}
        {data.original_label && (
          <Card className="mb-3">
            <Card.Header className="bg-light">
              <h6 className="mb-0">📋 原始标签</h6>
            </Card.Header>
            <Card.Body>
              <div className="text-muted">{data.original_label}</div>
            </Card.Body>
          </Card>
        )}

        {/* 模型和时间信息 */}
        <Card className="mb-3">
          <Card.Header className="bg-light">
            <h6 className="mb-0">ℹ️ 检测信息</h6>
          </Card.Header>
          <Card.Body>
            <div className="mb-2">
              <strong>检测模型:</strong>
              <div className="text-muted">{data.model_type}</div>
            </div>
            <div>
              <strong>检测时间:</strong>
              <div className="text-muted">
                {new Date(data.timestamp).toLocaleString()}
              </div>
            </div>
          </Card.Body>
        </Card>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          关闭
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default DetailPanel;
