import React, { useState } from "react";
import { Form, Button, Alert, Spinner, Card, Badge } from "react-bootstrap";
import {
  detectionApi,
  DetectResponse,
  getRiskLevelColor,
} from "../services/api";

interface AddressDetectionFormProps {
  onDetectionComplete?: (result: DetectResponse) => void;
}

const AddressDetectionForm: React.FC<AddressDetectionFormProps> = ({
  onDetectionComplete,
}) => {
  const [address, setAddress] = useState<string>("");
  const [addressType, setAddressType] = useState<string>("bitcoin");
  const [modelType, setModelType] = useState<string>("gnn");
  const [neighborDepth, setNeighborDepth] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<DetectResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const response = await detectionApi.detect({
        address: address.trim(),
        address_type: addressType,
        model_type: modelType,
        neighbor_depth: neighborDepth,
      });

      setResult(response);

      if (onDetectionComplete) {
        onDetectionComplete(response);
      }
    } catch (err: any) {
      console.error("检测失败:", err);
      const errorMessage =
        err.response?.data?.detail ||
        err.response?.data?.error ||
        "检测失败，请重试";
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setAddress("");
    setAddressType("bitcoin");
    setModelType("gnn");
    setNeighborDepth(1);
    setResult(null);
    setError("");
  };

  // 渲染检测结果
  const renderResult = () => {
    if (!result) return null;

    return (
      <Card
        className="mt-4"
        style={{
          border: `1px solid ${getRiskLevelColor(result.risk_label)}40`,
          background: `${getRiskLevelColor(result.risk_label)}08`,
        }}
      >
        <Card.Body>
          <div className="d-flex align-items-center justify-content-between mb-3">
            <h6 className="mb-0">📊 检测结果</h6>
            <Badge bg={result.is_suspicious ? "danger" : "success"}>
              {result.is_suspicious ? "可疑" : "正常"}
            </Badge>
          </div>

          {/* 风险等级显示 */}
          <div className="text-center mb-3">
            <div
              style={{
                fontSize: "48px",
                color: getRiskLevelColor(result.risk_label),
              }}
            >
              {result.is_suspicious ? "⚠️" : "✅"}
            </div>
            <div className="mt-2">
              <strong
                style={{
                  fontSize: "24px",
                  color: getRiskLevelColor(result.risk_label),
                }}
              >
                {result.risk_label === "high"
                  ? "高风险"
                  : result.risk_label === "medium"
                    ? "中风险"
                    : result.risk_label === "low"
                      ? "低风险"
                      : "正常"}
              </strong>
            </div>
            <div className="text-muted mt-1">
              异常概率: {(result.probability * 100).toFixed(2)}%
            </div>
          </div>

          {/* 子图信息 */}
          {result.subgraph_info && (
            <div className="mt-3 p-3 bg-light rounded">
              <div className="row text-center">
                <div className="col">
                  <div className="text-muted small">节点数</div>
                  <div className="fw-bold">
                    {result.subgraph_info.total_nodes}
                  </div>
                </div>
                <div className="col">
                  <div className="text-muted small">边数</div>
                  <div className="fw-bold">
                    {result.subgraph_info.total_edges}
                  </div>
                </div>
                <div className="col">
                  <div className="text-muted small">邻居深度</div>
                  <div className="fw-bold">
                    {result.subgraph_info.neighbor_depth}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 详细信息 */}
          <div className="mt-3 text-muted small">
            <div>地址: {result.address}</div>
            <div>模型: {result.model_type.toUpperCase()}</div>
            <div>检测时间: {new Date(result.timestamp).toLocaleString()}</div>
          </div>
        </Card.Body>
      </Card>
    );
  };

  return (
    <div className="address-detection-form" style={{ marginTop: "20px" }}>
      <Card>
        <Card.Body>
          <div className="text-center mb-4">
            <h3>🔍 地址风险检测</h3>
            <p className="text-secondary">输入区块链地址进行智能风险分析</p>
          </div>

          <Form onSubmit={handleSubmit}>
            {/* 地址输入 */}
            <Form.Group className="mb-3">
              <Form.Label>
                <span className="me-2">📍</span>
                区块链地址
              </Form.Label>
              <Form.Control
                type="text"
                placeholder="输入BTC地址 (bc1q...)"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                disabled={loading}
                style={{
                  fontFamily: 'Monaco, Consolas, "Courier New", monospace',
                  fontSize: "0.9rem",
                }}
              />
              <Form.Text className="text-muted">
                💡 请输入完整的比特币地址 (以 bc1q 开头)
              </Form.Text>
            </Form.Group>

            {/* 地址类型 */}
            <Form.Group className="mb-3">
              <Form.Label>
                <span className="me-2">🏷️</span>
                地址类型
              </Form.Label>
              <Form.Select
                value={addressType}
                onChange={(e) => setAddressType(e.target.value)}
                disabled={loading}
              >
                <option value="bitcoin">Bitcoin (BTC)</option>
                <option value="ethereum">Ethereum (ETH)</option>
              </Form.Select>
            </Form.Group>

            {/* 模型选择 */}
            <Form.Group className="mb-3">
              <Form.Label>
                <span className="me-2">🤖</span>
                检测模型
              </Form.Label>
              <Form.Select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                disabled={loading}
              >
                <option value="gnn">DGI + GIN + Random Forest</option>
              </Form.Select>
            </Form.Group>

            {/* 邻居深度 */}
            <Form.Group className="mb-4">
              <Form.Label>
                <span className="me-2">🔗</span>
                邻居深度: {neighborDepth}
              </Form.Label>
              <Form.Range
                min={1}
                max={3}
                value={neighborDepth}
                onChange={(e) => setNeighborDepth(parseInt(e.target.value))}
                disabled={loading}
              />
              <div className="d-flex justify-content-between text-muted small">
                <span>1</span>
                <span>2</span>
                <span>3</span>
              </div>
              <Form.Text className="text-muted">
                💡 决定从数据库中加载多少层邻居节点
              </Form.Text>
            </Form.Group>

            {/* 错误提示 */}
            {error && (
              <Alert variant="danger" className="mb-3">
                <div className="d-flex align-items-center">
                  <span className="me-2">⚠️</span>
                  <div>{error}</div>
                </div>
              </Alert>
            )}

            {/* 按钮组 */}
            <div className="d-grid gap-2">
              <Button
                variant="primary"
                type="submit"
                disabled={loading || !address.trim()}
                size="lg"
              >
                {loading ? (
                  <>
                    <Spinner
                      as="span"
                      animation="border"
                      size="sm"
                      className="me-2"
                    />
                    检测中...
                  </>
                ) : (
                  <>
                    <span className="me-2">🚀</span>
                    开始风险检测
                  </>
                )}
              </Button>

              {result && (
                <Button variant="outline-secondary" onClick={handleReset}>
                  <span className="me-2">🔄</span>
                  重新检测
                </Button>
              )}
            </div>
          </Form>

          {/* 检测结果 */}
          {renderResult()}
        </Card.Body>
      </Card>
    </div>
  );
};

export default AddressDetectionForm;
