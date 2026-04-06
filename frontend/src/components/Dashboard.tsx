import React, { useState, useEffect } from "react";
import { Card, Row, Col, Alert, Button, Spinner } from "react-bootstrap";
import { Helmet } from "react-helmet-async";
import {
  api,
  HealthResponse,
  ModelInfo,
  StatisticsResponse,
} from "../services/api";

const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [statistics, setStatistics] = useState<StatisticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    setError("");

    try {
      const [healthData, modelData, statsData] = await Promise.all([
        api.healthCheck(),
        api.getModelInfo().catch(() => null),
        api.getStatistics(),
      ]);

      setHealth(healthData);
      setModelInfo(modelData);
      setStatistics(statsData);
    } catch (err: any) {
      setError(err.response?.data?.error || "加载仪表板数据失败");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center p-4">
        <Spinner animation="border" />
        <p className="mt-2">加载中...</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <Helmet>
        <title>系统仪表板 - 区块链AML反洗钱系统</title>
      </Helmet>
      <div className="text-center mb-4">
        <h2>系统仪表板</h2>
        <p className="text-secondary">区块链AML反洗钱系统实时监控</p>
      </div>

      {error && <Alert variant="danger">{error}</Alert>}

      <Row className="g-4 mb-4">
        <Col xl={4} lg={6} md={12}>
          <Card className="h-100">
            <Card.Header>
              <div className="d-flex align-items-center">
                <div className="me-3">
                  <div
                    style={{
                      width: "48px",
                      height: "48px",
                      borderRadius: "12px",
                      background:
                        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "white",
                      fontSize: "24px",
                    }}
                  >
                    🖥️
                  </div>
                </div>
                <div>
                  <h5 className="mb-0">系统状态</h5>
                  <small className="text-muted">System Status</small>
                </div>
              </div>
            </Card.Header>
            <Card.Body>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">运行状态</span>
                <span
                  className={`badge ${
                    health?.status === "healthy" ? "bg-success" : "bg-danger"
                  }`}
                >
                  {health?.status === "healthy" ? "✓ 正常运行" : "✗ 系统异常"}
                </span>
              </div>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">系统版本</span>
                <span className="text-primary">{statistics?.version}</span>
              </div>
              <div className="d-flex justify-content-between align-items-center">
                <span className="text-secondary">最后更新</span>
                <small className="text-muted">
                  {new Date(health?.timestamp || "").toLocaleString()}
                </small>
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col xl={4} lg={6} md={12}>
          <Card className="h-100">
            <Card.Header>
              <div className="d-flex align-items-center">
                <div className="me-3">
                  <div
                    style={{
                      width: "48px",
                      height: "48px",
                      borderRadius: "12px",
                      background:
                        "linear-gradient(135deg, #13B497 0%, #59D4A4 100%)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "white",
                      fontSize: "24px",
                    }}
                  >
                    🤖
                  </div>
                </div>
                <div>
                  <h5 className="mb-0">模型状态</h5>
                  <small className="text-muted">Model Status</small>
                </div>
              </div>
            </Card.Header>
            <Card.Body>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">模型加载</span>
                <span
                  className={`badge ${
                    health?.model_loaded ? "bg-success" : "bg-warning"
                  }`}
                >
                  {health?.model_loaded ? "✓ 已加载" : "⚠ 未加载"}
                </span>
              </div>
              {modelInfo && (
                <>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="text-secondary">模型类型</span>
                    <span className="text-primary">{modelInfo.model_type}</span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="text-secondary">模型版本</span>
                    <span className="text-primary">
                      {modelInfo.model_version || "-"}
                    </span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center">
                    <span className="text-secondary">加载时间</span>
                    <small className="text-muted">
                      {modelInfo.loaded_at
                        ? new Date(modelInfo.loaded_at).toLocaleString()
                        : "-"}
                    </small>
                  </div>
                </>
              )}
            </Card.Body>
          </Card>
        </Col>

        <Col xl={4} lg={6} md={12}>
          <Card className="h-100">
            <Card.Header>
              <div className="d-flex align-items-center">
                <div className="me-3">
                  <div
                    style={{
                      width: "48px",
                      height: "48px",
                      borderRadius: "12px",
                      background:
                        "linear-gradient(135deg, #FFA726 0%, #FF7043 100%)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "white",
                      fontSize: "24px",
                    }}
                  >
                    📈
                  </div>
                </div>
                <div>
                  <h5 className="mb-0">性能指标</h5>
                  <small className="text-muted">Performance Metrics</small>
                </div>
              </div>
            </Card.Header>
            <Card.Body>
              {modelInfo?.performance_metrics ? (
                <>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="text-secondary">准确率</span>
                    <span className="text-success font-weight-bold">
                      {(
                        (modelInfo.performance_metrics.accuracy ?? 0) * 100
                      ).toFixed(2)}
                      %
                    </span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="text-secondary">auc</span>
                    <span className="text-info font-weight-bold">
                      {((modelInfo.performance_metrics.auc ?? 0) * 100).toFixed(
                        2,
                      )}
                      %
                    </span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="text-secondary">AP</span>
                    <span className="text-warning font-weight-bold">
                      {(
                        (modelInfo.performance_metrics.average_precision ?? 0) *
                        100
                      ).toFixed(2)}
                      %
                    </span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center">
                    <span className="text-secondary">F1分数</span>
                    <span className="text-primary font-weight-bold">
                      {(
                        (modelInfo.performance_metrics.f1_score ?? 0) * 100
                      ).toFixed(2)}
                      %
                    </span>
                  </div>
                </>
              ) : (
                <div className="text-center py-3">
                  <div style={{ fontSize: "48px", opacity: 0.3 }}>📊</div>
                  <p className="text-muted mt-2">暂无性能数据</p>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Card>
        <Card.Header>
          <div className="d-flex align-items-center">
            <div className="me-3">
              <div
                style={{
                  width: "48px",
                  height: "48px",
                  borderRadius: "12px",
                  background:
                    "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "white",
                  fontSize: "24px",
                }}
              >
                ℹ️
              </div>
            </div>
            <div>
              <h5 className="mb-0">系统信息</h5>
              <small className="text-muted">System Information</small>
            </div>
          </div>
        </Card.Header>
        <Card.Body>
          <Row className="g-4">
            <Col md={6}>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">API端点</span>
                <code>http://localhost:8080</code>
              </div>
              <div className="d-flex justify-content-between align-items-center">
                <span className="text-secondary">系统状态</span>
                <span className="badge bg-primary">
                  {statistics?.system_status}
                </span>
              </div>
            </Col>
            <Col md={6}>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">最后检查</span>
                <small className="text-muted">
                  {new Date(statistics?.timestamp || "").toLocaleString()}
                </small>
              </div>
              <Button
                variant="outline-primary"
                onClick={loadDashboardData}
                className="w-100"
              >
                🔄 刷新数据
              </Button>
            </Col>
          </Row>
        </Card.Body>
      </Card>
    </div>
  );
};

export default Dashboard;
