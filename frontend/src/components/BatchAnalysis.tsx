import React, { useState } from "react";
import { Button, Alert, Spinner, Card, Row, Col } from "react-bootstrap";
import { api } from "../services/api";

const BatchAnalysis: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string>("");

  const handleBatchAnalysis = async () => {
    setLoading(true);
    setError("");
    setResults(null);

    try {
      const batchResults = await api.batchPredict();
      setResults(batchResults);
    } catch (err: any) {
      const errorMessage = err.response?.data?.error || "批量分析失败，请重试";
      console.error("批量分析错误:", err);
      setError(errorMessage);

      // 如果是模型未加载的错误，提示用户先加载模型
      if (errorMessage.includes("模型") || errorMessage.includes("model")) {
        setError("模型未加载，请先在系统仪表板中加载模型后再进行批量分析");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="batch-analysis" style={{ marginTop: "20px" }}>
      <div className="text-center mb-4">
        <h3>📊 批量数据分析</h3>
        <p className="text-secondary">全量数据集智能分析引擎</p>
      </div>

      <Card className="border-0">
        <Card.Header>
          <div className="d-flex align-items-center">
            <div className="me-3">
              <div
                style={{
                  width: "64px",
                  height: "64px",
                  borderRadius: "16px",
                  background:
                    "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "white",
                  fontSize: "32px",
                }}
              >
                🚀
              </div>
            </div>
            <div className="flex-grow-1">
              <h4 className="mb-1">全量数据分析</h4>
              <p className="text-muted mb-0">
                对整个区块链数据集进行深度异常检测分析，发现潜在洗钱模式
              </p>
            </div>
          </div>
        </Card.Header>
        <Card.Body>
          <div className="text-center py-4">
            <div className="mb-4">
              <div
                style={{
                  width: "120px",
                  height: "120px",
                  borderRadius: "50%",
                  background:
                    "linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  margin: "0 auto",
                  border: "2px solid var(--card-border)",
                }}
              >
                <span style={{ fontSize: "48px" }}>🔍</span>
              </div>
            </div>

            <p className="text-muted mb-4">
              <strong>注意：</strong>
              批量分析将处理整个数据集，可能需要较长时间完成。
              系统会使用并行计算优化处理速度。
            </p>

            {error && (
              <Alert variant="danger" className="mb-4">
                <div className="d-flex align-items-center">
                  <span className="me-3">❌</span>
                  <div>{error}</div>
                </div>
              </Alert>
            )}

            <div className="d-grid gap-2 d-md-flex justify-content-md-center">
              <Button
                variant="primary"
                onClick={handleBatchAnalysis}
                disabled={loading}
                size="lg"
                className="px-5"
              >
                {loading ? (
                  <>
                    <Spinner
                      as="span"
                      animation="border"
                      size="sm"
                      className="me-2"
                    />
                    正在分析中，请稍候...
                  </>
                ) : (
                  <>
                    <span className="me-2">⚡</span>
                    开始批量分析
                  </>
                )}
              </Button>
            </div>
          </div>

          {results && (
            <div className="mt-5">
              <div className="text-center mb-4">
                <h4>🎉 分析完成</h4>
                <p className="text-secondary">全量数据分析报告</p>
              </div>

              <Alert variant="success">
                <div className="d-flex align-items-center">
                  <div className="me-3">
                    <div
                      style={{
                        width: "48px",
                        height: "48px",
                        borderRadius: "12px",
                        background: "var(--success-gradient)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "white",
                        fontSize: "24px",
                      }}
                    >
                      ✅
                    </div>
                  </div>
                  <div className="flex-grow-1">
                    <h6 className="mb-1">批量分析成功完成</h6>
                    <p className="mb-0">
                      共分析了{" "}
                      <strong>
                        {results.statistics?.total_transactions || 0}
                      </strong>{" "}
                      笔交易
                    </p>
                  </div>
                </div>
              </Alert>

              {results.statistics && (
                <Row className="g-4">
                  <Col md={4}>
                    <Card className="text-center">
                      <Card.Body>
                        <div style={{ fontSize: "36px", marginBottom: "1rem" }}>
                          📊
                        </div>
                        <h5 className="text-primary">
                          {results.statistics.total_transactions.toLocaleString()}
                        </h5>
                        <p className="text-muted mb-0">总交易数</p>
                      </Card.Body>
                    </Card>
                  </Col>
                  <Col md={4}>
                    <Card className="text-center">
                      <Card.Body>
                        <div style={{ fontSize: "36px", marginBottom: "1rem" }}>
                          ⚠️
                        </div>
                        <h5 className="text-danger">
                          {results.statistics.suspicious_count.toLocaleString()}
                        </h5>
                        <p className="text-muted mb-0">可疑交易数</p>
                      </Card.Body>
                    </Card>
                  </Col>
                  <Col md={4}>
                    <Card className="text-center">
                      <Card.Body>
                        <div style={{ fontSize: "36px", marginBottom: "1rem" }}>
                          📈
                        </div>
                        <h5 className="text-warning">
                          {(
                            (results.statistics.suspicious_count /
                              results.statistics.total_transactions) *
                            100
                          ).toFixed(2)}
                          %
                        </h5>
                        <p className="text-muted mb-0">可疑交易比例</p>
                      </Card.Body>
                    </Card>
                  </Col>
                </Row>
              )}
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
};

export default BatchAnalysis;
