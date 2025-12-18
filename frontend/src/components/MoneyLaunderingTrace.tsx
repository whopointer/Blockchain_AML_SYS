import React, { useState } from 'react';
import { Form, Button, Alert, Spinner, Card, Badge, Row, Col } from 'react-bootstrap';
import { api } from '../services/api';

interface TraceNode {
  tx_id: string;
  address: string;
  amount: number;
  timestamp: string;
  risk_score: number;
}

interface TracePath {
  path: TraceNode[];
  total_amount: number;
  risk_level: 'low' | 'medium' | 'high';
  suspicious_count: number;
}

interface TraceResult {
  source_tx: string;
  paths: TracePath[];
  total_paths: number;
  timestamp: string;
}

const MoneyLaunderingTrace: React.FC = () => {
  const [txId, setTxId] = useState<string>('');
  const [maxDepth, setMaxDepth] = useState<number>(3);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<TraceResult | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      if (!txId.trim()) {
        setError('请输入交易ID');
        return;
      }

      const traceResult = await api.traceMoneyLaundering(txId.trim(), maxDepth);
      setResult(traceResult);
    } catch (err: any) {
      const errorMessage = err.response?.data?.error || '路径追踪失败，请重试';
      console.error('路径追踪错误:', err);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getRiskBadgeVariant = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return 'danger';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'secondary';
    }
  };

  const getRiskColor = (riskScore: number) => {
    if (riskScore > 0.7) return '#dc3545';
    if (riskScore > 0.4) return '#ffc107';
    return '#28a745';
  };

  return (
    <div className="money-laundering-trace">
      <div className="text-center mb-4">
        <h3>🔗 洗钱路径追踪</h3>
        <p className="text-secondary">追踪可疑交易的资金流向，发现潜在洗钱网络</p>
      </div>

      <Row>
        <Col lg={4} className="mb-4">
          <Card>
            <Card.Header>
              <div className="d-flex align-items-center">
                <div className="me-3">
                  <div style={{ 
                    width: '48px', 
                    height: '48px', 
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    fontSize: '24px'
                  }}>
                    🎯
                  </div>
                </div>
                <div>
                  <h5 className="mb-0">追踪配置</h5>
                  <small className="text-muted">Trace Configuration</small>
                </div>
              </div>
            </Card.Header>
            <Card.Body>
              <Form onSubmit={handleSubmit}>
                <Form.Group className="mb-3">
                  <Form.Label>
                    <span className="me-2">🔗</span>
                    起始交易ID
                  </Form.Label>
                  <Form.Control
                    type="text"
                    placeholder="输入要追踪的交易ID"
                    value={txId}
                    onChange={(e) => setTxId(e.target.value)}
                    disabled={loading}
                    style={{ 
                      fontFamily: 'Monaco, Consolas, "Courier New", monospace',
                      fontSize: '0.9rem'
                    }}
                  />
                  <Form.Text className="text-muted">
                    输入可疑交易的ID作为追踪起点
                  </Form.Text>
                </Form.Group>

                <Form.Group className="mb-4">
                  <Form.Label>
                    <span className="me-2">📊</span>
                    追踪深度: {maxDepth} 层
                  </Form.Label>
                  <Form.Range
                    min={1}
                    max={5}
                    value={maxDepth}
                    onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                    disabled={loading}
                  />
                  <Form.Text className="text-muted">
                    设置资金流向的追踪层数（1-5层）
                  </Form.Text>
                </Form.Group>

                {error && (
                  <Alert variant="danger" className="mb-3">
                    <div className="d-flex align-items-center">
                      <span className="me-2">⚠️</span>
                      <div>{error}</div>
                    </div>
                  </Alert>
                )}

                <div className="d-grid">
                  <Button 
                    variant="primary" 
                    type="submit" 
                    disabled={loading}
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
                        追踪中...
                      </>
                    ) : (
                      <>
                        <span className="me-2">🚀</span>
                        开始追踪
                      </>
                    )}
                  </Button>
                </div>
              </Form>
            </Card.Body>
          </Card>
        </Col>

        <Col lg={8}>
          {result && (
            <Card>
              <Card.Header>
                <div className="d-flex align-items-center justify-content-between">
                  <div className="d-flex align-items-center">
                    <div className="me-3">
                      <div style={{ 
                        width: '48px', 
                        height: '48px', 
                        borderRadius: '12px',
                        background: 'linear-gradient(135deg, #13B497 0%, #59D4A4 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        fontSize: '24px'
                      }}>
                        📈
                      </div>
                    </div>
                    <div>
                      <h5 className="mb-0">追踪结果</h5>
                      <small className="text-muted">Trace Results</small>
                    </div>
                  </div>
                  <Badge bg="info" className="px-3 py-2">
                    发现 {result.total_paths} 条路径
                  </Badge>
                </div>
              </Card.Header>
              <Card.Body>
                <Alert variant="info" className="mb-4">
                  <div className="d-flex align-items-center">
                    <span className="me-3" style={{ fontSize: '24px' }}>🎯</span>
                    <div className="flex-grow-1">
                      <strong>起始交易:</strong> 
                      <code className="ms-2" style={{ 
                        background: 'rgba(26, 32, 53, 0.5)',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        border: '1px solid var(--card-border)'
                      }}>
                        {result.source_tx}
                      </code>
                    </div>
                  </div>
                </Alert>

                {result.paths.length === 0 ? (
                  <div className="text-center py-5">
                    <div style={{ fontSize: '64px', opacity: 0.3 }}>🔍</div>
                    <p className="text-muted mt-3">未发现可疑的资金流向路径</p>
                  </div>
                ) : (
                  <div className="paths-container">
                    {result.paths.map((path, pathIndex) => (
                      <Card key={pathIndex} className="mb-3">
                        <Card.Header>
                          <div className="d-flex justify-content-between align-items-center">
                            <div>
                              <strong>路径 #{pathIndex + 1}</strong>
                              <Badge 
                                bg={getRiskBadgeVariant(path.risk_level)} 
                                className="ms-2"
                              >
                                {path.risk_level === 'high' ? '高风险' : 
                                 path.risk_level === 'medium' ? '中风险' : '低风险'}
                              </Badge>
                            </div>
                            <div className="text-end">
                              <small className="text-muted">总金额: </small>
                              <strong>{path.total_amount.toFixed(2)} BTC</strong>
                            </div>
                          </div>
                        </Card.Header>
                        <Card.Body>
                          <div className="path-flow">
                            {path.path.map((node, nodeIndex) => (
                              <div key={nodeIndex}>
                                <div className="d-flex align-items-center mb-3">
                                  <div 
                                    className="me-3"
                                    style={{
                                      width: '40px',
                                      height: '40px',
                                      borderRadius: '50%',
                                      background: getRiskColor(node.risk_score),
                                      display: 'flex',
                                      alignItems: 'center',
                                      justifyContent: 'center',
                                      color: 'white',
                                      fontWeight: 'bold',
                                      fontSize: '14px'
                                    }}
                                  >
                                    {nodeIndex + 1}
                                  </div>
                                  <div className="flex-grow-1">
                                    <div className="d-flex justify-content-between align-items-start mb-1">
                                      <div>
                                        <small className="text-muted">交易ID:</small>
                                        <code className="ms-2" style={{ 
                                          fontSize: '0.85rem',
                                          background: 'rgba(26, 32, 53, 0.5)',
                                          padding: '2px 6px',
                                          borderRadius: '4px'
                                        }}>
                                          {node.tx_id}
                                        </code>
                                      </div>
                                      <Badge bg="secondary">
                                        {node.amount.toFixed(2)} BTC
                                      </Badge>
                                    </div>
                                    <div className="d-flex justify-content-between align-items-center">
                                      <small className="text-muted">
                                        地址: {node.address.substring(0, 16)}...
                                      </small>
                                      <small className="text-muted">
                                        风险: {(node.risk_score * 100).toFixed(1)}%
                                      </small>
                                    </div>
                                  </div>
                                </div>
                                {nodeIndex < path.path.length - 1 && (
                                  <div className="text-center mb-3">
                                    <div style={{ 
                                      fontSize: '20px', 
                                      color: 'var(--bs-primary)',
                                      opacity: 0.6
                                    }}>
                                      ⬇️
                                    </div>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                          {path.suspicious_count > 0 && (
                            <Alert variant="warning" className="mt-3 mb-0">
                              <small>
                                ⚠️ 该路径包含 <strong>{path.suspicious_count}</strong> 个可疑节点
                              </small>
                            </Alert>
                          )}
                        </Card.Body>
                      </Card>
                    ))}
                  </div>
                )}
              </Card.Body>
            </Card>
          )}

          {!result && !loading && (
            <Card>
              <Card.Body className="text-center py-5">
                <div style={{ fontSize: '80px', opacity: 0.2 }}>🔗</div>
                <h5 className="text-muted mt-3">输入交易ID开始追踪</h5>
                <p className="text-secondary">
                  系统将分析交易的资金流向，识别潜在的洗钱路径
                </p>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
};

export default MoneyLaunderingTrace;
