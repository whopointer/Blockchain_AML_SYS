import React, { useState } from 'react';
import { Form, Button, Alert, Spinner, Card, Table, Badge, Row, Col } from 'react-bootstrap';
import { api, detectionApi, DetectResponse, getRiskLevelColor } from '../../services/api';
import DetailPanel from './DetailPanel';

interface AddressResult extends DetectResponse {
  error?: string;
}

const AddressDetectionPanel: React.FC = () => {
  const [addresses, setAddresses] = useState<string>('');
  const [addressType, setAddressType] = useState<string>('bitcoin');
  const [neighborDepth, setNeighborDepth] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [results, setResults] = useState<any>(null);
  const [selectedDetail, setSelectedDetail] = useState<DetectResponse | null>(null);
  const [showDetail, setShowDetail] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setResults(null);
    setLoading(true);

    try {
      const addressList = addresses
        .split('\n')
        .map(addr => addr.trim())
        .filter(addr => addr);

      if (addressList.length === 0) {
        throw new Error('请至少输入一个地址');
      }

      let response;

      // 单个地址 - 使用 /detect
      if (addressList.length === 1) {
        const singleResult = await detectionApi.detect({
          address: addressList[0],
          address_type: addressType,
          model_type: 'gnn',
          neighbor_depth: neighborDepth
        });

        response = {
          results: [singleResult],
          statistics: {
            total: 1,
            success: 1,
            error: 0,
            suspicious: singleResult.is_suspicious ? 1 : 0,
            normal: singleResult.is_suspicious ? 0 : 1
          }
        };
      } else {
        // 多个地址 - 使用 /batch_detect
        response = await api.batchDetect({
          addresses: addressList,
          address_type: addressType,
          model_type: 'gnn',
          neighbor_depth: neighborDepth
        });
      }

      setResults(response);
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || '检测失败，请重试';
      console.error('检测错误:', err);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setAddresses('');
    setResults(null);
    setError('');
  };

  const handleRowClick = (result: AddressResult) => {
    if (!result.error) {
      setSelectedDetail(result as DetectResponse);
      setShowDetail(true);
    }
  };

  const getAddressCount = () => {
    return addresses.split('\n').filter(a => a.trim()).length;
  };

  return (
    <div className="address-detection-panel">
      {/* 标题 */}
      <div className="text-center mb-4">
        <h3>🔍 统一地址风险检测</h3>
        <p className="text-secondary">支持单地址详细分析和多地址批量检测</p>
      </div>

      {/* 输入卡片 */}
      <Card className="mb-4">
        <Card.Header className="bg-light">
          <h6 className="mb-0">📝 输入地址</h6>
        </Card.Header>
        <Card.Body>
          <Form onSubmit={handleSubmit}>
            <Form.Group className="mb-3">
              <Form.Label>区块链地址 (每行一个)</Form.Label>
              <Form.Control
                as="textarea"
                rows={6}
                placeholder={`输入一个或多个地址\n例如：\nbc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh\n3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy`}
                value={addresses}
                onChange={(e) => setAddresses(e.target.value)}
                disabled={loading}
              />
              <Form.Text className="text-muted">
                已输入: {getAddressCount()} 个地址
              </Form.Text>
            </Form.Group>

            {/* 参数设置 */}
            <Row>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>地址类型</Form.Label>
                  <Form.Select
                    value={addressType}
                    onChange={(e) => setAddressType(e.target.value)}
                    disabled={loading}
                  >
                    <option value="bitcoin">比特币 (Bitcoin)</option>
                    <option value="ethereum">以太坊 (Ethereum)</option>
                  </Form.Select>
                </Form.Group>
              </Col>
              <Col md={6}>
                <Form.Group className="mb-3">
                  <Form.Label>邻居深度 (1-3)</Form.Label>
                  <Form.Control
                    type="number"
                    min="1"
                    max="3"
                    value={neighborDepth}
                    onChange={(e) => setNeighborDepth(Number(e.target.value))}
                    disabled={loading}
                  />
                </Form.Group>
              </Col>
            </Row>

            {/* 操作按钮 */}
            <div className="d-grid gap-2 d-md-flex justify-content-md-end">
              <Button
                variant="outline-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                清空
              </Button>
              <Button
                variant="primary"
                type="submit"
                disabled={loading || getAddressCount() === 0}
              >
                {loading ? (
                  <>
                    <Spinner size="sm" className="me-2" animation="border" />
                    检测中...
                  </>
                ) : (
                  '🔍 开始检测'
                )}
              </Button>
            </div>
          </Form>
        </Card.Body>
      </Card>

      {/* 错误提示 */}
      {error && (
        <Alert variant="danger" dismissible onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* 结果展示 */}
      {results && (
        <div className="detection-results">
          {/* 统计卡片 */}
          <Row className="mb-4">
            <Col md={3}>
              <Card className="text-center">
                <Card.Body>
                  <div style={{ fontSize: '28px', color: '#667eea' }}>
                    {results.statistics.total}
                  </div>
                  <small className="text-muted">总检测数</small>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="text-center">
                <Card.Body>
                  <div style={{ fontSize: '28px', color: '#28a745' }}>
                    {results.statistics.success}
                  </div>
                  <small className="text-muted">成功数</small>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="text-center">
                <Card.Body>
                  <div style={{ fontSize: '28px', color: '#dc3545' }}>
                    {results.statistics.suspicious}
                  </div>
                  <small className="text-muted">可疑地址</small>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="text-center">
                <Card.Body>
                  <div style={{ fontSize: '28px', color: '#ffc107' }}>
                    {results.statistics.error}
                  </div>
                  <small className="text-muted">错误数</small>
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* 结果表格 */}
          <Card>
            <Card.Header className="bg-light">
              <h6 className="mb-0">📋 检测结果 (点击行查看详情)</h6>
            </Card.Header>
            <Card.Body>
              <div className="table-responsive">
                <Table hover striped>
                  <thead>
                    <tr>
                      <th>地址</th>
                      <th>风险等级</th>
                      <th>异常概率</th>
                      <th>状态</th>
                      <th>操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.results.map((result: AddressResult, idx: number) => (
                      <tr
                        key={idx}
                        style={{
                          cursor: result.error ? 'default' : 'pointer',
                          backgroundColor: result.error ? '#f8f9fa' : 'inherit'
                        }}
                        onClick={() => !result.error && handleRowClick(result)}
                      >
                        <td style={{ wordBreak: 'break-all', fontSize: '12px', maxWidth: '200px' }}>
                          {result.address}
                          {result.error && (
                            <div className="text-danger small mt-1">
                              ⚠️ {result.error}
                            </div>
                          )}
                        </td>
                        <td>
                          {!result.error ? (
                            <Badge
                              style={{
                                backgroundColor: getRiskLevelColor(result.risk_label)
                              }}
                            >
                              {result.risk_label === 'high' ? '高风险' :
                               result.risk_label === 'medium' ? '中风险' :
                               result.risk_label === 'low' ? '低风险' : '正常'}
                            </Badge>
                          ) : (
                            <Badge bg="secondary">N/A</Badge>
                          )}
                        </td>
                        <td>
                          {!result.error ? (
                            `${(result.probability * 100).toFixed(2)}%`
                          ) : (
                            'N/A'
                          )}
                        </td>
                        <td>
                          {!result.error ? (
                            <Badge bg={result.is_suspicious ? 'danger' : 'success'}>
                              {result.is_suspicious ? '可疑' : '正常'}
                            </Badge>
                          ) : (
                            <Badge bg="danger">失败</Badge>
                          )}
                        </td>
                        <td>
                          {!result.error && (
                            <Button
                              variant="link"
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleRowClick(result);
                              }}
                            >
                              详情
                            </Button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>
            </Card.Body>
          </Card>
        </div>
      )}

      {/* 详情面板 */}
      <DetailPanel
        show={showDetail}
        data={selectedDetail}
        onHide={() => setShowDetail(false)}
      />
    </div>
  );
};

export default AddressDetectionPanel;
