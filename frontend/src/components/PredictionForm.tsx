import React, { useState, useEffect } from 'react';
import { Form, Button, Alert, Spinner, Card } from 'react-bootstrap';
import { api, PredictionRequest, PredictionResponse, getModelDisplayName, getModelColor } from '../services/api';

interface PredictionFormProps {
  onPredictionComplete: (results: PredictionResponse) => void;
  currentModelType?: string;
}

const PredictionForm: React.FC<PredictionFormProps> = ({
  onPredictionComplete,
  currentModelType = 'gnn'
}) => {
  const [txIds, setTxIds] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    // 模拟快速加载完成
    const timer = setTimeout(() => {
      setLoading(false);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredicting(true);
    setError('');

    try {
      const txIdArray = txIds.split('\n').filter(id => id.trim()).map(id => id.trim());

      if (txIdArray.length === 0) {
        setError('请至少输入一个交易ID');
        return;
      }

      const request: PredictionRequest = {
        tx_ids: txIdArray
      };
      const results = await api.predictTransactions(request);
      onPredictionComplete(results);
    } catch (err: any) {
      const errorMessage = err.response?.data?.error || '预测失败，请重试';
      console.error('预测错误:', err);
      setError(errorMessage);

      // 如果是模型未加载的错误，提示用户先加载模型
      if (errorMessage.includes('模型') || errorMessage.includes('model')) {
        setError('模型未加载，请先在系统仪表板中加载模型后再进行检测');
      }
    } finally {
      setPredicting(false);
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


  const modelColor = getModelColor(currentModelType);

  return (
    <div className="prediction-form">
      <Card className="mb-4" style={{
        background: `linear-gradient(135deg, ${modelColor}15 0%, ${modelColor}08 100%)`,
        border: `1px solid ${modelColor}30`
      }}>
        <Card.Body className="py-3">
          <div className="d-flex align-items-center justify-content-between">
            <div className="d-flex align-items-center">
              <span className="me-2" style={{ fontSize: '24px' }}>🤖</span>
              <div>
                <div className="text-muted small">当前检测模型</div>
                <div className="fw-bold" style={{ color: modelColor }}>
                  {getModelDisplayName(currentModelType)}
                </div>
              </div>
            </div>
            <small className="text-muted">
              如需更换模型，请前往「系统仪表板」
            </small>
          </div>
        </Card.Body>
      </Card>

      <div className="text-center mb-4">
        <h3>🔍 交易异常检测</h3>
        <p className="text-secondary">输入区块链交易ID进行智能分析</p>
      </div>

      <Form onSubmit={handleSubmit}>
        {/* 交易ID输入 */}
        <Form.Group className="mb-4">
          <Form.Label>
            <span className="me-2">📝</span>
            交易ID列表
          </Form.Label>
          <Form.Control
            as="textarea"
            rows={8}
            placeholder="请输入交易ID，每行一个\n例如：\n0x1234567890abcdef...\n0x9876543210fedcba...\n0xabcdef1234567890..."
            value={txIds}
            onChange={(e) => setTxIds(e.target.value)}
            disabled={predicting}
            style={{
              fontFamily: 'Monaco, Consolas, "Courier New", monospace',
              fontSize: '0.9rem'
            }}
          />
          <Form.Text className="text-muted">
            💡 支持批量输入，每行一个交易ID，系统将并行处理
          </Form.Text>
        </Form.Group>

        {error && (
          <Alert variant="danger" className="mb-4">
            <div className="d-flex align-items-center">
              <span className="me-2">⚠️</span>
              <div>{error}</div>
            </div>
          </Alert>
        )}

        <div className="d-grid gap-2">
          <Button
            variant="primary"
            type="submit"
            disabled={predicting}
            size="lg"
            className="position-relative"
            style={{
              background: `linear-gradient(135deg, ${modelColor} 0%, ${modelColor}dd 100%)`,
              border: 'none'
            }}
          >
            {predicting ? (
              <>
                <Spinner
                  as="span"
                  animation="border"
                  size="sm"
                  className="me-2"
                />
                正在使用 {getModelDisplayName(currentModelType)} 检测中...
              </>
            ) : (
              <>
                <span className="me-2">🚀</span>
                使用 {getModelDisplayName(currentModelType)} 开始检测
              </>
            )}
          </Button>
        </div>

        {txIds && (
          <div className="mt-3 text-center">
            <small className="text-muted">
              已输入 {txIds.split('\n').filter(id => id.trim()).length} 个交易ID
            </small>
          </div>
        )}
      </Form>
    </div>
  );
};

export default PredictionForm;