import React, { useState } from 'react';
import { Form, Button, Alert, Spinner } from 'react-bootstrap';
import { api, PredictionRequest, PredictionResponse } from '../services/api';

interface PredictionFormProps {
  onPredictionComplete: (results: PredictionResponse) => void;
}

const PredictionForm: React.FC<PredictionFormProps> = ({ onPredictionComplete }) => {
  const [txIds, setTxIds] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const txIdArray = txIds.split('\n').filter(id => id.trim()).map(id => id.trim());
      
      if (txIdArray.length === 0) {
        setError('请至少输入一个交易ID');
        return;
      }

      const request: PredictionRequest = { tx_ids: txIdArray };
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
      setLoading(false);
    }
  };

  return (
    <div className="prediction-form">
      <div className="text-center mb-4">
        <h3>🔍 交易异常检测</h3>
        <p className="text-secondary">输入区块链交易ID进行智能分析</p>
      </div>
      
      <Form onSubmit={handleSubmit}>
        <Form.Group className="mb-4">
          <Form.Label>
            <span className="me-2">📝</span>
            交易ID列表
          </Form.Label>
          <Form.Control
            as="textarea"
            rows={8}
            placeholder="请输入交易ID，每行一个&#10;例如：&#10;0x1234567890abcdef...&#10;0x9876543210fedcba...&#10;0xabcdef1234567890..."
            value={txIds}
            onChange={(e) => setTxIds(e.target.value)}
            disabled={loading}
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
            disabled={loading}
            size="lg"
            className="position-relative"
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
                <span className="me-2">🚀</span>
                开始智能检测
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