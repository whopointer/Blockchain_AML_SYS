import React, { useState } from 'react';
import { Container, Nav, Navbar, Tab, Tabs, Row, Col } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

import Dashboard from './components/Dashboard';
import PredictionForm from './components/PredictionForm';
import ResultsTable from './components/ResultsTable';
import BatchAnalysis from './components/BatchAnalysis';
import MoneyLaunderingTrace from './components/MoneyLaunderingTrace';
import { PredictionResponse } from './services/api';

function App() {
  const [predictionResults, setPredictionResults] = useState<PredictionResponse | null>(null);

  const handlePredictionComplete = (results: PredictionResponse) => {
    setPredictionResults(results);
  };

  return (
    <div className="App">
      <Navbar expand="lg" className="fixed-top">
        <Container>
          <Navbar.Brand href="#home">
            区块链AML反洗钱系统
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link href="#dashboard">系统仪表板</Nav.Link>
              <Nav.Link href="#prediction">交易检测</Nav.Link>
              <Nav.Link href="#batch">批量分析</Nav.Link>
              <Nav.Link href="#trace">路径追踪</Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container fluid className="mt-5 pt-4">
        <Row className="justify-content-center">
          <Col xl={10} lg={11} md={12}>
            <Tabs
              defaultActiveKey="dashboard"
              id="main-tabs"
              className="mb-4"
              fill
              justify
            >
              <Tab eventKey="dashboard" title="🎯 系统仪表板">
                <Dashboard />
              </Tab>
              <Tab eventKey="prediction" title="🔍 交易异常检测">
                <Row>
                  <Col lg={5} className="mb-4">
                    <PredictionForm onPredictionComplete={handlePredictionComplete} />
                  </Col>
                  <Col lg={7}>
                    <ResultsTable results={predictionResults} />
                  </Col>
                </Row>
              </Tab>
              <Tab eventKey="batch" title="📊 批量分析">
                <BatchAnalysis />
              </Tab>
              <Tab eventKey="trace" title="🔗 洗钱路径追踪">
                <MoneyLaunderingTrace />
              </Tab>
            </Tabs>
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;
