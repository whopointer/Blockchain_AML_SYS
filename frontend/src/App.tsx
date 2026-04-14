import React, { useState } from "react";
import { Container, Nav, Navbar, Tab, Tabs, Row, Col } from "react-bootstrap";
import { Routes, Route, Navigate, useLocation, Link } from "react-router-dom";
import { Helmet } from "react-helmet-async";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

import Dashboard from "./components/Dashboard";
import ResultsTable from "./components/ResultsTable";
import MoneyLaunderingTrace from "./components/MoneyLaunderingTrace";
import TransactionGraph from "./components/TransactionGraph";
import CaseDetails from "./components/CaseDetails";
import PathTracking from "./components/PathTracking";
import AddressDetectionForm from "./components/AddressDetectionForm";
import { PredictionResponse } from "./services/api";
import AddressDetectionPanel from "./components/detection/AddressDetectionPanel";

const CustomNavLink = ({
  children,
  to,
  ...props
}: {
  children: React.ReactNode;
  to: string;
  [key: string]: any;
}) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Nav.Link as={Link} to={to} className={isActive ? "active" : ""} {...props}>
      {children}
    </Nav.Link>
  );
};

const DashboardPage = () => {
  const [predictionResults, setPredictionResults] =
    useState<PredictionResponse | null>(null);

  const handlePredictionComplete = (results: PredictionResponse) => {
    setPredictionResults(results);
  };

  return (
    <Tab.Container defaultActiveKey="dashboard">
      <Helmet>
        <title>系统仪表板 - 区块链AML反洗钱系统</title>
      </Helmet>
      <Tabs id="main-tabs" className="mb-4" fill justify>
        <Tab eventKey="dashboard" title="🎯 系统仪表板">
          <Dashboard />
        </Tab>
        <Tab eventKey="detection" title="🔍 地址检测">
          <AddressDetectionPanel />
        </Tab>
        <Tab eventKey="trace" title="🔗 洗钱路径追踪">
          <MoneyLaunderingTrace />
        </Tab>
      </Tabs>
    </Tab.Container>
  );
};

function App() {
  return (
    <div className="App">
      <Helmet>
        <title>区块链AML反洗钱系统</title>
      </Helmet>
      <Navbar expand="lg" className="fixed-top">
        <Container>
          <Navbar.Brand as={Link} to="/dashboard">
            区块链AML反洗钱系统
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <CustomNavLink to="/dashboard">系统仪表板</CustomNavLink>
              <CustomNavLink to="/address-detection">地址检测</CustomNavLink>
              <CustomNavLink to="/transaction-graph">交易图谱</CustomNavLink>
              <CustomNavLink to="/transaction-path">交易路径</CustomNavLink>
              <CustomNavLink to="/case-details">案件中心</CustomNavLink>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container fluid className="mt-5 pt-4">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route
            path="/dashboard"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={11} md={12}>
                  <DashboardPage />
                </Col>
              </Row>
            }
          />
          <Route
            path="/prediction"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={11} md={12}>
                  <DashboardPage />
                </Col>
              </Row>
            }
          />
          <Route
            path="/batch"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={11} md={12}>
                  <DashboardPage />
                </Col>
              </Row>
            }
          />
          <Route
            path="/trace"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={11} md={12}>
                  <DashboardPage />
                </Col>
              </Row>
            }
          />
          <Route
            path="/address-detection"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={11} md={12}>
                  <AddressDetectionForm />
                </Col>
              </Row>
            }
          />

          <Route
            path="/transaction-graph"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={12} md={12}>
                  <TransactionGraph />
                </Col>
              </Row>
            }
          />
          <Route
            path="/transaction-graph/:crypto/:address"
            element={
              <Row className="justify-content-center">
                <Col xl={12} lg={12} md={12}>
                  <TransactionGraph />
                </Col>
              </Row>
            }
          />
          <Route
            path="/transaction-path"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={12} md={12}>
                  <PathTracking />
                </Col>
              </Row>
            }
          />
          <Route
            path="/transaction-path/:crypto"
            element={
              <Row className="justify-content-center">
                <Col xl={12} lg={12} md={12}>
                  <PathTracking />
                </Col>
              </Row>
            }
          />

          <Route
            path="/case-details"
            element={
              <Row className="justify-content-center">
                <Col xl={20} lg={18} md={14}>
                  <CaseDetails />
                </Col>
              </Row>
            }
          />
          <Route
            path="/case-details/*"
            element={
              <Row className="justify-content-center">
                <CaseDetails />
              </Row>
            }
          />

          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Container>
    </div>
  );
}

export default App;
