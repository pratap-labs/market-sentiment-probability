import { BrowserRouter, NavLink, Route, Routes, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import PortfolioDashboard from "./pages/PortfolioDashboard";
import LongTerm from "./pages/LongTerm";
import Equities from "./pages/Equities";
import RiskBuckets from "./pages/RiskBuckets";
import RiskBucketsPortfolio from "./pages/RiskBucketsPortfolio";
import RiskBucketsBucket from "./pages/RiskBucketsBucket";
import RiskBucketsTrade from "./pages/RiskBucketsTrade";
import RiskBucketsMeta from "./pages/RiskBucketsMeta";
import RiskBucketsSettings from "./pages/RiskBucketsSettings";
import PreTradeStress from "./pages/PreTradeStress";
import TradeSelector from "./pages/TradeSelector";
import MarketRegime from "./pages/MarketRegime";
import HistoricalPerformance from "./pages/HistoricalPerformance";
import ProductOverview from "./pages/ProductOverview";
import DataSource from "./pages/DataSource";
import RiskBucketsSpotAnalysis from "./pages/RiskBucketsSpotAnalysis";
import SpotAnalysis from "./pages/SpotAnalysis";
import SpotAnalysisParticipants from "./pages/SpotAnalysisParticipants";
import SpotAnalysisModeling from "./pages/SpotAnalysisModeling";
import SpotAnalysisConstituents from "./pages/SpotAnalysisConstituents";
import SpotAnalysisGapEdge from "./pages/SpotAnalysisGapEdge";
import SpotAnalysisWorldIndexes from "./pages/SpotAnalysisWorldIndexes";
import { PortfolioProvider } from "./state/PortfolioContext";
import { NotificationProvider } from "./state/NotificationContext";
import AuthGate from "./components/AuthGate";
import { ControlsProvider } from "./state/ControlsContext";
import ControlsBar from "./components/ControlsBar";
import LiveTicker from "./components/LiveTicker";

const navItems = [
  { label: "Login", to: "/login" },
  { label: "Portfolio", to: "/portfolio" },
  { label: "Long-Term", to: "/long-term" },
  { label: "Equities", to: "/equities" },
  { label: "Spot Analysis", to: "/spot-analysis" },
  { label: "Risk Buckets", to: "/risk-buckets" },
  { label: "Pre-Trade", to: "/pre-trade" },
  { label: "Trade Selector", to: "/trade-selector" },
  { label: "Market Regime", to: "/market-regime" },
  { label: "Historical", to: "/historical" },
  { label: "Data Source", to: "/data-source" },
  { label: "Settings", to: "/settings" }
];

export default function App() {
  return (
    <BrowserRouter>
      <NotificationProvider>
        <ControlsProvider>
          <PortfolioProvider>
            <AuthGate>
              <div className="app-shell">
                <aside className="sidebar">
                  <div className="brand">
                    <img src="/gammashield-logo.png" alt="GammaShield" className="brand-logo" />
                    <div className="brand-subtitle">Options Risk Engine</div>
                  </div>
                  <nav className="nav">
                    {navItems.map((item) => (
                      <NavLink key={item.to} to={item.to} className={({ isActive }) => (isActive ? "active" : "")}
                      >
                        {item.label}
                      </NavLink>
                    ))}
                  </nav>
                </aside>
                <div className="main">
                  <header className="topbar">
                    <LiveTicker />
                  </header>
                  <ControlsBar />
                  <main className="content">
                    <Routes>
                      <Route path="/" element={<Login />} />
                      <Route path="/login" element={<Login />} />
                      <Route path="/portfolio" element={<PortfolioDashboard />} />
                      <Route path="/long-term" element={<LongTerm />} />
                      <Route path="/equities" element={<Equities />} />
                      <Route path="/spot-analysis" element={<SpotAnalysis />}>
                        <Route index element={<Navigate to="overview" replace />} />
                        <Route path="overview" element={<RiskBucketsSpotAnalysis />} />
                        <Route path="participants" element={<SpotAnalysisParticipants />} />
                        <Route path="gap-edge" element={<SpotAnalysisGapEdge />} />
                        <Route path="world-indexes" element={<SpotAnalysisWorldIndexes />} />
                        <Route path="constituents" element={<SpotAnalysisConstituents />} />
                        <Route path="modeling" element={<SpotAnalysisModeling />} />
                      </Route>
                      <Route path="/risk-buckets" element={<RiskBuckets />}>
                        <Route index element={<Navigate to="portfolio" replace />} />
                        <Route path="portfolio" element={<RiskBucketsPortfolio />} />
                        <Route path="bucket" element={<RiskBucketsBucket />} />
                        <Route path="trade" element={<RiskBucketsTrade />} />
                        <Route path="spot-analysis" element={<RiskBucketsSpotAnalysis />} />
                        <Route path="meta" element={<RiskBucketsMeta />} />
                      </Route>
                      <Route path="/pre-trade" element={<PreTradeStress />} />
                      <Route path="/trade-selector" element={<TradeSelector />} />
                      <Route path="/market-regime" element={<MarketRegime />} />
                      <Route path="/historical" element={<HistoricalPerformance />} />
                      <Route path="/product" element={<ProductOverview />} />
                      <Route path="/derivatives" element={<RiskBucketsSpotAnalysis />} />
                      <Route path="/data-source" element={<DataSource />} />
                      <Route path="/settings" element={<RiskBucketsSettings />} />
                    </Routes>
                  </main>
                </div>
              </div>
            </AuthGate>
          </PortfolioProvider>
        </ControlsProvider>
      </NotificationProvider>
    </BrowserRouter>
  );
}
