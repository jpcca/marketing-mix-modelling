import { Filter } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  DateTimeNumericAxis,
  EHorizontalAnchorPoint,
  ELegendPlacement,
  EVerticalAnchorPoint,
  LegendModifier,
  LogarithmicAxis,
  LogarithmicLabelProvider,
  MouseWheelZoomModifier,
  NumberRange,
  RolloverModifier,
  SciChartSurface,
  SplineMountainRenderableSeries,
  XyDataSeries,
  ZoomExtentsModifier,
  ZoomPanModifier
} from "scichart";
import { SciChartReact } from "scichart-react";
import { loadData } from './utils/dataLoader';
import { initSciChart } from './utils/scichartConfig';

// Initialize WASM paths
initSciChart();

class SmartTickLabelProvider extends LogarithmicLabelProvider {
  formatLabel(dataValue) {
    if (dataValue < 1000) return dataValue.toString();
    if (dataValue < 1000000) return (dataValue / 1000) + "k";
    if (dataValue < 1000000000) return (dataValue / 1000000) + "M";
    return (dataValue / 1000000000) + "B";
  }
}

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters state
  const [selectedVertical, setSelectedVertical] = useState('All');
  const [selectedTerritory, setSelectedTerritory] = useState('All Territories');

  // Chart References for updating data
  const trendChartRef = useRef(null);

  useEffect(() => {
    loadData()
      .then((parsedData) => {
        setData(parsedData);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Failed to load data.");
        setLoading(false);
      });
  }, []);

  // Compute unique filter options
  const verticals = useMemo(() => {
    if (!data.length) return [];
    const set = new Set(data.map(d => d.ORGANISATION_VERTICAL).filter(Boolean));
    return ['All', ...Array.from(set).sort()];
  }, [data]);

  const territories = useMemo(() => {
    if (!data.length) return [];
    const set = new Set(data.map(d => d.TERRITORY_NAME).filter(Boolean));
    return ['All Territories', ...Array.from(set).filter(t => t !== 'All Territories').sort()];
  }, [data]);

  // Filter Data
  const filteredData = useMemo(() => {
    return data.filter(row => {
      if (selectedVertical !== 'All' && row.ORGANISATION_VERTICAL !== selectedVertical) return false;
      if (selectedTerritory !== 'All Territories' && row.TERRITORY_NAME !== selectedTerritory) return false;
      return true;
    });
  }, [data, selectedVertical, selectedTerritory]);

  // Aggregations for Charts
  const timelineData = useMemo(() => {
    if (!filteredData.length) return [];
    
    // Group by Date
    const grouped = {};
    filteredData.forEach(row => {
        const date = row.DATE_DAY;
        if (!date) return;
        
        if (!grouped[date]) {
            grouped[date] = { 
                date, 
                spend: 0, 
                revenue: 0
            };
        }
        
        grouped[date].spend += (row.GOOGLE_PAID_SEARCH_SPEND || 0) + (row.GOOGLE_SHOPPING_SPEND || 0) + (row.GOOGLE_PMAX_SPEND || 0) + (row.GOOGLE_DISPLAY_SPEND || 0) + (row.GOOGLE_VIDEO_SPEND || 0) + (row.META_FACEBOOK_SPEND || 0) + (row.META_INSTAGRAM_SPEND || 0) + (row.META_OTHER_SPEND || 0) + (row.TIKTOK_SPEND || 0);
        grouped[date].revenue += (row.ALL_PURCHASES_ORIGINAL_PRICE || 0);
    });

    return Object.values(grouped).sort((a, b) => new Date(a.date) - new Date(b.date));
  }, [filteredData]);

  // Key Metrics
  const metrics = useMemo(() => {
      const totalSpend = timelineData.reduce((acc, curr) => acc + curr.spend, 0);
      const totalRevenue = timelineData.reduce((acc, curr) => acc + curr.revenue, 0);
      const roas = totalSpend > 0 ? totalRevenue / totalSpend : 0;
      return { totalSpend, totalRevenue, roas };
  }, [timelineData]);

  // Init Spend vs Revenue Chart
  const initTrendChart = async (divElement) => {
    const { sciChartSurface, wasmContext } = await SciChartSurface.create(divElement, {
      theme: { type: "Dark" }
    });

    const xAxis = new DateTimeNumericAxis(wasmContext);
    const yAxis = new LogarithmicAxis(wasmContext, {
      axisTitle: "Amount ($)",
      textStyle: { fontSize: 12 },
      logBase: 10,
      visibleRange: new NumberRange(1, 10000000000),
      growBy: new NumberRange(0.1, 0.1),
      labelProvider: new SmartTickLabelProvider()
    });

    sciChartSurface.xAxes.add(xAxis);
    sciChartSurface.yAxes.add(yAxis);

    // Data Series
    const revenueSeries = new XyDataSeries(wasmContext, { dataSeriesName: "Revenue" });
    const spendSeries = new XyDataSeries(wasmContext, { dataSeriesName: "Spend" });

    // Renderable Series
    const revenueRenderableSeries = new SplineMountainRenderableSeries(wasmContext, {
      dataSeries: revenueSeries,
      stroke: "#10b981",
      strokeThickness: 2,
      fill: "#10b98133", // hex with opacity
      opacity: 0.7
    });

    const spendRenderableSeries = new SplineMountainRenderableSeries(wasmContext, {
      dataSeries: spendSeries,
      stroke: "#ef4444",
      strokeThickness: 2,
      fill: "#ef444433",
      opacity: 0.7
    });

    sciChartSurface.renderableSeries.add(revenueRenderableSeries);
    sciChartSurface.renderableSeries.add(spendRenderableSeries);

    // Modifiers
    sciChartSurface.chartModifiers.add(
      new ZoomPanModifier(),
      new ZoomExtentsModifier(),
      new MouseWheelZoomModifier(),
      new RolloverModifier({
          modifierGroup: "trendGroup",
          showTooltip: true,
      }),
      new LegendModifier({ 
        placement: ELegendPlacement.TopLeft,
        verticalAnchorPoint: EVerticalAnchorPoint.Top,
        horizontalAnchorPoint: EHorizontalAnchorPoint.Left,
      })
    );

    trendChartRef.current = { sciChartSurface, revenueSeries, spendSeries };
    return { sciChartSurface };
  };

  // Update Data Effect
  useEffect(() => {
    console.log("Timeline Data:", timelineData);

    // Helper to clamp log values
    const clamp = (v) => Math.max(v, 1);

    // Trend Chart Update
    if (trendChartRef.current) {
        const { sciChartSurface, revenueSeries, spendSeries } = trendChartRef.current;
        const xValues = timelineData.map(d => new Date(d.date).getTime() / 1000); // Unix Timestamp
        const revenueValues = timelineData.map(d => clamp(d.revenue));
        const spendValues = timelineData.map(d => clamp(d.spend));

        console.log("Trend Chart Update:", { xValues, revenueValues, spendValues });

        revenueSeries.clear();
        revenueSeries.appendRange(xValues, revenueValues);
        
        spendSeries.clear();
        spendSeries.appendRange(xValues, spendValues);

        sciChartSurface.zoomExtents();
    }
  }, [timelineData]);

  if (loading) {
    return <div className="loading-container">Loading Conjura MMM Data...</div>;
  }

  if (error) {
    return <div className="loading-container text-red-500">{error}</div>;
  }

  return (
    <div className="container">
      <header style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 'bold' }}>MMM Data Explorer</h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          {filteredData.length.toLocaleString()} rows • {timelineData.length} days
        </p>
      </header>

      {/* Controls */}
      <div className="controls">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Filter size={18} />
          <span style={{ fontWeight: 600 }}>Filters:</span>
        </div>
        
        <select value={selectedVertical} onChange={e => setSelectedVertical(e.target.value)}>
          {verticals.map(v => <option key={v} value={v}>{v}</option>)}
        </select>

        <select value={selectedTerritory} onChange={e => setSelectedTerritory(e.target.value)}>
           {territories.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
      </div>

      {/* Scorecards */}
      <div className="dashboard-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <div className="card">
            <h2>Total Spend</h2>
            <div className="metric-value">
                ${metrics.totalSpend.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
        </div>
        <div className="card">
            <h2>Total Revenue</h2>
            <div className="metric-value">
                ${metrics.totalRevenue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
        </div>
        <div className="card">
            <h2>ROAS</h2>
            <div className="metric-value">
                {metrics.roas.toFixed(2)}x
            </div>
        </div>
      </div>

      {/* Charts */}
      <div className="dashboard-grid" style={{ gridTemplateColumns: '1fr' }}>
        <div className="card" style={{ height: '400px' }}>
            <h2>Spend vs Revenue (Trend)</h2>
            <div style={{ flex: 1, height: '100%', position: 'relative' }}>
              <SciChartReact 
                initChart={initTrendChart} 
                style={{ width: '100%', height: '100%' }}
              />
            </div>
        </div>
      </div>
    </div>
  )
}

export default App
