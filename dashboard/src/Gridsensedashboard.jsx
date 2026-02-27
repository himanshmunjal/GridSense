import { useState, useEffect, useCallback } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from "recharts";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";
const ZONES = [
  { id: 1, label: "Residential", color: "#00d4ff" },
  { id: 2, label: "Industrial",  color: "#ff6b35" },
  { id: 3, label: "Commercial",  color: "#39ff14" },
  { id: 4, label: "Mixed",       color: "#bf5fff" },
];
const POLL_INTERVAL = 30000; // 30s

// ─── UTILS ───────────────────────────────────────────────────────────────────
const fmtTime = (iso) => {
  const d = new Date(iso);
  return `${String(d.getHours()).padStart(2, "0")}:00`;
};
const fmtDateTime = (iso) => {
  const d = new Date(iso);
  return `${d.toLocaleDateString()} ${String(d.getHours()).padStart(2,"0")}:00`;
};

// ─── HEATMAP COMPONENT ───────────────────────────────────────────────────────
function DeviationHeatmap({ anomalies }) {
  const HOURS = Array.from({ length: 24 }, (_, i) => i);
  const DAYS  = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];

  // Build hour × day deviation grid
  const grid = {};
  (anomalies || []).forEach(a => {
    const d = new Date(a.timestamp);
    const key = `${d.getDay()}-${d.getHours()}`;
    grid[key] = (grid[key] || 0) + a.reconstruction_error;
  });

  const maxVal = Math.max(...Object.values(grid), 0.001);

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "grid", gridTemplateColumns: `40px repeat(24, 1fr)`, gap: 2, minWidth: 600 }}>
        {/* Header row */}
        <div />
        {HOURS.map(h => (
          <div key={h} style={{ fontSize: 10, color: "#666", textAlign: "center", padding: "2px 0" }}>
            {h}
          </div>
        ))}
        {/* Data rows */}
        {DAYS.map((day, di) => (
          <>
            <div key={day} style={{ fontSize: 11, color: "#aaa", display: "flex", alignItems: "center" }}>
              {day}
            </div>
            {HOURS.map(h => {
              const val = grid[`${di}-${h}`] || 0;
              const intensity = val / maxVal;
              const r = Math.round(255 * intensity);
              const g = Math.round(80 * (1 - intensity));
              const b = Math.round(100 * (1 - intensity));
              return (
                <div
                  key={`${di}-${h}`}
                  title={`${day} ${h}:00 — deviation: ${val.toFixed(4)}`}
                  style={{
                    height: 22,
                    borderRadius: 3,
                    background: val > 0
                      ? `rgba(${r},${g},${b},${0.2 + intensity * 0.8})`
                      : "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.04)",
                    cursor: "default",
                    transition: "opacity 0.2s",
                  }}
                />
              );
            })}
          </>
        ))}
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
        <span style={{ fontSize: 10, color: "#666" }}>Low</span>
        {[0.1,0.3,0.5,0.7,0.9].map(v => (
          <div key={v} style={{ width: 16, height: 16, borderRadius: 2,
            background: `rgba(${Math.round(255*v)},${Math.round(80*(1-v))},${Math.round(100*(1-v))},${0.2+v*0.8})` }} />
        ))}
        <span style={{ fontSize: 10, color: "#666" }}>High</span>
      </div>
    </div>
  );
}

// ─── ANOMALY BADGE ────────────────────────────────────────────────────────────
function SeverityBadge({ severity }) {
  const colors = { high: "#ff3b3b", medium: "#ff9500", low: "#ffd60a" };
  return (
    <span style={{
      padding: "2px 8px", borderRadius: 999, fontSize: 10, fontWeight: 700,
      background: colors[severity] + "22", color: colors[severity],
      border: `1px solid ${colors[severity]}44`, textTransform: "uppercase", letterSpacing: 1,
    }}>
      {severity}
    </span>
  );
}

// ─── MAIN DASHBOARD ──────────────────────────────────────────────────────────
export default function GridSenseDashboard() {
  const [activeZone, setActiveZone]     = useState(1);
  const [forecast, setForecast]         = useState(null);
  const [anomalies, setAnomalies]       = useState([]);
  const [loading, setLoading]           = useState(true);
  const [lastUpdated, setLastUpdated]   = useState(null);
  const [error, setError]               = useState(null);

  // ── Fetch data for active zone ─────────────────────────────────────────────
  const fetchData = useCallback(async (zone) => {
    try {
      const [fcRes, anRes] = await Promise.all([
        fetch(`${API_BASE}/forecast?zone=${zone}`),
        fetch(`${API_BASE}/anomalies?zone=${zone}`),
      ]);
      const fcData = await fcRes.json();
      const anData = await anRes.json();
      setForecast(fcData);
      setAnomalies(anData.anomalies || []);
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      setError("API unavailable — showing mock data");
      // Mock data so dashboard renders without a running backend
      setForecast(getMockForecast(zone));
      setAnomalies(getMockAnomalies());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    fetchData(activeZone);
    const interval = setInterval(() => fetchData(activeZone), POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [activeZone, fetchData]);

  // ── Build chart data ───────────────────────────────────────────────────────
  const chartData = forecast ? [
    ...forecast.actual_history.map((v, i) => ({
      time: fmtTime(forecast.history_timestamps[i]),
      actual: parseFloat(v.toFixed(2)),
      type: "history",
    })),
    ...forecast.forecast.map((v, i) => ({
      time: fmtTime(forecast.timestamps[i]),
      forecast: parseFloat(v.toFixed(2)),
      lower: parseFloat(forecast.lower_bound[i].toFixed(2)),
      upper: parseFloat(forecast.upper_bound[i].toFixed(2)),
      type: "forecast",
    })),
  ] : [];

  const zoneColor = ZONES.find(z => z.id === activeZone)?.color || "#00d4ff";

  // ─── RENDER ───────────────────────────────────────────────────────────────
  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0d14",
      color: "#e8eaf0",
      fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
      padding: "24px",
    }}>
      {/* Google Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; background: #111; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
        .zone-btn:hover { filter: brightness(1.2); }
        .anomaly-row { border-bottom: 1px solid rgba(255,255,255,0.05); transition: background 0.15s; }
        .anomaly-row:hover { background: rgba(255,255,255,0.03); }
      `}</style>

      {/* ── HEADER ─────────────────────────────────────────────────────────── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#39ff14",
              boxShadow: "0 0 8px #39ff14", animation: "pulse 2s infinite" }} />
            <span style={{ fontSize: 11, color: "#39ff14", letterSpacing: 3, fontWeight: 600 }}>LIVE</span>
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 700, letterSpacing: -1, color: "#fff" }}>
            Grid<span style={{ color: zoneColor }}>Sense</span>
          </h1>
          <p style={{ fontSize: 11, color: "#555", marginTop: 2, letterSpacing: 1 }}>
            SMART ELECTRICITY DEMAND FORECASTING & ANOMALY DETECTION
          </p>
        </div>
        <div style={{ textAlign: "right" }}>
          {error && <div style={{ fontSize: 11, color: "#ff9500", marginBottom: 4 }}>⚠ {error}</div>}
          {lastUpdated && (
            <div style={{ fontSize: 11, color: "#444" }}>
              Updated {lastUpdated.toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>

      {/* ── ZONE SELECTOR ──────────────────────────────────────────────────── */}
      <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        {ZONES.map(z => (
          <button
            key={z.id}
            className="zone-btn"
            onClick={() => setActiveZone(z.id)}
            style={{
              padding: "8px 18px", borderRadius: 6, cursor: "pointer", fontSize: 12,
              fontFamily: "inherit", fontWeight: 600, letterSpacing: 1,
              border: `1px solid ${activeZone === z.id ? z.color : "rgba(255,255,255,0.1)"}`,
              background: activeZone === z.id ? z.color + "18" : "transparent",
              color: activeZone === z.id ? z.color : "#666",
              transition: "all 0.2s",
            }}
          >
            Z{z.id} — {z.label.toUpperCase()}
          </button>
        ))}
      </div>

      {/* ── STAT CARDS ─────────────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
        {[
          { label: "ANOMALIES DETECTED", value: anomalies.length, unit: "events", color: anomalies.length > 0 ? "#ff3b3b" : "#39ff14" },
          { label: "HIGH SEVERITY", value: anomalies.filter(a => a.severity === "high").length, unit: "critical", color: "#ff3b3b" },
          { label: "AVG FORECAST", value: forecast ? Math.round(forecast.forecast.reduce((a,b)=>a+b,0)/forecast.forecast.length) : "—", unit: "kWh/hr", color: zoneColor },
          { label: "PEAK FORECAST", value: forecast ? Math.round(Math.max(...forecast.forecast)) : "—", unit: "kWh/hr", color: "#ffd60a" },
        ].map((card, i) => (
          <div key={i} style={{
            background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 8, padding: "16px 20px",
          }}>
            <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, marginBottom: 8 }}>{card.label}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: card.color, lineHeight: 1 }}>{card.value}</div>
            <div style={{ fontSize: 10, color: "#444", marginTop: 4 }}>{card.unit}</div>
          </div>
        ))}
      </div>

      {/* ── MAIN CHART ─────────────────────────────────────────────────────── */}
      <div style={{
        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)",
        borderRadius: 10, padding: "20px 20px 12px", marginBottom: 20,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#fff", letterSpacing: 1 }}>
              24-HOUR DEMAND FORECAST
            </div>
            <div style={{ fontSize: 10, color: "#555", marginTop: 2 }}>
              History (48h) + Forecast (24h) with 95% confidence interval
            </div>
          </div>
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            {[
              { label: "Actual", color: "#555", dashed: false },
              { label: "Forecast", color: zoneColor, dashed: false },
              { label: "95% CI", color: zoneColor, dashed: true },
            ].map(l => (
              <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 18, height: 2, background: l.dashed ? "transparent" : l.color,
                  border: l.dashed ? `1px dashed ${l.color}` : "none", borderRadius: 2 }} />
                <span style={{ fontSize: 10, color: "#666" }}>{l.label}</span>
              </div>
            ))}
          </div>
        </div>

        {loading ? (
          <div style={{ height: 280, display: "flex", alignItems: "center", justifyContent: "center", color: "#444" }}>
            Loading...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <defs>
                <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={zoneColor} stopOpacity={0.2} />
                  <stop offset="95%" stopColor={zoneColor} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#555" }} tickLine={false} axisLine={false}
                interval={5} />
              <YAxis tick={{ fontSize: 10, fill: "#555" }} tickLine={false} axisLine={false}
                tickFormatter={v => `${v}kWh`} width={60} />
              <Tooltip
                contentStyle={{ background: "#13161f", border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 6, fontSize: 11, color: "#ccc" }}
                labelStyle={{ color: "#888", marginBottom: 4 }}
              />
              {/* CI band — rendered as area between upper and lower */}
              <Area dataKey="upper" fill="url(#ciGradient)" stroke="none" />
              <Area dataKey="lower" fill="#0a0d14" stroke="none" />
              {/* Actual history */}
              <Line dataKey="actual" stroke="#3a3f50" strokeWidth={1.5} dot={false} />
              {/* Forecast line */}
              <Line dataKey="forecast" stroke={zoneColor} strokeWidth={2} dot={false}
                strokeDasharray="6 3" />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── BOTTOM PANELS ──────────────────────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Anomaly Alert Panel */}
        <div style={{
          background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: 10, padding: 20,
        }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#fff", letterSpacing: 1, marginBottom: 4 }}>
            ANOMALY ALERT PANEL
          </div>
          <div style={{ fontSize: 10, color: "#555", marginBottom: 16 }}>
            Events deviating &gt;2σ from expected reconstruction
          </div>

          {anomalies.length === 0 ? (
            <div style={{ color: "#39ff14", fontSize: 12, padding: "20px 0", textAlign: "center" }}>
              ✓ No anomalies detected in last 7 days
            </div>
          ) : (
            <div style={{ maxHeight: 280, overflowY: "auto" }}>
              {anomalies.slice(0, 20).map((a, i) => (
                <div key={i} className="anomaly-row" style={{ padding: "10px 4px",
                  display: "grid", gridTemplateColumns: "1fr auto auto", gap: 12, alignItems: "center" }}>
                  <div>
                    <div style={{ fontSize: 11, color: "#ccc" }}>{fmtDateTime(a.timestamp)}</div>
                    <div style={{ fontSize: 10, color: "#555", marginTop: 2 }}>
                      {a.demand_kwh.toFixed(1)} kWh · {a.anomaly_type}
                    </div>
                  </div>
                  <div style={{ fontSize: 10, color: "#666", textAlign: "right" }}>
                    Δ {a.reconstruction_error.toFixed(4)}
                  </div>
                  <SeverityBadge severity={a.severity} />
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Deviation Heatmap */}
        <div style={{
          background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: 10, padding: 20,
        }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#fff", letterSpacing: 1, marginBottom: 4 }}>
            ANOMALY HEATMAP
          </div>
          <div style={{ fontSize: 10, color: "#555", marginBottom: 16 }}>
            Deviation intensity by hour of day × day of week
          </div>
          <DeviationHeatmap anomalies={anomalies} />
        </div>
      </div>

      {/* Pulse animation */}
      <style>{`
        @keyframes pulse {
          0%,100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}

// ─── MOCK DATA (for when API is not running) ──────────────────────────────────
function getMockForecast(zone) {
  const now = new Date();
  const hist = Array.from({ length: 48 }, (_, i) => {
    const h = new Date(now - (48 - i) * 3600000);
    return { v: 200 + 100 * Math.sin(i / 6) + Math.random() * 20, t: h.toISOString() };
  });
  const future = Array.from({ length: 24 }, (_, i) => {
    const h = new Date(now.getTime() + (i + 1) * 3600000);
    const base = 220 + 90 * Math.sin((i + 48) / 6) + Math.random() * 15;
    return { v: base, l: base - 25, u: base + 25, t: h.toISOString() };
  });
  return {
    zone,
    actual_history: hist.map(x => x.v),
    history_timestamps: hist.map(x => x.t),
    forecast: future.map(x => x.v),
    lower_bound: future.map(x => x.l),
    upper_bound: future.map(x => x.u),
    timestamps: future.map(x => x.t),
  };
}

function getMockAnomalies() {
  const now = new Date();
  return Array.from({ length: 5 }, (_, i) => ({
    timestamp: new Date(now - i * 18 * 3600000).toISOString(),
    demand_kwh: 450 + Math.random() * 200,
    reconstruction_error: 0.05 + Math.random() * 0.1,
    anomaly_type: i % 2 === 0 ? "spike" : "dropout",
    severity: ["high", "medium", "low", "medium", "high"][i],
  }));
}