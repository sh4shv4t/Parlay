// ============================================================
// Parlay — Chart.js 4.4.1 Visualization System
// Bloomberg terminal meets poker app.
// Requires Chart.js 4.4.1 from cdnjs.
// ============================================================

const DEBUG = false;

// Resolve CSS variable to a concrete colour (for Chart.js which can't
// use var(--x) in dataset colours).
function _cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return v || fallback;
}

// Parlay colour palette pulled from CSS custom properties
function _palette() {
  return {
    ink:      _cssVar("--parlay-ink",      "#0f1117"),
    ink2:     _cssVar("--parlay-ink-2",    "#3d4151"),
    ink3:     _cssVar("--parlay-ink-3",    "#8a8f9e"),
    surface:  _cssVar("--parlay-surface",  "#ffffff"),
    surface2: _cssVar("--parlay-surface-2","#f4f5f7"),
    border:   _cssVar("--parlay-border",   "#e0e2e7"),
    green:    _cssVar("--parlay-green",    "#00a878"),
    greenBg:  _cssVar("--parlay-green-bg", "#e6f6f2"),
    red:      _cssVar("--parlay-red",      "#e03535"),
    redBg:    _cssVar("--parlay-red-bg",   "#fdf0f0"),
    amber:    _cssVar("--parlay-amber",    "#d97706"),
    amberBg:  _cssVar("--parlay-amber-bg", "#fef9ec"),
    blue:     _cssVar("--parlay-blue",     "#2563eb"),
    blueBg:   _cssVar("--parlay-blue-bg",  "#eff4ff"),
    purple:   _cssVar("--parlay-purple",   "#7c3aed"),
    purpleBg: _cssVar("--parlay-purple-bg","#f3eeff"),
  };
}

// Shared Chart.js defaults matching Parlay aesthetic
function _applyDefaults() {
  const p = _palette();
  Chart.defaults.color            = p.ink3;
  Chart.defaults.borderColor      = p.border;
  Chart.defaults.font.family      = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  Chart.defaults.font.size        = 11;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.tooltip.backgroundColor = p.surface;
  Chart.defaults.plugins.tooltip.titleColor       = p.ink;
  Chart.defaults.plugins.tooltip.bodyColor        = p.ink2;
  Chart.defaults.plugins.tooltip.borderColor      = p.border;
  Chart.defaults.plugins.tooltip.borderWidth      = 1;
  Chart.defaults.plugins.tooltip.padding          = 10;
  Chart.defaults.plugins.tooltip.cornerRadius     = 8;
  Chart.defaults.plugins.tooltip.titleFont        = { weight: "600" };
}

class ParlayCharts {
  constructor() {
    this.rewardChart      = null;
    this.efficiencyChart  = null;
    this.offerSparkline   = null;
    this.beliefChart      = null;
    this._sparklineData   = [];
    this._sparklineZopa   = { lower: 0, upper: 100, nash: 50 };
    this._liveRewardData  = { labels: [], values: [] };
    this._beliefHistory   = [];

    if (typeof Chart !== "undefined") {
      _applyDefaults();
    } else {
      if (DEBUG) console.log("[ParlayCharts] Chart.js not loaded yet");
    }
  }

  // ── Comparison Chart (train.html) ─────────────────────────
  initComparisonChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    _applyDefaults();
    const p = _palette();

    if (this.rewardChart) {
      this.rewardChart.destroy();
      this.rewardChart = null;
    }

    // data shape: { labels: [...], base: [...], sft: [...], grpo: [...] }
    const labels = data.labels || ["Reward", "Deal Rate", "Efficiency", "ToM Acc.", "Avg CP"];
    const baseD  = data.base  || [0.21, 0.34, 0.48, 0.31, 52];
    const sftD   = data.sft   || [0.44, 0.56, 0.63, 0.52, 61];
    const grpoD  = data.grpo  || [0.71, 0.74, 0.82, 0.69, 73];

    // Normalise to 0-1 for radar-like bar chart
    const maxVals = labels.map((_, i) => Math.max(baseD[i], sftD[i], grpoD[i]));

    this.rewardChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Base",
            data: baseD,
            backgroundColor: p.surface2,
            borderColor: p.border,
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: "SFT",
            data: sftD,
            backgroundColor: p.blueBg,
            borderColor: p.blue,
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: "GRPO",
            data: grpoD,
            backgroundColor: p.greenBg,
            borderColor: p.green,
            borderWidth: 2,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "top",
            align: "end",
          },
          title: {
            display: true,
            text: "Model Comparison — Base vs SFT vs GRPO",
            font: { size: 13, weight: "600" },
            color: p.ink,
            padding: { bottom: 16 },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(3)}`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: p.border },
            ticks: { color: p.ink3 },
          },
          y: {
            grid: { color: p.border },
            ticks: { color: p.ink3 },
            beginAtZero: true,
          },
        },
      },
    });

    if (DEBUG) console.log("[ParlayCharts] initComparisonChart done");
    return this.rewardChart;
  }

  // ── Live Reward Curve (train.html, streams during training) ─
  initLiveRewardChart(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    _applyDefaults();
    const p = _palette();

    if (this.efficiencyChart) {
      this.efficiencyChart.destroy();
      this.efficiencyChart = null;
    }

    this._liveRewardData = { labels: [], values: [], baseline: [] };

    this.efficiencyChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "GRPO Reward",
            data: [],
            borderColor: p.green,
            backgroundColor: p.greenBg + "44",
            fill: true,
            tension: 0.4,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
          },
          {
            label: "SFT Baseline",
            data: [],
            borderColor: p.blue,
            borderDash: [4, 4],
            borderWidth: 1.5,
            fill: false,
            tension: 0,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { position: "top", align: "end" },
          title: {
            display: true,
            text: "Live Training — GRPO Reward Curve",
            font: { size: 13, weight: "600" },
            color: p.ink,
            padding: { bottom: 16 },
          },
        },
        scales: {
          x: {
            grid: { color: p.border },
            ticks: {
              color: p.ink3,
              maxTicksLimit: 10,
              callback: (v, i) => `Step ${this._liveRewardData.labels[i] || i}`,
            },
          },
          y: {
            grid: { color: p.border },
            ticks: { color: p.ink3 },
            title: { display: true, text: "Reward", color: p.ink3 },
          },
        },
      },
    });

    if (DEBUG) console.log("[ParlayCharts] initLiveRewardChart done");
    return this.efficiencyChart;
  }

  updateLiveReward(step, reward, sftBaseline) {
    if (!this.efficiencyChart) return;

    this._liveRewardData.labels.push(step);
    this._liveRewardData.values.push(reward);
    if (sftBaseline !== undefined) {
      this._liveRewardData.baseline.push(sftBaseline);
    }

    const chart = this.efficiencyChart;
    chart.data.labels = this._liveRewardData.labels.map((_, i) => i);
    chart.data.datasets[0].data = this._liveRewardData.values;
    if (sftBaseline !== undefined) {
      chart.data.datasets[1].data = this._liveRewardData.baseline;
    }

    // Keep max 500 points for performance
    if (chart.data.labels.length > 500) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
      chart.data.datasets[1].data.shift();
    }

    chart.update("none");
    if (DEBUG) console.log("[ParlayCharts] updateLiveReward step=" + step + " reward=" + reward);
  }

  // ── Offer History Sparkline (game sidebar) ─────────────────
  initOfferSparkline(canvasId, zopaLower, zopaUpper, nashPoint) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    _applyDefaults();
    const p = _palette();

    if (this.offerSparkline) {
      this.offerSparkline.destroy();
      this.offerSparkline = null;
    }

    this._sparklineData  = [];
    this._sparklineZopa  = { lower: zopaLower, upper: zopaUpper, nash: nashPoint };

    const zopaAnnotations = {};
    if (typeof zopaLower === "number") {
      zopaAnnotations.zopaLower = {
        type: "line",
        yMin: zopaLower, yMax: zopaLower,
        borderColor: p.red + "88",
        borderWidth: 1,
        borderDash: [3, 3],
        label: { display: false },
      };
    }
    if (typeof zopaUpper === "number") {
      zopaAnnotations.zopaUpper = {
        type: "line",
        yMin: zopaUpper, yMax: zopaUpper,
        borderColor: p.green + "88",
        borderWidth: 1,
        borderDash: [3, 3],
        label: { display: false },
      };
    }
    if (typeof nashPoint === "number") {
      zopaAnnotations.nash = {
        type: "line",
        yMin: nashPoint, yMax: nashPoint,
        borderColor: p.amber,
        borderWidth: 1.5,
        label: { display: true, content: "Nash", font: { size: 9 }, color: p.amber, position: "end" },
      };
    }

    const plugins = [{ afterDraw: (chart) => this._drawSparklineZopa(chart) }];

    this.offerSparkline = new Chart(canvas, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Player",
            data: [],
            borderColor: p.blue,
            backgroundColor: "transparent",
            borderWidth: 2,
            pointRadius: 3,
            pointBackgroundColor: p.blue,
            tension: 0.3,
          },
          {
            label: "Opponent",
            data: [],
            borderColor: p.red,
            backgroundColor: "transparent",
            borderWidth: 2,
            pointRadius: 3,
            pointBackgroundColor: p.red,
            tension: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "bottom", labels: { padding: 8 } },
          title: {
            display: true,
            text: "Offer History",
            font: { size: 11, weight: "600" },
            color: p.ink,
            padding: { bottom: 8 },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { maxTicksLimit: 6, color: p.ink3, font: { size: 9 } },
            title: { display: true, text: "Turn", color: p.ink3, font: { size: 9 } },
          },
          y: {
            grid: { color: p.border },
            ticks: {
              color: p.ink3,
              font: { size: 9, family: "'JetBrains Mono', monospace" },
              callback: (v) => {
                if (v >= 1000000) return "$" + (v/1000000).toFixed(1) + "M";
                if (v >= 1000)    return "$" + (v/1000).toFixed(0) + "K";
                return "$" + v;
              },
            },
          },
        },
      },
      plugins,
    });

    if (DEBUG) console.log("[ParlayCharts] initOfferSparkline done");
    return this.offerSparkline;
  }

  _drawSparklineZopa(chart) {
    const { lower, upper } = this._sparklineZopa;
    if (lower == null || upper == null) return;

    const ctx   = chart.ctx;
    const yAxis = chart.scales.y;
    if (!yAxis) return;

    const yLow  = yAxis.getPixelForValue(lower);
    const yHigh = yAxis.getPixelForValue(upper);
    const xLeft = chart.chartArea.left;
    const xRight= chart.chartArea.right;

    const p = _palette();
    ctx.save();
    ctx.fillStyle = p.greenBg + "55";
    ctx.fillRect(xLeft, yHigh, xRight - xLeft, yLow - yHigh);
    ctx.restore();
  }

  updateOfferSparkline(playerOffer, opponentOffer, turn) {
    if (!this.offerSparkline) return;

    const chart = this.offerSparkline;
    const label = `T${turn != null ? turn : chart.data.labels.length + 1}`;

    chart.data.labels.push(label);
    if (playerOffer != null)   chart.data.datasets[0].data.push(playerOffer);
    if (opponentOffer != null) chart.data.datasets[1].data.push(opponentOffer);

    // Trim
    if (chart.data.labels.length > 20) {
      chart.data.labels.shift();
      chart.data.datasets.forEach(ds => { if (ds.data.length > 20) ds.data.shift(); });
    }

    chart.update();
    if (DEBUG) console.log("[ParlayCharts] updateOfferSparkline", { playerOffer, opponentOffer, turn });
  }

  // ── ToM Belief Chart (right panel) ────────────────────────
  initBeliefChart(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    _applyDefaults();
    const p = _palette();

    if (this.beliefChart) {
      this.beliefChart.destroy();
      this.beliefChart = null;
    }

    this._beliefHistory = [];

    this.beliefChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Cooperative",
            data: [],
            borderColor: p.green,
            borderWidth: 1.5,
            fill: false,
            tension: 0.4,
            pointRadius: 0,
          },
          {
            label: "Competitive",
            data: [],
            borderColor: p.red,
            borderWidth: 1.5,
            fill: false,
            tension: 0.4,
            pointRadius: 0,
          },
          {
            label: "Flexible",
            data: [],
            borderColor: p.blue,
            borderWidth: 1.5,
            fill: false,
            tension: 0.4,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 150 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "bottom",
            labels: { padding: 6, boxWidth: 8, font: { size: 9 } },
          },
          title: {
            display: true,
            text: "Belief Confidence Over Time",
            font: { size: 11, weight: "600" },
            color: p.ink,
            padding: { bottom: 6 },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { display: false },
          },
          y: {
            min: 0, max: 1,
            grid: { color: p.border },
            ticks: {
              color: p.ink3,
              font: { size: 9 },
              callback: (v) => `${(v * 100).toFixed(0)}%`,
            },
          },
        },
      },
    });

    if (DEBUG) console.log("[ParlayCharts] initBeliefChart done");
    return this.beliefChart;
  }

  updateBeliefChart(beliefState) {
    if (!this.beliefChart || !beliefState) return;

    const chart = this.beliefChart;
    const turn  = chart.data.labels.length + 1;
    chart.data.labels.push(`T${turn}`);
    chart.data.datasets[0].data.push(beliefState.cooperative  ?? 0);
    chart.data.datasets[1].data.push(beliefState.competitive  ?? 0);
    chart.data.datasets[2].data.push(beliefState.flexibility  ?? 0);

    if (chart.data.labels.length > 15) {
      chart.data.labels.shift();
      chart.data.datasets.forEach(ds => ds.data.shift());
    }

    chart.update("none");
    if (DEBUG) console.log("[ParlayCharts] updateBeliefChart", beliefState);
  }

  // ── Efficiency Radar (optional, training page) ─────────────
  initEfficiencyRadar(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    _applyDefaults();
    const p = _palette();

    const labels = ["Deal Rate", "ZOPA Eff.", "ToM Acc.", "Drift Adapt.", "CP Usage"];
    const baseD  = data.base  || [0.34, 0.44, 0.31, 0.28, 0.42];
    const grpoD  = data.grpo  || [0.74, 0.81, 0.69, 0.72, 0.78];

    return new Chart(canvas, {
      type: "radar",
      data: {
        labels,
        datasets: [
          {
            label: "Base",
            data: baseD,
            borderColor: p.border,
            backgroundColor: p.surface2 + "88",
            borderWidth: 1.5,
            pointRadius: 3,
          },
          {
            label: "GRPO",
            data: grpoD,
            borderColor: p.green,
            backgroundColor: p.greenBg + "88",
            borderWidth: 2,
            pointRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" },
          title: {
            display: true,
            text: "Efficiency Radar — Base vs GRPO",
            font: { size: 13, weight: "600" },
            color: p.ink,
          },
        },
        scales: {
          r: {
            min: 0, max: 1,
            ticks: {
              display: false,
              stepSize: 0.25,
            },
            pointLabels: { color: p.ink2, font: { size: 11 } },
            grid: { color: p.border },
            angleLines: { color: p.border },
          },
        },
      },
    });
  }

  // ── Destroy all charts ─────────────────────────────────────
  destroyAll() {
    [this.rewardChart, this.efficiencyChart, this.offerSparkline, this.beliefChart]
      .forEach(c => { if (c) c.destroy(); });
    this.rewardChart     = null;
    this.efficiencyChart = null;
    this.offerSparkline  = null;
    this.beliefChart     = null;
  }
}

window.ParlayCharts = ParlayCharts;
