// ============================================================
// Parlay Game Logic — app.js
// Connects frontend to FastAPI at /api/*
// Zero dependencies beyond native fetch / DOM APIs
// ============================================================

const DEBUG = false;
const API_BASE = "";  // same-origin

// ── State ─────────────────────────────────────────────────────
let gameState = {
  sessionId:   null,
  persona:     null,
  scenario:    null,
  observation: null,
  hand:        [],
  done:        false,
  playerName:  "Player",
  turnCount:   0,
  cp:          100,
  maxCp:       100,
};

let character   = null;   // NegotiatorCharacter instance
let charts      = null;   // ParlayCharts instance
let _driftTimer = null;

// ── Init ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  charts = new ParlayCharts();

  _initDarkMode();
  _initDarkModeToggle();
  loadScenarios();
  loadPersonas();

  // Landing modal
  const startBtn = document.getElementById("btn-start-game");
  if (startBtn) {
    startBtn.addEventListener("click", _handleModalStart);
  }

  // Input area
  const submitBtn = document.getElementById("btn-submit");
  if (submitBtn) submitBtn.addEventListener("click", submitMove);

  const acceptBtn = document.getElementById("btn-accept");
  if (acceptBtn) acceptBtn.addEventListener("click", acceptDeal);

  const walkBtn = document.getElementById("btn-walk");
  if (walkBtn) walkBtn.addEventListener("click", walkAway);

  const dismissBtn = document.getElementById("btn-dismiss-drift");
  if (dismissBtn) dismissBtn.addEventListener("click", dismissDriftAlert);

  // Offer input — Enter key submits
  const offerInput = document.getElementById("offer-input");
  if (offerInput) {
    offerInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") submitMove();
    });
  }

  // Leaderboard refresh
  loadLeaderboard();
});

// ── Dark Mode ──────────────────────────────────────────────────
function _initDarkMode() {
  const saved = localStorage.getItem("parlay-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = saved || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
}

function _initDarkModeToggle() {
  const toggle = document.getElementById("dark-toggle");
  if (!toggle) return;
  toggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next    = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("parlay-theme", next);
    if (DEBUG) console.log("[app] theme set to", next);
  });
}

// ── Modal ──────────────────────────────────────────────────────
function _handleModalStart() {
  const nameInput = document.getElementById("player-name-input");
  const name = nameInput ? nameInput.value.trim() : "Player";

  // Gather selection
  const selectedScenario = document.querySelector(".selector-card.selected");
  const selectedPersona  = document.querySelector(".persona-option.selected");

  if (!selectedScenario) {
    _showInlineError("modal-error", "Please select a scenario.");
    return;
  }
  if (!selectedPersona) {
    _showInlineError("modal-error", "Please choose a negotiation style.");
    return;
  }

  const scenarioId = selectedScenario.dataset.scenarioId;
  const persona    = selectedPersona.dataset.persona;

  startGame(scenarioId, persona, name || "Player");
}

function _showInlineError(containerId, msg) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.textContent = msg;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 3000);
}

function _closeModal() {
  const backdrop = document.getElementById("setup-modal");
  if (backdrop) backdrop.classList.add("hidden");
}

// ── API Calls ─────────────────────────────────────────────────
async function startGame(scenarioId, persona, playerName) {
  setLoading(true);
  gameState.persona    = persona;
  gameState.scenario   = scenarioId;
  gameState.playerName = playerName;
  gameState.done       = false;
  gameState.turnCount  = 0;

  try {
    const res = await fetch(`${API_BASE}/api/game/start`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenarioId, persona, player_name: playerName }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    gameState.sessionId  = data.session_id;
    gameState.observation = data.observation;
    gameState.hand        = data.hand || [];
    gameState.cp          = data.cp   ?? 100;
    gameState.maxCp       = data.max_cp ?? 100;

    _closeModal();
    updateUI(data);
    _initCharacter(persona);
    _initSparkline(data.observation);

    // First system message
    addMessage("system", `Game started. You are negotiating as the ${_personaLabel(persona)}.`);
    if (data.opening_message) {
      addMessage("opponent", data.opening_message, data.observation?.opponent_offer, null);
    }

    if (DEBUG) console.log("[app] game started", data);
  } catch (e) {
    _showError("Failed to start game: " + e.message);
    if (DEBUG) console.log("[app] startGame error", e);
  } finally {
    setLoading(false);
  }
}

async function submitMove() {
  if (gameState.done) return;
  if (!gameState.sessionId) return;

  const offerInput  = document.getElementById("offer-input");
  const moveSelect  = document.getElementById("move-select");
  const cardInput   = document.querySelector(".tactical-card.selected");

  const offerRaw  = offerInput ? offerInput.value.trim() : "";
  const move      = moveSelect ? moveSelect.value : "counter";
  const cardId    = cardInput ? cardInput.dataset.cardId : null;
  const offer     = offerRaw ? parseFloat(offerRaw.replace(/[$,]/g, "")) : null;

  if (offer === null && move === "counter") {
    offerInput && offerInput.focus();
    return;
  }
  if (isNaN(offer) && offer !== null) return;

  // Render player message immediately
  addMessage("player", _moveSummary(move, offer), offer, move);

  // Show thinking indicator
  const thinkId = _showThinkingBubble();

  setLoading(true);
  try {
    const body = { session_id: gameState.sessionId, move, offer_amount: offer, card_id: cardId };
    const res  = await fetch(`${API_BASE}/api/game/step`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    gameState.observation = data.observation;
    gameState.done        = data.done ?? false;
    gameState.hand        = data.hand || gameState.hand;
    gameState.cp          = data.cp   ?? gameState.cp;
    gameState.turnCount  += 1;

    _removeThinkingBubble(thinkId);
    updateUI(data);

    if (data.opponent_message) {
      addMessage("opponent", data.opponent_message, data.observation?.opponent_offer, data.opponent_move);
    }

    if (gameState.done) {
      _handleGameOver(data);
    }

    // Clear input
    if (offerInput) offerInput.value = "";
    if (cardInput)  cardInput.classList.remove("selected");

    if (DEBUG) console.log("[app] step result", data);
  } catch (e) {
    _removeThinkingBubble(thinkId);
    _showError("Move failed: " + e.message);
    if (DEBUG) console.log("[app] submitMove error", e);
  } finally {
    setLoading(false);
  }
}

async function acceptDeal() {
  if (gameState.done || !gameState.sessionId) return;
  const offerInput = document.getElementById("offer-input");
  const offer      = offerInput ? parseFloat(offerInput.value.replace(/[$,]/g, "")) : null;

  addMessage("player", "I accept the deal.", offer, "accept");
  const thinkId = _showThinkingBubble();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/api/game/step`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: gameState.sessionId, move: "accept", offer_amount: offer }),
    });

    const data = await res.json();
    _removeThinkingBubble(thinkId);
    gameState.done = true;
    updateUI(data);
    if (data.opponent_message) addMessage("opponent", data.opponent_message, null, "accept");
    _handleGameOver(data);
    if (DEBUG) console.log("[app] acceptDeal", data);
  } catch (e) {
    _removeThinkingBubble(thinkId);
    _showError(e.message);
  } finally {
    setLoading(false);
  }
}

async function walkAway() {
  if (gameState.done || !gameState.sessionId) return;
  addMessage("player", "I'm walking away from the table.", null, "walk");
  const thinkId = _showThinkingBubble();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/api/game/step`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: gameState.sessionId, move: "walk_away" }),
    });

    const data = await res.json();
    _removeThinkingBubble(thinkId);
    gameState.done = true;
    updateUI(data);
    if (data.opponent_message) addMessage("opponent", data.opponent_message, null, "walk");
    _handleGameOver(data);
    if (DEBUG) console.log("[app] walkAway", data);
  } catch (e) {
    _removeThinkingBubble(thinkId);
    _showError(e.message);
  } finally {
    setLoading(false);
  }
}

async function loadScenarios() {
  try {
    const res = await fetch(`${API_BASE}/api/scenarios`);
    if (!res.ok) return;
    const data = await res.json();
    const scenarios = data.scenarios || data || [];
    _renderScenarioOptions(scenarios);
    if (DEBUG) console.log("[app] scenarios loaded", scenarios);
  } catch (e) {
    if (DEBUG) console.log("[app] loadScenarios error", e);
  }
}

async function loadPersonas() {
  try {
    const res = await fetch(`${API_BASE}/api/personas`);
    if (!res.ok) return;
    const data = await res.json();
    const personas = data.personas || data || [];
    _renderPersonaOptions(personas);
    if (DEBUG) console.log("[app] personas loaded", personas);
  } catch (e) {
    if (DEBUG) console.log("[app] loadPersonas error", e);
  }
}

async function loadLeaderboard() {
  try {
    const res = await fetch(`${API_BASE}/api/leaderboard?limit=5`);
    if (!res.ok) return;
    const data = await res.json();
    const entries = data.entries || data || [];
    renderLeaderboard(entries);
    if (DEBUG) console.log("[app] leaderboard loaded", entries);
  } catch (e) {
    if (DEBUG) console.log("[app] loadLeaderboard error", e);
  }
}

// ── UI Updates ────────────────────────────────────────────────
function updateUI(response) {
  const obs = response.observation || gameState.observation;

  if (obs) {
    updateZOPABar(obs);
    updateBeliefBars(obs.belief_state || obs.beliefState);
    updateCharacterState(obs);
  }

  const tension = obs?.tension_score ?? obs?.tension ?? 0;
  updateTensionMeter(tension);
  updateCPBar(gameState.cp);

  const drift = response.drift_event || obs?.drift_event;
  if (drift) showDriftAlert(drift);

  const act = obs?.current_act ?? obs?.act ?? 1;
  _updateActPills(act);

  if (gameState.hand && gameState.hand.length > 0) {
    renderHand(gameState.hand);
  }

  // Sparkline update
  if (charts && charts.offerSparkline && obs) {
    const playerOffer   = obs.player_offer   ?? obs.your_offer   ?? null;
    const opponentOffer = obs.opponent_offer ?? null;
    if (playerOffer !== null || opponentOffer !== null) {
      charts.updateOfferSparkline(playerOffer, opponentOffer, gameState.turnCount);
    }
  }

  // ToM belief chart update
  if (charts && charts.beliefChart && obs?.belief_state) {
    charts.updateBeliefChart(obs.belief_state);
  }

  // Achievements
  const achievements = response.achievements || obs?.achievements;
  if (achievements) showAchievements(achievements);

  // Update player avatar initial
  const avatarEl = document.getElementById("player-avatar");
  if (avatarEl && gameState.playerName) {
    avatarEl.textContent = gameState.playerName.charAt(0).toUpperCase();
  }

  const nameEl = document.getElementById("player-name-display");
  if (nameEl) nameEl.textContent = gameState.playerName;

  // Disable inputs when done
  _setInputsDisabled(gameState.done);
}

// ── Message Bubbles ────────────────────────────────────────────
function addMessage(role, text, offer, move) {
  const thread = document.getElementById("chat-thread");
  if (!thread) return;

  // Remove thinking bubble if any
  const existing = thread.querySelector(".thinking-bubble");
  if (existing) existing.remove();

  if (role === "system") {
    const sys = document.createElement("div");
    sys.className = "message-system";
    sys.textContent = text;
    thread.appendChild(sys);
    _scrollThread(thread);
    return;
  }

  const bubble = document.createElement("div");
  bubble.className = `message-bubble ${role}`;

  // Meta row
  const meta = document.createElement("div");
  meta.className = "bubble-meta";

  const nameSpan = document.createElement("span");
  nameSpan.textContent = role === "player" ? gameState.playerName : "Opponent";
  meta.appendChild(nameSpan);

  if (move) {
    const pill = document.createElement("span");
    pill.className = `move-pill ${move}`;
    pill.textContent = move.replace("_", " ");
    meta.appendChild(pill);
  }

  // Body
  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = text;

  // Offer chip
  if (offer != null && !isNaN(offer)) {
    const chip = document.createElement("div");
    chip.className = "offer-chip";
    chip.textContent = formatCurrency(offer, "USD");
    body.appendChild(chip);
  }

  bubble.appendChild(meta);
  bubble.appendChild(body);
  thread.appendChild(bubble);
  _scrollThread(thread);
}

function _showThinkingBubble() {
  const thread = document.getElementById("chat-thread");
  if (!thread) return null;

  const id = "thinking-" + Date.now();
  const wrap = document.createElement("div");
  wrap.className = "thinking-bubble";
  wrap.id = id;

  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("div");
    dot.className = "thinking-dot";
    wrap.appendChild(dot);
  }

  thread.appendChild(wrap);
  _scrollThread(thread);
  return id;
}

function _removeThinkingBubble(id) {
  if (!id) return;
  const el = document.getElementById(id);
  if (el) el.remove();
}

function _scrollThread(thread) {
  requestAnimationFrame(() => {
    thread.scrollTop = thread.scrollHeight;
  });
}

// ── ZOPA Bar ──────────────────────────────────────────────────
function updateZOPABar(observation) {
  const track = document.getElementById("zopa-track");
  if (!track) return;

  const batnaPlayer   = observation.player_batna   ?? observation.your_batna   ?? 0;
  const batnaOpponent = observation.opponent_batna ?? observation.opp_batna    ?? 100;
  const currentOffer  = observation.opponent_offer ?? observation.player_offer  ?? null;
  const nash          = observation.nash_point     ?? null;

  // Determine scale
  const minVal = Math.min(batnaPlayer, batnaOpponent) * 0.9;
  const maxVal = Math.max(batnaPlayer, batnaOpponent) * 1.1;
  const range  = maxVal - minVal || 1;

  const pct = (v) => `${Math.max(0, Math.min(100, ((v - minVal) / range) * 100)).toFixed(1)}%`;

  // ZOPA zone
  const zopaZone = document.getElementById("zopa-zone");
  if (zopaZone) {
    const lo = Math.min(batnaPlayer, batnaOpponent);
    const hi = Math.max(batnaPlayer, batnaOpponent);
    zopaZone.style.left  = pct(lo);
    zopaZone.style.width = `${(((hi - lo) / range) * 100).toFixed(1)}%`;
  }

  // Player BATNA marker
  const mPlayer = document.getElementById("marker-player");
  if (mPlayer) mPlayer.style.left = pct(batnaPlayer);

  // Opponent BATNA marker
  const mOpponent = document.getElementById("marker-opponent");
  if (mOpponent) mOpponent.style.left = pct(batnaOpponent);

  // Current offer marker
  const mCurrent = document.getElementById("marker-current");
  if (mCurrent && currentOffer != null) {
    mCurrent.style.left    = pct(currentOffer);
    mCurrent.style.display = "flex";
  }

  // Nash diamond
  const nashEl = document.getElementById("nash-diamond");
  if (nashEl && nash != null) {
    nashEl.style.left    = pct(nash);
    nashEl.style.display = "block";
  }

  // Labels
  const lblLow  = document.getElementById("zopa-label-low");
  const lblHigh = document.getElementById("zopa-label-high");
  if (lblLow)  lblLow.textContent  = formatCurrency(minVal, "USD");
  if (lblHigh) lblHigh.textContent = formatCurrency(maxVal, "USD");

  if (DEBUG) console.log("[app] updateZOPABar", { batnaPlayer, batnaOpponent, currentOffer, nash });
}

// ── Tension Meter ─────────────────────────────────────────────
function updateTensionMeter(tensionScore) {
  const fill  = document.getElementById("tension-fill");
  const value = document.getElementById("tension-value");
  if (!fill) return;

  const pct = Math.max(0, Math.min(100, (tensionScore || 0) * 100));
  fill.style.width = `${pct}%`;

  let level = "low";
  if (pct >= 70)      level = "high";
  else if (pct >= 40) level = "medium";

  fill.setAttribute("data-level", level);

  if (value) {
    value.textContent = `${Math.round(pct)}%`;
    value.style.color = level === "high"   ? "var(--parlay-red)"
                      : level === "medium" ? "var(--parlay-amber)"
                      : "var(--parlay-green)";
  }
}

// ── Belief Bars ───────────────────────────────────────────────
function updateBeliefBars(beliefState) {
  if (!beliefState) return;

  const mapping = {
    "belief-cooperative": beliefState.cooperative ?? beliefState.cooperative_prob ?? 0,
    "belief-competitive": beliefState.competitive ?? beliefState.competitive_prob ?? 0,
    "belief-reservation": beliefState.reservation ?? beliefState.reservation_sensitivity ?? 0,
    "belief-flexibility": beliefState.flexibility ?? beliefState.concession_rate ?? 0,
  };

  Object.entries(mapping).forEach(([id, val]) => {
    const fill    = document.getElementById(id + "-fill");
    const pctEl   = document.getElementById(id + "-pct");
    const confEl  = document.getElementById(id + "-conf");

    if (!fill) return;
    const pct = Math.max(0, Math.min(100, val * 100));
    fill.style.width = `${pct.toFixed(1)}%`;
    if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;

    // Confidence dot
    if (confEl) {
      const conf = pct > 60 ? "high" : pct > 30 ? "medium" : "low";
      confEl.className = `belief-confidence confidence-${conf}`;
    }
  });

  // Update chart
  if (charts && charts.beliefChart) {
    charts.updateBeliefChart(beliefState);
  }

  if (DEBUG) console.log("[app] updateBeliefBars", beliefState);
}

// ── CP Bar ────────────────────────────────────────────────────
function updateCPBar(cp) {
  const fill  = document.getElementById("cp-fill");
  const value = document.getElementById("cp-value");
  if (!fill) return;

  const maxCp = gameState.maxCp || 100;
  const pct   = Math.max(0, Math.min(100, (cp / maxCp) * 100));
  fill.style.width = `${pct}%`;
  if (value) value.textContent = `${Math.round(cp)} / ${maxCp}`;
}

// ── Character State ───────────────────────────────────────────
function updateCharacterState(observation) {
  if (!character) return;

  const tension = observation?.tension_score ?? observation?.tension ?? 0;
  const drift   = observation?.drift_event;

  let state = "idle";
  if (drift)          state = "shocked";
  else if (tension > 0.7) state = "aggressive";
  else if (tension > 0.4) state = "thinking";
  else if (tension < 0.2 && gameState.turnCount > 0) state = "pleased";

  character.setState(state);
  if (DEBUG) console.log("[app] character state:", state);
}

// ── Drift Alert ───────────────────────────────────────────────
function showDriftAlert(driftEvent) {
  const bar = document.getElementById("drift-alert");
  if (!bar) return;

  const text = document.getElementById("drift-alert-text");
  if (text) {
    if (typeof driftEvent === "string") {
      text.textContent = driftEvent;
    } else {
      text.textContent = driftEvent.description || driftEvent.event || "Market conditions have shifted.";
    }
  }

  bar.classList.remove("hidden");

  // Auto-dismiss after 8 seconds
  if (_driftTimer) clearTimeout(_driftTimer);
  _driftTimer = setTimeout(dismissDriftAlert, 8000);
}

function dismissDriftAlert() {
  const bar = document.getElementById("drift-alert");
  if (bar) bar.classList.add("hidden");
  if (_driftTimer) clearTimeout(_driftTimer);
}

// ── Hand Rendering ────────────────────────────────────────────
function renderHand(hand) {
  const container = document.getElementById("hand-container");
  if (!container) return;

  container.innerHTML = "";  // safe — no user data in card definitions

  hand.forEach((card) => {
    const wrapper = document.createElement("div");
    wrapper.className = "tactical-card";
    wrapper.dataset.cardId = card.id || card.card_id || "";

    const inner = document.createElement("div");
    inner.className = "card-inner";

    // Front
    const front = document.createElement("div");
    front.className = "card-face";

    const name = document.createElement("div");
    name.className = "card-name";
    name.textContent = card.name || "Tactic";

    const type = document.createElement("div");
    type.className = "card-type";
    type.textContent = card.type || "";

    const cost = document.createElement("div");
    cost.className = "card-cost";
    cost.textContent = card.cp_cost ?? card.cost ?? "1";

    front.appendChild(name);
    front.appendChild(type);
    front.appendChild(cost);

    // Back
    const back = document.createElement("div");
    back.className = "card-back";

    const backLabel = document.createElement("div");
    backLabel.className = "card-back-label";
    backLabel.textContent = "Game Theory";

    const gt = document.createElement("div");
    gt.className = "card-game-theory";
    gt.textContent = card.game_theory_basis || card.description || "";

    back.appendChild(backLabel);
    back.appendChild(gt);

    inner.appendChild(front);
    inner.appendChild(back);
    wrapper.appendChild(inner);

    // Selection
    wrapper.addEventListener("click", () => {
      const already = wrapper.classList.contains("selected");
      container.querySelectorAll(".tactical-card").forEach(c => c.classList.remove("selected"));
      if (!already) wrapper.classList.add("selected");
    });

    container.appendChild(wrapper);
  });

  if (DEBUG) console.log("[app] renderHand", hand.length, "cards");
}

// ── Leaderboard ───────────────────────────────────────────────
function renderLeaderboard(entries) {
  const tbody = document.getElementById("leaderboard-body");
  if (!tbody) return;

  tbody.innerHTML = "";

  if (!entries || entries.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "empty-state text-muted";
    td.textContent = "No games yet";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  entries.forEach((entry, idx) => {
    const tr = document.createElement("tr");

    // Highlight current player
    if (entry.player_name === gameState.playerName) {
      tr.classList.add("highlight-player");
    }

    // Rank
    const rankTd = document.createElement("td");
    const rankSpan = document.createElement("span");
    rankSpan.className = "lb-rank" +
      (idx === 0 ? " gold" : idx === 1 ? " silver" : idx === 2 ? " bronze" : "");
    rankSpan.textContent = `#${idx + 1}`;
    rankTd.appendChild(rankSpan);
    tr.appendChild(rankTd);

    // Name
    const nameTd = document.createElement("td");
    nameTd.textContent = entry.player_name || "—";
    tr.appendChild(nameTd);

    // Score
    const scoreTd = document.createElement("td");
    scoreTd.className = "num";
    scoreTd.textContent = (entry.score ?? entry.reward ?? 0).toFixed(2);
    tr.appendChild(scoreTd);

    // Deals
    const dealsTd = document.createElement("td");
    dealsTd.className = "num";
    dealsTd.textContent = entry.deals ?? "—";
    tr.appendChild(dealsTd);

    tbody.appendChild(tr);
  });

  if (DEBUG) console.log("[app] renderLeaderboard", entries.length, "entries");
}

// ── Achievements ──────────────────────────────────────────────
function showAchievements(achievements) {
  if (!achievements || !Array.isArray(achievements)) return;

  achievements.forEach((ach) => {
    const el = document.querySelector(`.badge[data-achievement="${ach.id}"]`);
    if (el && !el.classList.contains("earned")) {
      el.classList.add("earned");
      _showToast(`Achievement unlocked: ${ach.name || ach.id}`);
    }
  });
}

function _showToast(msg) {
  const toast = document.createElement("div");
  toast.style.cssText = [
    "position:fixed", "bottom:24px", "right:24px", "z-index:9999",
    "background:var(--parlay-surface)", "border:1px solid var(--parlay-border)",
    "border-radius:8px", "padding:12px 16px",
    "font-size:0.875rem", "font-weight:500",
    "color:var(--parlay-ink)", "box-shadow:0 4px 16px rgba(0,0,0,0.12)",
    "animation:slide-down 200ms ease",
  ].join(";");
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// ── Game Over ─────────────────────────────────────────────────
function _handleGameOver(data) {
  const deal    = data.deal_reached ?? data.deal ?? false;
  const amount  = data.deal_amount  ?? data.final_offer ?? null;
  const score   = data.reward       ?? data.score       ?? 0;

  // Show result in chat
  const resultMsg = deal
    ? `Deal reached at ${formatCurrency(amount, "USD")}! Score: ${score.toFixed(2)}`
    : "No deal. You walked away.";

  addMessage("system", resultMsg);

  // Show result banner
  const banner = document.getElementById("result-banner");
  if (banner) {
    banner.className = `result-banner ${deal ? "deal" : "walk"}`;
    banner.classList.remove("hidden");

    const title = banner.querySelector(".result-title");
    if (title) title.textContent = deal ? "Deal Closed" : "Walked Away";

    const amountEl = banner.querySelector(".result-amount");
    if (amountEl && amount != null) amountEl.textContent = formatCurrency(amount, "USD");
    else if (amountEl) amountEl.textContent = "—";

    const scoreEl = banner.querySelector(".result-score");
    if (scoreEl) scoreEl.textContent = `Score: ${score.toFixed(2)}`;
  }

  // Character state
  if (character) character.setState(deal ? "pleased" : "shocked");

  // Refresh leaderboard after short delay
  setTimeout(loadLeaderboard, 1200);

  if (DEBUG) console.log("[app] game over", { deal, amount, score });
}

// ── Scenario / Persona Selectors ──────────────────────────────
function _renderScenarioOptions(scenarios) {
  const grid = document.getElementById("scenario-grid");
  if (!grid) return;

  grid.innerHTML = "";

  if (!scenarios.length) {
    // Fallback defaults so UI isn't empty
    scenarios = [
      { id: "merger",     name: "M&A Deal",         description: "Acquire a fintech startup", difficulty: "Medium" },
      { id: "salary",     name: "Salary Negotiation",description: "Land the offer you deserve", difficulty: "Easy" },
      { id: "licensing",  name: "IP Licensing",      description: "Patent licensing deal",      difficulty: "Hard" },
      { id: "supply",     name: "Supply Contract",   description: "Commodity procurement",      difficulty: "Medium" },
    ];
  }

  scenarios.forEach((s) => {
    const card = document.createElement("div");
    card.className = "selector-card";
    card.dataset.scenarioId = s.id || s.scenario_id;

    const name = document.createElement("div");
    name.className = "selector-card-name";
    name.textContent = s.name || s.title;

    const meta = document.createElement("div");
    meta.className = "selector-card-meta";
    meta.textContent = (s.description || "") + (s.difficulty ? ` · ${s.difficulty}` : "");

    card.appendChild(name);
    card.appendChild(meta);

    card.addEventListener("click", () => {
      grid.querySelectorAll(".selector-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
    });

    grid.appendChild(card);
  });
}

function _renderPersonaOptions(personas) {
  const grid = document.getElementById("persona-grid");
  if (!grid) return;

  grid.innerHTML = "";

  const defaults = [
    { id: "shark",    name: "Shark",    icon: "🦈" },
    { id: "diplomat", name: "Diplomat", icon: "🕊️" },
    { id: "analyst",  name: "Analyst",  icon: "📊" },
    { id: "wildcard", name: "Wildcard", icon: "🃏" },
    { id: "veteran",  name: "Veteran",  icon: "⚔️" },
  ];

  const list = (personas && personas.length) ? personas : defaults;

  list.forEach((p) => {
    const opt = document.createElement("div");
    opt.className = "persona-option";
    opt.dataset.persona = p.id || p.persona_id;

    const icon = document.createElement("div");
    icon.className = "persona-option-icon";
    icon.textContent = p.icon || "🃏";

    const name = document.createElement("div");
    name.className = "persona-option-name";
    name.textContent = p.name || p.id;

    opt.appendChild(icon);
    opt.appendChild(name);

    opt.addEventListener("click", () => {
      grid.querySelectorAll(".persona-option").forEach(c => c.classList.remove("selected"));
      opt.classList.add("selected");
    });

    grid.appendChild(opt);
  });
}

// ── Character init ────────────────────────────────────────────
function _initCharacter(persona) {
  if (typeof NegotiatorCharacter === "undefined") {
    if (DEBUG) console.log("[app] NegotiatorCharacter not available");
    return;
  }
  if (character) character.destroy();
  character = new NegotiatorCharacter("character-canvas", persona || "shark");
  if (DEBUG) console.log("[app] character created for persona:", persona);
}

// ── Sparkline init ────────────────────────────────────────────
function _initSparkline(observation) {
  if (!charts || !observation) return;

  const lo = observation.player_batna   ?? observation.your_batna   ?? 0;
  const hi = observation.opponent_batna ?? observation.opp_batna    ?? 0;
  const nash = observation.nash_point   ?? ((lo + hi) / 2);

  charts.initOfferSparkline("offer-sparkline", lo, hi, nash);
  charts.initBeliefChart("belief-chart");
  if (DEBUG) console.log("[app] sparkline init", { lo, hi, nash });
}

// ── Helpers ───────────────────────────────────────────────────
function formatCurrency(amount, currency) {
  if (amount == null || isNaN(amount)) return "—";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: currency || "USD",
      maximumFractionDigits: 0,
    }).format(amount);
  } catch {
    return `$${Number(amount).toLocaleString()}`;
  }
}

function setLoading(isLoading) {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) {
    overlay.classList.toggle("hidden", !isLoading);
  }

  // Disable action buttons during load
  ["btn-submit", "btn-accept", "btn-walk"].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.disabled = isLoading;
  });
}

function _setInputsDisabled(disabled) {
  const ids = ["offer-input", "move-select", "btn-submit", "btn-accept", "btn-walk"];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
  });
}

function _updateActPills(act) {
  [1, 2, 3].forEach(n => {
    const pill = document.getElementById(`act-pill-${n}`);
    if (!pill) return;
    pill.classList.remove("active", "completed");
    if (n < act)       pill.classList.add("completed");
    else if (n === act) pill.classList.add("active");
  });
}

function _moveSummary(move, offer) {
  switch (move) {
    case "anchor":    return `Anchoring at ${formatCurrency(offer, "USD")}.`;
    case "counter":   return `Counter offer: ${formatCurrency(offer, "USD")}.`;
    case "concede":   return `Concession: ${formatCurrency(offer, "USD")}.`;
    case "package":   return `Package deal offer: ${formatCurrency(offer, "USD")}.`;
    case "accept":    return "Accepting the deal.";
    case "walk_away": return "Walking away.";
    default:          return offer != null ? formatCurrency(offer, "USD") : move;
  }
}

function _personaLabel(persona) {
  const labels = {
    shark:    "Shark (aggressive closer)",
    diplomat: "Diplomat (relationship-first)",
    analyst:  "Analyst (data-driven)",
    wildcard: "Wildcard (unpredictable)",
    veteran:  "Veteran (experience-led)",
  };
  return labels[persona] || persona;
}

function _showError(msg) {
  const el = document.getElementById("global-error");
  if (!el) { _showToast(msg); return; }
  el.textContent = msg;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 4000);
}
