// ============================================================
// Parlay Game Logic — app.js
// "The Deal Room" — 3-step onboarding, mock mode, Mad Men theme.
// Zero dependencies beyond native fetch / DOM APIs.
// ============================================================

const APP_DEBUG = false; // not "DEBUG" — chart.js is also a global script
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

let character      = null;   // NegotiatorCharacter instance
let charts         = null;   // ParlayCharts instance
let _driftTimer    = null;
let _previewChars  = {};     // PersonaPreviewCharacter instances keyed by persona id

// ── Onboarding step tracking ──────────────────────────────────
let _currentStep = 1;

// ── Init ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Wire onboarding first so a Chart / fetch failure cannot block the wizard.
  _initOnboarding();
  _syncOnboardingInert();

  try {
    if (typeof ParlayCharts !== "undefined") {
      charts = new ParlayCharts();
    }
  } catch (e) {
    if (typeof console !== "undefined" && console.error) {
      console.error("[Parlay] ParlayCharts init failed:", e);
    }
  }

  loadScenarios();
  loadPersonas();
  _checkDemoMode();

  // Game action buttons
  const submitBtn = document.getElementById("btn-submit");
  if (submitBtn) submitBtn.addEventListener("click", submitMove);

  const acceptBtn = document.getElementById("btn-accept");
  if (acceptBtn) acceptBtn.addEventListener("click", acceptDeal);

  const walkBtn = document.getElementById("btn-walk");
  if (walkBtn) walkBtn.addEventListener("click", walkAway);

  const dismissDrift = document.getElementById("btn-dismiss-drift");
  if (dismissDrift) dismissDrift.addEventListener("click", dismissDriftAlert);

  const dismissDemo = document.getElementById("btn-dismiss-demo");
  if (dismissDemo) {
    dismissDemo.addEventListener("click", () => {
      const banner = document.getElementById("demo-banner");
      if (banner) banner.classList.add("hidden");
      document.body.classList.remove("demo-mode");
    });
  }

  const offerInput = document.getElementById("offer-input");
  if (offerInput) {
    offerInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) submitMove();
    });
  }

  const darkToggle = document.getElementById("dark-toggle");
  if (darkToggle) darkToggle.addEventListener("click", _toggleDisplayMode);

  loadLeaderboard();
});

// ── Demo mode detection ────────────────────────────────────────
async function _checkDemoMode() {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("health check failed");
    const data = await res.json();
    if (data.gemini === "mock") _showDemoBanner();
  } catch {
    // No backend reachable — still show demo banner
    _showDemoBanner();
  }
}

function _showDemoBanner() {
  const banner = document.getElementById("demo-banner");
  if (banner) banner.classList.remove("hidden");
  document.body.classList.add("demo-mode");
  if (APP_DEBUG) console.log("[app] demo mode active");
}

// ── Display mode toggle (replacing dark-mode concept) ─────────
function _toggleDisplayMode() {
  const html = document.documentElement;
  const cur  = html.getAttribute("data-theme") || "felt";
  const next = cur === "felt" ? "light" : "felt";
  html.setAttribute("data-theme", next);
  localStorage.setItem("parlay-theme", next);
}

// ── Onboarding (3-step wizard) ────────────────────────────────
function _initOnboarding() {
  // Step 1 — name
  const step1Input    = document.getElementById("step1-name");
  const step1Continue = document.getElementById("step1-continue");

  if (step1Input) {
    step1Input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") _goToStep(2);
    });
  }
  if (step1Continue) step1Continue.addEventListener("click", () => _goToStep(2));

  // Step 2 — scenario
  const step2Back     = document.getElementById("step2-back");
  const step2Continue = document.getElementById("step2-continue");
  if (step2Back)     step2Back.addEventListener("click",     () => _goToStep(1));
  if (step2Continue) step2Continue.addEventListener("click", () => _goToStep(3));

  // Step 3 — persona
  const step3Back  = document.getElementById("step3-back");
  const step3Start = document.getElementById("step3-start");
  if (step3Back)  step3Back.addEventListener("click",  () => _goToStep(2));
  if (step3Start) step3Start.addEventListener("click", _handleStep3Start);
}

/**
 * Inert hides non-active steps from the accessibility tree and blocks interaction
 * in modern browsers, complementing z-index and pointer-events.
 */
function _syncOnboardingInert() {
  for (let n = 1; n <= 3; n += 1) {
    const el = document.getElementById(`onboarding-step-${n}`);
    if (!el) continue;
    if (n === _currentStep) {
      el.removeAttribute("inert");
    } else {
      el.setAttribute("inert", "");
    }
  }
}

function _goToStep(step) {
  if (APP_DEBUG) console.log("[app] going to step", step);

  // Validate before advancing
  if (step === 2) {
    const name = (document.getElementById("step1-name")?.value ?? "").trim();
    if (!name) {
      _showStepError(1, "Please enter your name.");
      return;
    }
    _showStepError(1, "");
  }

  if (step === 3) {
    const sel = document.querySelector(".scenario-dossier.selected");
    if (!sel) {
      _showStepError(2, "Please select a scenario.");
      return;
    }
    _showStepError(2, "");
  }

  // Exit current step
  const currentEl = document.getElementById(`onboarding-step-${_currentStep}`);
  if (currentEl) {
    currentEl.classList.remove("active", "start-active");
    currentEl.classList.add("exiting");
    setTimeout(() => {
      currentEl.classList.remove("exiting");
    }, 300);
  }

  _currentStep = step;
  _syncOnboardingInert();

  const nextEl = document.getElementById(`onboarding-step-${step}`);
  if (nextEl) {
    nextEl.classList.remove("exiting");
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        nextEl.classList.add("active");
      });
    });
  }
}

function _showStepError(step, msg) {
  const el = document.getElementById(`step${step}-error`);
  if (!el) return;
  el.textContent = msg;
  if (msg) {
    setTimeout(() => { el.textContent = ""; }, 3000);
  }
}

function _closeOnboarding() {
  [1, 2, 3].forEach(n => {
    const el = document.getElementById(`onboarding-step-${n}`);
    if (el) {
      el.classList.remove("active", "start-active", "exiting");
      el.removeAttribute("inert");
      el.style.display = "none";
    }
  });
}

function _handleStep3Start() {
  const nameInput = document.getElementById("step1-name");
  const selScenario = document.querySelector(".scenario-dossier.selected");
  const selPersona  = document.querySelector(".persona-card-option.selected");

  if (!selPersona) {
    _showStepError(3, "Please choose an opponent.");
    return;
  }

  const name       = nameInput?.value.trim() || "Player";
  const scenarioId = selScenario?.dataset.scenarioId || "saas_enterprise";
  const persona    = selPersona.dataset.persona;

  startGame(scenarioId, persona, name);
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

    let data;
    if (!res.ok) {
      // If backend is down, load with mock data
      if (APP_DEBUG) console.log("[app] backend down — using mock data");
      data = _mockStartData(scenarioId, persona, playerName);
      _showDemoBanner();
    } else {
      data = await res.json();
    }

    gameState.sessionId   = data.session_id;
    gameState.observation = data.observation;
    gameState.hand        = data.hand || [];
    gameState.cp          = data.cp   ?? data.observation?.credibility_points ?? 100;
    gameState.maxCp       = data.max_cp ?? 100;

    _closeOnboarding();
    _destroyPreviewChars();
    updateUI(data);
    _initCharacter(persona);
    _initSparkline(data.observation);
    _updateScenarioHeader(data);
    _updatePersonaPanel(data);

    addMessage("system", `Game started. You are negotiating as the ${_personaLabel(persona)}.`);
    const opener = data.opening_message || data.persona?.opening_line;
    if (opener) {
      addMessage("opponent", opener, data.observation?.opponent_offer ?? null, null);
    }

    if (APP_DEBUG) console.log("[app] game started", data);
  } catch (e) {
    // Backend completely unreachable — use mock data
    if (APP_DEBUG) console.log("[app] startGame error, falling back to mock:", e);
    const data = _mockStartData(scenarioId, persona, playerName);
    gameState.sessionId   = data.session_id;
    gameState.observation = data.observation;
    gameState.hand        = data.hand;
    gameState.cp          = 100;

    _closeOnboarding();
    _destroyPreviewChars();
    updateUI(data);
    _initCharacter(persona);
    _showDemoBanner();
    addMessage("system", `Demo mode: running mock game for ${_personaLabel(persona)}.`);
  } finally {
    setLoading(false);
  }
}

function _mockStartData(scenarioId, persona, playerName) {
  const mockScenarios = {
    saas_enterprise:        { title: "Enterprise SaaS Contract",   lower: 125000, upper: 165000 },
    consulting_retainer:    { title: "Consulting Retainer",        lower: 25000,  upper: 40000  },
    hiring_package:         { title: "Senior Engineer Offer",      lower: 195000, upper: 230000 },
    vendor_hardware:        { title: "Hardware Vendor Contract",   lower: 1750000,upper: 2200000},
    acquisition_term_sheet: { title: "Startup Acquisition",        lower: 10500000,upper:16000000},
  };
  const s = mockScenarios[scenarioId] || mockScenarios.saas_enterprise;
  const nash = (s.lower + s.upper) / 2;
  const sid  = "mock-" + Math.random().toString(36).slice(2);

  return {
    session_id: sid,
    scenario: { id: scenarioId, title: s.title },
    observation: {
      step_count: 0, zopa_lower: s.lower, zopa_upper: s.upper,
      nash_point: nash, tension_score: 10, credibility_points: 100, act: 1,
      belief_state: { cooperative: 0.5, competitive: 0.3, reservation: 0.4, flexibility: 0.6 },
    },
    persona: { id: persona, name: _personaLabel(persona), symbol: "◈", emoji: "🎯",
               opening_line: "Let's see what you've got." },
    hand: [],
    opening_message: "Welcome to the room. Let's negotiate.",
    cp: 100,
    max_cp: 100,
  };
}

async function submitMove() {
  if (gameState.done) return;
  if (!gameState.sessionId) return;

  const offerInput = document.getElementById("offer-input");
  const moveSelect = document.getElementById("move-select");
  const cardEl     = document.querySelector(".tactical-card.selected");

  const offerRaw = offerInput?.value.trim() ?? "";
  const move     = moveSelect?.value ?? "counter";
  const cardId   = cardEl?.dataset.cardId ?? null;
  const offer    = offerRaw ? parseFloat(offerRaw.replace(/[$,]/g, "")) : null;

  if (offer === null && move === "counter") {
    offerInput && offerInput.focus();
    return;
  }
  if (offer !== null && isNaN(offer)) return;

  addMessage("player", _moveSummary(move, offer), offer, move);
  const thinkId = _showThinkingBubble();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/api/game/step`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: gameState.sessionId,
        move,
        offer_amount: offer,
        card_id: cardId,
      }),
    });

    const data = res.ok ? await res.json() : _mockStepData(offer, move);

    gameState.observation = data.observation;
    gameState.done        = data.done ?? false;
    gameState.hand        = data.hand || gameState.hand;
    gameState.cp          = data.cp   ?? data.observation?.credibility_points ?? gameState.cp;
    gameState.turnCount  += 1;

    _removeThinkingBubble(thinkId);
    updateUI(data);

    // Opponent message from /api/game/step unified endpoint response format
    const oppMsg  = data.opponent_message ?? data.opponent?.utterance;
    const oppOffer = data.observation?.opponent_offer ?? data.opponent?.offer;
    const oppMove  = data.opponent_move ?? data.opponent?.tactical_move;
    if (oppMsg) {
      const bubble = addMessage("opponent", oppMsg, oppOffer, oppMove);
      if (bubble && gameState.persona) {
        bubble.setAttribute("data-persona", gameState.persona);
      }
    }

    if (gameState.done) _handleGameOver(data);

    if (offerInput) offerInput.value = "";
    if (cardEl) cardEl.classList.remove("selected");

    if (APP_DEBUG) console.log("[app] step result", data);
  } catch (e) {
    _removeThinkingBubble(thinkId);
    _showError("Move failed: " + e.message);
    if (APP_DEBUG) console.log("[app] submitMove error", e);
  } finally {
    setLoading(false);
  }
}

function _mockStepData(offer, move) {
  gameState.turnCount += 1;
  const obs = gameState.observation || {};
  const tension = Math.min(100, (obs.tension_score || 10) + 5);
  return {
    observation: { ...obs, tension_score: tension, step_count: (obs.step_count || 0) + 1 },
    opponent_message: "That's an interesting position. Let me consider it.",
    opponent: { utterance: "That's an interesting position. Let me consider it.", offer: null },
    done: false,
  };
}

async function acceptDeal() {
  if (gameState.done || !gameState.sessionId) return;
  const offerInput = document.getElementById("offer-input");
  const offer = offerInput ? parseFloat(offerInput.value.replace(/[$,]/g, "")) : null;

  addMessage("player", "I accept the deal.", offer, "accept");
  const thinkId = _showThinkingBubble();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/api/game/accept`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: gameState.sessionId }),
    });

    const data = res.ok ? await res.json() : { deal_reached: true, deal_amount: offer, reward: 0 };
    _removeThinkingBubble(thinkId);
    gameState.done = true;
    updateUI(data);
    addMessage("system", "Deal accepted.");
    _handleGameOver({ ...data, deal_reached: true, deal_amount: offer ?? data.final_price });
    if (APP_DEBUG) console.log("[app] acceptDeal", data);
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
    const res = await fetch(`${API_BASE}/api/game/walkaway`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: gameState.sessionId }),
    });

    const data = res.ok ? await res.json() : { result: "walk_away" };
    _removeThinkingBubble(thinkId);
    gameState.done = true;
    updateUI(data);
    _handleGameOver({ deal_reached: false, reward: 0 });
    if (APP_DEBUG) console.log("[app] walkAway", data);
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
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const scenarios = data.scenarios || data || [];
    _renderScenarioDossiers(scenarios);
    if (APP_DEBUG) console.log("[app] scenarios loaded", scenarios.length);
  } catch (e) {
    // Render defaults so onboarding UI isn't empty
    _renderScenarioDossiers([]);
    if (APP_DEBUG) console.log("[app] loadScenarios error", e);
  }
}

async function loadPersonas() {
  try {
    const res = await fetch(`${API_BASE}/api/personas`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const personas = data.personas || data || [];
    _renderPersonaCards(personas);
    if (APP_DEBUG) console.log("[app] personas loaded", personas.length);
  } catch (e) {
    _renderPersonaCards([]);
    if (APP_DEBUG) console.log("[app] loadPersonas error", e);
  }
}

async function loadLeaderboard() {
  try {
    const res = await fetch(`${API_BASE}/api/leaderboard?limit=5`);
    if (!res.ok) return;
    const data = await res.json();
    renderLeaderboard(data.entries || data || []);
    if (APP_DEBUG) console.log("[app] leaderboard loaded");
  } catch (e) {
    if (APP_DEBUG) console.log("[app] loadLeaderboard error", e);
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
  if (charts && obs) {
    const playerOffer   = obs.player_offer   ?? obs.your_offer   ?? null;
    const opponentOffer = obs.opponent_offer ?? null;
    if (playerOffer !== null || opponentOffer !== null) {
      charts.updateOfferSparkline && charts.updateOfferSparkline(playerOffer, opponentOffer, gameState.turnCount);
    }
    if (obs.belief_state) {
      charts.updateBeliefChart && charts.updateBeliefChart(obs.belief_state);
    }
  }

  const achievements = response.achievements || obs?.achievements;
  if (achievements) showAchievements(achievements);

  const avatarEl = document.getElementById("player-avatar");
  if (avatarEl && gameState.playerName) {
    avatarEl.textContent = gameState.playerName.charAt(0).toUpperCase();
  }
  const nameEl = document.getElementById("player-name-display");
  if (nameEl) nameEl.textContent = gameState.playerName;

  _setInputsDisabled(gameState.done);
}

function _updateScenarioHeader(data) {
  const titleEl = document.getElementById("scenario-title");
  const metaEl  = document.getElementById("scenario-meta");
  const sc = data.scenario || {};
  if (titleEl) titleEl.textContent = sc.title || data.scenario_id || "Negotiation";
  if (metaEl)  metaEl.textContent  = sc.description || "";
}

function _updatePersonaPanel(data) {
  const p = data.persona || {};
  const nameEl = document.getElementById("persona-name");
  const descEl = document.getElementById("persona-desc");
  const avatEl = document.getElementById("persona-avatar");
  if (nameEl) nameEl.textContent = p.name || _personaLabel(gameState.persona);
  if (descEl) descEl.textContent = p.style || "";
  if (avatEl) avatEl.textContent = p.symbol || p.emoji || "◈";
}

// ── Message Bubbles ────────────────────────────────────────────
function addMessage(role, text, offer, move) {
  const thread = document.getElementById("chat-thread");
  if (!thread) return null;

  const existing = thread.querySelector(".thinking-bubble");
  if (existing) existing.remove();

  if (role === "system") {
    const sys = document.createElement("div");
    sys.className = "message-system";
    sys.textContent = text;
    thread.appendChild(sys);
    _scrollThread(thread);
    return null;
  }

  const bubble = document.createElement("div");
  bubble.className = `message-bubble ${role}`;
  if (role === "opponent" && gameState.persona) {
    bubble.setAttribute("data-persona", gameState.persona);
  }

  const meta = document.createElement("div");
  meta.className = "bubble-meta";

  const nameSpan = document.createElement("span");
  nameSpan.textContent = role === "player" ? gameState.playerName : "Opponent";
  meta.appendChild(nameSpan);

  if (move) {
    const pill = document.createElement("span");
    pill.className = `move-pill ${move}`;
    pill.textContent = move.replace(/_/g, " ");
    meta.appendChild(pill);
  }

  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = text;

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
  return bubble;
}

function _showThinkingBubble() {
  const thread = document.getElementById("chat-thread");
  if (!thread) return null;

  const id   = "thinking-" + Date.now();
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
  requestAnimationFrame(() => { thread.scrollTop = thread.scrollHeight; });
}

// ── ZOPA Bar ──────────────────────────────────────────────────
function updateZOPABar(observation) {
  const track = document.getElementById("zopa-track");
  if (!track) return;

  const batnaPlayer   = observation.player_batna   ?? observation.your_batna   ?? observation.zopa_lower  ?? 0;
  const batnaOpponent = observation.opponent_batna ?? observation.opp_batna    ?? observation.zopa_upper  ?? 100;
  const currentOffer  = observation.opponent_offer ?? observation.player_offer  ?? null;
  const nash          = observation.nash_point     ?? null;

  const minVal = Math.min(batnaPlayer, batnaOpponent) * 0.9;
  const maxVal = Math.max(batnaPlayer, batnaOpponent) * 1.1;
  const range  = maxVal - minVal || 1;
  const pct    = (v) => `${Math.max(0, Math.min(100, ((v - minVal) / range) * 100)).toFixed(1)}%`;

  const zopaZone = document.getElementById("zopa-zone");
  if (zopaZone) {
    const lo = Math.min(batnaPlayer, batnaOpponent);
    const hi = Math.max(batnaPlayer, batnaOpponent);
    zopaZone.style.left  = pct(lo);
    zopaZone.style.width = `${(((hi - lo) / range) * 100).toFixed(1)}%`;
  }

  const mPlayer = document.getElementById("marker-player");
  if (mPlayer) mPlayer.style.left = pct(batnaPlayer);

  const mOpponent = document.getElementById("marker-opponent");
  if (mOpponent) mOpponent.style.left = pct(batnaOpponent);

  const mCurrent = document.getElementById("marker-current");
  if (mCurrent && currentOffer != null) {
    mCurrent.style.left    = pct(currentOffer);
    mCurrent.style.display = "flex";
  }

  const nashEl = document.getElementById("nash-diamond");
  if (nashEl && nash != null) {
    nashEl.style.left    = pct(nash);
    nashEl.style.display = "block";
  }

  const lblLow  = document.getElementById("zopa-label-low");
  const lblHigh = document.getElementById("zopa-label-high");
  if (lblLow)  lblLow.textContent  = formatCurrency(minVal, "USD");
  if (lblHigh) lblHigh.textContent = formatCurrency(maxVal, "USD");
}

// ── Tension Meter ─────────────────────────────────────────────
function updateTensionMeter(tensionScore) {
  const fill  = document.getElementById("tension-fill");
  const value = document.getElementById("tension-value");
  if (!fill) return;

  // tensionScore arrives as 0–100 from the server
  const pct = Math.max(0, Math.min(100, tensionScore || 0));
  fill.style.width = `${pct}%`;

  let level = "low";
  if (pct >= 75)      level = "high";
  else if (pct >= 50) level = "medium";

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
    const fill  = document.getElementById(id + "-fill");
    const pctEl = document.getElementById(id + "-pct");
    const confEl= document.getElementById(id + "-conf");
    if (!fill) return;
    const pct = Math.max(0, Math.min(100, val * 100));
    fill.style.width = `${pct.toFixed(1)}%`;
    if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
    if (confEl) {
      const conf = pct > 60 ? "high" : pct > 30 ? "medium" : "low";
      confEl.className = `belief-confidence confidence-${conf}`;
    }
  });

  if (charts && charts.updateBeliefChart) charts.updateBeliefChart(beliefState);
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
  else if (tension > 75) state = "aggressive";
  else if (tension > 50) state = "thinking";
  else if (tension < 25 && gameState.turnCount > 0) state = "pleased";

  character.setState(state);

  // After drift: revert to aggressive after 2s
  if (drift && character) {
    setTimeout(() => {
      if (character && gameState.sessionId) character.setState("aggressive");
    }, 2000);
  }

  const label = document.getElementById("character-state-label");
  if (label) label.textContent = state;

  if (APP_DEBUG) console.log("[app] character state:", state, "tension:", tension);
}

// ── Drift Alert ───────────────────────────────────────────────
function showDriftAlert(driftEvent) {
  const bar = document.getElementById("drift-alert");
  if (!bar) return;

  const text = document.getElementById("drift-alert-text");
  if (text) {
    text.textContent = typeof driftEvent === "string"
      ? driftEvent
      : (driftEvent.description || driftEvent.event || "Market conditions have shifted.");
  }

  bar.classList.remove("hidden");

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

  container.innerHTML = "";  // safe — card definitions contain no user data

  if (!hand || !hand.length) return;

  hand.forEach((card) => {
    const wrapper = document.createElement("div");
    wrapper.className = "tactical-card";
    wrapper.dataset.cardId = card.id || card.card_id || card.move || "";

    const inner = document.createElement("div");
    inner.className = "card-inner";

    const front = document.createElement("div");
    front.className = "card-face";

    const name = document.createElement("div");
    name.className = "card-name";
    name.textContent = card.name || card.move || "Tactic";
    front.appendChild(name);

    const type = document.createElement("div");
    type.className = "card-type";
    type.textContent = card.type || card.move || "";
    front.appendChild(type);

    const cost = document.createElement("div");
    cost.className = "card-cost";
    cost.textContent = `${card.cp_cost ?? card.cost ?? "—"} CP`;
    front.appendChild(cost);

    const back = document.createElement("div");
    back.className = "card-back";

    const backLabel = document.createElement("div");
    backLabel.className = "card-back-label";
    backLabel.textContent = "Game Theory";
    back.appendChild(backLabel);

    const gt = document.createElement("div");
    gt.className = "card-game-theory";
    gt.textContent = card.game_theory_basis || card.description || "";
    back.appendChild(gt);

    inner.appendChild(front);
    inner.appendChild(back);
    wrapper.appendChild(inner);

    wrapper.addEventListener("click", () => {
      const already = wrapper.classList.contains("selected");
      container.querySelectorAll(".tactical-card").forEach(c => c.classList.remove("selected"));
      if (!already) wrapper.classList.add("selected");
    });

    container.appendChild(wrapper);
  });
}

// ── Leaderboard ───────────────────────────────────────────────
function renderLeaderboard(entries) {
  const tbody = document.getElementById("leaderboard-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!entries || !entries.length) {
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
    if (entry.player_name === gameState.playerName) tr.classList.add("highlight-player");

    const rankTd = document.createElement("td");
    const rankSpan = document.createElement("span");
    rankSpan.className = "lb-rank" + (idx === 0 ? " gold" : idx === 1 ? " silver" : idx === 2 ? " bronze" : "");
    rankSpan.textContent = `#${idx + 1}`;
    rankTd.appendChild(rankSpan);
    tr.appendChild(rankTd);

    const nameTd = document.createElement("td");
    nameTd.textContent = entry.player_name || "—";
    tr.appendChild(nameTd);

    const scoreTd = document.createElement("td");
    scoreTd.className = "num";
    scoreTd.textContent = (entry.score ?? entry.total_reward ?? entry.reward ?? 0).toFixed(2);
    tr.appendChild(scoreTd);

    const dealsTd = document.createElement("td");
    dealsTd.className = "num";
    dealsTd.textContent = entry.deals ?? (entry.deal_closed ? "✓" : "—");
    tr.appendChild(dealsTd);

    tbody.appendChild(tr);
  });
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
    "background:var(--mahogany)", "border:1px solid var(--gold)",
    "border-radius:6px", "padding:12px 20px",
    "font-family:var(--font-display)", "font-style:italic",
    "font-size:0.9rem", "color:var(--cream)",
    "box-shadow:0 4px 16px rgba(0,0,0,0.4)",
    "animation:slide-down 200ms ease",
  ].join(";");
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// ── Game Over ─────────────────────────────────────────────────
function _handleGameOver(data) {
  const deal   = data.deal_reached ?? data.deal ?? false;
  const amount = data.deal_amount  ?? data.final_price ?? null;
  const score  = data.reward       ?? data.total_reward ?? data.score ?? 0;

  const resultMsg = deal
    ? `Deal closed at ${formatCurrency(amount, "USD")}! Score: ${score.toFixed(2)}`
    : "No deal. You walked away from the table.";
  addMessage("system", resultMsg);

  const banner = document.getElementById("result-banner");
  if (banner) {
    banner.className = `result-banner ${deal ? "deal" : "walk"}`;
    banner.classList.remove("hidden");

    const title = banner.querySelector(".result-title");
    if (title) title.textContent = deal ? "Deal Closed" : "Walked Away";

    const amountEl = banner.querySelector(".result-amount");
    if (amountEl) amountEl.textContent = amount != null ? formatCurrency(amount, "USD") : "—";

    const scoreEl = banner.querySelector(".result-score");
    if (scoreEl) scoreEl.textContent = `Score: ${score.toFixed(2)}`;
  }

  if (character) character.setState(deal ? "pleased" : "shocked");
  setTimeout(loadLeaderboard, 1200);

  if (APP_DEBUG) console.log("[app] game over", { deal, amount, score });
}

// ── Scenario Dossier Cards ────────────────────────────────────
function _renderScenarioDossiers(scenarios) {
  const grid = document.getElementById("scenario-dossier-grid");
  if (!grid) return;
  grid.innerHTML = "";

  const defaults = [
    { id: "saas_enterprise",        title: "Enterprise SaaS",       description: "500-seat analytics platform",     zopa_lower: 125000,  zopa_upper: 165000,  difficulty: 2 },
    { id: "consulting_retainer",    title: "Consulting Retainer",   description: "Monthly strategy retainer",       zopa_lower: 25000,   zopa_upper: 40000,   difficulty: 1 },
    { id: "hiring_package",         title: "Senior Eng. Offer",     description: "Total comp negotiation",          zopa_lower: 195000,  zopa_upper: 230000,  difficulty: 2 },
    { id: "vendor_hardware",        title: "Hardware Contract",     description: "200-unit bulk purchase",          zopa_lower: 1750000, zopa_upper: 2200000, difficulty: 3 },
    { id: "acquisition_term_sheet", title: "Startup Acquisition",   description: "Acqui-hire term sheet",           zopa_lower: 10500000,zopa_upper:16000000, difficulty: 3 },
  ];

  const list = (scenarios && scenarios.length) ? scenarios : defaults;
  const diffLabels = ["", "Easy", "Medium", "Hard"];
  let caseNum = 1;

  list.forEach((s) => {
    const card = document.createElement("div");
    card.className = "scenario-dossier";
    card.dataset.scenarioId = s.id || s.scenario_id;
    card.setAttribute("role", "radio");
    card.setAttribute("tabindex", "0");

    const caseEl = document.createElement("div");
    caseEl.className = "dossier-case";
    caseEl.textContent = `CASE ${String(caseNum++).padStart(3, "0")}`;
    card.appendChild(caseEl);

    const diffEl = document.createElement("div");
    diffEl.className = "dossier-difficulty";
    diffEl.textContent = diffLabels[s.difficulty ?? 2] || "Medium";
    card.appendChild(diffEl);

    const titleEl = document.createElement("div");
    titleEl.className = "dossier-title";
    titleEl.textContent = s.title || s.name;
    card.appendChild(titleEl);

    const descEl = document.createElement("div");
    descEl.className = "dossier-desc";
    descEl.textContent = s.description || "";
    card.appendChild(descEl);

    const zopaLo = s.zopa_lower ?? s.zopa?.[0] ?? 0;
    const zopaHi = s.zopa_upper ?? s.zopa?.[1] ?? 0;
    const zopaEl = document.createElement("div");
    zopaEl.className = "dossier-zopa";
    zopaEl.textContent = `ZOPA ${formatCurrency(zopaLo, "USD")} – ${formatCurrency(zopaHi, "USD")}`;
    card.appendChild(zopaEl);

    card.addEventListener("click", () => {
      grid.querySelectorAll(".scenario-dossier").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
    });
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); card.click(); }
    });

    grid.appendChild(card);
  });
}

// ── Persona Cards ──────────────────────────────────────────────
function _renderPersonaCards(personas) {
  const grid = document.getElementById("persona-cards-grid");
  if (!grid) return;
  grid.innerHTML = "";

  const defaults = [
    { id: "shark",    name: "The Shark",    symbol: "◈", aggression: 0.88, patience: 0.18, bluff_rate: 0.72 },
    { id: "diplomat", name: "The Diplomat", symbol: "◎", aggression: 0.20, patience: 0.85, bluff_rate: 0.15 },
    { id: "analyst",  name: "The Analyst",  symbol: "◻", aggression: 0.35, patience: 0.90, bluff_rate: 0.10 },
    { id: "wildcard", name: "The Wildcard", symbol: "◇", aggression: 0.60, patience: 0.25, bluff_rate: 0.65 },
    { id: "veteran",  name: "The Veteran",  symbol: "◆", aggression: 0.50, patience: 0.95, bluff_rate: 0.45 },
  ];

  const list = (personas && personas.length) ? personas : defaults;

  list.forEach((p) => {
    const pid = p.id || p.persona_id;

    const card = document.createElement("div");
    card.className = "persona-card-option";
    card.dataset.persona = pid;
    card.setAttribute("role", "radio");
    card.setAttribute("tabindex", "0");

    // Mini Three.js preview canvas
    const canvasWrap = document.createElement("div");
    canvasWrap.className = "persona-card-canvas-wrap";
    const previewCanvas = document.createElement("canvas");
    previewCanvas.width  = 280;
    previewCanvas.height = 200;
    canvasWrap.appendChild(previewCanvas);
    card.appendChild(canvasWrap);

    const nameEl = document.createElement("div");
    nameEl.className = "persona-card-name";
    nameEl.textContent = p.name || pid;
    card.appendChild(nameEl);

    const symEl = document.createElement("div");
    symEl.className = "persona-card-symbol";
    symEl.textContent = p.symbol || "◈";
    card.appendChild(symEl);

    // Trait bars
    const traits = document.createElement("div");
    traits.className = "persona-trait-bars";
    [
      { label: "AGG", val: p.aggression ?? 0.5 },
      { label: "PAT", val: p.patience   ?? 0.5 },
    ].forEach(({ label, val }) => {
      const row = document.createElement("div");
      row.className = "persona-trait-row";
      const lbl = document.createElement("div");
      lbl.className = "persona-trait-label";
      lbl.textContent = label;
      const bar = document.createElement("div");
      bar.className = "persona-trait-bar";
      const fill = document.createElement("div");
      fill.className = "persona-trait-fill";
      fill.style.width = `${Math.round(val * 100)}%`;
      bar.appendChild(fill);
      row.appendChild(lbl);
      row.appendChild(bar);
      traits.appendChild(row);
    });
    card.appendChild(traits);

    // Click selection
    card.addEventListener("click", () => {
      grid.querySelectorAll(".persona-card-option").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
    });
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); card.click(); }
    });

    grid.appendChild(card);

    // Spin up a PersonaPreviewCharacter after DOM insert
    requestAnimationFrame(() => {
      if (typeof PersonaPreviewCharacter !== "undefined") {
        const prev = new PersonaPreviewCharacter(previewCanvas, pid);
        _previewChars[pid] = prev;
      }
    });
  });
}

function _destroyPreviewChars() {
  Object.values(_previewChars).forEach(c => { try { c.destroy(); } catch {} });
  _previewChars = {};
}

// ── Character init ────────────────────────────────────────────
function _initCharacter(persona) {
  if (typeof NegotiatorCharacter === "undefined") return;
  if (character) character.destroy();
  character = new NegotiatorCharacter("character-canvas", persona || "shark");
  if (APP_DEBUG) console.log("[app] character created for persona:", persona);
}

// ── Sparkline init ────────────────────────────────────────────
function _initSparkline(observation) {
  if (!charts || !observation) return;
  const lo   = observation.player_batna   ?? observation.your_batna   ?? observation.zopa_lower  ?? 0;
  const hi   = observation.opponent_batna ?? observation.opp_batna    ?? observation.zopa_upper  ?? 0;
  const nash = observation.nash_point     ?? ((lo + hi) / 2);
  charts.initOfferSparkline && charts.initOfferSparkline("offer-sparkline", lo, hi, nash);
  charts.initBeliefChart    && charts.initBeliefChart("belief-chart");
}

// ── Helpers ───────────────────────────────────────────────────
function formatCurrency(amount, currency) {
  if (amount == null || isNaN(amount)) return "—";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency", currency: currency || "USD",
      maximumFractionDigits: 0,
    }).format(amount);
  } catch {
    return `$${Number(amount).toLocaleString()}`;
  }
}

function setLoading(isLoading) {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.classList.toggle("hidden", !isLoading);
  ["btn-submit", "btn-accept", "btn-walk"].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.disabled = isLoading;
  });
}

function _setInputsDisabled(disabled) {
  ["offer-input", "move-select", "btn-submit", "btn-accept", "btn-walk"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
  });
}

function _updateActPills(act) {
  [1, 2, 3].forEach(n => {
    const pill = document.getElementById(`act-pill-${n}`);
    if (!pill) return;
    pill.classList.remove("active", "completed");
    if (n < act)        pill.classList.add("completed");
    else if (n === act) pill.classList.add("active");
  });
}

function _moveSummary(move, offer) {
  switch (move) {
    case "anchor":    return `Anchoring at ${formatCurrency(offer, "USD")}.`;
    case "counter":   return `Counter offer: ${formatCurrency(offer, "USD")}.`;
    case "concede":   return `Concession: ${formatCurrency(offer, "USD")}.`;
    case "package":   return `Package deal: ${formatCurrency(offer, "USD")}.`;
    case "accept":    return "Accepting the deal.";
    case "walk_away": return "Walking away from the table.";
    default:          return offer != null ? formatCurrency(offer, "USD") : move;
  }
}

function _personaLabel(persona) {
  const labels = {
    shark:    "The Shark",
    diplomat: "The Diplomat",
    analyst:  "The Analyst",
    wildcard: "The Wildcard",
    veteran:  "The Veteran",
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
