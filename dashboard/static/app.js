// ============================================================
// Parlay Game Logic — app.js
// "The Deal Room" — 3-step onboarding, mock mode, Mad Men theme.
// Items: 7-27 (UI fixes, deal flow, theme toggle, tooltips)
// ============================================================

const APP_DEBUG = false;
const API_BASE  = "";

// ── State ─────────────────────────────────────────────────────
let gameState = {
  sessionId:    null,
  persona:      null,
  scenario:     null,
  scenarioData: null,  // full scenario object for briefing
  observation:  null,
  hand:         [],
  done:         false,
  playerName:   "Player",
  turnCount:    0,
  cp:           100,
  maxCp:        100,
  currentAct:   1,
  actTranscripts: { 1: [] },
  lastPlayerOffer: null,
  lastOpponentOffer: null,
  pendingAiDeal:  false,   // AI has offered a deal, awaiting player response
  stepperAmount:  145000,  // current stepper value
  selectedTactic: null,
};

let character      = null;
let charts         = null;
let _driftTimer    = null;
let _previewChars  = {};

let _currentStep = 1;

// ── Init ──────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  _restoreTheme();  // item 27 — restore saved theme on load
  _initOnboarding();
  _syncOnboardingInert();

  try {
    if (typeof ParlayCharts !== "undefined") charts = new ParlayCharts();
  } catch (e) {
    if (APP_DEBUG) console.error("[Parlay] ParlayCharts init failed:", e);
  }

  loadScenarios();
  loadPersonas();
  _checkDemoMode();

  // Main submit
  const submitBtn = document.getElementById("btn-submit");
  if (submitBtn) submitBtn.addEventListener("click", submitMove);

  // Quick action chips — item 14
  const chipAccept = document.getElementById("chip-accept");
  if (chipAccept) chipAccept.addEventListener("click", acceptDeal);
  const chipWalk = document.getElementById("chip-walk");
  if (chipWalk) chipWalk.addEventListener("click", walkAway);
  const chipOffer = document.getElementById("chip-offer");
  if (chipOffer) chipOffer.addEventListener("click", _toggleOfferStepper);

  document.querySelectorAll(".tactic-btn").forEach((btn) => {
    btn.addEventListener("click", () => _toggleTacticButton(btn));
  });

  // Stepper controls
  document.getElementById("stepper-down")?.addEventListener("click", () => _stepOffer(-5000));
  document.getElementById("stepper-up")?.addEventListener("click",   () => _stepOffer(+5000));
  document.getElementById("stepper-use")?.addEventListener("click",  _useStepperOffer);
  document.getElementById("stepper-cancel")?.addEventListener("click", _hideOfferStepper);

  // Briefing begin
  document.getElementById("btn-briefing-begin")?.addEventListener("click", _dismissBriefing);

  // Drift dismiss
  document.getElementById("btn-dismiss-drift")?.addEventListener("click", dismissDriftAlert);

  // Demo banner dismiss
  document.getElementById("btn-dismiss-demo")?.addEventListener("click", () => {
    document.getElementById("demo-banner")?.classList.add("hidden");
    document.body.classList.remove("demo-mode");
  });

  // Offer input enter key
  const offerInput = document.getElementById("offer-input");
  if (offerInput) offerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) submitMove();
  });

  // Theme toggle — item 27
  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) themeBtn.addEventListener("click", _toggleTheme);

  loadLeaderboard();
});

// ── Theme toggle — item 27 ─────────────────────────────────────
function _restoreTheme() {
  const saved = localStorage.getItem("parlay-theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = saved === "light" ? "◑" : "●";
}

function _toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute("data-theme") !== "light";
  const next = isDark ? "light" : "dark";
  html.setAttribute("data-theme", next);
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = isDark ? "◑" : "●";
  localStorage.setItem("parlay-theme", next);
}

// ── Demo mode detection ────────────────────────────────────────
async function _checkDemoMode() {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("health");
    const data = await res.json();
    if (data.gemini === "mock") _showDemoBanner();
  } catch { _showDemoBanner(); }
}

function _showDemoBanner() {
  document.getElementById("demo-banner")?.classList.remove("hidden");
  document.body.classList.add("demo-mode");
}

// ── Onboarding ────────────────────────────────────────────────
function _initOnboarding() {
  document.getElementById("step1-name")?.addEventListener("keydown", e => {
    if (e.key === "Enter") _goToStep(2);
  });
  document.getElementById("step1-continue")?.addEventListener("click", () => _goToStep(2));
  document.getElementById("step2-back")?.addEventListener("click",     () => _goToStep(1));
  document.getElementById("step2-continue")?.addEventListener("click", () => _goToStep(3));
  document.getElementById("step3-back")?.addEventListener("click",     () => _goToStep(2));
  document.getElementById("step3-start")?.addEventListener("click",    _handleStep3Start);
}

function _syncOnboardingInert() {
  for (let n = 1; n <= 3; n++) {
    const el = document.getElementById(`onboarding-step-${n}`);
    if (!el) continue;
    n === _currentStep ? el.removeAttribute("inert") : el.setAttribute("inert", "");
  }
}

function _goToStep(step) {
  if (step === 2) {
    const name = (document.getElementById("step1-name")?.value ?? "").trim();
    if (!name) { _showStepError(1, "Please enter your name."); return; }
    _showStepError(1, "");
  }
  if (step === 3) {
    const sel = document.querySelector(".scenario-dossier.selected");
    if (!sel) { _showStepError(2, "Please select a scenario."); return; }
    _showStepError(2, "");
  }

  const currentEl = document.getElementById(`onboarding-step-${_currentStep}`);
  if (currentEl) {
    currentEl.classList.remove("active", "start-active");
    currentEl.classList.add("exiting");
    setTimeout(() => currentEl.classList.remove("exiting"), 300);
  }

  _currentStep = step;
  _syncOnboardingInert();

  const nextEl = document.getElementById(`onboarding-step-${step}`);
  if (nextEl) {
    nextEl.classList.remove("exiting");
    requestAnimationFrame(() => requestAnimationFrame(() => nextEl.classList.add("active")));
  }
}

function _showStepError(step, msg) {
  const el = document.getElementById(`step${step}-error`);
  if (!el) return;
  el.textContent = msg;
  if (msg) setTimeout(() => { el.textContent = ""; }, 3000);
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
  const nameInput   = document.getElementById("step1-name");
  const selScenario = document.querySelector(".scenario-dossier.selected");
  const selPersona  = document.querySelector(".persona-card-option.selected");

  if (!selPersona) { _showStepError(3, "Please choose an opponent."); return; }

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
  gameState.currentAct = 1;
  gameState.lastPlayerOffer   = null;
  gameState.lastOpponentOffer = null;
  gameState.pendingAiDeal = false;
  gameState.selectedTactic = null;

  try {
    const res = await fetch(`${API_BASE}/api/game/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenarioId, persona, player_name: playerName }),
    });

    let data;
    if (!res.ok) {
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
    gameState.scenarioData = data.scenario || null;

    _closeOnboarding();
    _destroyPreviewChars();
    updateUI(data);
    _initCharacter(persona);
    _initSparkline(data.observation);
    _updateScenarioHeader(data);
    _updatePersonaPanel(data);
    _updateNameplate(data);

    // Show deal briefing — item 15
    _showBriefing(data);

    if (APP_DEBUG) console.log("[app] game started", data);
  } catch (e) {
    if (APP_DEBUG) console.error("[app] startGame error:", e);
    const data = _mockStartData(scenarioId, persona, playerName);
    gameState.sessionId   = data.session_id;
    gameState.observation = data.observation;
    gameState.hand        = data.hand;
    gameState.cp          = 100;
    gameState.scenarioData = data.scenario;

    _closeOnboarding();
    _destroyPreviewChars();
    updateUI(data);
    _initCharacter(persona);
    _showDemoBanner();
    _showBriefing(data);
  } finally {
    setLoading(false);
  }
}

// ── Deal Briefing — item 15 ────────────────────────────────────
function _showBriefing(data) {
  const overlay = document.getElementById("briefing-overlay");
  if (!overlay) return;

  const sc = data.scenario || {};
  const obs = data.observation || {};

  const zopaLo = obs.zopa_lower ?? sc.zopa_lower ?? 125000;
  const zopaHi = obs.zopa_upper ?? sc.zopa_upper ?? 165000;
  const scenarioId = sc.id || gameState.scenario || "";
  const caseNum = scenarioId.replace(/_/g, "-").toUpperCase();

  const _s = (v) => formatCurrency(v, "USD");

  const caseNumEl = document.getElementById("briefing-case-num");
  const titleEl   = document.getElementById("briefing-title");
  const goalEl    = document.getElementById("briefing-your-goal");
  const theirEl   = document.getElementById("briefing-their-goal");
  const rangeEl   = document.getElementById("briefing-range");

  if (caseNumEl) caseNumEl.textContent = `CASE FILE #${caseNum || "001"}`;
  if (titleEl)   titleEl.textContent   = sc.title || "Negotiation";
  if (goalEl)    goalEl.textContent    = `Close the deal above ${_s(zopaLo)}. Your ideal: ${_s(zopaHi)}.`;
  if (theirEl)   theirEl.textContent   = `Pay as little as possible. They'll push hard on price from around ${_s(zopaLo)}.`;
  if (rangeEl)   rangeEl.textContent   = `A deal is possible between ${_s(zopaLo)} and ${_s(zopaHi)}.`;

  // Initialize stepper at midpoint
  gameState.stepperAmount = Math.round((zopaLo + zopaHi) / 2 / 1000) * 1000;
  _updateStepperDisplay();

  overlay.style.display = "flex";
}

function _dismissBriefing() {
  const overlay = document.getElementById("briefing-overlay");
  if (overlay) overlay.style.display = "none";

  // Enable inputs
  _setInputsDisabled(false);

  // Post system message and AI opening — item 7
  const thread = document.getElementById("chat-thread");
  if (thread) {
    // Clear placeholder
    const existing = thread.querySelector(".system-msg");
    if (existing) existing.remove();
  }
  addMessage("system", `Game started. You are negotiating as ${_personaLabel(gameState.persona)}.`);

  // Load AI opening
  const opener = gameState.observation?.opening_message
              || gameState._pendingOpener;
  if (opener) {
    addMessage("opponent", opener, gameState.observation?.opponent_offer ?? null, null);
  } else {
    // In mock mode, add a default opener
    const persona = gameState.persona || "shark";
    const mockOpeners = {
      shark:    "Let's not waste each other's time. What's your opening number?",
      diplomat: "I believe we can find a solution that works for both of us. Shall we begin?",
      veteran:  "I've been in rooms like this before. Let's get to it.",
    };
    addMessage("opponent", mockOpeners[persona] || "Let's begin.", null, null);
  }
}

function _mockStartData(scenarioId, persona, playerName) {
  const mockScenarios = {
    saas_enterprise:        { id: "saas_enterprise",        title: "Enterprise SaaS Contract",   zopa_lower: 125000,   zopa_upper: 165000   },
    hiring_package:         { id: "hiring_package",         title: "Senior Engineer Offer",       zopa_lower: 195000,   zopa_upper: 230000   },
    acquisition_term_sheet: { id: "acquisition_term_sheet", title: "Startup Acquisition",         zopa_lower: 10500000, zopa_upper: 16000000 },
  };
  const s = mockScenarios[scenarioId] || mockScenarios.saas_enterprise;
  const nash = (s.zopa_lower + s.zopa_upper) / 2;
  const sid  = "mock-" + Math.random().toString(36).slice(2);

  return {
    session_id: sid,
    scenario: s,
    observation: {
      step_count: 0, zopa_lower: s.zopa_lower, zopa_upper: s.zopa_upper,
      nash_point: nash, tension_score: 10, credibility_points: 100, zopa_width_pct_remaining: 1.0,
      belief_state: { cooperative: 0.5, competitive: 0.5,
                      reservation: s.zopa_lower / s.zopa_upper, flexibility: 0.5 },
    },
    persona: { id: persona, name: _personaLabel(persona), symbol: _personaSymbol(persona), emoji: "◈" },
    hand: [],
    opening_message: null,
    cp: 100, max_cp: 100,
  };
}

// ── Submit move ────────────────────────────────────────────────
async function submitMove() {
  if (gameState.done || !gameState.sessionId) return;

  const offerInput = document.getElementById("offer-input");

  const raw    = offerInput?.value.trim() ?? "";
  const cardId = gameState.selectedTactic;
  const offer  = raw ? parseFloat(raw.replace(/[$,]/g, "")) : null;

  if (offer !== null && isNaN(offer)) return;

  // Build message text
  const msgText = offer != null
    ? `Counter offer: ${formatCurrency(offer, "USD")}.`
    : raw || "Let me think about that.";

  if (offer != null) gameState.lastPlayerOffer = offer;

  addMessage("player", msgText, offer, cardId ? cardId : null);
  const thinkId = _showThinkingBubble();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/api/game/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: gameState.sessionId,
        move: offer ? "counter" : "chat",
        offer_amount: offer,
        card_id: cardId,
        message: raw,
      }),
    });

    const data = res.ok ? await res.json() : _mockStepData(offer, "counter");

    gameState.observation = data.observation;
    gameState.done        = data.done ?? false;
    gameState.hand        = data.hand || gameState.hand;
    gameState.cp          = data.cp   ?? data.observation?.credibility_points ?? gameState.cp;
    gameState.turnCount  += 1;

    _removeThinkingBubble(thinkId);
    updateUI(data);

    const oppMsg   = data.opponent_message ?? data.opponent?.utterance;
    const oppOffer = data.observation?.opponent_offer ?? data.opponent?.offer ?? null;
    const oppMove  = data.opponent_move ?? data.opponent?.tactical_move ?? null;

    if (oppOffer != null) gameState.lastOpponentOffer = oppOffer;

    if (oppMsg) {
      // Check if AI is offering a deal close — item 22
      const aiClose = _checkAiDealClose(oppOffer, oppMsg, data);
      if (aiClose) {
        _showAiDealOffer(oppMsg, oppOffer);
      } else {
        const bubble = addMessage("opponent", oppMsg, oppOffer, oppMove);
        if (bubble && gameState.persona) bubble.setAttribute("data-persona", gameState.persona);
      }
    }

    // Update character from game state — item 31
    if (data.observation) updateCharacterFromGameState(data.observation);

    if (gameState.done) _handleGameOver(data);

    if (offerInput) offerInput.value = "";
    gameState.selectedTactic = null;
    _updateTacticalButtons();

  } catch (e) {
    _removeThinkingBubble(thinkId);
    _showError("Move failed: " + e.message);
  } finally {
    setLoading(false);
  }
}

function _mockStepData(offer, move) {
  const obs  = gameState.observation || {};
  const tens = Math.min(100, (obs.tension_score || 10) + 7);
  const mockOffer = offer ? Math.round(offer * 0.97) : null;
  if (mockOffer) gameState.lastOpponentOffer = mockOffer;
  return {
    observation: { ...obs, tension_score: tens, step_count: (obs.step_count || 0) + 1,
                   opponent_offer: mockOffer },
    opponent_message: "That's an interesting position. Here's my counter.",
    opponent: { utterance: "That's an interesting position. Here's my counter.", offer: mockOffer },
    done: false,
  };
}

// ── AI deal close detection — item 22 ─────────────────────────
function _checkAiDealClose(oppOffer, oppMsg, data) {
  if (!oppOffer || !gameState.lastPlayerOffer) return false;
  const diff = Math.abs(oppOffer - gameState.lastPlayerOffer);
  const pct  = diff / Math.max(gameState.lastPlayerOffer, 1);
  if (pct < 0.03 && Math.random() < 0.30) return true;

  // High tension walk-away check
  const tension = data.observation?.tension_score ?? 0;
  const cumRew  = data.observation?.cumulative_reward ?? 0;
  if (tension >= 90 && cumRew < -30 && Math.random() < 0.20) {
    _showAiWalkAway();
    return true;
  }

  return false;
}

function _showAiDealOffer(msg, offer) {
  const thread = document.getElementById("chat-thread");
  if (!thread) return;

  // Gold-bordered bubble — item 22
  const bubble = document.createElement("div");
  bubble.className = `message-bubble opponent deal-offer`;
  if (gameState.persona) bubble.setAttribute("data-persona", gameState.persona);

  const meta = document.createElement("div");
  meta.className = "bubble-meta";
  const nameSpan = document.createElement("span");
  nameSpan.textContent = "Opponent";
  meta.appendChild(nameSpan);
  const pill = document.createElement("span");
  pill.className = "move-pill";
  pill.textContent = "◈ Deal offered";
  meta.appendChild(pill);

  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = msg || "Deal — let's close this.";

  if (offer != null && !isNaN(offer)) {
    const chip = document.createElement("div");
    chip.className = "offer-chip";
    chip.textContent = formatCurrency(offer, "USD");
    body.appendChild(chip);
  }

  // Accept / Counter buttons
  const actions = document.createElement("div");
  actions.className = "ai-deal-prompt";
  const acceptBtn = document.createElement("button");
  acceptBtn.className = "ai-deal-btn accept";
  acceptBtn.textContent = "Accept ✓";
  acceptBtn.onclick = () => acceptDeal();
  const counterBtn = document.createElement("button");
  counterBtn.className = "ai-deal-btn counter";
  counterBtn.textContent = "Counter";
  counterBtn.onclick = () => { actions.remove(); };

  actions.appendChild(acceptBtn);
  actions.appendChild(counterBtn);

  bubble.appendChild(meta);
  bubble.appendChild(body);
  bubble.appendChild(actions);
  thread.appendChild(bubble);
  _scrollThread(thread);

  gameState.pendingAiDeal = true;
}

function _showAiWalkAway() {
  const walkMsg = "I'm afraid we're too far apart. I'm walking away.";
  addMessage("opponent", walkMsg, null, null);
  setTimeout(() => _triggerWalkAwayModal(false), 800);
}

// ── Accept Deal — item 21 ──────────────────────────────────────
async function acceptDeal() {
  if (gameState.done || !gameState.sessionId) return;

  // Validate: must have an offer on the table
  if (!gameState.lastPlayerOffer && !gameState.lastOpponentOffer) {
    _showInlineWarning("Make an offer first before accepting.");
    return;
  }

  const dealAmount = gameState.lastOpponentOffer || gameState.lastPlayerOffer;

  addMessage("player", "I accept the deal.", dealAmount, "accept");
  const thinkId = _showThinkingBubble();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/api/game/accept`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: gameState.sessionId }),
    });
    const data = res.ok ? await res.json() : { deal_reached: true, deal_amount: dealAmount, reward: 0 };

    _removeThinkingBubble(thinkId);
    gameState.done = true;
    updateUI(data);

    const finalPrice = data.final_price ?? data.deal_amount ?? dealAmount;
    _handleGameOver({ ...data, deal_reached: true, deal_amount: finalPrice });
  } catch (e) {
    _removeThinkingBubble(thinkId);
    _showError(e.message);
  } finally {
    setLoading(false);
  }
}

// ── Act Transition Modal — item 21 ────────────────────────────
function _showActTransition(act, price, obs) {
  const zopaLo = obs.zopa_lower ?? 125000;
  const zopaHi = obs.zopa_upper ?? 165000;
  const zopaWidth = Math.max(1, zopaHi - zopaLo);
  const efficiency = price ? Math.max(0, Math.min(1, (price - zopaLo) / zopaWidth)) : 0;
  const captured = price ? Math.round(price - zopaLo) : 0;
  const available = Math.round(zopaWidth);
  const nash = (zopaLo + zopaHi) / 2;
  const nashDiff = price ? Math.abs(price - nash) / nash : 1;
  const nashNote = nashDiff < 0.05 ? "You nailed it." : nashDiff < 0.15 ? "Close to optimal." : "Room to improve next time.";

  const overlay = document.createElement("div");
  overlay.className = "act-modal-overlay";
  overlay.id = "act-modal";

  overlay.innerHTML = `
    <div class="act-modal-card" role="dialog" aria-label="Act ${act} complete">
      <div class="act-modal-title">ACT ${['I','II','III'][act-1]} COMPLETE</div>
      <div class="act-modal-price">${formatCurrency(price, "USD")}</div>
      <div class="act-modal-eff">
        <div class="act-modal-eff-label">Efficiency: ${Math.round(efficiency * 100)}%</div>
        <div class="act-modal-eff-track">
          <div class="act-modal-eff-fill" style="width:0%;"></div>
        </div>
      </div>
      <div class="act-modal-caption">
        You captured ${formatCurrency(captured, "USD")} of the
        ${formatCurrency(available, "USD")} available.
      </div>
      <div class="act-modal-nash">Nash optimal was ${formatCurrency(nash, "USD")}. ${nashNote}</div>
      <button class="act-modal-btn" type="button">Continue to Act ${['II','III'][act-1] || 'End'}: ${['Terms →','Coalition →'][act-1] || '→'}</button>
    </div>
  `;

  document.body.appendChild(overlay);

  // Animate efficiency bar
  requestAnimationFrame(() => {
    const fill = overlay.querySelector(".act-modal-eff-fill");
    if (fill) fill.style.width = `${Math.round(efficiency * 100)}%`;
  });

  overlay.querySelector(".act-modal-btn").onclick = () => {
    overlay.remove();
    _advanceAct(act + 1, price);
  };
}

function _advanceAct(nextAct, priceFromPrev) {
  gameState.currentAct = nextAct;
  gameState.actsCompleted = nextAct - 1;
  gameState.done = false;
  gameState.lastPlayerOffer = null;
  gameState.lastOpponentOffer = null;

  _updateActPills(nextAct);

  if (nextAct === 2) {
    addMessage("system", `Act II — Terms. Price locked at ${formatCurrency(priceFromPrev, "USD")}. Now negotiate the package.`);
    _showActIIBriefing(priceFromPrev);
  } else if (nextAct === 3) {
    addMessage("system", "Act III — Coalition. A third party has entered the room.");
    _showActIIIIntro();
  } else {
    // Game over
    _handleGameOver({ deal_reached: true, deal_amount: priceFromPrev, reward: 0 });
  }

  _setInputsDisabled(false);
}

// ── Act II Briefing — item 24 ──────────────────────────────────
function _showActIIBriefing(lockedPrice) {
  addMessage("opponent", `Price is locked at ${formatCurrency(lockedPrice, "USD")}. Now let's talk about the terms — payment schedule, SLA, and contract length.`, null, null);
}

// ── Act III Intro — item 25 ────────────────────────────────────
function _showActIIIIntro() {
  addMessage("system", "The Board has entered. A third party with its own interests will interject every few turns.");
  addMessage("opponent", "I've been authorized to proceed, but The Board will have the final word on certain terms.", null, null);
  // Board interjects every 3 turns (tracked via turnCount)
  gameState._boardInterjectionTurns = [
    gameState.turnCount + 3,
    gameState.turnCount + 6,
    gameState.turnCount + 9,
  ];
}

// ── Walk Away — item 23 ────────────────────────────────────────
async function walkAway() {
  if (gameState.done || !gameState.sessionId) return;
  addMessage("player", "I'm walking away from the table.", null, "walk");
  const thinkId = _showThinkingBubble();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/api/game/walkaway`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: gameState.sessionId }),
    });
    const data = res.ok ? await res.json() : { result: "walk_away" };
    _removeThinkingBubble(thinkId);
    gameState.done = true;
    updateUI(data);
    _triggerWalkAwayModal(true, data);
  } catch (e) {
    _removeThinkingBubble(thinkId);
    _showError(e.message);
  } finally {
    setLoading(false);
  }
}

function _triggerWalkAwayModal(playerInitiated, data) {
  const obs     = gameState.observation || {};
  const zopaLo  = obs.zopa_lower ?? 125000;
  const zopaHi  = obs.zopa_upper ?? 165000;
  const left    = zopaHi - zopaLo;
  const reasons = _buildWalkAwayReasons(obs, data);

  const overlay = document.createElement("div");
  overlay.className = "walk-modal-overlay";
  overlay.id = "walk-modal";

  const reasonsHtml = reasons.map(r => `<div class="walk-reason-item">${r}</div>`).join("");

  overlay.innerHTML = `
    <div class="walk-modal-card" role="dialog" aria-label="No deal">
      <div class="walk-modal-title">No Deal</div>
      <div class="walk-modal-sub">The negotiation collapsed.</div>
      <div class="walk-modal-left">You left ${formatCurrency(left, "USD")} on the table.</div>
      <div class="walk-modal-range">The deal was possible between ${formatCurrency(zopaLo,"USD")} and ${formatCurrency(zopaHi,"USD")}.</div>
      ${reasons.length ? `<div class="walk-modal-reasons"><div class="walk-modal-reasons-label">What went wrong:</div>${reasonsHtml}</div>` : ""}
      <div class="walk-modal-actions">
        <button class="walk-modal-btn primary" id="walk-try-again">Try Again</button>
        <button class="walk-modal-btn" id="walk-change-opp">Change Opponent</button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  overlay.querySelector("#walk-try-again").onclick = () => {
    overlay.remove();
    startGame(gameState.scenario, gameState.persona, gameState.playerName);
  };
  overlay.querySelector("#walk-change-opp").onclick = () => {
    overlay.remove();
    location.reload();
  };
}

function _buildWalkAwayReasons(obs, data) {
  const reasons = [];
  const tension     = obs.tension_score ?? 0;
  const stepCount   = obs.step_count ?? 0;
  const concRate    = data?.concession_rate ?? 0;
  const tomAcc      = data?.tom_accuracy ?? obs.tom_accuracy ?? 1;
  const tacticsUsed = gameState.hand && gameState.hand.length > 0;

  if (concRate > 0.15) reasons.push("Too many concessions early");
  if (tension > 80)    reasons.push(`Tension peaked at turn ${stepCount}`);
  if (!tacticsUsed && gameState.turnCount > 3) reasons.push("No tactical moves played");
  if (tomAcc < 0.4)    reasons.push("Misread the opponent's constraints");
  return reasons;
}

// ── Board interject — item 25 ──────────────────────────────────
function _checkBoardInterject() {
  return;
  const turns = gameState._boardInterjectionTurns || [];
  const idx = turns.indexOf(gameState.turnCount);
  if (idx === -1) return;

  const msgs = [
    "I've been told another vendor can deliver in 4 weeks. That changes our calculus.",
    "The budget committee approved up to the higher range — but they want delivery certainty.",
    "The Board is watching this closely. We need to wrap this up.",
  ];
  const msg = msgs[idx % msgs.length];
  setTimeout(() => {
    addMessage("system", `◉ The Board: "${msg}"`);
  }, 500);
}

// ── Inline warning — item 21 ───────────────────────────────────
function _showInlineWarning(msg) {
  const thread = document.getElementById("chat-thread");
  if (!thread) return;
  const div = document.createElement("div");
  div.className = "system-msg";
  div.style.color = "var(--scarlet-light)";
  div.textContent = msg;
  thread.appendChild(div);
  _scrollThread(thread);
  setTimeout(() => div.remove(), 3000);
}

// ── Offer Stepper — item 14 ────────────────────────────────────
function _toggleOfferStepper() {
  const stepper = document.getElementById("offer-stepper");
  if (!stepper) return;
  const visible = stepper.classList.contains("visible");
  stepper.classList.toggle("visible", !visible);
}

function _hideOfferStepper() {
  document.getElementById("offer-stepper")?.classList.remove("visible");
}

function _stepOffer(delta) {
  const obs = gameState.observation || {};
  const lo  = obs.zopa_lower ?? 0;
  const hi  = obs.zopa_upper ?? 999999999;
  gameState.stepperAmount = Math.max(lo, Math.min(hi, (gameState.stepperAmount || lo) + delta));
  _updateStepperDisplay();
}

function _updateStepperDisplay() {
  const el = document.getElementById("stepper-value");
  if (el) el.textContent = formatCurrency(gameState.stepperAmount, "USD");
}

function _useStepperOffer() {
  const offerInput = document.getElementById("offer-input");
  if (offerInput) offerInput.value = String(gameState.stepperAmount);
  _hideOfferStepper();
  offerInput?.focus();
}

// ── Scenario/Persona loaders ───────────────────────────────────
async function loadScenarios() {
  try {
    const res = await fetch(`${API_BASE}/api/scenarios`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    _renderScenarioDossiers(data.scenarios || data || []);
  } catch {
    _renderScenarioDossiers([]);
  }
}

async function loadPersonas() {
  try {
    const res = await fetch(`${API_BASE}/api/personas`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    _renderPersonaCards(data.personas || data || []);
  } catch {
    _renderPersonaCards([]);
  }
}

async function loadLeaderboard() {
  try {
    const res = await fetch(`${API_BASE}/api/leaderboard?limit=5`);
    if (!res.ok) return;
    const data = await res.json();
    renderLeaderboard(data.entries || data || []);
  } catch {}
}

// ── UI Updates ────────────────────────────────────────────────
function updateUI(response) {
  const obs = response.observation || gameState.observation;

  if (obs) {
    updateZOPABar(obs);
    updateBeliefBars(obs.belief_state || obs.beliefState);
    updateCharacterState(obs);
    updateCharacterFromGameState(obs);  // item 31
  }

  const tension = obs?.tension_score ?? obs?.tension ?? 0;
  updateTensionMeter(tension);
  updateCPBar(gameState.cp);
  _updateTacticalButtons();

  const drift = response.drift_event || obs?.drift_event;
  if (drift) showDriftAlert(drift);

  const act = 1;
  _updateActPills(act);

  if (charts && obs) {
    const playerOffer   = obs.player_offer   ?? obs.your_offer  ?? null;
    const opponentOffer = obs.opponent_offer ?? null;
    if (playerOffer !== null || opponentOffer !== null) {
      charts.updateOfferSparkline?.call(charts, playerOffer, opponentOffer, gameState.turnCount);
    }
    if (obs.belief_state) charts.updateBeliefChart?.call(charts, obs.belief_state);
  }

  const avatarEl = document.getElementById("player-avatar");
  if (avatarEl && gameState.playerName) avatarEl.textContent = gameState.playerName.charAt(0).toUpperCase();

  const nameEl = document.getElementById("player-name-display");
  if (nameEl) nameEl.textContent = gameState.playerName;

  _setInputsDisabled(gameState.done);

  // Board interject check — item 25
  _checkBoardInterject();
}

// item 31 — update character from game state observation
function updateCharacterFromGameState(obs) {
  if (!character) return;
  const tension   = obs?.tension_score ?? obs?.tension ?? 0;
  const drift     = obs?.drift_event;
  const cumReward = obs?.cumulative_reward ?? obs?.reward ?? 0;

  let state = "idle";
  if (drift)             state = "shocked";
  else if (tension > 80) state = "aggressive";
  else if (tension > 55) state = "thinking";
  else if (cumReward > 15) state = "pleased";

  character.setState(state);

  if (drift) setTimeout(() => {
    if (character && gameState.sessionId) character.setState("aggressive");
  }, 2000);

  const label = document.getElementById("character-state-label");
  if (label) label.textContent = state;
  const badge = document.querySelector(".character-state-badge");
  if (badge) badge.textContent = state;
}

function _updateScenarioHeader(data) {
  const titleEl = document.getElementById("scenario-title");
  const metaEl  = document.getElementById("scenario-meta");
  const sessionEl = document.getElementById("session-id-label");
  const sc = data.scenario || {};
  if (titleEl) titleEl.textContent = sc.title || data.scenario_id || "Negotiation";
  if (metaEl)  metaEl.textContent  = sc.description || "";
  if (sessionEl) sessionEl.textContent = `Session: ${gameState.sessionId || data.session_id || "—"}`;
}

function _updatePersonaPanel(data) {
  const p = data.persona || {};
  const nameEl = document.getElementById("persona-name");
  const descEl = document.getElementById("persona-desc");
  const avatEl = document.getElementById("persona-avatar");
  if (nameEl) nameEl.textContent = p.name || _personaLabel(gameState.persona);
  if (descEl) descEl.textContent = p.style || "";
  if (avatEl) avatEl.textContent = p.symbol || p.emoji || _personaSymbol(gameState.persona);
}

function _updateNameplate(data) {
  const p = data.persona || {};
  const sym  = document.getElementById("nameplate-symbol");
  const name = document.getElementById("nameplate-name");
  const tag  = document.getElementById("nameplate-tag");

  const personaId = p.id || gameState.persona || "shark";
  const pName = p.name || _personaLabel(personaId);
  const pSym  = p.symbol || _personaSymbol(personaId);
  const pTag  = _personaTag(personaId);

  if (sym) {
    sym.textContent  = pSym;
    sym.style.color  = _personaColor(personaId);
  }
  if (name) name.textContent = pName;
  if (tag)  tag.textContent  = pTag;
}

// ── Message Bubbles — item 7 (system-msg fix) ─────────────────
function addMessage(role, text, offer, move) {
  const thread = document.getElementById("chat-thread");
  if (!thread) return null;

  const existing = thread.querySelector(".thinking-bubble");
  if (existing) existing.remove();

  // System messages — item 7: use .system-msg, no blue highlight
  if (role === "system") {
    const sys = document.createElement("div");
    sys.className = "system-msg";
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

  // Save to act transcript — item 26
  const actKey = 1;
  if (!gameState.actTranscripts[actKey]) gameState.actTranscripts[actKey] = [];
  gameState.actTranscripts[actKey].push({ role, text, offer, move });

  _scrollThread(thread);
  return bubble;
}

function _showThinkingBubble() {
  const thread = document.getElementById("chat-thread");
  if (!thread) return null;
  const id   = "thinking-" + Date.now();
  const wrap = document.createElement("div");
  wrap.className = "thinking-bubble"; wrap.id = id;
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement("div");
    dot.className = "thinking-dot"; wrap.appendChild(dot);
  }
  thread.appendChild(wrap);
  _scrollThread(thread);
  return id;
}

function _removeThinkingBubble(id) {
  if (!id) return;
  document.getElementById(id)?.remove();
}

function _scrollThread(thread) {
  requestAnimationFrame(() => { thread.scrollTop = thread.scrollHeight; });
}

// ── Tactic chips row — item 14 ─────────────────────────────────
function _renderTacticChips(hand) {
  const row = document.getElementById("tactic-chips-row");
  if (!row) return;
  row.innerHTML = "";

  if (!hand || !hand.length) return;

  hand.forEach(card => {
    const chip = document.createElement("button");
    chip.className = "tactic-chip";
    chip.dataset.cardId = card.id || card.card_id || card.move || "";
    chip.type = "button";

    const costDiv = document.createElement("span");
    costDiv.className = "tactic-chip-cost";
    costDiv.textContent = String(card.cp_cost ?? card.cost ?? "?");
    chip.appendChild(costDiv);

    const labelSpan = document.createElement("span");
    labelSpan.textContent = card.name || card.move || "Tactic";
    chip.appendChild(labelSpan);

    chip.addEventListener("click", () => {
      row.querySelectorAll(".tactic-chip").forEach(c => c.classList.remove("selected"));
      chip.classList.toggle("selected");
    });

    row.appendChild(chip);
  });
}

function _toggleTacticButton(button) {
  const cardId = button?.dataset.card || null;
  if (!cardId || button.disabled) return;
  gameState.selectedTactic = gameState.selectedTactic === cardId ? null : cardId;
  _updateTacticalButtons();
}

function _updateTacticalButtons() {
  document.querySelectorAll(".tactic-btn").forEach((btn) => {
    const cost = Number(btn.dataset.cost || "0");
    const cardId = btn.dataset.card || "";
    btn.disabled = gameState.done || gameState.cp < cost || !gameState.sessionId;
    btn.classList.toggle("selected", gameState.selectedTactic === cardId);
  });
}

function getZopaColor(pctRemaining) {
  if (pctRemaining > 0.7) return "var(--gold)";
  if (pctRemaining > 0.4) return "#c8860a";
  return "var(--scarlet)";
}

// ── ZOPA Bar — item 10 ─────────────────────────────────────────
function updateZOPABar(observation) {
  const track = document.getElementById("zopa-track");
  if (!track) return;

  const batnaPlayer   = observation.player_batna   ?? observation.your_batna   ?? observation.zopa_lower  ?? 0;
  const batnaOpponent = observation.opponent_batna ?? observation.opp_batna    ?? observation.zopa_upper  ?? 100;
  const playerOffer   = observation.player_offer   ?? observation.your_offer   ?? null;
  const opponentOffer = observation.opponent_offer ?? null;
  const nash          = observation.nash_point     ?? null;
  const pctRemaining  = observation.zopa_width_pct_remaining ?? 1.0;
  const zopaColor     = getZopaColor(pctRemaining);

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
    zopaZone.style.background = zopaColor;
    zopaZone.style.borderLeftColor = zopaColor;
    zopaZone.style.borderRightColor = zopaColor;
  }
  track.style.borderColor = zopaColor;

  const mPlayer = document.getElementById("marker-player");
  if (mPlayer) mPlayer.style.left = pct(batnaPlayer);

  const mOpponent = document.getElementById("marker-opponent");
  if (mOpponent) mOpponent.style.left = pct(batnaOpponent);

  // Your offer marker
  const mCurrent = document.getElementById("marker-current");
  if (mCurrent) {
    const currentOffer = playerOffer ?? opponentOffer;
    if (currentOffer != null) {
      mCurrent.style.left    = pct(currentOffer);
      mCurrent.style.display = "flex";
    }
  }

  // Nash — always visible when known
  const nashMarker = document.getElementById("nash-marker");
  if (nashMarker && nash != null) {
    nashMarker.style.left    = pct(nash);
    nashMarker.style.display = "block";
  }

  const lblLow  = document.getElementById("zopa-label-low");
  const lblHigh = document.getElementById("zopa-label-high");
  const widthEl = document.getElementById("zopa-width-indicator");
  if (lblLow)  lblLow.textContent  = formatCurrency(minVal, "USD");
  if (lblHigh) lblHigh.textContent = formatCurrency(maxVal, "USD");
  if (widthEl) widthEl.textContent = `Deal zone: ${Math.round(pctRemaining * 100)}%`;
}

// ── Tension Meter — item 12 ────────────────────────────────────
function updateTensionMeter(tensionScore) {
  const fill  = document.getElementById("tension-fill");
  const value = document.getElementById("tension-value");
  const desc  = document.getElementById("tension-descriptor");
  if (!fill) return;

  const pct = Math.max(0, Math.min(100, tensionScore || 0));
  fill.style.width = `${pct}%`;

  let level = "low";
  let word  = "Calm";
  if (pct >= 85)      { level = "high";   word = "Critical"; }
  else if (pct >= 70) { level = "high";   word = "Intense";  }
  else if (pct >= 55) { level = "medium"; word = "Heated";   }
  else if (pct >= 35) { level = "medium"; word = "Warming";  }

  fill.setAttribute("data-level", level);

  if (value) value.textContent = `${Math.round(pct)}%`;
  if (desc)  desc.textContent  = `· ${word}`;
}

// ── Belief Bars — item 9 (non-zero priors) ─────────────────────
function updateBeliefBars(beliefState) {
  if (!beliefState) return;

  const obs = gameState.observation || {};
  const zopaLo = obs.zopa_lower ?? 125000;
  const zopaHi = obs.zopa_upper ?? 165000;
  // Reservation prior: batna_seller / anchor * 100
  const resPrior = zopaLo / Math.max(zopaHi, 1);

  const mapping = {
    "belief-cooperative": beliefState.cooperative ?? beliefState.cooperative_prob ?? 0.5,
    "belief-competitive": beliefState.competitive ?? beliefState.competitive_prob ?? 0.5,
    "belief-reservation": beliefState.reservation ?? beliefState.reservation_sensitivity ?? resPrior,
    "belief-flexibility": beliefState.flexibility ?? beliefState.concession_rate ?? 0.5,
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

  if (charts?.updateBeliefChart) charts.updateBeliefChart(beliefState);
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

// ── Character State ────────────────────────────────────────────
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
  if (drift && character) {
    setTimeout(() => { if (character && gameState.sessionId) character.setState("aggressive"); }, 2000);
  }
}

// ── Drift Alert ───────────────────────────────────────────────
function showDriftAlert(driftEvent) {
  const bar  = document.getElementById("drift-alert");
  const text = document.getElementById("drift-alert-text");
  if (!bar) return;
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
  document.getElementById("drift-alert")?.classList.add("hidden");
  if (_driftTimer) clearTimeout(_driftTimer);
}

// ── Hand Rendering (card cards) — item 8 (no enum strings) ────
function renderHand(hand) {
  const container = document.getElementById("hand-container");
  if (!container) return;
  container.innerHTML = "";
  if (!hand || !hand.length) return;

  // Card descriptions (plain English) — item 8
  const CARD_DESCS = {
    anchor_high:      "Set a bold opening number to frame the negotiation.",
    batna_reveal:     "Reveal your walk-away option to signal credibility.",
    silence:          "Say nothing — let the pressure work for you.",
  };

  hand.forEach((card) => {
    const wrapper = document.createElement("div");
    wrapper.className = "tactical-card";
    wrapper.dataset.cardId = card.id || card.card_id || card.move || "";

    const inner = document.createElement("div");
    inner.className = "card-inner";

    // Front face
    const front = document.createElement("div");
    front.className = "card-face";

    const sym = document.createElement("div");
    sym.className = "card-symbol";
    sym.textContent = card.symbol || "◈";
    front.appendChild(sym);

    const name = document.createElement("div");
    name.className = "card-name";
    name.textContent = card.name || _tidyCardName(card.move || card.id || "Tactic");
    front.appendChild(name);

    // Cost badge (bottom right)
    const cost = document.createElement("div");
    cost.className = "card-cost";
    cost.textContent = `${card.cp_cost ?? card.cost ?? "—"}`;
    front.appendChild(cost);

    // Back face
    const back = document.createElement("div");
    back.className = "card-back";
    const backLabel = document.createElement("div");
    backLabel.className = "card-back-label";
    backLabel.textContent = "Game Theory";
    back.appendChild(backLabel);
    const gt = document.createElement("div");
    gt.className = "card-game-theory";
    // Use plain English description — item 8 (no raw enum strings)
    const moveKey = (card.move || card.id || "").toLowerCase();
    gt.textContent = CARD_DESCS[moveKey] || card.game_theory_basis || card.description || "";
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

function _tidyCardName(raw) {
  return raw.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

// ── Leaderboard ───────────────────────────────────────────────
function renderLeaderboard(entries) {
  const tbody = document.getElementById("leaderboard-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!entries || !entries.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4; td.className = "empty-state text-muted"; td.textContent = "No games yet";
    tr.appendChild(td); tbody.appendChild(tr);
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
    "position:fixed","bottom:24px","right:24px","z-index:9999",
    "background:var(--mahogany)","border:1px solid var(--gold)",
    "border-radius:6px","padding:12px 20px",
    "font-family:var(--font-display)","font-style:italic",
    "font-size:0.9rem","color:var(--cream)",
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
    const title   = banner.querySelector(".result-title");
    const amountEl= banner.querySelector(".result-amount");
    const scoreEl = banner.querySelector(".result-score");
    if (title)   title.textContent   = deal ? "Deal Closed" : "Walked Away";
    if (amountEl) amountEl.textContent = amount != null ? formatCurrency(amount, "USD") : "—";
    if (scoreEl)  scoreEl.textContent  = `Score: ${score.toFixed(2)}`;
  }

  if (character) character.setState(deal ? "pleased" : "shocked");
  setTimeout(loadLeaderboard, 1200);
}

// ── Act Pills — item 26 ────────────────────────────────────────
function _updateActPills(act) {
  const labels = {
    1: "I · Price", 2: "II · Terms", 3: "III · Coalition",
  };
  [1, 2, 3].forEach(n => {
    const pill = document.getElementById(`act-pill-${n}`);
    if (!pill) return;
    pill.classList.remove("active", "completed", "locked");

    if (n < act) {
      pill.classList.add("completed");
      pill.textContent = `✓ ${labels[n]}`;
      pill.style.cursor = "pointer";
      pill.onclick = () => _showActTranscript(n);
    } else if (n === act) {
      pill.classList.add("active");
      pill.textContent = labels[n];
      pill.style.cursor = "default";
      pill.onclick = null;
    } else {
      pill.classList.add("locked");
      pill.textContent = labels[n];
      pill.style.cursor = "not-allowed";
      pill.setAttribute("data-lock-msg", `Complete Act ${['I','II'][n-2]} first`);
      pill.onclick = null;
    }
  });
}

function _showActTranscript(act) {
  const msgs = gameState.actTranscripts[act] || [];
  if (!msgs.length) return;
  _showToast(`Act ${['I','II','III'][act-1]} — ${msgs.length} messages (read-only view)`);
}

// ── Scenario + Persona render ──────────────────────────────────
function _renderScenarioDossiers(scenarios) {
  const grid = document.getElementById("scenario-dossier-grid");
  if (!grid) return;
  grid.innerHTML = "";

  const defaults = [
    { id: "saas_enterprise",        title: "Enterprise SaaS",       description: "500-seat analytics platform",   zopa_lower: 125000,   zopa_upper: 165000,   difficulty: 2 },
    { id: "hiring_package",         title: "Senior Eng. Offer",     description: "Total comp negotiation",        zopa_lower: 195000,   zopa_upper: 230000,   difficulty: 2 },
    { id: "acquisition_term_sheet", title: "Startup Acquisition",   description: "Acqui-hire term sheet",         zopa_lower: 10500000, zopa_upper: 16000000, difficulty: 3 },
  ];

  const list = (scenarios && scenarios.length) ? scenarios : defaults;
  const diffLabels = ["", "Easy", "Medium", "Hard"];
  let caseNum = 1;

  list.forEach((s) => {
    const card = document.createElement("div");
    card.className = "scenario-dossier";
    card.dataset.scenarioId = s.id || s.scenario_id;
    card.setAttribute("role", "radio"); card.setAttribute("tabindex", "0");

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
      gameState.scenarioData = s;
    });
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); card.click(); }
    });

    grid.appendChild(card);
  });
}

function _renderPersonaCards(personas) {
  const grid = document.getElementById("persona-cards-grid");
  if (!grid) return;
  grid.innerHTML = "";

  const defaults = [
    { id: "shark",    name: "The Shark",    symbol: "◈", aggression: 0.88, patience: 0.18 },
    { id: "diplomat", name: "The Diplomat", symbol: "◎", aggression: 0.20, patience: 0.85 },
    { id: "veteran",  name: "The Veteran",  symbol: "◆", aggression: 0.50, patience: 0.95 },
  ];

  const list = (personas && personas.length) ? personas : defaults;

  list.forEach((p) => {
    const pid  = p.id || p.persona_id;
    const card = document.createElement("div");
    card.className = "persona-card-option";
    card.dataset.persona = pid;
    card.setAttribute("role", "radio"); card.setAttribute("tabindex", "0");

    const canvasWrap = document.createElement("div");
    canvasWrap.className = "persona-card-canvas-wrap";
    const previewCanvas = document.createElement("canvas");
    previewCanvas.width = 280; previewCanvas.height = 200;
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

    const traits = document.createElement("div");
    traits.className = "persona-trait-bars";
    [
      { label: "AGG", val: p.aggression ?? 0.5 },
      { label: "PAT", val: p.patience   ?? 0.5 },
    ].forEach(({ label, val }) => {
      const row   = document.createElement("div"); row.className = "persona-trait-row";
      const lbl   = document.createElement("div"); lbl.className = "persona-trait-label"; lbl.textContent = label;
      const bar   = document.createElement("div"); bar.className = "persona-trait-bar";
      const fill  = document.createElement("div"); fill.className = "persona-trait-fill";
      fill.style.width = `${Math.round(val * 100)}%`;
      bar.appendChild(fill); row.appendChild(lbl); row.appendChild(bar);
      traits.appendChild(row);
    });
    card.appendChild(traits);

    card.addEventListener("click", () => {
      grid.querySelectorAll(".persona-card-option").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
    });
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); card.click(); }
    });

    grid.appendChild(card);

    requestAnimationFrame(() => {
      if (typeof PersonaPreviewCharacter !== "undefined") {
        _previewChars[pid] = new PersonaPreviewCharacter(previewCanvas, pid);
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
}

// ── Sparkline init ────────────────────────────────────────────
function _initSparkline(observation) {
  if (!charts || !observation) return;
  const lo   = observation.player_batna   ?? observation.your_batna   ?? observation.zopa_lower  ?? 0;
  const hi   = observation.opponent_batna ?? observation.opp_batna    ?? observation.zopa_upper  ?? 0;
  const nash = observation.nash_point     ?? ((lo + hi) / 2);
  charts.initOfferSparkline?.("offer-sparkline", lo, hi, nash);
  charts.initBeliefChart?.("belief-chart");
}

// ── Helpers ───────────────────────────────────────────────────
function formatCurrency(amount, currency) {
  if (amount == null || isNaN(amount)) return "—";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency", currency: currency || "USD", maximumFractionDigits: 0,
    }).format(amount);
  } catch {
    return `$${Number(amount).toLocaleString()}`;
  }
}

function setLoading(isLoading) {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.classList.toggle("hidden", !isLoading);
  ["btn-submit", "chip-accept", "chip-walk", "chip-offer"].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.disabled = isLoading;
  });
}

function _setInputsDisabled(disabled) {
  ["offer-input", "btn-submit", "chip-accept", "chip-walk", "chip-offer"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
  });
}

function _personaLabel(persona) {
  const labels = { shark: "The Shark", diplomat: "The Diplomat", veteran: "The Veteran" };
  return labels[persona] || persona;
}

function _personaSymbol(persona) {
  const syms = { shark: "◈", diplomat: "◎", veteran: "◆" };
  return syms[persona] || "◈";
}

function _personaColor(persona) {
  const cols = { shark: "var(--scarlet-light)", diplomat: "var(--emerald)", veteran: "var(--parlay-purple)" };
  return cols[persona] || "var(--gold)";
}

function _personaTag(persona) {
  const tags = { shark: "Aggressive · Low A", diplomat: "Cooperative · High A", veteran: "Experienced · Patient" };
  return tags[persona] || "Negotiator";
}

function _showError(msg) {
  const el = document.getElementById("global-error");
  if (!el) { _showToast(msg); return; }
  el.textContent = msg;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 4000);
}
