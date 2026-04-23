// ============================================================
// Parlay — Three.js r128 Character System
// Items 28-32: canvas size, per-persona detail, animation,
// state transitions with lerping, persona name plate.
// NO CapsuleGeometry. NO OrbitControls.
// ============================================================

// ── State definitions ─────────────────────────────────────────────────────
const CHARACTER_STATES = {
  idle:       { headTilt: 0,     eyeScale: 1.0, animSpeed: 0.6,  bodyLean: 0,     mouthOpen: 0,   mouthCurve: 0,    breathAmp: 0.025, breathHz: 0.7  },
  thinking:   { headTilt: 0.17,  eyeScale: 0.8, animSpeed: 0.3,  bodyLean: 0,     mouthOpen: 0,   mouthCurve: 0,    breathAmp: 0.020, breathHz: 0.3  },
  aggressive: { headTilt: -0.08, eyeScale: 1.3, animSpeed: 1.5,  bodyLean: -0.08, mouthOpen: 0.4, mouthCurve: -0.2, breathAmp: 0.030, breathHz: 1.8  },
  pleased:    { headTilt: 0.08,  eyeScale: 1.1, animSpeed: 0.8,  bodyLean: 0.03,  mouthOpen: 0.2, mouthCurve: 0.3,  breathAmp: 0.040, breathHz: 0.9  },
  shocked:    { headTilt: 0.25,  eyeScale: 1.6, animSpeed: 2.0,  bodyLean: 0,     mouthOpen: 0.7, mouthCurve: 0,    breathAmp: 0.025, breathHz: 1.5  },
};

// ── Persona definitions — item 29 ─────────────────────────────────────────
const PERSONA_DEFS = {
  shark: {
    suitColor:  0x1a1a2e,  // deep navy
    hairColor:  0x1a1a1a,  // black
    tieColor:   0x8b1a1a,  // scarlet
    skinColor:  0xf0c8a0,
    badge:      "◈",
    badgeColor: 0xc9a84c,
    // Leans forward — predatory posture
    torsoRotX:  -0.08,
    // Eyebrows angled inward for scowl
    browAngleZ: 0.3,
    buildAccessory(group) {
      const mat = new THREE.MeshStandardMaterial({ color: 0x2c1810, roughness: 0.7, metalness: 0.1 });
      const gc  = new THREE.BoxGeometry(0.45, 0.35, 0.12);
      const bc  = new THREE.Mesh(gc, mat);
      bc.position.set(0.65, 0.25, 0.1);
      bc.rotation.z = 0.05;
      group.add(bc);
      const hgeo = new THREE.BoxGeometry(0.12, 0.03, 0.06);
      const handle = new THREE.Mesh(hgeo, new THREE.MeshStandardMaterial({ color: 0xc9a84c, roughness: 0.3, metalness: 0.5 }));
      handle.position.set(0.65, 0.435, 0.1);
      group.add(handle);
      return { briefcase: bc, handle };
    },
  },
  diplomat: {
    suitColor:  0x2a3d28,  // forest green
    hairColor:  0x5c4020,  // warm brown
    tieColor:   0xc9a84c,  // gold
    skinColor:  0xf0c8a0,
    badge:      "◎",
    badgeColor: 0xc9a84c,
    torsoRotX:  0.03,   // slight backward lean — relaxed
    browAngleZ: 0,
    buildAccessory(group) {
      // Pocket square — cream strip in breast pocket
      const mat = new THREE.MeshStandardMaterial({ color: 0xf5efe0, roughness: 0.9 });
      const sq  = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.02), mat);
      sq.position.set(-0.28, 0.5, 0.33);
      group.add(sq);
      return { pocketSquare: sq };
    },
  },
  analyst: {
    suitColor:  0x1a2535,  // charcoal blue
    hairColor:  0x3a3020,  // dark
    tieColor:   0x1a5fa8,  // blue
    skinColor:  0xf0c8a0,
    badge:      "◻",
    badgeColor: 0x3a8fd8,
    torsoRotX:  0.06,   // slight hunch
    browAngleZ: 0,
    buildAccessory(group) {
      const frameMat = new THREE.MeshStandardMaterial({ color: 0x5c4020, roughness: 0.8 });
      const lensGeo  = new THREE.BoxGeometry(0.18, 0.12, 0.02);
      const lensL = new THREE.Mesh(lensGeo, frameMat);
      lensL.position.set(-0.19, 1.49, 0.4);
      group.add(lensL);
      const lensR = new THREE.Mesh(lensGeo, frameMat);
      lensR.position.set(0.19, 1.49, 0.4);
      group.add(lensR);
      // Bridge
      const bridgeGeo = new THREE.BoxGeometry(0.07, 0.02, 0.02);
      const bridge = new THREE.Mesh(bridgeGeo, frameMat);
      bridge.position.set(0, 1.49, 0.4);
      group.add(bridge);
      return { lensL, lensR, bridge };
    },
  },
  wildcard: {
    suitColor:  0x3d2810,  // warm brown
    hairColor:  0xc8a050,  // golden
    tieColor:   0xd08020,  // amber
    skinColor:  0xf0c8a0,
    badge:      "◇",
    badgeColor: 0xd08020,
    torsoRotX:  0,
    browAngleZ: 0,
    // Loose, askew tie — wider (0.18) and slightly rotated
    buildAccessory(group) {
      const tieMat = new THREE.MeshStandardMaterial({ color: 0xd08020, roughness: 0.6 });
      const looseTie = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.72, 0.06), tieMat);
      looseTie.position.set(0.02, 0.1, 0.37);
      looseTie.rotation.z = 0.15;
      group.add(looseTie);
      return { looseTie };
    },
  },
  veteran: {
    suitColor:  0x1a1a1a,  // near-black
    hairColor:  0xd8d8d0,  // silver-white
    tieColor:   0x5c3d9e,  // purple
    skinColor:  0xd8b090,  // slightly older skin
    badge:      "◆",
    badgeColor: 0xc9a84c,
    torsoRotX:  0,        // most upright — never leans
    browAngleZ: 0,
    buildAccessory(group) {
      // Gold cufflinks at wrists
      const cuffMat = new THREE.MeshStandardMaterial({ color: 0xc9a84c, roughness: 0.2, metalness: 0.7 });
      const cuffGeo = new THREE.BoxGeometry(0.08, 0.06, 0.28);
      const cuffL = new THREE.Mesh(cuffGeo, cuffMat);
      cuffL.position.set(-0.72, -0.73, 0);
      group.add(cuffL);
      const cuffR = new THREE.Mesh(cuffGeo, cuffMat);
      cuffR.position.set(0.72, -0.73, 0);
      group.add(cuffR);
      return { cuffL, cuffR };
    },
  },
};

// ── NegotiatorCharacter class — items 28-31 ──────────────────────────────
class NegotiatorCharacter {
  constructor(canvasId, persona = "shark") {
    this.canvasId   = canvasId;
    this.persona    = persona;
    this.state      = "idle";
    this.targetState = CHARACTER_STATES.idle;
    this.clock      = 0;
    this.animFrameId = null;
    this.scene      = null;
    this.camera     = null;
    this.renderer   = null;
    this.meshes     = {};
    this.characterGroup = null;

    // Smooth lerp accumulators
    this._curTilt      = 0;
    this._curEyeScale  = 1.0;
    this._curLean      = 0;
    this._curMouthY    = 1.33;
    this._curMouthScaleY = 1.0;
    this._curArmRot    = 0;

    // Blink state
    this._lastBlink    = 0;
    this._blinkInterval = 3.5;
    this._blinking     = false;
    this._blinkTimer   = 0;

    // Shocked-state timer
    this._shockedUntil    = 0;
    this._postShockState  = "idle";

    // Wildcard random snap
    this._wildcardAngle = 0;
    this._wildcardNextSnap = 0;

    this._init(canvasId);
    this._buildScene();
    this._buildCharacter();
    this._animate();
  }

  // ── item 28 — canvas 280×380, PerspectiveCamera(38), felt background
  _init(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) { console.warn("[Character] Canvas not found:", canvasId); return; }

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1c2b1a);  // felt green

    this.camera = new THREE.PerspectiveCamera(38, 280 / 380, 0.1, 100);
    this.camera.position.set(0, 1.8, 5.8);
    this.camera.lookAt(0, 1.4, 0);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setSize(280, 380);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  }

  // ── item 28 — lighting drama, mahogany table
  _buildScene() {
    if (!this.scene) return;

    // Key — warm brass pendant lamp
    const key = new THREE.DirectionalLight(0xffe8b0, 1.5);
    key.position.set(4, 7, 4);
    key.castShadow = true;
    this.scene.add(key);

    // Rim — cool blue
    const rim = new THREE.DirectionalLight(0x4060a0, 0.4);
    rim.position.set(-4, 3, -3);
    this.scene.add(rim);

    // Under — bounced mahogany light
    const under = new THREE.PointLight(0x8b4010, 0.3);
    under.position.set(0, -0.3, 1.5);
    this.scene.add(under);

    // Ambient
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.35));

    // Mahogany table — item 28
    const tableMat = new THREE.MeshStandardMaterial({ color: 0x3d1f0f, roughness: 0.5, metalness: 0.05 });
    const table = new THREE.Mesh(new THREE.BoxGeometry(8, 0.12, 2.5), tableMat);
    table.position.set(0, -0.55, 0);
    table.receiveShadow = true;
    this.scene.add(table);

    // Felt runner on table
    const runnerMat = new THREE.MeshStandardMaterial({ color: 0x1c2b1a, roughness: 0.9 });
    const runner = new THREE.Mesh(new THREE.BoxGeometry(8, 0.02, 0.8), runnerMat);
    runner.position.set(0, -0.48, 0);
    this.scene.add(runner);

    // Table edge gold strip
    const edgeMat = new THREE.MeshStandardMaterial({ color: 0xc9a84c, roughness: 0.3, metalness: 0.5 });
    const edge = new THREE.Mesh(new THREE.BoxGeometry(8, 0.02, 0.04), edgeMat);
    edge.position.set(0, -0.49, 1.25);
    this.scene.add(edge);
  }

  _mat(color, roughness = 0.7, metalness = 0.05) {
    return new THREE.MeshStandardMaterial({ color, roughness, metalness });
  }

  _clearCharacter() {
    if (this.characterGroup && this.scene) this.scene.remove(this.characterGroup);
    this.characterGroup = null;
    this.meshes = {};
  }

  // ── item 29 — per-persona detail ─────────────────────────────────────────
  _buildCharacter() {
    this._clearCharacter();
    if (!this.scene) return;

    const def   = PERSONA_DEFS[this.persona] || PERSONA_DEFS.shark;
    const group = new THREE.Group();
    this.characterGroup = group;
    this.scene.add(group);

    const skin  = this._mat(def.skinColor, 0.85, 0.0);
    const suit  = this._mat(def.suitColor, 0.75, 0.05);
    const tie   = this._mat(def.tieColor,  0.55, 0.05);
    const hair  = this._mat(def.hairColor, 0.85, 0.0);
    const shirt = this._mat(0xf5efe0, 0.9, 0.0);
    const lapel = this._mat(Math.max(0, def.suitColor - 0x050505), 0.8, 0.0);

    // ── TORSO
    const torso = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.3, 0.6), suit);
    torso.castShadow = true;
    torso.rotation.x = def.torsoRotX || 0;
    group.add(torso);
    this.meshes.torso = torso;

    // Shirt front
    const shirtMesh = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.9, 0.32), shirt);
    shirtMesh.position.set(0, 0.1, 0.32);
    group.add(shirtMesh);

    // Main tie (slim) — wildcard gets wider from accessory
    if (this.persona !== "wildcard") {
      const tieMesh = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.74, 0.06), tie);
      tieMesh.position.set(0, 0.1, 0.36);
      group.add(tieMesh);
      this.meshes.tie = tieMesh;
    }

    // Lapels
    const lapelGeo = new THREE.BoxGeometry(0.22, 0.7, 0.08);
    const lapelL = new THREE.Mesh(lapelGeo, lapel);
    lapelL.position.set(-0.22, 0.25, 0.3); lapelL.rotation.z = 0.2;
    group.add(lapelL);
    const lapelR = new THREE.Mesh(lapelGeo, lapel);
    lapelR.position.set(0.22, 0.25, 0.3); lapelR.rotation.z = -0.2;
    group.add(lapelR);

    // Badge
    const badgeTex = this._badgeTex(def.badge, def.badgeColor, 0xffffff);
    const badge = new THREE.Mesh(
      new THREE.BoxGeometry(0.2, 0.2, 0.04),
      new THREE.MeshStandardMaterial({ map: badgeTex, roughness: 0.4 })
    );
    badge.position.set(-0.27, 0.4, 0.32);
    group.add(badge);

    // ── NECK
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.15, 0.3, 8), skin);
    neck.position.set(0, 0.8, 0);
    group.add(neck);

    // ── HEAD
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.85, 0.65), skin);
    head.scale.set(1, 1.05, 0.95);
    head.position.set(0, 1.45, 0);
    head.castShadow = true;
    group.add(head);
    this.meshes.head = head;

    // ── HAIR (veteran gets silver)
    const hairGeoTop = this.persona === "veteran"
      ? new THREE.BoxGeometry(0.76, 0.22, 0.66)
      : new THREE.BoxGeometry(0.78, 0.26, 0.68);
    const hairTop = new THREE.Mesh(hairGeoTop, hair);
    hairTop.position.set(0, 1.82, -0.02);
    group.add(hairTop);

    const hSideGeo = new THREE.BoxGeometry(0.12, 0.44, 0.6);
    const hL = new THREE.Mesh(hSideGeo, hair);
    hL.position.set(-0.37, 1.6, -0.02); group.add(hL);
    const hR = new THREE.Mesh(hSideGeo, hair);
    hR.position.set(0.37, 1.6, -0.02);  group.add(hR);

    // ── EYES
    const eyeW = new THREE.SphereGeometry(0.095, 12, 8);
    const eyeP = new THREE.SphereGeometry(0.052, 8, 6);
    const eyeWhite = this._mat(0xffffff, 0.95, 0.0);
    const pupil    = this._mat(0x111111, 0.9, 0.0);

    const eyeLW = new THREE.Mesh(eyeW, eyeWhite);
    eyeLW.position.set(-0.19, 1.50, 0.33); group.add(eyeLW);
    this.meshes.eyeLeft = eyeLW;

    const eyeRW = new THREE.Mesh(eyeW, eyeWhite);
    eyeRW.position.set(0.19, 1.50, 0.33); group.add(eyeRW);
    this.meshes.eyeRight = eyeRW;

    const eyeLP = new THREE.Mesh(eyeP, pupil);
    eyeLP.position.set(-0.19, 1.50, 0.38); group.add(eyeLP);
    this.meshes.pupilLeft = eyeLP;

    const eyeRP = new THREE.Mesh(eyeP, pupil);
    eyeRP.position.set(0.19, 1.50, 0.38); group.add(eyeRP);
    this.meshes.pupilRight = eyeRP;

    // ── EYEBROWS — veteran gets thicker; shark angled for scowl
    const browH   = this.persona === "veteran" ? 0.055 : 0.04;
    const browGeo = new THREE.BoxGeometry(0.18, browH, 0.04);
    const browMat = this._mat(def.hairColor === 0xd8d8d0 ? 0x888878 : def.hairColor, 0.9, 0.0);
    const browL = new THREE.Mesh(browGeo, browMat);
    browL.position.set(-0.19, 1.63, 0.33);
    browL.rotation.z = def.browAngleZ || 0;
    group.add(browL); this.meshes.browL = browL;
    const browR = new THREE.Mesh(browGeo, browMat);
    browR.position.set(0.19, 1.63, 0.33);
    browR.rotation.z = -(def.browAngleZ || 0);
    group.add(browR); this.meshes.browR = browR;

    // ── MOUTH — diplomat gets slight upward curve
    const mouthMat = this._mat(0x8b3a3a, 0.8, 0.0);
    const mouth = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.04, 0.04), mouthMat);
    mouth.position.set(0, 1.33, 0.33);
    if (this.persona === "diplomat") mouth.rotation.z = 0.15;
    group.add(mouth); this.meshes.mouth = mouth;

    // ── NOSE
    const nose = new THREE.Mesh(
      new THREE.BoxGeometry(0.09, 0.09, 0.12),
      this._mat(def.skinColor * 0.95 | 0, 0.85, 0.0)
    );
    nose.position.set(0, 1.44, 0.36);
    group.add(nose);

    // ── SHOULDERS + ARMS
    const shlGeo = new THREE.BoxGeometry(0.32, 0.28, 0.45);
    const shlL = new THREE.Mesh(shlGeo, suit); shlL.position.set(-0.7, 0.5, 0); group.add(shlL);
    const shlR = new THREE.Mesh(shlGeo, suit); shlR.position.set(0.7, 0.5, 0);  group.add(shlR);

    const uArmGeo = new THREE.BoxGeometry(0.28, 0.6, 0.3);
    const uArmL = new THREE.Mesh(uArmGeo, suit); uArmL.position.set(-0.72, 0.05, 0); group.add(uArmL); this.meshes.upperArmL = uArmL;
    const uArmR = new THREE.Mesh(uArmGeo, suit); uArmR.position.set(0.72, 0.05, 0);  group.add(uArmR); this.meshes.upperArmR = uArmR;

    const lArmL = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.55, 0.26), suit); lArmL.position.set(-0.72, -0.5, 0); group.add(lArmL);
    const lArmR = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.55, 0.26), suit); lArmR.position.set(0.72, -0.5, 0);  group.add(lArmR);
    this.meshes.lowerArmL = lArmL;
    this.meshes.lowerArmR = lArmR;

    // Hands
    const handL = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 6), skin);
    handL.position.set(-0.72, -0.85, 0); group.add(handL);
    const handR = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 6), skin);
    handR.position.set(0.72, -0.85, 0);  group.add(handR);

    // ── LEGS
    const pelvis = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.3, 0.55), suit);
    pelvis.position.set(0, -0.8, 0); group.add(pelvis);
    const legL = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.5, 0.38), suit); legL.position.set(-0.3, -1.15, 0); group.add(legL);
    const legR = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.5, 0.38), suit); legR.position.set(0.3, -1.15, 0);  group.add(legR);

    // ── PERSONA ACCESSORY — item 29
    if (typeof def.buildAccessory === "function") {
      this.meshes.accessory = def.buildAccessory(group);
    }

    this.meshes._isVeteran  = this.persona === "veteran";
    this.meshes._isWildcard = this.persona === "wildcard";
    this.meshes._isShark    = this.persona === "shark";
    this.meshes._isAnalyst  = this.persona === "analyst";

    this._wildcardAngle    = 0;
    this._wildcardNextSnap = performance.now() + 3000 + Math.random() * 2000;

    group.position.set(0, 0, 0);
    this._eyeBaseY = 1.50;
  }

  _badgeTex(symbol, bgColor, fgColor) {
    const size = 128;
    const cvs  = document.createElement("canvas");
    cvs.width = size; cvs.height = size;
    const ctx  = cvs.getContext("2d");
    ctx.fillStyle = `#${bgColor.toString(16).padStart(6, "0")}`;
    ctx.beginPath();
    ctx.roundRect(16, 16, size - 32, size - 32, 12);
    ctx.fill();
    ctx.fillStyle = `#${fgColor.toString(16).padStart(6, "0")}`;
    ctx.font = "bold 58px serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(symbol, size / 2, size / 2 + 3);
    return new THREE.CanvasTexture(cvs);
  }

  // ── Public API ──────────────────────────────────────────────────────────
  setState(newState) {
    if (!(newState in CHARACTER_STATES)) return;
    if (newState === "shocked") {
      this._shockedUntil   = performance.now() + 1500;
      this._postShockState = this.state === "shocked" ? "idle" : this.state;
    }
    this.state       = newState;
    this.targetState = CHARACTER_STATES[newState];
    this._updateBadge(newState);
  }

  setPersona(persona) {
    if (!(persona in PERSONA_DEFS)) return;
    this.persona = persona;
    this._buildCharacter();
  }

  destroy() {
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    this._clearCharacter();
    if (this.renderer) this.renderer.dispose();
  }

  // ── Private ───────────────────────────────────────────────────────────────
  _updateBadge(state) {
    const b = document.querySelector(".character-state-badge");
    if (b) b.textContent = state;
    const chip = document.getElementById("character-state-label");
    if (chip) chip.textContent = state;
  }

  _lerp(a, b, t) { return a + (b - a) * t; }

  // ── item 30 — idle/state animations ─────────────────────────────────────
  _animate() {
    this.animFrameId = requestAnimationFrame(() => this._animate());
    if (!this.scene || !this.camera || !this.renderer) return;

    const now = performance.now();
    this.clock += 0.016;  // ~60fps tick

    // Auto-revert shocked after 1.5s → aggressive
    if (this.state === "shocked" && now > this._shockedUntil) {
      this.setState(this._postShockState || "idle");
    }

    if (!this.characterGroup) { this.renderer.render(this.scene, this.camera); return; }

    const g      = this.characterGroup;
    const target = this.targetState;
    const isVet  = !!this.meshes._isVeteran;

    // ── Breathing — item 30 ──────────────────────────────────────────────
    const breathAmp = isVet ? 0.01 : target.breathAmp;
    g.position.y = Math.sin(this.clock * target.breathHz) * breathAmp;

    // ── Head ─────────────────────────────────────────────────────────────
    if (this.meshes.head) {
      const h = this.meshes.head;

      // Subtle side-to-side swing — item 30 idle
      const headSwing = Math.sin(this.clock * 0.22) * 0.06;
      h.rotation.y = headSwing;

      // Forward tilt — lerp to target (thinking leans forward)
      const forwardTarget = this.state === "thinking" ? 0.18 : 0;
      h.rotation.x = this._lerp(h.rotation.x, forwardTarget, 0.05);

      // Z-tilt (headTilt) — item 31 lerp
      this._curTilt = this._lerp(this._curTilt, target.headTilt, 0.08);
      h.rotation.z = this._curTilt;

      // Shocked: rapid head shake — item 30
      if (this.state === "shocked") {
        h.rotation.z = Math.sin(this.clock * 8) * 0.15;
      }

      // Wildcard — random micro-snap — item 29
      if (this.meshes._isWildcard && this.state === "idle") {
        if (now > this._wildcardNextSnap) {
          this._wildcardAngle    = (Math.random() - 0.5) * 0.3;
          this._wildcardNextSnap = now + 3000 + Math.random() * 2000;
        }
        h.rotation.z = this._lerp(h.rotation.z, this._wildcardAngle, 0.08);
      }
    }

    // ── Body lean — item 31 lerp, veteran never leans ─────────────────────
    const targetLean = isVet ? 0 : target.bodyLean;
    this._curLean = this._lerp(this._curLean, targetLean, 0.04);
    g.rotation.x  = this._curLean;

    // Aggressive shark — whole group leans forward — item 29
    if (this.meshes._isShark && this.state === "aggressive") {
      g.position.z = this._lerp(g.position.z, 0.3, 0.05);
    } else {
      g.position.z = this._lerp(g.position.z, 0, 0.05);
    }

    // ── Eye scale — item 31 lerp ──────────────────────────────────────────
    this._curEyeScale = this._lerp(this._curEyeScale, target.eyeScale, 0.06);
    const es = this._curEyeScale;
    if (this.meshes.eyeLeft)  this.meshes.eyeLeft.scale.setScalar(es);
    if (this.meshes.eyeRight) this.meshes.eyeRight.scale.setScalar(es);

    // ── Eye blink — item 30 ───────────────────────────────────────────────
    const t = this.clock;
    if (!this._blinking && t - this._lastBlink > this._blinkInterval) {
      this._blinking    = true;
      this._blinkTimer  = 0;
      this._lastBlink   = t;
      this._blinkInterval = 3 + Math.random() * 2;
    }
    if (this._blinking) {
      this._blinkTimer += 0.016;
      const blinkScale = this._blinkTimer < 0.04 ? 0.1 : 1.0;
      if (this.meshes.eyeLeft)  this.meshes.eyeLeft.scale.y  = this._curEyeScale * blinkScale;
      if (this.meshes.eyeRight) this.meshes.eyeRight.scale.y = this._curEyeScale * blinkScale;
      if (this._blinkTimer > 0.08) this._blinking = false;
    }

    // ── Pupils wander ─────────────────────────────────────────────────────
    if (this.meshes.pupilLeft && this.meshes.pupilRight) {
      const wx = Math.sin(this.clock * 0.7) * 0.025;
      const wy = Math.cos(this.clock * 0.5) * 0.015;
      this.meshes.pupilLeft.position.x  = -0.19 + wx;
      this.meshes.pupilRight.position.x =  0.19 + wx;
      this.meshes.pupilLeft.position.y  = this._eyeBaseY + wy;
      this.meshes.pupilRight.position.y = this._eyeBaseY + wy;
    }

    // ── Eyebrows ──────────────────────────────────────────────────────────
    if (this.meshes.browL && this.meshes.browR) {
      const browDelta = this.state === "aggressive" ? -0.1
                      : this.state === "shocked"    ?  0.12
                      : this.state === "pleased"    ?  0.04
                      : 0;
      const tY = 1.63 + browDelta;
      this.meshes.browL.position.y = this._lerp(this.meshes.browL.position.y, tY, 0.07);
      this.meshes.browR.position.y = this._lerp(this.meshes.browR.position.y, tY, 0.07);

      // Aggressive scowl on browZ — shark has inherent browAngleZ already
      const browZAdd = this.state === "aggressive" ? 0.15 : 0;
      const def = PERSONA_DEFS[this.persona] || PERSONA_DEFS.shark;
      this.meshes.browL.rotation.z = this._lerp(this.meshes.browL.rotation.z,  (def.browAngleZ || 0) + browZAdd, 0.07);
      this.meshes.browR.rotation.z = this._lerp(this.meshes.browR.rotation.z, -(def.browAngleZ || 0) - browZAdd, 0.07);
    }

    // ── Mouth morph — item 31 lerp ────────────────────────────────────────
    if (this.meshes.mouth) {
      this.meshes.mouth.scale.y = this._lerp(this.meshes.mouth.scale.y, 1 + target.mouthOpen * 2.2, 0.07);
      const targetMY = 1.33 - target.mouthOpen * 0.04 + target.mouthCurve * 0.04;
      this.meshes.mouth.position.y = this._lerp(this.meshes.mouth.position.y, targetMY, 0.07);
      // Pleased — mouth curves up
      if (this.state === "pleased") {
        this.meshes.mouth.rotation.z = this._lerp(this.meshes.mouth.rotation.z, 0.25, 0.05);
      } else {
        const def = PERSONA_DEFS[this.persona] || PERSONA_DEFS.shark;
        const baseCurve = this.persona === "diplomat" ? 0.15 : 0;
        this.meshes.mouth.rotation.z = this._lerp(this.meshes.mouth.rotation.z, baseCurve, 0.05);
      }
    }

    // ── Arms ──────────────────────────────────────────────────────────────
    if (this.meshes.upperArmL && this.meshes.upperArmR) {
      const swayAmp  = this.state === "aggressive" ? 0.07 : 0.015;
      const swayFreq = this.state === "aggressive" ? 1.5  : 0.35;
      const sway = Math.sin(this.clock * swayFreq) * swayAmp;
      this.meshes.upperArmL.rotation.z = this._lerp(this.meshes.upperArmL.rotation.z,  sway, 0.1);
      this.meshes.upperArmR.rotation.z = this._lerp(this.meshes.upperArmR.rotation.z, -sway, 0.1);

      // Thinking — right arm rises — item 30
      if (this.state === "thinking" && this.meshes._isAnalyst) {
        this.meshes.upperArmR.rotation.x = this._lerp(this.meshes.upperArmR.rotation.x, -0.3, 0.06);
      } else {
        this.meshes.upperArmR.rotation.x = this._lerp(this.meshes.upperArmR.rotation.x, 0, 0.06);
      }

      // Shocked — both arms raise — item 30 (wildcard)
      if (this.state === "shocked") {
        this.meshes.upperArmL.rotation.x = this._lerp(this.meshes.upperArmL.rotation.x, -0.6, 0.08);
        this.meshes.upperArmR.rotation.x = this._lerp(this.meshes.upperArmR.rotation.x, -0.6, 0.08);
      } else {
        this.meshes.upperArmL.rotation.x = this._lerp(this.meshes.upperArmL.rotation.x, 0, 0.05);
        if (this.state !== "thinking") {
          this.meshes.upperArmR.rotation.x = this._lerp(this.meshes.upperArmR.rotation.x, 0, 0.05);
        }
      }
    }

    this.renderer.render(this.scene, this.camera);
  }
}

// ── PersonaPreviewCharacter (mini, for onboarding cards) ──────────────────
class PersonaPreviewCharacter {
  constructor(canvasEl, persona = "shark") {
    this.persona  = persona;
    this.clock    = 0;
    this.animId   = null;
    this.scene    = null; this.camera = null; this.renderer = null;
    this.group    = null;
    this._setup(canvasEl);
    this._build();
    this._animate();
  }

  _setup(canvas) {
    const w = canvas.width  || 280;
    const h = canvas.height || 200;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1c2b1a);
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
    this.camera.position.set(0, 1.4, 4.5);
    this.camera.lookAt(0, 1.1, 0);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setSize(w, h, false);
    const key = new THREE.DirectionalLight(0xffe8b0, 1.3);
    key.position.set(2, 5, 3); this.scene.add(key);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const t = new THREE.Mesh(
      new THREE.BoxGeometry(8, 0.08, 2),
      new THREE.MeshStandardMaterial({ color: 0x3d1f0f, roughness: 0.5 })
    );
    t.position.y = -0.8; this.scene.add(t);
  }

  _build() {
    if (!this.scene) return;
    if (this.group) this.scene.remove(this.group);
    const def  = PERSONA_DEFS[this.persona] || PERSONA_DEFS.shark;
    const g    = new THREE.Group();
    this.group = g; this.scene.add(g);
    const mat = (c, r = 0.75) => new THREE.MeshStandardMaterial({ color: c, roughness: r, metalness: 0.04 });
    const torso = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.3, 0.6), mat(def.suitColor));
    g.add(torso);
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.85, 0.65), mat(def.skinColor, 0.85));
    head.position.set(0, 1.45, 0); g.add(head); this._head = head;
    const hTop = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.26, 0.68), mat(def.hairColor, 0.85));
    hTop.position.set(0, 1.82, -0.02); g.add(hTop);
    const ew = new THREE.SphereGeometry(0.09, 10, 7);
    const ep = new THREE.SphereGeometry(0.05, 8, 6);
    const eWM = mat(0xffffff, 0.95); const ePM = mat(0x111111, 0.9);
    [-0.19, 0.19].forEach(x => {
      const ew_ = new THREE.Mesh(ew, eWM); ew_.position.set(x, 1.5, 0.33); g.add(ew_);
      const ep_ = new THREE.Mesh(ep, ePM); ep_.position.set(x, 1.5, 0.38); g.add(ep_);
    });
  }

  _animate() {
    this.animId = requestAnimationFrame(() => this._animate());
    if (!this.scene || !this.renderer || !this.camera) return;
    this.clock += 0.016 * 0.5;
    if (this.group) this.group.position.y = Math.sin(this.clock) * 0.01;
    if (this._head) this._head.rotation.y = Math.sin(this.clock * 0.25) * 0.03;
    this.renderer.render(this.scene, this.camera);
  }

  destroy() {
    if (this.animId) cancelAnimationFrame(this.animId);
    if (this.group && this.scene) this.scene.remove(this.group);
    if (this.renderer) this.renderer.dispose();
  }
}

// ── Global exports ─────────────────────────────────────────────────────────
window.NegotiatorCharacter    = NegotiatorCharacter;
window.PersonaPreviewCharacter = PersonaPreviewCharacter;
window.PERSONA_DEFS           = PERSONA_DEFS;
