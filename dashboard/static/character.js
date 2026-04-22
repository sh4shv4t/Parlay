// ============================================================
// Parlay — Three.js r128 Character System
// "The Deal Room" — 1960s Madison Avenue boardroom aesthetic.
// Loaded AFTER Three.js r128 from cdnjs.
// NO CapsuleGeometry. NO OrbitControls.
// ============================================================

// ── State definitions ─────────────────────────────────────────────────────
const CHARACTER_STATES = {
  idle:       { headTilt: 0,     eyeScale: 1.0, animSpeed: 0.6,  bodyLean: 0,     mouthOpen: 0,   mouthCurve: 0    },
  thinking:   { headTilt: 0.17,  eyeScale: 0.8, animSpeed: 0.3,  bodyLean: 0,     mouthOpen: 0,   mouthCurve: 0    },
  aggressive: { headTilt: -0.08, eyeScale: 1.3, animSpeed: 1.5,  bodyLean: -0.08, mouthOpen: 0.4, mouthCurve: -0.2 },
  pleased:    { headTilt: 0.08,  eyeScale: 1.1, animSpeed: 0.8,  bodyLean: 0.03,  mouthOpen: 0.2, mouthCurve: 0.3  },
  shocked:    { headTilt: 0.25,  eyeScale: 1.5, animSpeed: 2.0,  bodyLean: 0,     mouthOpen: 0.7, mouthCurve: 0    },
};

// ── Persona definitions (1960s boardroom) ─────────────────────────────────
const PERSONA_DEFS = {
  shark: {
    suitColor:  0x1a1a2e,  // deep navy
    hairColor:  0x1a1a1a,  // black
    tieColor:   0x8b1a1a,  // scarlet
    skinColor:  0xf0c8a0,
    badge:      "◈",
    badgeColor: 0xc9a84c,
    // accessory: slim briefcase left hand
    buildAccessory(group, mats) {
      const geo = new THREE.BoxGeometry(0.2, 0.28, 0.06);
      const mat = new THREE.MeshStandardMaterial({ color: 0x2c1810, roughness: 0.7, metalness: 0.1 });
      const briefcase = new THREE.Mesh(geo, mat);
      briefcase.position.set(-0.82, -0.75, 0.04);
      group.add(briefcase);
      // handle
      const hgeo = new THREE.BoxGeometry(0.08, 0.03, 0.04);
      const handle = new THREE.Mesh(hgeo, new THREE.MeshStandardMaterial({ color: 0xc9a84c, roughness: 0.4, metalness: 0.4 }));
      handle.position.set(-0.82, -0.59, 0.04);
      group.add(handle);
      return { briefcase, handle };
    },
  },
  diplomat: {
    suitColor:  0x2a3d28,  // dark forest
    hairColor:  0x5c4020,  // brown
    tieColor:   0xc9a84c,  // gold
    skinColor:  0xf0c8a0,
    badge:      "◎",
    badgeColor: 0xc9a84c,
    // accessory: pocket watch fob (squashed sphere)
    buildAccessory(group, mats) {
      const geo = new THREE.SphereGeometry(0.06, 8, 6);
      const mat = new THREE.MeshStandardMaterial({ color: 0xc9a84c, roughness: 0.3, metalness: 0.6 });
      const watch = new THREE.Mesh(geo, mat);
      watch.scale.set(1, 0.5, 0.5);
      watch.position.set(0.38, 0.35, 0.34);
      group.add(watch);
      // chain
      const cgeo = new THREE.BoxGeometry(0.02, 0.12, 0.02);
      const chain = new THREE.Mesh(cgeo, mat);
      chain.position.set(0.38, 0.44, 0.33);
      group.add(chain);
      return { watch, chain };
    },
  },
  analyst: {
    suitColor:  0x1a2535,  // charcoal blue
    hairColor:  0x3a3020,  // dark
    tieColor:   0x1a5fa8,  // blue
    skinColor:  0xf0c8a0,
    badge:      "◻",
    badgeColor: 0x3a8fd8,
    // accessory: glasses frames
    buildAccessory(group, mats) {
      const frameMat = new THREE.MeshStandardMaterial({ color: 0x2c1810, roughness: 0.8 });
      const lensGeo = new THREE.BoxGeometry(0.16, 0.1, 0.02);
      const lensL = new THREE.Mesh(lensGeo, frameMat);
      lensL.position.set(-0.19, 1.49, 0.4);
      group.add(lensL);
      const lensR = new THREE.Mesh(lensGeo, frameMat);
      lensR.position.set(0.19, 1.49, 0.4);
      group.add(lensR);
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
    // accessory: wide loose tie (wider geo, slight rotation)
    buildAccessory(group, mats) {
      const tieGeo = new THREE.BoxGeometry(0.22, 0.72, 0.06);
      const tieMat = new THREE.MeshStandardMaterial({ color: 0xd08020, roughness: 0.6 });
      const looseTie = new THREE.Mesh(tieGeo, tieMat);
      looseTie.position.set(0.02, 0.1, 0.37);
      looseTie.rotation.z = 0.05;
      group.add(looseTie);
      return { looseTie };
    },
  },
  veteran: {
    suitColor:  0x1a1a1a,  // near-black charcoal
    hairColor:  0xe0e0d8,  // silver-white
    tieColor:   0x5c3d9e,  // deep purple
    skinColor:  0xd8b090,  // slightly older skin tone
    badge:      "◆",
    badgeColor: 0xc9a84c,
    // accessory: gold cufflinks at wrist
    buildAccessory(group, mats) {
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

// ── NegotiatorCharacter class ─────────────────────────────────────────────
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

    // Lerp accumulators
    this._curTilt     = 0;
    this._curEyeScale = 1.0;
    this._curLean     = 0;

    // Shocked-state timer
    this._shockedUntil = 0;

    this._init(canvasId);
    this._buildScene();
    this._buildCharacter();
    this._animate();
  }

  _init(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
      console.warn("[Character] Canvas not found:", canvasId);
      return;
    }

    this.scene = new THREE.Scene();
    // Felt-green boardroom background
    this.scene.background = new THREE.Color(0x1c2b1a);

    this.camera = new THREE.PerspectiveCamera(45, 280 / 380, 0.1, 100);
    this.camera.position.set(0, 1.6, 5.2);
    this.camera.lookAt(0, 1.3, 0);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setSize(280, 380);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  }

  _buildScene() {
    if (!this.scene) return;

    // Warm key light — brass pendant lamp
    const key = new THREE.DirectionalLight(0xffe8b0, 1.4);
    key.position.set(3, 6, 4);
    key.castShadow = true;
    this.scene.add(key);

    // Cool rim light
    const rim = new THREE.DirectionalLight(0x8090c0, 0.3);
    rim.position.set(-3, 2, -2);
    this.scene.add(rim);

    // Ambient
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambient);

    // Mahogany table surface
    const tableGeo = new THREE.BoxGeometry(10, 0.1, 3);
    const tableMat = new THREE.MeshStandardMaterial({ color: 0x2c1810, roughness: 0.5, metalness: 0.05 });
    const table = new THREE.Mesh(tableGeo, tableMat);
    table.position.set(0, -0.8, 0);
    table.receiveShadow = true;
    this.scene.add(table);

    // Table edge highlight (thin gold strip)
    const edgeGeo = new THREE.BoxGeometry(10, 0.02, 0.04);
    const edgeMat = new THREE.MeshStandardMaterial({ color: 0xc9a84c, roughness: 0.3, metalness: 0.5 });
    const edge = new THREE.Mesh(edgeGeo, edgeMat);
    edge.position.set(0, -0.745, 1.5);
    this.scene.add(edge);
  }

  _mat(color, roughness = 0.7, metalness = 0.05) {
    return new THREE.MeshStandardMaterial({ color, roughness, metalness });
  }

  _badgeTex(symbol, bgColor, fgColor) {
    const size = 128;
    const cvs  = document.createElement("canvas");
    cvs.width  = size; cvs.height = size;
    const ctx  = cvs.getContext("2d");

    ctx.fillStyle = `#${bgColor.toString(16).padStart(6, "0")}`;
    ctx.beginPath();
    ctx.roundRect(16, 16, size - 32, size - 32, 12);
    ctx.fill();

    ctx.fillStyle = `#${fgColor.toString(16).padStart(6, "0")}`;
    ctx.font = "bold 58px serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(symbol, size / 2, size / 2 + 3);

    return new THREE.CanvasTexture(cvs);
  }

  _clearCharacter() {
    if (this.characterGroup) {
      this.scene && this.scene.remove(this.characterGroup);
    }
    this.characterGroup = null;
    this.meshes = {};
  }

  _buildCharacter() {
    this._clearCharacter();
    if (!this.scene) return;

    const def = PERSONA_DEFS[this.persona] || PERSONA_DEFS.shark;
    const group = new THREE.Group();
    this.characterGroup = group;
    this.scene.add(group);

    const skin = this._mat(def.skinColor, 0.85, 0.0);
    const suit = this._mat(def.suitColor, 0.75, 0.05);
    const tie  = this._mat(def.tieColor,  0.55, 0.05);
    const hair = this._mat(def.hairColor, 0.85, 0.0);
    const shirt= this._mat(0xf5efe0, 0.9, 0.0);
    const lapel= this._mat(Math.max(0, def.suitColor - 0x050505), 0.8, 0.0);

    // ── TORSO ──────────────────────────────────────────────
    const torso = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.3, 0.6), suit);
    torso.castShadow = true;
    group.add(torso);
    this.meshes.torso = torso;

    // Shirt front
    const shirtMesh = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.9, 0.32), shirt);
    shirtMesh.position.set(0, 0.1, 0.32);
    group.add(shirtMesh);

    // Tie
    const tieMesh = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.74, 0.06), tie);
    tieMesh.position.set(0, 0.1, 0.36);
    group.add(tieMesh);
    this.meshes.tie = tieMesh;

    // Lapels
    const lapelGeo = new THREE.BoxGeometry(0.22, 0.7, 0.08);
    const lapelL = new THREE.Mesh(lapelGeo, lapel);
    lapelL.position.set(-0.22, 0.25, 0.3); lapelL.rotation.z = 0.2;
    group.add(lapelL);
    const lapelR = new THREE.Mesh(lapelGeo, lapel);
    lapelR.position.set(0.22, 0.25, 0.3); lapelR.rotation.z = -0.2;
    group.add(lapelR);

    // Badge on left breast
    const badgeTex = this._badgeTex(def.badge, def.badgeColor, 0xffffff);
    const badge = new THREE.Mesh(
      new THREE.BoxGeometry(0.2, 0.2, 0.04),
      new THREE.MeshStandardMaterial({ map: badgeTex, roughness: 0.4 })
    );
    badge.position.set(-0.27, 0.4, 0.32);
    group.add(badge);

    // Analyst: slightly hunched torso
    if (this.persona === "analyst") torso.rotation.x = 0.05;

    // ── NECK ───────────────────────────────────────────────
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.15, 0.3, 8), skin);
    neck.position.set(0, 0.8, 0);
    group.add(neck);

    // ── HEAD ───────────────────────────────────────────────
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.85, 0.65), skin);
    head.scale.set(1, 1.05, 0.95);
    head.position.set(0, 1.45, 0);
    head.castShadow = true;
    group.add(head);
    this.meshes.head = head;

    // Hair top slab
    const hairTop = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.26, 0.68), hair);
    hairTop.position.set(0, 1.82, -0.02);
    group.add(hairTop);

    // Hair sides
    const hSideGeo = new THREE.BoxGeometry(0.12, 0.44, 0.6);
    const hL = new THREE.Mesh(hSideGeo, hair);
    hL.position.set(-0.37, 1.6, -0.02);
    group.add(hL);
    const hR = new THREE.Mesh(hSideGeo, hair);
    hR.position.set(0.37, 1.6, -0.02);
    group.add(hR);

    // ── EYES ───────────────────────────────────────────────
    const eyeW = new THREE.SphereGeometry(0.095, 12, 8);
    const eyeP = new THREE.SphereGeometry(0.052, 8, 6);
    const eyeWhite = this._mat(0xffffff, 0.95, 0.0);
    const pupil    = this._mat(0x111111, 0.9, 0.0);

    const eyeLW = new THREE.Mesh(eyeW, eyeWhite);
    eyeLW.position.set(-0.19, 1.50, 0.33);
    group.add(eyeLW);
    this.meshes.eyeLeft = eyeLW;

    const eyeRW = new THREE.Mesh(eyeW, eyeWhite);
    eyeRW.position.set(0.19, 1.50, 0.33);
    group.add(eyeRW);
    this.meshes.eyeRight = eyeRW;

    const eyeLP = new THREE.Mesh(eyeP, pupil);
    eyeLP.position.set(-0.19, 1.50, 0.38);
    group.add(eyeLP);
    this.meshes.pupilLeft = eyeLP;

    const eyeRP = new THREE.Mesh(eyeP, pupil);
    eyeRP.position.set(0.19, 1.50, 0.38);
    group.add(eyeRP);
    this.meshes.pupilRight = eyeRP;

    // Eyebrows (Veteran gets thicker)
    const browH = this.persona === "veteran" ? 0.055 : 0.04;
    const browGeo = new THREE.BoxGeometry(0.18, browH, 0.04);
    const browMat = this._mat(def.hairColor === 0xe0e0d8 ? 0x888878 : def.hairColor, 0.9, 0.0);
    const browL = new THREE.Mesh(browGeo, browMat);
    browL.position.set(-0.19, 1.63, 0.33);
    group.add(browL);
    this.meshes.browL = browL;
    const browR = new THREE.Mesh(browGeo, browMat);
    browR.position.set(0.19, 1.63, 0.33);
    group.add(browR);
    this.meshes.browR = browR;

    // Mouth
    const mouthMat = this._mat(0x8b3a3a, 0.8, 0.0);
    const mouth = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.04, 0.04), mouthMat);
    mouth.position.set(0, 1.33, 0.33);
    group.add(mouth);
    this.meshes.mouth = mouth;

    // Nose
    const nose = new THREE.Mesh(
      new THREE.BoxGeometry(0.09, 0.09, 0.12),
      this._mat(def.skinColor * 0.95 | 0, 0.85, 0.0)
    );
    nose.position.set(0, 1.44, 0.36);
    group.add(nose);

    // ── SHOULDERS / ARMS ───────────────────────────────────
    const shlGeo = new THREE.BoxGeometry(0.32, 0.28, 0.45);
    const shlL = new THREE.Mesh(shlGeo, suit); shlL.position.set(-0.7, 0.5, 0); group.add(shlL);
    const shlR = new THREE.Mesh(shlGeo, suit); shlR.position.set(0.7, 0.5, 0);  group.add(shlR);

    const uArmGeo = new THREE.BoxGeometry(0.28, 0.6, 0.3);
    const uArmL = new THREE.Mesh(uArmGeo, suit);
    uArmL.position.set(-0.72, 0.05, 0); group.add(uArmL);
    this.meshes.upperArmL = uArmL;
    const uArmR = new THREE.Mesh(uArmGeo, suit);
    uArmR.position.set(0.72, 0.05, 0);  group.add(uArmR);
    this.meshes.upperArmR = uArmR;

    const lArmGeo = new THREE.BoxGeometry(0.24, 0.55, 0.26);
    const lArmL = new THREE.Mesh(lArmGeo, suit); lArmL.position.set(-0.72, -0.5, 0); group.add(lArmL);
    const lArmR = new THREE.Mesh(lArmGeo, suit); lArmR.position.set(0.72, -0.5, 0);  group.add(lArmR);

    // Hands
    const handGeo = new THREE.SphereGeometry(0.14, 8, 6);
    const handL = new THREE.Mesh(handGeo, skin); handL.position.set(-0.72, -0.85, 0); group.add(handL);
    const handR = new THREE.Mesh(handGeo, skin); handR.position.set(0.72, -0.85, 0);  group.add(handR);

    // ── LEGS ───────────────────────────────────────────────
    const pelvis = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.3, 0.55), suit);
    pelvis.position.set(0, -0.8, 0); group.add(pelvis);

    const legGeo = new THREE.BoxGeometry(0.38, 0.5, 0.38);
    const legL = new THREE.Mesh(legGeo, suit); legL.position.set(-0.3, -1.15, 0); group.add(legL);
    const legR = new THREE.Mesh(legGeo, suit); legR.position.set(0.3, -1.15, 0);  group.add(legR);

    // ── PERSONA ACCESSORY ──────────────────────────────────
    if (typeof def.buildAccessory === "function") {
      this.meshes.accessory = def.buildAccessory(group, { suit, skin, hair, tie });
    }

    // Veteran: most upright — no lean even in aggressive
    this.meshes._veteranPosture = this.persona === "veteran";

    // Wildcard: random head tilt state
    this._wildcardAngle = 0;
    this._wildcardNext  = 0;

    group.position.set(0, 0, 0);
    this._eyeBaseY = 1.50;
  }

  // ── Public API ──────────────────────────────────────────────────────────
  setState(newState) {
    if (!(newState in CHARACTER_STATES)) return;

    if (newState === "shocked") {
      this._shockedUntil = performance.now() + 1500;
      const stateBefore = this.state;
      this._postShockState = stateBefore === "shocked" ? "idle" : stateBefore;
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

  // ── Private helpers ──────────────────────────────────────────────────────
  _updateBadge(state) {
    const b = document.querySelector(".character-state-badge");
    if (b) b.textContent = state;
    const chip = document.getElementById("character-state-label");
    if (chip) chip.textContent = state;
  }

  _lerp(a, b, t) { return a + (b - a) * t; }

  _animate() {
    this.animFrameId = requestAnimationFrame(() => this._animate());
    if (!this.scene || !this.camera || !this.renderer) return;

    const now = performance.now();
    const dt  = 0.016; // ~60fps
    const target = this.targetState;
    const speed  = target.animSpeed;

    // Shocked state auto-revert after 1.5s
    if (this.state === "shocked" && now > this._shockedUntil) {
      this.setState(this._postShockState || "idle");
    }

    this.clock += dt * speed;

    if (!this.characterGroup) { this.renderer.render(this.scene, this.camera); return; }

    const g = this.characterGroup;

    // ── Breathing (y-oscillation 0.6Hz idle, 1.5Hz aggressive) ─────────────
    const breathFreq = this.state === "aggressive" ? 1.5 : 0.6;
    const breathAmp  = this.state === "aggressive" ? 0.025 : 0.012;
    g.position.y = Math.sin(this.clock * breathFreq) * breathAmp;

    // ── Head side-to-side ±2° at 0.25Hz ──────────────────────────────────
    if (this.meshes.head) {
      const headSwing = Math.sin(this.clock * 0.25) * (3.14159 / 90); // ±2°
      this.meshes.head.rotation.y = headSwing;

      // Thinking: head tilts forward 10°
      const forwardTilt = this.state === "thinking" ? -(10 * 3.14159 / 180) : 0;
      this.meshes.head.rotation.x = this._lerp(this.meshes.head.rotation.x, forwardTilt, 0.05);

      // Head tilt (z) lerp
      this._curTilt = this._lerp(this._curTilt, target.headTilt, 0.05);
      this.meshes.head.rotation.z = this._curTilt;

      // Shocked: rapid head shake
      if (this.state === "shocked") {
        this.meshes.head.rotation.z = Math.sin(this.clock * 4) * (8 * 3.14159 / 180);
      }
    }

    // ── Body lean ─────────────────────────────────────────────────────────
    const isVeteran = this.meshes._veteranPosture;
    const targetLean = isVeteran ? 0 : target.bodyLean;
    this._curLean = this._lerp(this._curLean, targetLean, 0.04);
    g.rotation.x = this._curLean;

    // ── Eye scale lerp ────────────────────────────────────────────────────
    this._curEyeScale = this._lerp(this._curEyeScale, target.eyeScale, 0.06);
    const es = this._curEyeScale;
    if (this.meshes.eyeLeft)  this.meshes.eyeLeft.scale.setScalar(es);
    if (this.meshes.eyeRight) this.meshes.eyeRight.scale.setScalar(es);

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

      const browAngle = this.state === "aggressive" ? 0.2 : 0;
      this.meshes.browL.rotation.z = this._lerp(this.meshes.browL.rotation.z,  browAngle, 0.07);
      this.meshes.browR.rotation.z = this._lerp(this.meshes.browR.rotation.z, -browAngle, 0.07);
    }

    // ── Mouth morph ───────────────────────────────────────────────────────
    if (this.meshes.mouth) {
      this.meshes.mouth.scale.y = this._lerp(
        this.meshes.mouth.scale.y,
        1 + target.mouthOpen * 2.2,
        0.07
      );
      this.meshes.mouth.position.y = this._lerp(
        this.meshes.mouth.position.y,
        1.33 - target.mouthOpen * 0.04 + target.mouthCurve * 0.04,
        0.07
      );
    }

    // ── Arm sway ──────────────────────────────────────────────────────────
    if (this.meshes.upperArmL && this.meshes.upperArmR) {
      const swayAmp  = this.state === "aggressive" ? 0.07 : 0.015;
      const swayFreq = this.state === "aggressive" ? 1.5  : 0.35;
      const sway = Math.sin(this.clock * swayFreq) * swayAmp;
      this.meshes.upperArmL.rotation.z =  sway;
      this.meshes.upperArmR.rotation.z = -sway;
    }

    // ── Wildcard: random head snap ─────────────────────────────────────────
    if (this.persona === "wildcard" && this.state === "idle") {
      if (Math.random() < 0.005) {
        this._wildcardAngle = (Math.random() - 0.5) * 0.3;
      }
      if (this.meshes.head) {
        this.meshes.head.rotation.z = this._lerp(
          this.meshes.head.rotation.z,
          this._wildcardAngle,
          0.08
        );
      }
    }

    this.renderer.render(this.scene, this.camera);
  }
}

// ── Mini render for persona-picker cards (180px wide) ─────────────────────
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
    key.position.set(2, 5, 3);
    this.scene.add(key);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.45));

    // mini table
    const t = new THREE.Mesh(
      new THREE.BoxGeometry(8, 0.08, 2),
      new THREE.MeshStandardMaterial({ color: 0x2c1810, roughness: 0.5 })
    );
    t.position.y = -0.8;
    this.scene.add(t);
  }

  _build() {
    if (!this.scene) return;
    if (this.group) this.scene.remove(this.group);

    const def  = PERSONA_DEFS[this.persona] || PERSONA_DEFS.shark;
    const g    = new THREE.Group();
    this.group = g;
    this.scene.add(g);

    const mat = (c, r=0.75) => new THREE.MeshStandardMaterial({ color: c, roughness: r, metalness: 0.04 });

    const torso = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.3, 0.6), mat(def.suitColor));
    torso.position.y = 0;
    g.add(torso);

    const head = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.85, 0.65), mat(def.skinColor, 0.85));
    head.position.set(0, 1.45, 0);
    g.add(head);
    this._head = head;

    // Hair
    const hTop = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.26, 0.68), mat(def.hairColor, 0.85));
    hTop.position.set(0, 1.82, -0.02);
    g.add(hTop);

    // Eyes
    const ew = new THREE.SphereGeometry(0.09, 10, 7);
    const ep = new THREE.SphereGeometry(0.05, 8, 6);
    const eWM = mat(0xffffff, 0.95);
    const ePM = mat(0x111111, 0.9);
    [-0.19, 0.19].forEach(x => {
      const ew_ = new THREE.Mesh(ew, eWM); ew_.position.set(x, 1.5, 0.33); g.add(ew_);
      const ep_ = new THREE.Mesh(ep, ePM); ep_.position.set(x, 1.5, 0.38); g.add(ep_);
    });
  }

  _animate() {
    this.animId = requestAnimationFrame(() => this._animate());
    if (!this.scene || !this.renderer || !this.camera) return;

    this.clock += 0.016 * 0.5;
    if (this.group) {
      this.group.position.y = Math.sin(this.clock) * 0.01;
    }
    if (this._head) {
      this._head.rotation.y = Math.sin(this.clock * 0.25) * 0.03;
    }

    this.renderer.render(this.scene, this.camera);
  }

  destroy() {
    if (this.animId) cancelAnimationFrame(this.animId);
    if (this.group) this.scene && this.scene.remove(this.group);
    if (this.renderer) this.renderer.dispose();
  }
}

// ── Global exports ─────────────────────────────────────────────────────────
window.NegotiatorCharacter   = NegotiatorCharacter;
window.PersonaPreviewCharacter = PersonaPreviewCharacter;
window.PERSONA_DEFS          = PERSONA_DEFS;
