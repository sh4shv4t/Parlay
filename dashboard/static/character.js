// ============================================================
// Parlay — Three.js r128 Negotiator Character System
// Bloomberg terminal meets poker app.
// MUST be loaded AFTER Three.js r128 from cdnjs.
// ============================================================

const CHARACTER_STATES = {
  idle:       { headTilt: 0,     eyeScale: 1.0, animSpeed: 0.5,  mouthOpen: 0,   mouthCurve: 0    },
  thinking:   { headTilt: 0.15,  eyeScale: 0.8, animSpeed: 0.3,  mouthOpen: 0,   mouthCurve: 0    },
  aggressive: { headTilt: -0.1,  eyeScale: 1.3, animSpeed: 1.2,  mouthOpen: 0.4, mouthCurve: -0.2 },
  pleased:    { headTilt: 0.08,  eyeScale: 1.1, animSpeed: 0.8,  mouthOpen: 0.2, mouthCurve: 0.3  },
  shocked:    { headTilt: 0.25,  eyeScale: 1.5, animSpeed: 2.0,  mouthOpen: 0.7, mouthCurve: 0    },
};

const PERSONA_COLORS = {
  shark:    { primary: 0xb83030, secondary: 0x8a2020, accent: 0xe05050, suit: 0x2a1010, tie: 0xb83030 },
  diplomat: { primary: 0x2d7a4f, secondary: 0x1d5a3a, accent: 0x4daa70, suit: 0x1a2e22, tie: 0x2d7a4f },
  analyst:  { primary: 0x1a5fa8, secondary: 0x104080, accent: 0x3a8fd8, suit: 0x0f1e30, tie: 0x1a5fa8 },
  wildcard: { primary: 0xa05c00, secondary: 0x7a4400, accent: 0xd08020, suit: 0x2a1c00, tie: 0xa05c00 },
  veteran:  { primary: 0x5c3d9e, secondary: 0x3c2070, accent: 0x8c60d0, suit: 0x1a0f30, tie: 0x5c3d9e },
};

const PERSONA_SYMBOLS = {
  shark:    "▲",
  diplomat: "◆",
  analyst:  "●",
  wildcard: "★",
  veteran:  "⬟",
};

class NegotiatorCharacter {
  constructor(canvasId, persona = "shark") {
    this.canvasId = canvasId;
    this.persona = persona;
    this.state = "idle";
    this.targetState = CHARACTER_STATES.idle;
    this.currentTilt = 0;
    this.currentEyeScale = 1.0;
    this.clock = 0;
    this.animFrameId = null;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.meshes = {};
    this._init(canvasId);
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
    this.scene.background = null;

    this.camera = new THREE.PerspectiveCamera(45, 280 / 380, 0.1, 100);
    this.camera.position.set(0, 1.0, 5.5);
    this.camera.lookAt(0, 0.8, 0);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
    });
    this.renderer.setSize(280, 380);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Ambient light
    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    this.scene.add(ambient);

    // Key light
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(3, 5, 5);
    key.castShadow = true;
    this.scene.add(key);

    // Fill light
    const fill = new THREE.DirectionalLight(0x8899ff, 0.35);
    fill.position.set(-3, 2, -2);
    this.scene.add(fill);

    // Rim light
    const rim = new THREE.DirectionalLight(0xffffff, 0.2);
    rim.position.set(0, -2, -4);
    this.scene.add(rim);
  }

  _makeMat(color, roughness = 0.7, metalness = 0.1) {
    return new THREE.MeshStandardMaterial({ color, roughness, metalness });
  }

  _makeTextureBadge(symbol, bgColor, fgColor) {
    const size = 128;
    const cvs = document.createElement("canvas");
    cvs.width = size;
    cvs.height = size;
    const ctx = cvs.getContext("2d");

    ctx.fillStyle = `#${bgColor.toString(16).padStart(6, "0")}`;
    ctx.beginPath();
    ctx.roundRect(16, 16, size - 32, size - 32, 12);
    ctx.fill();

    ctx.fillStyle = `#${fgColor.toString(16).padStart(6, "0")}`;
    ctx.font = "bold 56px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(symbol, size / 2, size / 2);

    const tex = new THREE.CanvasTexture(cvs);
    return tex;
  }

  _clearScene() {
    // Remove character meshes
    Object.values(this.meshes).forEach(m => {
      if (m) this.scene.remove(m);
    });
    this.meshes = {};
    if (this.characterGroup) {
      this.scene.remove(this.characterGroup);
    }
    this.characterGroup = null;
  }

  _buildCharacter() {
    this._clearScene();

    const colors = PERSONA_COLORS[this.persona];
    const group = new THREE.Group();
    this.characterGroup = group;
    this.scene.add(group);

    // Skin tone — neutral
    const skinColor = 0xf5d0a9;
    const skinMat  = this._makeMat(skinColor, 0.85, 0.0);
    const suitMat  = this._makeMat(colors.suit,      0.8, 0.05);
    const shirtMat = this._makeMat(0xf0f0f0,         0.9, 0.0);
    const tieMat   = this._makeMat(colors.tie,       0.6, 0.05);
    const accentMat= this._makeMat(colors.accent,    0.5, 0.2);
    const eyeWhite = this._makeMat(0xffffff,         0.95, 0.0);
    const eyePupil = this._makeMat(0x111111,         0.9, 0.0);
    const hairMat  = this._makeMat(0x2a1a0a,         0.9, 0.0);

    // --- TORSO ---
    const torsoGeo = new THREE.BoxGeometry(1.1, 1.3, 0.6);
    const torso = new THREE.Mesh(torsoGeo, suitMat);
    torso.position.set(0, 0, 0);
    group.add(torso);
    this.meshes.torso = torso;

    // Shirt (white strip on chest)
    const shirtGeo = new THREE.BoxGeometry(0.36, 0.9, 0.32);
    const shirt = new THREE.Mesh(shirtGeo, shirtMat);
    shirt.position.set(0, 0.1, 0.32);
    group.add(shirt);

    // Tie
    const tieGeo = new THREE.BoxGeometry(0.12, 0.75, 0.06);
    const tie = new THREE.Mesh(tieGeo, tieMat);
    tie.position.set(0, 0.1, 0.36);
    group.add(tie);

    // Suit lapels (two thin boxes)
    const lapelGeo = new THREE.BoxGeometry(0.22, 0.7, 0.08);
    const lapelL = new THREE.Mesh(lapelGeo, suitMat);
    lapelL.position.set(-0.22, 0.25, 0.3);
    lapelL.rotation.z = 0.2;
    group.add(lapelL);

    const lapelR = new THREE.Mesh(lapelGeo, suitMat);
    lapelR.position.set(0.22, 0.25, 0.3);
    lapelR.rotation.z = -0.2;
    group.add(lapelR);

    // Persona badge on left breast
    const badgeTex = this._makeTextureBadge(
      PERSONA_SYMBOLS[this.persona],
      colors.primary,
      0xffffff
    );
    const badgeMat = new THREE.MeshStandardMaterial({ map: badgeTex, roughness: 0.5 });
    const badgeGeo = new THREE.BoxGeometry(0.2, 0.2, 0.04);
    const badge = new THREE.Mesh(badgeGeo, badgeMat);
    badge.position.set(-0.28, 0.4, 0.32);
    group.add(badge);

    // --- NECK ---
    const neckGeo = new THREE.CylinderGeometry(0.13, 0.15, 0.3, 8);
    const neck = new THREE.Mesh(neckGeo, skinMat);
    neck.position.set(0, 0.8, 0);
    group.add(neck);
    this.meshes.neck = neck;

    // --- HEAD ---
    const headGeo = new THREE.BoxGeometry(0.75, 0.85, 0.65);
    const head = new THREE.Mesh(headGeo, skinMat);
    head.position.set(0, 1.45, 0);
    group.add(head);
    this.meshes.head = head;

    // Hair
    const hairGeo = new THREE.BoxGeometry(0.78, 0.28, 0.68);
    const hair = new THREE.Mesh(hairGeo, hairMat);
    hair.position.set(0, 1.82, -0.02);
    group.add(hair);

    // Hair sides
    const hairSideGeo = new THREE.BoxGeometry(0.12, 0.45, 0.6);
    const hairL = new THREE.Mesh(hairSideGeo, hairMat);
    hairL.position.set(-0.37, 1.6, -0.02);
    group.add(hairL);
    const hairR = new THREE.Mesh(hairSideGeo, hairMat);
    hairR.position.set(0.37, 1.6, -0.02);
    group.add(hairR);

    // --- EYES ---
    const eyeW = new THREE.SphereGeometry(0.1, 12, 8);
    const eyeP = new THREE.SphereGeometry(0.055, 8, 6);

    const eyeLW = new THREE.Mesh(eyeW, eyeWhite);
    eyeLW.position.set(-0.19, 1.5, 0.33);
    group.add(eyeLW);
    this.meshes.eyeLeft = eyeLW;

    const eyeRW = new THREE.Mesh(eyeW, eyeWhite);
    eyeRW.position.set(0.19, 1.5, 0.33);
    group.add(eyeRW);
    this.meshes.eyeRight = eyeRW;

    const eyeLP = new THREE.Mesh(eyeP, eyePupil);
    eyeLP.position.set(-0.19, 1.5, 0.38);
    group.add(eyeLP);
    this.meshes.pupilLeft = eyeLP;

    const eyeRP = new THREE.Mesh(eyeP, eyePupil);
    eyeRP.position.set(0.19, 1.5, 0.38);
    group.add(eyeRP);
    this.meshes.pupilRight = eyeRP;

    // Eyebrows
    const browGeo = new THREE.BoxGeometry(0.18, 0.04, 0.04);
    const browMat = this._makeMat(0x2a1a0a, 0.9, 0.0);
    const browL = new THREE.Mesh(browGeo, browMat);
    browL.position.set(-0.19, 1.63, 0.33);
    group.add(browL);
    this.meshes.browL = browL;

    const browR = new THREE.Mesh(browGeo, browMat);
    browR.position.set(0.19, 1.63, 0.33);
    group.add(browR);
    this.meshes.browR = browR;

    // --- MOUTH ---
    // We represent the mouth as a thin box that we morph
    const mouthGeo = new THREE.BoxGeometry(0.22, 0.04, 0.04);
    const mouthMat = this._makeMat(0x8b3a3a, 0.8, 0.0);
    const mouth = new THREE.Mesh(mouthGeo, mouthMat);
    mouth.position.set(0, 1.33, 0.33);
    group.add(mouth);
    this.meshes.mouth = mouth;

    // Nose (small box)
    const noseGeo = new THREE.BoxGeometry(0.1, 0.1, 0.12);
    const nose = new THREE.Mesh(noseGeo, this._makeMat(0xe8b890, 0.85, 0.0));
    nose.position.set(0, 1.44, 0.35);
    group.add(nose);

    // --- SHOULDERS / UPPER ARMS ---
    const shoulderGeo = new THREE.BoxGeometry(0.32, 0.28, 0.45);
    const shlL = new THREE.Mesh(shoulderGeo, suitMat);
    shlL.position.set(-0.7, 0.5, 0);
    group.add(shlL);
    const shlR = new THREE.Mesh(shoulderGeo, suitMat);
    shlR.position.set(0.7, 0.5, 0);
    group.add(shlR);

    // Upper arms
    const uArmGeo = new THREE.BoxGeometry(0.28, 0.6, 0.3);
    const uArmL = new THREE.Mesh(uArmGeo, suitMat);
    uArmL.position.set(-0.72, 0.05, 0);
    group.add(uArmL);
    this.meshes.upperArmL = uArmL;

    const uArmR = new THREE.Mesh(uArmGeo, suitMat);
    uArmR.position.set(0.72, 0.05, 0);
    group.add(uArmR);
    this.meshes.upperArmR = uArmR;

    // Lower arms
    const lArmGeo = new THREE.BoxGeometry(0.24, 0.55, 0.26);
    const lArmL = new THREE.Mesh(lArmGeo, suitMat);
    lArmL.position.set(-0.72, -0.5, 0);
    group.add(lArmL);

    const lArmR = new THREE.Mesh(lArmGeo, suitMat);
    lArmR.position.set(0.72, -0.5, 0);
    group.add(lArmR);

    // Hands (spheres)
    const handGeo = new THREE.SphereGeometry(0.14, 8, 6);
    const handL = new THREE.Mesh(handGeo, skinMat);
    handL.position.set(-0.72, -0.85, 0);
    group.add(handL);

    const handR = new THREE.Mesh(handGeo, skinMat);
    handR.position.set(0.72, -0.85, 0);
    group.add(handR);

    // Accent cufflinks
    const cuffGeo = new THREE.BoxGeometry(0.26, 0.06, 0.28);
    const cuffL = new THREE.Mesh(cuffGeo, accentMat);
    cuffL.position.set(-0.72, -0.74, 0);
    group.add(cuffL);
    const cuffR = new THREE.Mesh(cuffGeo, accentMat);
    cuffR.position.set(0.72, -0.74, 0);
    group.add(cuffR);

    // --- PELVIS / LEGS (partial, cropped by camera) ---
    const pelvisGeo = new THREE.BoxGeometry(1.0, 0.3, 0.55);
    const pelvis = new THREE.Mesh(pelvisGeo, suitMat);
    pelvis.position.set(0, -0.8, 0);
    group.add(pelvis);

    const legGeo = new THREE.BoxGeometry(0.38, 0.5, 0.38);
    const legL = new THREE.Mesh(legGeo, suitMat);
    legL.position.set(-0.3, -1.15, 0);
    group.add(legL);

    const legR = new THREE.Mesh(legGeo, suitMat);
    legR.position.set(0.3, -1.15, 0);
    group.add(legR);

    // Centre group
    group.position.set(0, 0, 0);

    // Store per-state base positions for lerp
    this._eyeBaseY = 1.5;
  }

  setState(newState) {
    if (!(newState in CHARACTER_STATES)) return;
    this.state = newState;
    this.targetState = CHARACTER_STATES[newState];
    this._updateStateBadge(newState);
  }

  setPersona(persona) {
    if (!(persona in PERSONA_COLORS)) return;
    this.persona = persona;
    this._buildCharacter();
  }

  _updateStateBadge(state) {
    const badge = document.querySelector(".character-state-badge");
    if (badge) badge.textContent = state;
  }

  _lerp(a, b, t) {
    return a + (b - a) * t;
  }

  _animate() {
    this.animFrameId = requestAnimationFrame(() => this._animate());

    if (!this.scene || !this.camera || !this.renderer) return;

    const target = this.targetState;
    const speed  = target.animSpeed;
    const dt     = 0.016; // ~60fps assumption

    this.clock += dt * speed;

    // Breathing: subtle y oscillation on whole group
    if (this.characterGroup) {
      const breathAmp  = this.state === "aggressive" ? 0.025 : 0.012;
      const breathFreq = this.state === "aggressive" ? 1.8 : 0.9;
      this.characterGroup.position.y = Math.sin(this.clock * breathFreq) * breathAmp;

      // Head tilt lerp
      this.currentTilt = this._lerp(this.currentTilt, target.headTilt, 0.05);
      if (this.meshes.head) {
        this.meshes.head.rotation.z = this.currentTilt;
      }

      // Eye scale lerp
      this.currentEyeScale = this._lerp(this.currentEyeScale, target.eyeScale, 0.06);
      const es = this.currentEyeScale;
      if (this.meshes.eyeLeft)  this.meshes.eyeLeft.scale.setScalar(es);
      if (this.meshes.eyeRight) this.meshes.eyeRight.scale.setScalar(es);

      // Pupils — subtle idle wander, large for shocked
      if (this.meshes.pupilLeft && this.meshes.pupilRight) {
        const wanderX = Math.sin(this.clock * 0.7) * 0.025;
        const wanderY = Math.cos(this.clock * 0.5) * 0.015;
        this.meshes.pupilLeft.position.x  = -0.19 + wanderX;
        this.meshes.pupilRight.position.x =  0.19 + wanderX;
        this.meshes.pupilLeft.position.y  = this._eyeBaseY + wanderY;
        this.meshes.pupilRight.position.y = this._eyeBaseY + wanderY;
      }

      // Eyebrow expression
      if (this.meshes.browL && this.meshes.browR) {
        const browTarget = this.state === "aggressive" ? -0.12
                         : this.state === "shocked"    ?  0.12
                         : this.state === "pleased"    ?  0.05
                         : 0;
        this.meshes.browL.position.y = this._lerp(this.meshes.browL.position.y, 1.63 + browTarget, 0.07);
        this.meshes.browR.position.y = this._lerp(this.meshes.browR.position.y, 1.63 + browTarget, 0.07);

        // Angle brows for aggressive (inner raised) vs. shocked (both raised)
        const browAngle = this.state === "aggressive" ? 0.2 : 0;
        this.meshes.browL.rotation.z =  this._lerp(this.meshes.browL.rotation.z,  browAngle, 0.07);
        this.meshes.browR.rotation.z = this._lerp(this.meshes.browR.rotation.z,  -browAngle, 0.07);
      }

      // Mouth
      if (this.meshes.mouth) {
        const openTarget = target.mouthOpen;
        const curveTarget = target.mouthCurve;
        this.meshes.mouth.scale.y = this._lerp(this.meshes.mouth.scale.y, 1 + openTarget * 2, 0.07);
        this.meshes.mouth.position.y = this._lerp(
          this.meshes.mouth.position.y,
          1.33 - openTarget * 0.04 + curveTarget * 0.04,
          0.07
        );
      }

      // Arm sway idle
      if (this.meshes.upperArmL && this.meshes.upperArmR) {
        const swayAmp = this.state === "aggressive" ? 0.08 : 0.02;
        const swayFreq = this.state === "aggressive" ? 1.2 : 0.4;
        const sway = Math.sin(this.clock * swayFreq) * swayAmp;
        this.meshes.upperArmL.rotation.z =  sway;
        this.meshes.upperArmR.rotation.z = -sway;
      }

      // Thinking: periodic head bob
      if (this.state === "thinking") {
        const bobY = Math.sin(this.clock * 1.5) * 0.015;
        if (this.meshes.head) {
          this.meshes.head.position.y = 1.45 + bobY;
        }
      } else if (this.meshes.head) {
        this.meshes.head.position.y = this._lerp(this.meshes.head.position.y, 1.45, 0.05);
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  destroy() {
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    this._clearScene();
    if (this.renderer) this.renderer.dispose();
  }
}

window.NegotiatorCharacter = NegotiatorCharacter;
