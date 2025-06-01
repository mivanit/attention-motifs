/* PointCloud.js – snap-to-point cross-hair, optional hover UI (“k”),
   optional click-to-select (“b”), and better picking accuracy. */

class PointCloud {
    /** @param {DataModel} model */
    constructor(model) {
        this.model = model;

        /* ── THREE basics ─────────────────────────────────────── */
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75, window.innerWidth / window.innerHeight, 0.1, 2_000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });

        /* ── picking helpers ──────────────────────────────────── */
        this.raycaster = new THREE.Raycaster();
        this.raycaster.params.Points = { threshold: 0.15 };   // world-unit tolerance
        this.pointerNDC = new THREE.Vector2();
        this.pointerScreen = { x: 0, y: 0 };

        this.hoverId = null;
        this.prevHoverId = null;

        /* ── behaviour flags (toggled by UIManager) ───────────── */
        this.hoverActive = true;   // “k”
        this.selectOnClick = true;   // “b”

        /* ── colour / selection state ─────────────────────────── */
        this.state = new VisState(model);
        this.selMgr = new SelectionManager(model, this.state);
        this.state.addEventListener('selection', () => this._updateColors());
        this.state.addEventListener('vis', () => this._updateColors());

        /* ── viewer settings ------------------------------------ */
        this.settings = { pointSize: 0.1, opacity: 0.8, speed: 10 };

        /* ── movement bookkeeping -------------------------------- */
        this.keys = {};
        this.pitch = 0;
        this.mouseDX = 0;
        this.mouseDY = 0;
        this.rollSpeed = 0.02;
        this.velocity = new THREE.Vector3();

        /* ── cross-hair objects ───────────────────────────────── */
        this.crossH = null;
        this.crossV = null;
        this._createCrosshairs();

        /* bootstrap */
        this._init();
    }

    /* ========================================================= */
    _init() {
        this._setupRenderer();
        this._setupInput();
        this._buildGeometry();
        this._animate();
    }

    _setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000011);
        document.getElementById('container').appendChild(this.renderer.domElement);
        this.camera.position.set(0, 0, 0);
    }

    /* ---------- in-scene cross-hair --------------------------- */
    _createCrosshairs() {
        const mat = new THREE.LineBasicMaterial({
            color: 0xffff00, transparent: true, opacity: 0.4, depthTest: false
        });

        /* horizontal (X-axis) */
        const gH = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-1_000, 0, 0),
            new THREE.Vector3(1_000, 0, 0)
        ]);
        this.crossH = new THREE.Line(gH, mat);
        this.crossH.visible = false;
        this.scene.add(this.crossH);

        /* vertical (Y-axis) */
        const gV = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, -1_000, 0),
            new THREE.Vector3(0, 1_000, 0)
        ]);
        this.crossV = new THREE.Line(gV, mat);
        this.crossV.visible = false;
        this.scene.add(this.crossV);
    }

    _updateCrosshairs(id) {
        if (!this.hoverActive || id == null) {
            this.crossH.visible = this.crossV.visible = false;
            return;
        }
        const a = this.state.axis;
        const x = this.model.getCoord(id, a.x);
        const y = this.model.getCoord(id, a.y);
        const z = this.model.getCoord(id, a.z);

        this.crossH.position.set(0, y, z);
        this.crossV.position.set(x, 0, z);
        this.crossH.visible = this.crossV.visible = true;
    }

    /* ---------- input & interaction -------------------------- */
    _setupInput() {
        /* keyboard state */
        document.addEventListener('keydown', e => this.keys[e.code] = true);
        document.addEventListener('keyup', e => this.keys[e.code] = false);

        /* mouse movement */
        document.addEventListener('mousemove', e => {
            if (document.pointerLockElement === document.body) {
                this.mouseDX += e.movementX;
                this.mouseDY += e.movementY;
            }
            this.pointerScreen.x = e.clientX;
            this.pointerScreen.y = e.clientY;
            this.pointerNDC.x = (e.clientX / window.innerWidth) * 2 - 1;
            this.pointerNDC.y = -(e.clientY / window.innerHeight) * 2 + 1;
        });

        /* click-to-select (can be disabled) */
        window.addEventListener('click', () => {
            if (!this.selectOnClick || this.hoverId == null) return;
            const v = this.model.row(this.hoverId)[this.state.selectBy];
            this.state.toggleValue(v);
        });

        /* pointer-lock helpers */
        document.addEventListener('dblclick', () => {
            (document.pointerLockElement === document.body)
                ? document.exitPointerLock()
                : document.body.requestPointerLock();
        });
        document.addEventListener('keydown', e => {
            if (e.code === 'Escape' && document.pointerLockElement === document.body)
                document.exitPointerLock();
        });

        /* resize */
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    /* ---------- build / rebuild point geometry --------------- */
    _buildGeometry() {
        if (this.points) {
            this.scene.remove(this.points);
            this.points.geometry.dispose();
            this.points.material.dispose();
        }

        const pos = [];
        const col = [];

        for (let i = 0; i < this.model.rowCount; ++i) {
            pos.push(
                this.model.getCoord(i, 0),
                this.model.getCoord(i, 1),
                this.model.getCoord(i, 2)
            );
            col.push(0.6, 0.6, 0.6);
        }

        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        geom.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));

        const mat = new THREE.PointsMaterial({
            size: this.settings.pointSize,
            opacity: this.settings.opacity,
            transparent: this.settings.opacity < 1,
            vertexColors: true,
            sizeAttenuation: true
        });

        this.points = new THREE.Points(geom, mat);
        this.colorAttr = geom.getAttribute('color');
        this.scene.add(this.points);

        if (this.uiManager) this.uiManager.onPointsRegenerated();
        this._updateColors();
    }

    handleSettingChange(prop) {
        if (prop === 'pointSize' || prop === 'opacity') {
            this.points.material.size = this.settings.pointSize;
            this.points.material.opacity = this.settings.opacity;
            this.points.material.transparent = this.settings.opacity < 1;
            this.points.material.needsUpdate = true;

            /* update picking tolerance to roughly match size */
            this.raycaster.params.Points.threshold = this.settings.pointSize * 3;
        }
    }

    /* ---------- recolour all points --------------------------- */
    _updateColors() {
        const A = this.colorAttr.array;
        for (let i = 0; i < this.colorAttr.count; ++i) {
            const { r, g, b } = this.selMgr.attrs(i);
            A[i * 3] = r; A[i * 3 + 1] = g; A[i * 3 + 2] = b;
        }
        this.colorAttr.needsUpdate = true;
    }

    /* ---------- camera motion --------------------------------- */
    _moveCamera() {
        const sens = 0.002;
        const yaw = -this.mouseDX * sens;
        const dp = -this.mouseDY * sens;

        if (yaw) this.camera.rotateY(yaw);
        if (dp) {
            const np = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitch + dp));
            this.camera.rotateX(np - this.pitch);
            this.pitch = np;
        }
        this.mouseDX = this.mouseDY = 0;

        if (this.keys['KeyQ']) this.camera.rotateZ(this.rollSpeed);
        if (this.keys['KeyE']) this.camera.rotateZ(-this.rollSpeed);

        this.velocity.set(0, 0, 0);
        if (this.keys['KeyW']) this.velocity.z -= 1;
        if (this.keys['KeyS']) this.velocity.z += 1;
        if (this.keys['KeyA']) this.velocity.x -= 1;
        if (this.keys['KeyD']) this.velocity.x += 1;

        if (this.velocity.lengthSq()) {
            const speed = this.settings.speed *
                (this.keys['ShiftLeft'] ? 3 : 1) * 0.016;
            this.velocity.normalize().multiplyScalar(speed)
                .applyQuaternion(this.camera.quaternion);
            this.camera.position.add(this.velocity);
        }
    }

    /* ---------- render loop ----------------------------------- */
    _animate() {
        requestAnimationFrame(() => this._animate());

        this._moveCamera();

        /* picking */
        this.raycaster.setFromCamera(this.pointerNDC, this.camera);
        const hit = this.raycaster.intersectObject(this.points, false)[0];
        this.hoverId = hit ? hit.index : null;

        if (this.hoverId !== this.prevHoverId) {
            this.prevHoverId = this.hoverId;
            this._updateCrosshairs(this.hoverId);
        }

        if (this.uiManager) this.uiManager.updateUI();
        this.renderer.render(this.scene, this.camera);
    }

    /* ---------- allow UIManager to attach --------------------- */
    setUIManager(ui) { this.uiManager = ui; }
}
