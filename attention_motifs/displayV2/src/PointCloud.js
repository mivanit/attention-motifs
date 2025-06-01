/* PointCloud.js – adds in-scene cross-hair that snaps to the hovered
   point (no colour change on hover). */

class PointCloud {
    /** @param {DataModel} model */
    constructor(model) {
        this.model = model;

        /* ── THREE essentials ─────────────────────────────────── */
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75, window.innerWidth / window.innerHeight, 0.1, 2_000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });

        /* ── helpers ──────────────────────────────────────────── */
        this.raycaster = new THREE.Raycaster();
        this.pointerNDC = new THREE.Vector2();
        this.pointerScreen = { x: 0, y: 0 };
        this.hoverId = null;
        this.prevHoverId = null;

        /* ── state / colour management -------------------------- */
        this.state = new VisState(model);
        this.selMgr = new SelectionManager(model, this.state);
        this.state.addEventListener('selection', () => this._updateColors());
        this.state.addEventListener('vis', () => this._updateColors());

        /* ── viewer tunables ------------------------------------ */
        this.settings = { pointSize: 0.1, opacity: 0.8, speed: 10 };

        /* ── movement bookkeeping ------------------------------- */
        this.keys = {};
        this.pitch = 0;
        this.mouseDX = 0;
        this.mouseDY = 0;
        this.rollSpeed = 0.02;
        this.velocity = new THREE.Vector3();

        /* ── cross-hair lines (in-scene) ------------------------ */
        this.crosshairH = null;
        this.crosshairV = null;
        this._initCrosshairs();

        /* bootstrap */
        this._init();
    }

    /* ========================================================= */
    _init() {
        this._setupRenderer();
        this._setupInput();
        this._generatePoints();
        this._animate();
    }

    _setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000011);
        document.getElementById('container').appendChild(this.renderer.domElement);
        this.camera.position.set(0, 0, 0);
    }

    /* ---------- cross-hair helpers --------------------------- */
    _initCrosshairs() {
        const mat = new THREE.LineBasicMaterial({
            color: 0xffff00,
            transparent: true,
            opacity: 0.35,
            depthTest: false
        });

        /* horizontal line */
        const hGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-1_000, 0, 0),
            new THREE.Vector3(1_000, 0, 0)
        ]);
        this.crosshairH = new THREE.Line(hGeom, mat);
        this.crosshairH.visible = false;
        this.scene.add(this.crosshairH);

        /* vertical line */
        const vGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, -1_000, 0),
            new THREE.Vector3(0, 1_000, 0)
        ]);
        this.crosshairV = new THREE.Line(vGeom, mat);
        this.crosshairV.visible = false;
        this.scene.add(this.crosshairV);
    }

    _updateCrosshairs(rowId) {
        if (rowId == null) {
            this.crosshairH.visible = false;
            this.crosshairV.visible = false;
            return;
        }

        const a = this.state.axis;
        const x = this.model.getCoord(rowId, a.x);
        const y = this.model.getCoord(rowId, a.y);
        const z = this.model.getCoord(rowId, a.z);

        /* position cross-hair planes so they intersect at the point */
        this.crosshairH.position.set(0, y, z);  // spans X-axis
        this.crosshairV.position.set(x, 0, z);  // spans Y-axis

        this.crosshairH.visible = true;
        this.crosshairV.visible = true;
    }

    /* ---------- input & interaction -------------------------- */
    _setupInput() {
        /* keyboard */
        document.addEventListener('keydown', e => this.keys[e.code] = true);
        document.addEventListener('keyup', e => this.keys[e.code] = false);

        /* mouse */
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

        /* click selects category */
        window.addEventListener('click', () => {
            if (this.hoverId == null) return;
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

    /* ---------- geometry ------------------------------------- */
    _generatePoints() {
        if (this.points) {
            this.scene.remove(this.points);
            this.points.geometry.dispose();
            this.points.material.dispose();
        }

        const pos = [];
        const col = [];

        if (this.model) {
            const rows = this.model.rowCount;
            for (let i = 0; i < rows; ++i) {
                pos.push(
                    this.model.getCoord(i, 0),
                    this.model.getCoord(i, 1),
                    this.model.getCoord(i, 2)
                );
                col.push(0.6, 0.6, 0.6);      // will be recoloured below
            }
        } else {
            console.warn('No data loaded');
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

    handleSettingChange(prop, _val) {
        if (prop === 'pointSize' || prop === 'opacity') {
            this.points.material.size = this.settings.pointSize;
            this.points.material.opacity = this.settings.opacity;
            this.points.material.transparent = this.settings.opacity < 1;
            this.points.material.needsUpdate = true;
        }
    }

    /* ---------- recolour points (no hover tint) --------------- */
    _updateColors() {
        if (!this.colorAttr) return;
        const arr = this.colorAttr.array;

        for (let i = 0; i < this.colorAttr.count; ++i) {
            const { r, g, b } = this.selMgr.attrs(i);
            arr[i * 3] = r;
            arr[i * 3 + 1] = g;
            arr[i * 3 + 2] = b;
        }

        this.colorAttr.needsUpdate = true;
    }

    /* ---------- camera movement ------------------------------- */
    _updateMovement() {
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

    /* ---------- main render loop ------------------------------ */
    _animate() {
        requestAnimationFrame(() => this._animate());

        this._updateMovement();

        /* hover picking */
        this.raycaster.setFromCamera(this.pointerNDC, this.camera);
        const hit = this.raycaster.intersectObject(this.points, false)[0];
        this.hoverId = hit ? hit.index : null;

        if (this.hoverId !== this.prevHoverId) {
            this.prevHoverId = this.hoverId;
            this._updateCrosshairs(this.hoverId);   // move / hide lines
        }

        if (this.uiManager) this.uiManager.updateUI();
        this.renderer.render(this.scene, this.camera);
    }

    /* ---------- hook for UI ----------------------------------- */
    setUIManager(ui) { this.uiManager = ui; }
}
