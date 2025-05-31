class PointCloud {
    /** @param {DataModel|null} model */
    constructor(model) {
        this.model = model;

        /* -- THREE essentials --------------------------------- */
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });

        /* -- helpers ------------------------------------------ */
        this.raycaster = new THREE.Raycaster();
        this.pointerNDC = new THREE.Vector2();
        this.pointerScreen = { x: 0, y: 0 };     // for tooltip
        this.hoverId = null;
        this.prevHoverId = null;

        /* -- colouring / selection ---------------------------- */
        this.state = new VisState(model);
        this.selMgr = new SelectionManager(model, this.state);
        this.state.addEventListener('selection', () => this._updateColors());
        this.state.addEventListener('vis', () => this._updateColors());

        /* -- viewer tunables ---------------------------------- */
        this.settings = { pointSize: 0.1, opacity: 0.8, speed: 10 };

        /* movement state */
        this.keys = {};
        this.pitch = 0;
        this.mouseDX = 0; this.mouseDY = 0;
        this.rollSpeed = 0.02;
        this.velocity = new THREE.Vector3();

        /* bootstrap */
        this.init();
    }

    /* --------------------------------------------------------- */
    init() {
        this.setupRenderer();
        this.setupInput();
        this.generatePoints();
        this.animate();
    }

    setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000011);
        document.getElementById('container').appendChild(this.renderer.domElement);
        this.camera.position.set(0, 0, 0);
    }

    /* ---------- input & interaction ------------------------- */
    setupInput() {
        /* keyboard state */
        document.addEventListener('keydown', e => this.keys[e.code] = true);
        document.addEventListener('keyup', e => this.keys[e.code] = false);

        /* mouse move (absolute & relative) */
        document.addEventListener('mousemove', e => {
            if (document.pointerLockElement === document.body) {
                this.mouseDX += e.movementX;
                this.mouseDY += e.movementY;
            }
            /* always store absolute for tooltip & picking */
            this.pointerScreen.x = e.clientX;
            this.pointerScreen.y = e.clientY;
            this.pointerNDC.x = (e.clientX / window.innerWidth) * 2 - 1;
            this.pointerNDC.y = -(e.clientY / window.innerHeight) * 2 + 1;
        });

        /* click toggles *category* selection */
        window.addEventListener('click', () => {
            if (this.hoverId == null) return;
            const val = this.model.row(this.hoverId)[this.state.selectBy];
            this.state.toggleValue(val);          // VisState owns Set now
            /* colours update via event listener */
        });

        /* double-click pointer-lock on/off */
        document.addEventListener('dblclick', () => {
            (document.pointerLockElement === document.body)
                ? document.exitPointerLock()
                : document.body.requestPointerLock();
        });

        /* escape exits pointer-lock */
        document.addEventListener('keydown', e => {
            if (e.code === 'Escape' && document.pointerLockElement === document.body)
                document.exitPointerLock();
        });

        /* window resize */
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    /* ---------- generate / regenerate geometry -------------- */
    generatePoints() {
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
                col.push(0.6, 0.6, 0.6);   // grey – updated per frame
            }
        } else {
            /* fallback random cloud */
            const N = 50000, R = 1000;
            for (let i = 0; i < N; ++i) {
                pos.push(
                    (Math.random() - 0.5) * R,
                    (Math.random() - 0.5) * R,
                    (Math.random() - 0.5) * R
                );
                const c = new THREE.Color().setHSL(Math.random(), 0.7, 0.6);
                col.push(c.r, c.g, c.b);
            }
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

    /* ---------- recolour all points ------------------------- */
    _updateColors() {
        if (!this.colorAttr) return;
        const arr = this.colorAttr.array;

        for (let i = 0; i < this.colorAttr.count; ++i) {
            const { r, g, b } = this.selMgr.attrs(i);
            arr[i * 3] = r; arr[i * 3 + 1] = g; arr[i * 3 + 2] = b;
        }

        /* highlight hovered point bright yellow */
        if (this.hoverId != null) {
            arr[this.hoverId * 3] = 1.0;
            arr[this.hoverId * 3 + 1] = 1.0;
            arr[this.hoverId * 3 + 2] = 0.0;
        }

        this.colorAttr.needsUpdate = true;
    }

    /* ---------- camera movement ----------------------------- */
    updateMovement() {
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

    /* ---------- render loop --------------------------------- */
    animate() {
        requestAnimationFrame(() => this.animate());

        this.updateMovement();

        /* update hover picking */
        this.raycaster.setFromCamera(this.pointerNDC, this.camera);
        const hit = this.raycaster.intersectObject(this.points, false)[0];
        this.hoverId = hit ? hit.index : null;

        if (this.hoverId !== this.prevHoverId) {
            this.prevHoverId = this.hoverId;
            this._updateColors();      // recolour only when hover changes
        }

        if (this.uiManager) this.uiManager.updateUI();
        this.renderer.render(this.scene, this.camera);
    }

    /* allow UI to hook after construction */
    setUIManager(ui) { this.uiManager = ui; }
}
