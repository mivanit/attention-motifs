class PointCloud {
    /**
     * @param {DataModel|null} model  –  DataModel with PCA rows,
     *                                  or null to fall back to a random demo cloud.
     */
    constructor(model) {
        this.model = model;                     // <-- keep the model reference
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });

        /* ---------- point-cloud settings ---------- */
        this.points = null;
        this.settings = {
            pointSize: 3.0,
            opacity: 0.8,
            speed: 10,
            pointCount: 25000          // max points to render
        };

        /* ---------- movement state ---------- */
        this.keys = {};
        this.pitch = 0;
        this.mouseDX = 0;
        this.mouseDY = 0;
        this.rollSpeed = 0.02;
        this.velocity = new THREE.Vector3();

        this.init();
    }

    /* ===== initialisation ===== */
    init() {
        this.setupRenderer();
        this.setupMovement();
        this.generatePoints();
        this.animate();
    }

    setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000011);
        document.getElementById('container').appendChild(this.renderer.domElement);
        this.camera.position.set(0, 0, 0);   // start inside the cloud
    }

    /* ===== input / pointer-lock ===== */
    setupMovement() {
        document.addEventListener('keydown', e => { this.keys[e.code] = true; });
        document.addEventListener('keyup', e => { this.keys[e.code] = false; });

        document.addEventListener('mousemove', e => {
            if (document.pointerLockElement === document.body) {
                this.mouseDX += e.movementX;
                this.mouseDY += e.movementY;
            }
        });

        document.addEventListener('dblclick', () => {
            (document.pointerLockElement === document.body)
                ? document.exitPointerLock()
                : document.body.requestPointerLock();
        });

        document.addEventListener('keydown', e => {
            if (e.code === 'Escape' && document.pointerLockElement === document.body) {
                document.exitPointerLock();
            }
        });

        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    /* ===== build the geometry ===== */
    generatePoints() {
        if (this.points) {
            this.scene.remove(this.points);
            this.points.geometry.dispose();
            this.points.material.dispose();
        }

        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];

        /* ---- use real PCA rows if we have a DataModel ---- */
        if (this.model) {
            const rows = Math.min(this.settings.pointCount, this.model.rowCount);
            for (let i = 0; i < rows; ++i) {
                positions.push(
                    this.model.getCoord(i, 0),   // x
                    this.model.getCoord(i, 1),   // y
                    this.model.getCoord(i, 2)    // z
                );
                // flat grey placeholder; SelectionManager will recolour later
                colors.push(0.6, 0.6, 0.6);
            }
        } else {
            /* fallback demo cloud --------------------------- */
            const range = 1000;
            for (let i = 0; i < this.settings.pointCount; ++i) {
                positions.push(
                    (Math.random() - 0.5) * range,
                    (Math.random() - 0.5) * range,
                    (Math.random() - 0.5) * range
                );
                const clr = new THREE.Color().setHSL(Math.random(), 0.7, 0.6);
                colors.push(clr.r, clr.g, clr.b);
            }
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: this.settings.pointSize,
            opacity: this.settings.opacity,
            transparent: this.settings.opacity < 1,
            vertexColors: true,
            sizeAttenuation: true
        });

        this.points = new THREE.Points(geometry, material);
        this.scene.add(this.points);

        if (this.uiManager) this.uiManager.onPointsRegenerated();
    }

    handleSettingChange(prop, _value) {
        if (prop === 'pointCount') {
            this.generatePoints();
        } else if (prop === 'pointSize' || prop === 'opacity') {
            if (this.points?.material) {
                this.points.material.size = this.settings.pointSize;
                this.points.material.opacity = this.settings.opacity;
                this.points.material.transparent = this.settings.opacity < 1;
                this.points.material.needsUpdate = true;
            }
        }
    }

    /* ===== per-frame update (movement + look) ===== */
    updateMovement() {
        const sens = 0.002;
        const yaw = -this.mouseDX * sens;
        const dPitch = -this.mouseDY * sens;

        if (yaw) this.camera.rotateY(yaw);
        if (dPitch) {
            const newPitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitch + dPitch));
            this.camera.rotateX(newPitch - this.pitch);
            this.pitch = newPitch;
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
            const speed = this.settings.speed * (this.keys['ShiftLeft'] ? 3 : 1) * 0.016;
            this.velocity.normalize().multiplyScalar(speed);
            this.velocity.applyQuaternion(this.camera.quaternion);
            this.camera.position.add(this.velocity);
        }
    }

    /* ===== render loop ===== */
    animate() {
        requestAnimationFrame(() => this.animate());
        this.updateMovement();
        if (this.uiManager) this.uiManager.updateUI();
        this.renderer.render(this.scene, this.camera);
    }

    /* attach UI after construction */
    setUIManager(uiManager) { this.uiManager = uiManager; }
}
