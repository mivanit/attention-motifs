class PointCloud {
    constructor() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            2000
        );
        this.renderer = new THREE.WebGLRenderer({ antialias: true });

        /* ---------- point-cloud settings ---------- */
        this.points = null;
        this.settings = {
            pointSize: 3.0,
            opacity: 0.8,
            speed: 10,
            pointCount: 25000
        };

        /* ---------- movement state ---------- */
        this.keys = {};
        this.pitch = 0;            // current total pitch, for clamping
        this.mouseDX = 0;            // frame-accumulated mouse delta X
        this.mouseDY = 0;            // frame-accumulated mouse delta Y
        this.rollSpeed = 0.02;         // radians per frame when Q/E held
        this.velocity = new THREE.Vector3();

        /* ---------- UI state ---------- */
        this.uiState = {
            helpVisible: false,
            menuVisible: false,
            navbarVisible: false,
            statsVisible: false
        };

        /* ---------- FPS tracking ---------- */
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 60;

        this.init();
    }

    /* ===== initialisation ===== */
    init() {
        this.setupRenderer();
        this.setupControls();
        this.setupMovement();
        this.setupUI();
        this.generatePoints();
        this.animate();
    }

    setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000011);
        document.getElementById('container').appendChild(this.renderer.domElement);
        /* start inside the cloud, looking −Z */
        this.camera.position.set(0, 0, 0);
    }

    setupControls() {
        const controls = {
            pointSize: {
                element: document.getElementById('pointSize'),
                display: document.getElementById('pointSizeValue')
            },
            opacity: {
                element: document.getElementById('opacity'),
                display: document.getElementById('opacityValue')
            },
            speed: {
                element: document.getElementById('speed'),
                display: document.getElementById('speedValue')
            },
            pointCount: {
                element: document.getElementById('pointCount'),
                display: document.getElementById('pointCountValue')
            }
        };

        Object.keys(controls).forEach(key => {
            const c = controls[key];
            c.element.addEventListener('input', () => {
                const v = parseFloat(c.element.value);
                this.settings[key] = v;
                c.display.textContent = v;
                this.handleSettingChange(key, v);
            });
        });

        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    setupUI() {
        // UI toggle functionality - keyboard
        document.addEventListener('keydown', (e) => {
            // Handle UI keys regardless of pointer lock state
            switch (e.code) {
                case 'KeyH':
                    e.preventDefault();
                    this.toggleUI('help');
                    break;
                case 'KeyM':
                    e.preventDefault();
                    this.toggleUI('menu');
                    break;
                case 'KeyN':
                    e.preventDefault();
                    this.toggleUI('navbar');
                    break;
                case 'KeyS':
                    e.preventDefault();
                    this.toggleUI('stats');
                    break;
            }
        });

        // UI toggle functionality - mouse clicks on shortcuts
        document.getElementById('shortcuts').addEventListener('click', (e) => {
            const action = e.target.getAttribute('data-action');
            if (action) {
                this.toggleUI(action);
            }
        });
    }

    toggleUI(type) {
        const elements = {
            help: document.getElementById('helpMenu'),
            menu: document.getElementById('controlsMenu'),
            navbar: document.getElementById('navbar'),
            stats: document.getElementById('statsMenu')
        };

        const stateKey = type + 'Visible';
        this.uiState[stateKey] = !this.uiState[stateKey];
        elements[type].style.display = this.uiState[stateKey] ? 'block' : 'none';
    }

    /* ===== input / pointer-lock ===== */
    setupMovement() {
        document.addEventListener('keydown', e => {
            this.keys[e.code] = true;
        });
        document.addEventListener('keyup', e => {
            this.keys[e.code] = false;
        });

        /* accumulate raw mouse deltas only while in pointer-lock */
        document.addEventListener('mousemove', e => {
            if (document.pointerLockElement === document.body) {
                this.mouseDX += e.movementX;
                this.mouseDY += e.movementY;
            }
        });

        /* double-click toggles pointer-lock */
        document.addEventListener('dblclick', () => {
            (document.pointerLockElement === document.body)
                ? document.exitPointerLock()
                : document.body.requestPointerLock();
        });

        /* ESC releases pointer-lock */
        document.addEventListener('keydown', e => {
            if (e.code === 'Escape' && document.pointerLockElement === document.body) {
                document.exitPointerLock();
            }
        });
    }

    /* ===== point-cloud generation ===== */
    generatePoints() {
        if (this.points) {
            this.scene.remove(this.points);
            this.points.geometry.dispose();
            this.points.material.dispose();
        }

        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
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

        // Update navbar if visible
        if (this.uiState.navbarVisible) {
            document.getElementById('renderedCount').textContent = this.settings.pointCount;
        }

        // Update stats if visible
        if (this.uiState.statsVisible) {
            this.updateStatsDisplay();
        }
    }

    handleSettingChange(prop, _value) {
        if (prop === 'pointCount') {
            this.generatePoints();
        } else if (prop === 'pointSize' || prop === 'opacity') {
            if (this.points && this.points.material) {
                this.points.material.size = this.settings.pointSize;
                this.points.material.opacity = this.settings.opacity;
                this.points.material.transparent = this.settings.opacity < 1;
                this.points.material.needsUpdate = true;
            }
        }
    }

    /* ===== per-frame update (movement + look) ===== */
    updateMovement() {
        /* --- local yaw / pitch from mouse deltas --- */
        const sens = 0.002;
        const yaw = -this.mouseDX * sens;
        const dPitch = -this.mouseDY * sens;

        if (yaw !== 0) this.camera.rotateY(yaw);          // local-Y axis
        if (dPitch !== 0) {
            const newPitch = Math.max(
                -Math.PI / 2,
                Math.min(Math.PI / 2, this.pitch + dPitch)
            );
            this.camera.rotateX(newPitch - this.pitch);   // local-X axis
            this.pitch = newPitch;
        }
        this.mouseDX = this.mouseDY = 0;                  // reset per frame

        /* --- roll keys (local-Z axis) --- */
        if (this.keys['KeyQ']) this.camera.rotateZ(this.rollSpeed);
        if (this.keys['KeyE']) this.camera.rotateZ(-this.rollSpeed);

        /* --- translation (WASD, Shift for sprint) --- */
        this.velocity.set(0, 0, 0);
        if (this.keys['KeyW']) this.velocity.z -= 1;
        if (this.keys['KeyS']) this.velocity.z += 1;
        if (this.keys['KeyA']) this.velocity.x -= 1;
        if (this.keys['KeyD']) this.velocity.x += 1;

        if (this.velocity.lengthSq() !== 0) {
            const speed = this.settings.speed *
                (this.keys['ShiftLeft'] ? 3 : 1) *
                0.016;                 // ~16 ms/frame scalar
            this.velocity.normalize().multiplyScalar(speed);
            this.velocity.applyQuaternion(this.camera.quaternion);   // to world space
            this.camera.position.add(this.velocity);
        }
    }

    updateUI() {
        // Update navbar
        if (this.uiState.navbarVisible) {
            const pos = this.camera.position;
            document.getElementById('position').textContent =
                `${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)}`;

            // Update FPS
            this.frameCount++;
            const currentTime = performance.now();
            if (currentTime - this.lastTime >= 1000) {
                this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime));
                document.getElementById('fps').textContent = this.fps;
                this.frameCount = 0;
                this.lastTime = currentTime;
            }
        }

        // Update stats display
        if (this.uiState.statsVisible) {
            this.updateStatsDisplay();
        }
    }

    updateStatsDisplay() {
        // Update FPS calculation
        this.frameCount++;
        const currentTime = performance.now();
        if (currentTime - this.lastTime >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime));
            const frameTime = (currentTime - this.lastTime) / this.frameCount;

            document.getElementById('fps').textContent = this.fps;
            document.getElementById('frameTime').textContent = frameTime.toFixed(1) + 'ms';
            document.getElementById('renderedCount').textContent = this.settings.pointCount;

            this.frameCount = 0;
            this.lastTime = currentTime;
        }

        // Update position
        const pos = this.camera.position;
        document.getElementById('posX').textContent = pos.x.toFixed(1);
        document.getElementById('posY').textContent = pos.y.toFixed(1);
        document.getElementById('posZ').textContent = pos.z.toFixed(1);
    }

    /* ===== render loop ===== */
    animate() {
        requestAnimationFrame(() => this.animate());
        this.updateMovement();
        this.updateUI();
        this.renderer.render(this.scene, this.camera);
    }
}