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

        this.init();
    }

    /* ===== initialisation ===== */
    init() {
        this.setupRenderer();
        this.setupControls();
        this.setupMovement();
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

    /* ===== input / pointer-lock ===== */
    setupMovement() {
        document.addEventListener('keydown', e => { this.keys[e.code] = true; });
        document.addEventListener('keyup', e => { this.keys[e.code] = false; });

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
            if (e.code === 'Escape' && document.pointerLockElement === document.body)
                document.exitPointerLock();
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

        document.getElementById('renderedCount').textContent = this.settings.pointCount;
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

    /* ===== render loop ===== */
    animate() {
        requestAnimationFrame(() => this.animate());
        this.updateMovement();
        this.renderer.render(this.scene, this.camera);
    }
}
