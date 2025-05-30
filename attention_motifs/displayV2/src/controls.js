class Controls {
    constructor(camera) {
        this.camera = camera;
        this.settings = {
            pointSize: 2.0,
            opacity: 0.8,
            renderDistance: 50,
            speed: 5,
            pointCount: 10000
        };

        this.keys = {};
        this.mouseX = 0;
        this.mouseY = 0;
        this.velocity = new THREE.Vector3();
        this.direction = new THREE.Vector3();

        this.onSettingChange = null;
        this.onWindowResize = null;
        this.onMouseMove = null;

        this.setupEventListeners();
        this.setupControls();
    }

    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            this.keys[e.code] = true;
        });

        document.addEventListener('keyup', (e) => {
            this.keys[e.code] = false;
        });

        document.addEventListener('mousemove', (e) => {
            if (document.pointerLockElement === document.body) {
                // Fixed: Reverse mouse X axis and Y axis
                this.mouseX -= e.movementX * 0.002;
                this.mouseY -= e.movementY * 0.002;
                this.mouseY = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.mouseY));
            }

            if (this.onMouseMove) {
                this.onMouseMove(e);
            }
        });

        // Fixed: Use double-click instead of single click for pointer lock
        document.addEventListener('dblclick', () => {
            if (document.pointerLockElement === document.body) {
                document.exitPointerLock();
            } else {
                document.body.requestPointerLock();
            }
        });

        // Add escape key to exit pointer lock
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Escape' && document.pointerLockElement === document.body) {
                document.exitPointerLock();
            }
        });

        window.addEventListener('resize', () => {
            if (this.onWindowResize) {
                this.onWindowResize();
            }
        });
    }

    setupControls() {
        const controlElements = {
            pointSize: document.getElementById('pointSize'),
            opacity: document.getElementById('opacity'),
            renderDistance: document.getElementById('lodDistance'),
            speed: document.getElementById('speed'),
            pointCount: document.getElementById('pointCount')
        };

        const valueElements = {
            pointSize: document.getElementById('pointSizeValue'),
            opacity: document.getElementById('opacityValue'),
            renderDistance: document.getElementById('lodDistanceValue'),
            speed: document.getElementById('speedValue'),
            pointCount: document.getElementById('pointCountValue')
        };

        Object.keys(controlElements).forEach(key => {
            const control = controlElements[key];
            const valueDisplay = valueElements[key];

            if (control && valueDisplay) {
                control.addEventListener('input', () => {
                    const value = parseFloat(control.value);
                    this.settings[key] = value;
                    valueDisplay.textContent = value;

                    if (this.onSettingChange) {
                        this.onSettingChange(key, value);
                    }
                });
            }
        });
    }

    updateMovement() {
        // Mouse look
        this.camera.rotation.set(this.mouseY, this.mouseX, 0, 'YXZ');

        // Movement
        this.velocity.set(0, 0, 0);

        // Fixed: Reverse forward/back movement
        if (this.keys['KeyW']) this.velocity.z += 1;  // Forward
        if (this.keys['KeyS']) this.velocity.z -= 1;  // Back
        if (this.keys['KeyA']) this.velocity.x -= 1;  // Left
        if (this.keys['KeyD']) this.velocity.x += 1;  // Right

        const speedMultiplier = this.keys['ShiftLeft'] ? 3 : 1;
        this.velocity.normalize().multiplyScalar(this.settings.speed * speedMultiplier * 0.016);

        // Apply movement relative to camera direction
        this.direction.set(0, 0, -1);
        this.direction.applyQuaternion(this.camera.quaternion);

        const right = new THREE.Vector3();
        right.crossVectors(this.direction, this.camera.up).normalize();

        const forward = new THREE.Vector3();
        forward.crossVectors(this.camera.up, right).normalize();

        this.camera.position.addScaledVector(right, this.velocity.x);
        this.camera.position.addScaledVector(forward, this.velocity.z);
    }

    isMoving() {
        return this.velocity.length() > 0;
    }

    setOnSettingChange(callback) {
        this.onSettingChange = callback;
    }

    setOnWindowResize(callback) {
        this.onWindowResize = callback;
    }

    setOnMouseMove(callback) {
        this.onMouseMove = callback;
    }

    checkHover(event, activeVoxels, allVoxels) {
        // Simple hover implementation - you can expand this
        const hoverInfo = document.getElementById('hover-info');
        if (hoverInfo) {
            hoverInfo.style.display = 'none';
        }
    }
}