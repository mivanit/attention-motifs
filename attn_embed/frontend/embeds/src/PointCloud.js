/* PointCloud.js – Enhanced with mobile touch controls for pinch zoom, pan, tap, and hold */

class PointCloud {
    /** @param {DataModel} model */
    constructor(model) {
        this.model = model;

        /* ── THREE basics - using CONFIG values ──────────────────── */
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            CONFIG.rendering.cameraFov,
            window.innerWidth / window.innerHeight,
            CONFIG.rendering.cameraNear,
            CONFIG.rendering.cameraFar
        );
        this.renderer = new THREE.WebGLRenderer({ antialias: CONFIG.rendering.antialiasing });

        /* ── picking helpers ──────────────────────────────────── */
        this.raycaster = new THREE.Raycaster();
        this.raycaster.params.Points = { threshold: CONFIG.interaction.raycastThreshold };
        this.pointerNDC = new THREE.Vector2();
        this.pointerScreen = { x: 0, y: 0 };

        this.hoverId = null;
        this.prevHoverId = null;

        /* ── behaviour flags (from CONFIG) ────────────────────── */
        this.hoverActive = CONFIG.interaction.hoverActive;
        this.selectOnClick = CONFIG.interaction.selectOnClick;
        this.rightClickActive = CONFIG.interaction.rightClickActive;

        /* ── mobile touch state ─────────────────────────────── */
        this.touchState = {
            touching: false,
            touches: new Map(), // trackingId -> {x, y, startTime}
            lastDistance: 0,
            isPinching: false,
            isPanning: false,
            panDisablesInteraction: false,
            singleTapTimer: null,
            holdTimer: null,
            lastTapTime: 0,
            tapThreshold: 200, // ms for tap vs hold
            doubleTapThreshold: 300, // ms for double tap detection
            holdThreshold: 500, // ms for hold detection
            moveThreshold: 10, // pixels before it's considered a drag
            zoomSensitivity: 0.01,
            panSensitivity: 0.002
        };

        /* ── colour / selection state ─────────────────────────── */
        this.state = new VisState(model);
        this.selMgr = new SelectionManager(model, this.state);
        this.state.addEventListener('selection', () => this._updateColors());
        this.state.addEventListener('vis', () => this._updateColors());

        /* ── viewer settings - using CONFIG ──────────────────── */
        this.settings = {
            pointSize: 0.1,
            opacity: 0.8,
            speed: CONFIG.movement.speed
        };

        /* ── movement bookkeeping ──────────────────────────────── */
        this.keys = {};
        this.pitch = CONFIG.camera.rotation.pitch; // Initialize from CONFIG
        this.mouseDX = 0;
        this.mouseDY = 0;
        this.rollSpeed = CONFIG.movement.rollSpeed;
        this.velocity = new THREE.Vector3();

        /* camera sync debouncing */
        this.cameraSyncTimeout = null;
        this.cameraSyncDelay = 1000; // Sync camera to URL after 1 second of no movement

        /* ── cross-hair objects ───────────────────────────────── */
        this.crossH = null;
        this.crossV = null;
        this._createCrosshairs();

        /* bootstrap */
        this._init();
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
            this._updatePointerPosition(e.clientX, e.clientY);
        });

        /* click-to-select (can be disabled) */
        window.addEventListener('click', (e) => {
            // Don't process clicks during touch interactions
            if (this.touchState.panDisablesInteraction) return;

            if (!this.selectOnClick || this.hoverId == null) return;
            const v = this.model.row(this.hoverId)[this.state.selectBy];
            this.state.toggleValue(v);
        });

        /* right-click handler */
        window.addEventListener('contextmenu', (e) => {
            if (!this.rightClickActive || this.hoverId == null) return;
            e.preventDefault(); // Prevent default context menu

            const row = this.model.row(this.hoverId);
            this._handleRightClick(row, this.hoverId);
        });

        /* ── Touch event handlers ─────────────────────────────── */
        this._setupTouchEvents();

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

    _setupTouchEvents() {
        const canvas = this.renderer.domElement;

        // Prevent default touch behaviors that interfere with our controls
        canvas.addEventListener('touchstart', e => e.preventDefault(), { passive: false });
        canvas.addEventListener('touchmove', e => e.preventDefault(), { passive: false });
        canvas.addEventListener('touchend', e => e.preventDefault(), { passive: false });

        canvas.addEventListener('touchstart', this._onTouchStart.bind(this));
        canvas.addEventListener('touchmove', this._onTouchMove.bind(this));
        canvas.addEventListener('touchend', this._onTouchEnd.bind(this));
        canvas.addEventListener('touchcancel', this._onTouchEnd.bind(this));
    }

    _onTouchStart(event) {
        const now = performance.now();
        this.touchState.touching = true;

        // Update touch tracking
        for (const touch of event.touches) {
            this.touchState.touches.set(touch.identifier, {
                x: touch.clientX,
                y: touch.clientY,
                startX: touch.clientX,
                startY: touch.clientY,
                startTime: now
            });
        }

        const touchCount = event.touches.length;

        if (touchCount === 1) {
            // Single touch - potential tap, hold, or pan start
            const touch = event.touches[0];
            this._updatePointerPosition(touch.clientX, touch.clientY);

            // Set up hold timer
            this.touchState.holdTimer = setTimeout(() => {
                this._onTouchHold(touch);
            }, this.touchState.holdThreshold);

            // Check for double tap
            const timeSinceLastTap = now - this.touchState.lastTapTime;
            if (timeSinceLastTap < this.touchState.doubleTapThreshold) {
                this._onDoubleTap(touch);
                this.touchState.lastTapTime = 0; // Reset to prevent triple tap
            }

        } else if (touchCount === 2) {
            // Two touches - pinch zoom setup
            this._clearTouchTimers();
            this.touchState.isPinching = true;
            this.touchState.panDisablesInteraction = true;

            const touch1 = event.touches[0];
            const touch2 = event.touches[1];
            this.touchState.lastDistance = this._getTouchDistance(touch1, touch2);
        }
    }

    _onTouchMove(event) {
        if (!this.touchState.touching) return;

        const touchCount = event.touches.length;

        if (touchCount === 1 && !this.touchState.isPinching) {
            // Single touch movement - pan or cancel tap
            const touch = event.touches[0];
            const stored = this.touchState.touches.get(touch.identifier);

            if (stored) {
                const moveDistance = Math.sqrt(
                    Math.pow(touch.clientX - stored.startX, 2) +
                    Math.pow(touch.clientY - stored.startY, 2)
                );

                // If moved beyond threshold, start panning
                if (moveDistance > this.touchState.moveThreshold) {
                    this._clearTouchTimers(); // Cancel tap/hold
                    this.touchState.isPanning = true;
                    this.touchState.panDisablesInteraction = true;

                    // Apply pan rotation
                    const deltaX = touch.clientX - stored.x;
                    const deltaY = touch.clientY - stored.y;

                    this._applyTouchPan(deltaX, deltaY);
                }

                // Update stored position
                stored.x = touch.clientX;
                stored.y = touch.clientY;
                this._updatePointerPosition(touch.clientX, touch.clientY);
            }

        } else if (touchCount === 2 && this.touchState.isPinching) {
            // Two finger pinch zoom
            const touch1 = event.touches[0];
            const touch2 = event.touches[1];
            const distance = this._getTouchDistance(touch1, touch2);

            if (this.touchState.lastDistance > 0) {
                const deltaDistance = distance - this.touchState.lastDistance;
                this._applyTouchZoom(deltaDistance);
            }

            this.touchState.lastDistance = distance;

            // Update touch positions
            this.touchState.touches.set(touch1.identifier, {
                ...this.touchState.touches.get(touch1.identifier),
                x: touch1.clientX,
                y: touch1.clientY
            });
            this.touchState.touches.set(touch2.identifier, {
                ...this.touchState.touches.get(touch2.identifier),
                x: touch2.clientX,
                y: touch2.clientY
            });
        }
    }

    _onTouchEnd(event) {
        const now = performance.now();

        // Remove ended touches from tracking
        const activeTouchIds = new Set(Array.from(event.touches).map(t => t.identifier));
        for (const [id] of this.touchState.touches) {
            if (!activeTouchIds.has(id)) {
                const stored = this.touchState.touches.get(id);

                // Check if this was a quick tap (not moved, not held)
                if (stored && !this.touchState.isPanning && !this.touchState.isPinching) {
                    const duration = now - stored.startTime;
                    const moveDistance = Math.sqrt(
                        Math.pow(stored.x - stored.startX, 2) +
                        Math.pow(stored.y - stored.startY, 2)
                    );

                    if (duration < this.touchState.tapThreshold &&
                        moveDistance < this.touchState.moveThreshold) {
                        this._onSingleTap(stored);
                    }
                }

                this.touchState.touches.delete(id);
            }
        }

        // Reset states when no touches remain
        if (event.touches.length === 0) {
            this.touchState.touching = false;
            this.touchState.isPinching = false;

            // Re-enable interaction after a short delay to prevent accidental clicks
            setTimeout(() => {
                this.touchState.isPanning = false;
                this.touchState.panDisablesInteraction = false;
            }, 100);

            this._clearTouchTimers();
        }
    }

    _onSingleTap(touchData) {
        // Single tap acts like hover - update pointer and show hover info
        this._updatePointerPosition(touchData.x, touchData.y);
        this.touchState.lastTapTime = performance.now();

        // Briefly show hover for touch devices
        if (this.hoverActive && this.hoverId !== null) {
            // Force a hover update
            this._updateCrosshairs(this.hoverId);
        }
    }

    _onDoubleTap(touch) {
        // Double tap acts like click-to-select
        this._clearTouchTimers();

        if (this.selectOnClick && this.hoverId !== null) {
            const v = this.model.row(this.hoverId)[this.state.selectBy];
            this.state.toggleValue(v);
        }
    }

    _onTouchHold(touch) {
        // Hold acts like right-click
        if (this.rightClickActive && this.hoverId !== null) {
            const row = this.model.row(this.hoverId);
            this._handleRightClick(row, this.hoverId);
        }
    }

    _applyTouchPan(deltaX, deltaY) {
        const sens = this.touchState.panSensitivity;
        const yaw = -deltaX * sens;
        const pitchDelta = -deltaY * sens;

        if (yaw) {
            this.camera.rotateY(yaw);
            CONFIG.camera.rotation.yaw += yaw;
            this._debounceCameraSync();
        }
        if (pitchDelta) {
            const newPitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitch + pitchDelta));
            this.camera.rotateX(newPitch - this.pitch);
            this.pitch = newPitch;
            CONFIG.camera.rotation.pitch = this.pitch;
            this._debounceCameraSync();
        }
    }

    _applyTouchZoom(deltaDistance) {
        // Convert pinch distance to forward/backward movement
        const zoomAmount = deltaDistance * this.touchState.zoomSensitivity;

        // Move camera forward/backward along its local Z axis
        const zoomVector = new THREE.Vector3(0, 0, -zoomAmount);
        zoomVector.applyQuaternion(this.camera.quaternion);
        this.camera.position.add(zoomVector);

        // Update CONFIG to sync position to URL
        CONFIG.camera.position.x = this.camera.position.x;
        CONFIG.camera.position.y = this.camera.position.y;
        CONFIG.camera.position.z = this.camera.position.z;
        this._debounceCameraSync();
    }

    _getTouchDistance(touch1, touch2) {
        const dx = touch1.clientX - touch2.clientX;
        const dy = touch1.clientY - touch2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    _updatePointerPosition(x, y) {
        this.pointerScreen.x = x;
        this.pointerScreen.y = y;
        this.pointerNDC.x = (x / window.innerWidth) * 2 - 1;
        this.pointerNDC.y = -(y / window.innerHeight) * 2 + 1;
    }

    _clearTouchTimers() {
        if (this.touchState.singleTapTimer) {
            clearTimeout(this.touchState.singleTapTimer);
            this.touchState.singleTapTimer = null;
        }
        if (this.touchState.holdTimer) {
            clearTimeout(this.touchState.holdTimer);
            this.touchState.holdTimer = null;
        }
    }

    /* ---------- right-click handling and template system ----- */
    _handleRightClick(row, pointId) {
        const templateData = this._buildTemplateData(row, pointId);

        if (CONFIG.rightClick.mode === "content") {
            this._openContentTab(templateData);
        } else if (CONFIG.rightClick.mode === "url") {
            this._openUrlTab(templateData);
        }
    }

    _buildTemplateData(row, pointId) {
        const a = this.state.axis;
        const coords = {
            x: this.model.getCoord(pointId, a.x).toFixed(3),
            y: this.model.getCoord(pointId, a.y).toFixed(3),
            z: this.model.getCoord(pointId, a.z).toFixed(3)
        };

        return {
            // All row data (flattened with dot notation for nested objects)
            ...this._flattenObject(row),

            // Coordinate data
            'coord.x': coords.x,
            'coord.y': coords.y,
            'coord.z': coords.z,

            // Axis information
            'axis.x.name': this.model.numericCols[a.x],
            'axis.x.index': a.x,
            'axis.y.name': this.model.numericCols[a.y],
            'axis.y.index': a.y,
            'axis.z.name': this.model.numericCols[a.z],
            'axis.z.index': a.z,

            // Config data (flattened)
            ...this._flattenObject(CONFIG, 'config.')
        };
    }

    _flattenObject(obj, prefix = '') {
        const flattened = {};

        for (const key in obj) {
            const value = obj[key];
            const newKey = prefix + key;

            if (value && typeof value === 'object' && !Array.isArray(value)) {
                Object.assign(flattened, this._flattenObject(value, newKey + '.'));
            } else {
                flattened[newKey] = value;
            }
        }

        return flattened;
    }

    _replaceTemplate(template, data) {
        return template.replace(/\{([^}]+)\}/g, (match, key) => {
            const value = data[key];
            if (value !== undefined && value !== null) {
                return String(value);
            }
            return match; // Keep original if no replacement found
        });
    }

    _openContentTab(templateData) {
        const title = this._replaceTemplate(CONFIG.rightClick.content.title, templateData);
        const content = this._replaceTemplate(CONFIG.rightClick.content.template, templateData);

        // Create HTML content with monospace styling
        const htmlContent = `<!DOCTYPE html>
<html>
<head>
	<title>${this._escapeHtml(title)}</title>
	<style>
		body {
			font-family: 'Courier New', monospace;
			background: #000;
			color: #00ff00;
			padding: 20px;
			margin: 0;
			white-space: pre-wrap;
			line-height: 1.4;
		}
	</style>
</head>
<body>${this._escapeHtml(content)}</body>
</html>`;

        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const newWindow = window.open(url, '_blank');

        // Clean up after a delay
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    _openUrlTab(templateData) {
        const url = this._replaceTemplate(CONFIG.rightClick.url.template, templateData);

        // URL encode the final URL to handle special characters
        try {
            window.open(url, '_blank');
        } catch (error) {
            console.error('Failed to open URL:', url, error);
            NOTIF.error(`Failed to open URL: ${url}`);
        }
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
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
        this.renderer.setClearColor(CONFIG.rendering.clearColor);
        document.getElementById('container').appendChild(this.renderer.domElement);

        // Set camera position and rotation from CONFIG
        this.camera.position.set(
            CONFIG.camera.position.x,
            CONFIG.camera.position.y,
            CONFIG.camera.position.z
        );

        // Apply rotation (yaw, pitch, roll)
        this.camera.rotation.set(0, 0, 0); // Reset first
        this.camera.rotateY(CONFIG.camera.rotation.yaw);
        this.camera.rotateX(CONFIG.camera.rotation.pitch);
        this.camera.rotateZ(CONFIG.camera.rotation.roll);

        // Set internal pitch tracking to match config
        this.pitch = CONFIG.camera.rotation.pitch;
    }

    /* ---------- in-scene cross-hair - using CONFIG ----------- */
    _createCrosshairs() {
        const mat = new THREE.LineBasicMaterial({
            color: CONFIG.crosshair.color,
            transparent: true,
            opacity: CONFIG.crosshair.opacity,
            depthTest: false
        });

        /* horizontal (X-axis) */
        const gH = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-CONFIG.crosshair.length, 0, 0),
            new THREE.Vector3(CONFIG.crosshair.length, 0, 0)
        ]);
        this.crossH = new THREE.Line(gH, mat);
        this.crossH.visible = false;
        this.scene.add(this.crossH);

        /* vertical (Y-axis) */
        const gV = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, -CONFIG.crosshair.length, 0),
            new THREE.Vector3(0, CONFIG.crosshair.length, 0)
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

    /* ---------- build / rebuild point geometry --------------- */
    _buildGeometry() {
        if (this.points) {
            this.scene.remove(this.points);
            this.points.geometry.dispose();
            this.points.material.dispose();
        }

        const pos = [];
        const col = [];
        const sizes = [];
        const opacities = [];

        const a = this.state.axis;
        for (let i = 0; i < this.model.rowCount; ++i) {
            pos.push(
                this.model.getCoord(i, a.x),
                this.model.getCoord(i, a.y),
                this.model.getCoord(i, a.z)
            );
            col.push(0.6, 0.6, 0.6);
            sizes.push(1.0); // Default size, will be updated in _updateColors
            opacities.push(1.0); // Default opacity, will be updated in _updateColors
        }

        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
        geom.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
        geom.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
        geom.setAttribute('opacity', new THREE.Float32BufferAttribute(opacities, 1));

        // Use shader material for per-point sizes
        const mat = new THREE.ShaderMaterial({
            uniforms: {
                baseSize: { value: 0.1 }
            },
            vertexShader: `
                attribute float size;
                attribute vec3 color;
                attribute float opacity;
                uniform float baseSize;
                varying vec3 vColor;
                varying float vOpacity;
                
                void main() {
                    vColor = color;
                    vOpacity = opacity;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = size * baseSize * (300.0 / -mvPosition.z);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                varying vec3 vColor;
                varying float vOpacity;
                
                void main() {
                    // Optimized circle calculation - avoid expensive sqrt
                    vec2 center = gl_PointCoord - 0.5;
                    float dist2 = dot(center, center);
                    if (dist2 > 0.25) discard; // 0.25 = 0.5^2
                    
                    gl_FragColor = vec4(vColor, vOpacity);
                }
            `,
            transparent: false, // Will be dynamically set based on opacity values
            depthWrite: true   // Will be set to false when transparent
        });

        this.points = new THREE.Points(geom, mat);
        this.colorAttr = geom.getAttribute('color');
        this.sizeAttr = geom.getAttribute('size');
        this.opacityAttr = geom.getAttribute('opacity');
        this.scene.add(this.points);

        if (this.uiManager) this.uiManager.onPointsRegenerated();
        this._updateColors();
    }

    handleSettingChange(prop) {
        if (prop === 'pointSize' || prop === 'opacity') {
            this.points.material.uniforms.baseSize.value = this.settings.pointSize;
            this.points.material.needsUpdate = true;

            /* update picking tolerance to roughly match size - using CONFIG */
            this.raycaster.params.Points.threshold = this.settings.pointSize * CONFIG.interaction.raycastThresholdMultiplier;
        }
    }

    /* ---------- recolour all points --------------------------- */
    _updateColors() {
        const colorArray = this.colorAttr.array;
        const sizeArray = this.sizeAttr.array;
        const opacityArray = this.opacityAttr.array;

        // Check if we need transparency for any points
        let needsTransparency = false;

        for (let i = 0; i < this.colorAttr.count; ++i) {
            const attrs = this.selMgr.attrs(i);
            // Set RGB color values (don't multiply by opacity)
            colorArray[i * 3] = attrs.r;
            colorArray[i * 3 + 1] = attrs.g;
            colorArray[i * 3 + 2] = attrs.b;

            // Set per-point size and opacity
            sizeArray[i] = attrs.size;
            opacityArray[i] = attrs.opacity;

            // Check if we need transparency
            if (attrs.opacity < 0.95) {
                needsTransparency = true;
            }
        }

        // Update material transparency setting
        if (this.points.material.transparent !== needsTransparency) {
            this.points.material.transparent = needsTransparency;
            this.points.material.depthWrite = !needsTransparency; // Disable depth write for transparency
            this.points.material.needsUpdate = true;
        }

        this.colorAttr.needsUpdate = true;
        this.sizeAttr.needsUpdate = true;
        this.opacityAttr.needsUpdate = true;
    }

    /* ---------- camera motion - using CONFIG values ---------- */
    _moveCamera() {
        const sens = CONFIG.movement.mouseSensitivity;
        const yaw = -this.mouseDX * sens;
        const dp = -this.mouseDY * sens;

        if (yaw) {
            this.camera.rotateY(yaw);
            // Update CONFIG and debounce URL sync
            CONFIG.camera.rotation.yaw += yaw;
            this._debounceCameraSync();
        }
        if (dp) {
            const np = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitch + dp));
            this.camera.rotateX(np - this.pitch);
            this.pitch = np;
            // Update CONFIG and debounce URL sync
            CONFIG.camera.rotation.pitch = this.pitch;
            this._debounceCameraSync();
        }
        this.mouseDX = this.mouseDY = 0;

        if (this.keys['KeyQ']) {
            this.camera.rotateZ(this.rollSpeed);
            CONFIG.camera.rotation.roll += this.rollSpeed;
            this._debounceCameraSync();
        }
        if (this.keys['KeyE']) {
            this.camera.rotateZ(-this.rollSpeed);
            CONFIG.camera.rotation.roll -= this.rollSpeed;
            this._debounceCameraSync();
        }

        this.velocity.set(0, 0, 0);
        if (this.keys['KeyW']) this.velocity.z -= 1;
        if (this.keys['KeyS']) this.velocity.z += 1;
        if (this.keys['KeyA']) this.velocity.x -= 1;
        if (this.keys['KeyD']) this.velocity.x += 1;

        if (this.velocity.lengthSq()) {
            const speed = this.settings.speed *
                (this.keys['ShiftLeft'] ? CONFIG.movement.sprintMultiplier : 1) * 0.016;
            this.velocity.normalize().multiplyScalar(speed)
                .applyQuaternion(this.camera.quaternion);
            this.camera.position.add(this.velocity);

            // Update CONFIG and debounce URL sync
            CONFIG.camera.position.x = this.camera.position.x;
            CONFIG.camera.position.y = this.camera.position.y;
            CONFIG.camera.position.z = this.camera.position.z;
            this._debounceCameraSync();
        }
    }

    /* ---------- render loop ----------------------------------- */
    _animate() {
        requestAnimationFrame(() => this._animate());

        this._moveCamera();

        /* picking - skip during touch interactions to improve performance */
        if (!this.touchState.panDisablesInteraction) {
            this.raycaster.setFromCamera(this.pointerNDC, this.camera);
            const hit = this.raycaster.intersectObject(this.points, false)[0];
            this.hoverId = hit ? hit.index : null;

            if (this.hoverId !== this.prevHoverId) {
                this.prevHoverId = this.hoverId;
                this._updateCrosshairs(this.hoverId);
            }
        }

        if (this.uiManager) this.uiManager.updateUI();
        this.renderer.render(this.scene, this.camera);
    }

    /* ---------- debounced camera sync to URL ----------------- */
    _debounceCameraSync() {
        if (this.cameraSyncTimeout) {
            clearTimeout(this.cameraSyncTimeout);
        }
        this.cameraSyncTimeout = setTimeout(() => {
            updateURL();
            this.cameraSyncTimeout = null;
        }, this.cameraSyncDelay);
    }

    /* ---------- allow UIManager to attach --------------------- */
    setUIManager(ui) { this.uiManager = ui; }
}