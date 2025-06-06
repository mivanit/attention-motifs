/* PointCloud.js – snap-to-point cross-hair, optional hover UI ("k"),
   optional click-to-select ("b"), optional right-click action ("o"), and better picking accuracy using CONFIG. */

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
        this.pitch = 0;
        this.mouseDX = 0;
        this.mouseDY = 0;
        this.rollSpeed = CONFIG.movement.rollSpeed;
        this.velocity = new THREE.Vector3();

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

        /* right-click handler */
        window.addEventListener('contextmenu', (e) => {
            if (!this.rightClickActive || this.hoverId == null) return;
            e.preventDefault(); // Prevent default context menu

            const row = this.model.row(this.hoverId);
            this._handleRightClick(row, this.hoverId);
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
        this.camera.position.set(0, 0, 0);
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
                (this.keys['ShiftLeft'] ? CONFIG.movement.sprintMultiplier : 1) * 0.016;
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