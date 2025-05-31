class UIManager {
    constructor(pointCloud) {
        this.pointCloud = pointCloud;

        // UI configuration - single source of truth for all menus
        this.uiConfig = {
            help: {
                key: 'KeyH',
                elementId: 'helpMenu',
                shortcutText: 'h - help',
                visible: false
            },
            menu: {
                key: 'KeyM',
                elementId: 'controlsMenu',
                shortcutText: 'm - menu',
                visible: false
            },
            navbar: {
                key: 'KeyN',
                elementId: 'navbar',
                shortcutText: 'n - navbar',
                visible: false
            },
            stats: {
                key: 'KeyJ',
                elementId: 'statsMenu',
                shortcutText: 'j - stats',
                visible: false
            }
        };

        /* ---------- FPS tracking ---------- */
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 60;

        this.init();
    }

    init() {
        this.generateShortcutsHTML();
        this.setupControls();
        this.setupUI();
        this.setupNavball();
    }

    setupNavball() {
        // Create navball instance
        this.navball = new Navball('navball-container');
    }

    generateShortcutsHTML() {
        const shortcutsContainer = document.getElementById('shortcuts');

        // Clear existing content except for movement text
        shortcutsContainer.innerHTML = '<div>wasd to move</div><div>mouse+Q/E to change view</div>';

        // Add shortcuts based on config
        Object.entries(this.uiConfig).forEach(([action, config]) => {
            const shortcutDiv = document.createElement('div');
            shortcutDiv.className = 'shortcut-link';
            shortcutDiv.setAttribute('data-action', action);
            shortcutDiv.textContent = config.shortcutText;
            shortcutsContainer.appendChild(shortcutDiv);
        });
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
                this.pointCloud.settings[key] = v;
                c.display.textContent = v;
                this.pointCloud.handleSettingChange(key, v);
            });
        });

        window.addEventListener('resize', () => {
            this.pointCloud.camera.aspect = window.innerWidth / window.innerHeight;
            this.pointCloud.camera.updateProjectionMatrix();
            this.pointCloud.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    setupUI() {
        // UI toggle functionality - keyboard
        document.addEventListener('keydown', (e) => {
            // Handle UI keys regardless of pointer lock state
            Object.entries(this.uiConfig).forEach(([action, config]) => {
                if (e.code === config.key) {
                    e.preventDefault();
                    this.toggleUI(action);
                }
            });
        });

        // UI toggle functionality - mouse clicks on shortcuts
        document.getElementById('shortcuts').addEventListener('click', (e) => {
            const action = e.target.getAttribute('data-action');
            if (action && this.uiConfig[action]) {
                this.toggleUI(action);
            }
        });
    }

    toggleUI(type) {
        const config = this.uiConfig[type];
        if (!config) return;

        config.visible = !config.visible;
        const element = document.getElementById(config.elementId);
        element.style.display = config.visible ? 'block' : 'none';
    }

    updateUI() {
        // Update navbar
        if (this.uiConfig.navbar.visible) {
            const pos = this.pointCloud.camera.position;
            const camera = this.pointCloud.camera;

            // Get forward direction vector
            const forward = new THREE.Vector3(0, 0, -1);
            forward.applyQuaternion(camera.quaternion);

            // Calculate yaw and pitch from forward vector
            const yaw = Math.atan2(forward.x, -forward.z) * 180 / Math.PI;
            const pitch = Math.asin(forward.y) * 180 / Math.PI;

            // Update position
            document.getElementById('posX').textContent = pos.x.toFixed(1);
            document.getElementById('posY').textContent = pos.y.toFixed(1);
            document.getElementById('posZ').textContent = pos.z.toFixed(1);

            // Sync navball with camera
            if (this.navball) {
                this.navball.syncWithCamera(this.pointCloud.camera);
            }
        }

        // Update stats display
        if (this.uiConfig.stats.visible) {
            this.updateStatsDisplay();
        }
    }

    updateFPS() {
        this.frameCount++;
        const currentTime = performance.now();
        if (currentTime - this.lastTime >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime));
            this.frameCount = 0;
            this.lastTime = currentTime;
        }
    }

    updateStatsDisplay() {
        this.updateFPS();

        const frameTime = this.frameCount > 0 ? (performance.now() - this.lastTime) / this.frameCount : 16.7;

        // Update all stats elements
        const statsElements = {
            fps: document.querySelector('#statsMenu #fps'),
            frameTime: document.getElementById('frameTime'),
            renderedCount: document.querySelector('#statsMenu #renderedCount'),
            posX: document.getElementById('posX'),
            posY: document.getElementById('posY'),
            posZ: document.getElementById('posZ')
        };

        if (statsElements.fps) {
            statsElements.fps.textContent = this.fps;
        }
        if (statsElements.frameTime) {
            statsElements.frameTime.textContent = frameTime.toFixed(1) + 'ms';
        }
        if (statsElements.renderedCount) {
            statsElements.renderedCount.textContent = this.pointCloud.settings.pointCount;
        }

        // Update position
        const pos = this.pointCloud.camera.position;
        if (statsElements.posX) statsElements.posX.textContent = pos.x.toFixed(1);
        if (statsElements.posY) statsElements.posY.textContent = pos.y.toFixed(1);
        if (statsElements.posZ) statsElements.posZ.textContent = pos.z.toFixed(1);
    }

    // Called when point cloud regenerates
    onPointsRegenerated() {
        // Update stats if visible
        if (this.uiConfig.stats.visible) {
            const statsCount = document.querySelector('#statsMenu #renderedCount');
            if (statsCount) {
                statsCount.textContent = this.pointCloud.settings.pointCount;
            }
        }
    }
}