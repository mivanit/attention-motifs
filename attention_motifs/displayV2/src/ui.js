class UIManager {
    constructor(pointCloud) {
        this.pointCloud = pointCloud;

        /* ── central UI config ── */
        this.uiConfig = {
            help: { key: 'KeyH', elementId: 'helpMenu', shortcutText: 'h - help', visible: false },
            menu: { key: 'KeyM', elementId: 'controlsMenu', shortcutText: 'm - menu', visible: false },
            navbar: { key: 'KeyN', elementId: 'navbar', shortcutText: 'n - navbar', visible: false },
            stats: { key: 'KeyJ', elementId: 'statsMenu', shortcutText: 'j - stats', visible: false }
        };

        /* ── FPS tracking ── */
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 60;

        this.init();
    }

    /* ─────────────────────────────────────────── */

    init() {
        this.generateShortcutsHTML();
        this.setupControls();
        this.setupUI();
        this.setupNavball();

        /* hover panel */
        this.hoverPanel = document.createElement('div');
        this.hoverPanel.className = 'hover-panel';
        document.body.appendChild(this.hoverPanel);
    }

    setupNavball() { this.navball = new Navball('navball-container'); }

    generateShortcutsHTML() {
        const sc = document.getElementById('shortcuts');
        sc.innerHTML =
            '<div>wasd to move</div><div>mouse&nbsp;+&nbsp;Q/E&nbsp;to roll view</div>';

        Object.entries(this.uiConfig).forEach(([action, cfg]) => {
            const div = document.createElement('div');
            div.className = 'shortcut-link';
            div.dataset.action = action;
            div.textContent = cfg.shortcutText;
            sc.appendChild(div);
        });
    }

    /* only pointSize / opacity / speed controls now */
    setupControls() {
        const controls = {
            pointSize: { el: 'pointSize', disp: 'pointSizeValue' },
            opacity: { el: 'opacity', disp: 'opacityValue' },
            speed: { el: 'speed', disp: 'speedValue' }
        };

        Object.values(controls).forEach(cfg => {
            const slider = document.getElementById(cfg.el);
            const display = document.getElementById(cfg.disp);
            slider.addEventListener('input', () => {
                const v = parseFloat(slider.value);
                display.textContent = v;
                this.pointCloud.settings[cfg.el] = v;
                this.pointCloud.handleSettingChange(cfg.el, v);
            });
        });

        window.addEventListener('resize', () => {
            this.pointCloud.camera.aspect = window.innerWidth / window.innerHeight;
            this.pointCloud.camera.updateProjectionMatrix();
            this.pointCloud.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    setupUI() {
        document.addEventListener('keydown', e => {
            for (const [action, cfg] of Object.entries(this.uiConfig)) {
                if (e.code === cfg.key) { e.preventDefault(); this.toggleUI(action); }
            }
        });

        document.getElementById('shortcuts')
            .addEventListener('click', e => {
                const action = e.target.dataset.action;
                if (action) this.toggleUI(action);
            });
    }

    toggleUI(type) {
        const cfg = this.uiConfig[type];
        cfg.visible = !cfg.visible;
        document.getElementById(cfg.elementId).style.display =
            cfg.visible ? 'block' : 'none';
    }

    /* ───────── per-frame update ───────── */
    updateUI() {
        /* navbar position & navball */
        if (this.uiConfig.navbar.visible) {
            const p = this.pointCloud.camera.position;
            ['posX', 'posY', 'posZ'].forEach((id, i) =>
                document.getElementById(id).textContent = p[['x', 'y', 'z'][i]].toFixed(1));

            if (this.navball)
                this.navball.syncWithCameraQuaternion(this.pointCloud.camera.quaternion);
        }

        /* stats box */
        if (this.uiConfig.stats.visible) this.updateStatsDisplay();

        /* hover tooltip */
        this.showHover(this.pointCloud.hoverId);
    }

    /* FPS + counts */
    updateFPS() {
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastTime >= 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (now - this.lastTime));
            this.frameCount = 0;
            this.lastTime = now;
        }
    }

    updateStatsDisplay() {
        this.updateFPS();
        const frameTime = (1000 / Math.max(this.fps, 1)).toFixed(1) + ' ms';

        document.getElementById('fps').textContent = this.fps;
        document.getElementById('frameTime').textContent = frameTime;
        document.getElementById('renderedCount').textContent =
            this.pointCloud.points.geometry.getAttribute('position').count;

        const p = this.pointCloud.camera.position;
            document.getElementById('statsPosX').textContent = p.x.toFixed(3);
            document.getElementById('statsPosY').textContent = p.y.toFixed(3);
            document.getElementById('statsPosZ').textContent = p.z.toFixed(3);
    }

    /* called by PointCloud when geometry is rebuilt */
    onPointsRegenerated() {
        if (this.uiConfig.stats.visible)
            document.getElementById('renderedCount').textContent =
                this.pointCloud.points.geometry.getAttribute('position').count;
    }

    /* ───────── hover panel ───────── */
    showHover(rowId) {
        if (rowId == null) { this.hoverPanel.style.display = 'none'; return; }

        const row = this.pointCloud.model.row(rowId);
        this.hoverPanel.innerHTML =
            CONFIG.hoverColumns.map(c => `<b>${c}</b>: ${row[c]}`).join('<br>');

        const { x, y } = this.pointCloud.pointerScreen;
        this.hoverPanel.style.left = (x + 12) + 'px';
        this.hoverPanel.style.top = (y + 12) + 'px';
        this.hoverPanel.style.display = 'block';
    }
}
