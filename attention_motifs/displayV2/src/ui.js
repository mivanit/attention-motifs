/* UIManager – adds "k" to toggle hover UI / cross-hair
   and "b" to toggle click-to-select. */

class UIManager {
    constructor(pointCloud) {
        this.pointCloud = pointCloud;

        /* panel metadata ------------------------------------------------ */
        this.uiConfig = {
            help: { key: 'KeyH', elementId: 'helpMenu', shortcutText: 'h – help', visible: false },
            menu: { key: 'KeyM', elementId: 'controlsMenu', shortcutText: 'm – menu', visible: false },
            navbar: { key: 'KeyN', elementId: 'navbar', shortcutText: 'n – navball', visible: true },
            stats: { key: 'KeyJ', elementId: 'statsMenu', shortcutText: 'j – stats', visible: false }
        };

        /* categorical columns for c / v cycling ------------------------ */
        this.cats = this.pointCloud.model.df.columns
            .filter(c => !c.startsWith(CONFIG.numericalPrefix));
        this.colorIdx = Math.max(0, this.cats.indexOf(this.pointCloud.state.colorBy));
        this.selectIdx = Math.max(0, this.cats.indexOf(this.pointCloud.state.selectBy));

        /* FPS counters -------------------------------------------------- */
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 60;

        /* build static UI */
        this._init();
    }

    /* ========================================================= */
    _init() {
        this._buildShortcutsLegend();
        this._setupControlSliders();
        this._bindKeys();
        this._setupNavball();

        /* hover tooltip */
        this.hoverPanel = document.createElement('div');
        this.hoverPanel.className = 'hover-panel';
        document.body.appendChild(this.hoverPanel);
    }

    _setupNavball() {
        this.navball = new Navball('navball-container');
        // Show navball by default
        document.getElementById('navbar').style.display = 'block';
    }

    /* ---------- shortcuts legend (top-right) ------------------ */
    _buildShortcutsLegend() {
        const sc = document.getElementById('shortcuts');
        sc.innerHTML = '<div>wasd – move</div><div>mouse + Q/E – roll</div>';

        Object.values(this.uiConfig).forEach(cfg => {
            const d = document.createElement('div');
            d.className = 'shortcut-link';
            d.dataset.action = cfg.elementId;
            d.innerHTML = `${cfg.shortcutText} <span class="status-indicator ${cfg.visible ? 'status-enabled' : 'status-disabled'}">(${cfg.visible ? 'enabled' : 'disabled'})</span>`;
            sc.appendChild(d);
        });

        // Add hover and click-select shortcuts with status indicators
        sc.insertAdjacentHTML('beforeend', `
            <div class="shortcut-link" data-action="hover-toggle">k – hover UI <span class="status-indicator status-enabled" id="hover-status">(enabled)</span></div>
            <div class="shortcut-link" data-action="click-select-toggle">b – click-select <span class="status-indicator status-enabled" id="click-select-status">(enabled)</span></div>`);

        sc.addEventListener('click', e => {
            const target = e.target.closest('[data-action]');
            if (!target) return;

            const action = target.dataset.action;

            if (action === 'hover-toggle') {
                this.pointCloud.hoverActive = !this.pointCloud.hoverActive;
                this._updateStatusIndicator('hover-status', this.pointCloud.hoverActive);
                if (!this.pointCloud.hoverActive) this.hoverPanel.style.display = 'none';
            } else if (action === 'click-select-toggle') {
                this.pointCloud.selectOnClick = !this.pointCloud.selectOnClick;
                this._updateStatusIndicator('click-select-status', this.pointCloud.selectOnClick);
            } else {
                const entry = Object.entries(this.uiConfig)
                    .find(([, cfg]) => cfg.elementId === action);
                if (entry) {
                    this._togglePanel(entry[0]);
                    this._updatePanelStatusIndicator(target, this.uiConfig[entry[0]].visible);
                }
            }
        });
    }

    _updateStatusIndicator(elementId, enabled) {
        const statusEl = document.getElementById(elementId);
        statusEl.textContent = enabled ? '(enabled)' : '(disabled)';
        statusEl.className = `status-indicator ${enabled ? 'status-enabled' : 'status-disabled'}`;
    }

    _updatePanelStatusIndicator(element, visible) {
        const statusEl = element.querySelector('.status-indicator');
        statusEl.textContent = visible ? '(enabled)' : '(disabled)';
        statusEl.className = `status-indicator ${visible ? 'status-enabled' : 'status-disabled'}`;
    }

    /* ---------- sliders for size / opacity / speed ------------ */
    _setupControlSliders() {
        const map = {
            pointSize: { el: 'pointSize', disp: 'pointSizeValue' },
            opacity: { el: 'opacity', disp: 'opacityValue' },
            speed: { el: 'speed', disp: 'speedValue' }
        };

        Object.values(map).forEach(cfg => {
            const s = document.getElementById(cfg.el);
            const d = document.getElementById(cfg.disp);
            s.addEventListener('input', () => {
                const v = parseFloat(s.value);
                d.textContent = v;
                this.pointCloud.settings[cfg.el] = v;
                this.pointCloud.handleSettingChange(cfg.el, v);
            });
        });

        /* keep renderer sized */
        window.addEventListener('resize', () => {
            this.pointCloud.camera.aspect = window.innerWidth / window.innerHeight;
            this.pointCloud.camera.updateProjectionMatrix();
            this.pointCloud.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    /* ---------- key bindings ---------------------------------- */
    _bindKeys() {
        document.addEventListener('keydown', e => {
            /* panel toggles */
            for (const [name, cfg] of Object.entries(this.uiConfig)) {
                if (e.code === cfg.key) {
                    e.preventDefault();
                    this._togglePanel(name);
                    // Update the corresponding shortcut status
                    const shortcutEl = document.querySelector(`[data-action="${cfg.elementId}"]`);
                    if (shortcutEl) {
                        this._updatePanelStatusIndicator(shortcutEl, cfg.visible);
                    }
                }
            }

            /* colour / selection cycling */
            if (e.code === 'KeyC') {
                this.colorIdx = (this.colorIdx + 1) % this.cats.length;
                this.pointCloud.state.setColorBy(this.cats[this.colorIdx]);
            }
            if (e.code === 'KeyV') {
                this.selectIdx = (this.selectIdx + 1) % this.cats.length;
                this.pointCloud.state.setSelectBy(this.cats[this.selectIdx]);
            }

            /* hover UI toggle */
            if (e.code === 'KeyK') {
                this.pointCloud.hoverActive = !this.pointCloud.hoverActive;
                this._updateStatusIndicator('hover-status', this.pointCloud.hoverActive);
                if (!this.pointCloud.hoverActive) this.hoverPanel.style.display = 'none';
            }

            /* click-select toggle */
            if (e.code === 'KeyB') {
                this.pointCloud.selectOnClick = !this.pointCloud.selectOnClick;
                this._updateStatusIndicator('click-select-status', this.pointCloud.selectOnClick);
            }
        });
    }

    _togglePanel(name) {
        const cfg = this.uiConfig[name];
        cfg.visible = !cfg.visible;
        document.getElementById(cfg.elementId).style.display = cfg.visible ? 'block' : 'none';
    }

    /* ---------- per-frame UI refresh -------------------------- */
    updateUI() {
        /* navbar + navball */
        if (this.uiConfig.navbar.visible) {
            const p = this.pointCloud.camera.position;
            ['posX', 'posY', 'posZ'].forEach((id, i) =>
                document.getElementById(id).textContent = p[['x', 'y', 'z'][i]].toFixed(1));
            this.navball.syncWithCameraQuaternion(this.pointCloud.camera.quaternion);
        }

        /* stats */
        if (this.uiConfig.stats.visible) this._updateStats();

        /* hover tooltip */
        this._showHover(this.pointCloud.hoverId);
    }

    /* ---------- FPS / stats ----------------------------------- */
    _updateStats() {
        this._tickFPS();
        const ms = (1000 / Math.max(this.fps, 1)).toFixed(1) + ' ms';

        document.getElementById('fps').textContent = this.fps;
        document.getElementById('frameTime').textContent = ms;
        document.getElementById('renderedCount').textContent =
            this.pointCloud.points.geometry.getAttribute('position').count;

        const p = this.pointCloud.camera.position;
        document.getElementById('statsPosX').textContent = p.x.toFixed(3);
        document.getElementById('statsPosY').textContent = p.y.toFixed(3);
        document.getElementById('statsPosZ').textContent = p.z.toFixed(3);
    }
    _tickFPS() {
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastTime >= 1000) {
            this.fps = Math.round(this.frameCount * 1000 / (now - this.lastTime));
            this.frameCount = 0;
            this.lastTime = now;
        }
    }

    onPointsRegenerated() {
        if (this.uiConfig.stats.visible)
            document.getElementById('renderedCount').textContent =
                this.pointCloud.points.geometry.getAttribute('position').count;
    }

    /* ---------- hover tooltip --------------------------------- */
    _showHover(id) {
        if (!this.pointCloud.hoverActive || id == null) {
            this.hoverPanel.style.display = 'none';
            return;
        }

        const row = this.pointCloud.model.row(id);
        const a = this.pointCloud.state.axis;
        const xyz = [
            this.pointCloud.model.getCoord(id, a.x).toFixed(2),
            this.pointCloud.model.getCoord(id, a.y).toFixed(2),
            this.pointCloud.model.getCoord(id, a.z).toFixed(2)
        ];
        const html = CONFIG.hoverColumns
            .map(c => `<b>${c}</b>: ${row[c]}`)
            .concat([`<b>coord</b>: [${xyz.join(', ')}]`])
            .join('<br>');

        this.hoverPanel.innerHTML = html;
        const { x, y } = this.pointCloud.pointerScreen;
        this.hoverPanel.style.left = (x + 15) + 'px';
        this.hoverPanel.style.top = (y + 15) + 'px';
        this.hoverPanel.style.display = 'block';
    }
}