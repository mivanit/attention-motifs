/* UIManager – adds "k" to toggle hover UI / cross-hair
   and "b" to toggle click-to-select. */

class UIManager {
    constructor(pointCloud) {
        this.pointCloud = pointCloud;

        /* panel metadata ------------------------------------------------ */
        this.uiConfig = {
            help: { key: 'KeyH', elementId: 'helpMenu', shortcutText: 'h – help', visible: false },
            menu: { key: 'KeyM', elementId: 'controlsMenu', shortcutText: 'm – menu', visible: false },
            info: { key: 'KeyI', elementId: 'infoMenu', shortcutText: 'i – info', visible: false },
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
        // Point size controls
        const pointSizeSlider = document.getElementById('pointSize');
        const pointSizeValue = document.getElementById('pointSizeValue');
        const nonSelPointSizeSlider = document.getElementById('nonSelPointSize');
        const nonSelPointSizeValue = document.getElementById('nonSelPointSizeValue');

        // Opacity controls
        const opacitySlider = document.getElementById('opacity');
        const opacityValue = document.getElementById('opacityValue');
        const nonSelOpacitySlider = document.getElementById('nonSelOpacity');
        const nonSelOpacityValue = document.getElementById('nonSelOpacityValue');

        // Speed control
        const speedSlider = document.getElementById('speed');
        const speedValue = document.getElementById('speedValue');

        // Color controls
        const nonSelColorPicker = document.getElementById('nonSelColor');
        const randomizeColorsBtn = document.getElementById('randomizeColors');

        // Selected point size
        if (pointSizeSlider && pointSizeValue) {
            pointSizeSlider.addEventListener('input', () => {
                const v = parseFloat(pointSizeSlider.value);
                pointSizeValue.textContent = v.toFixed(2);
                this.pointCloud.state.setVisParam('selSize', v);
            });
        }

        // Non-selected point size
        if (nonSelPointSizeSlider && nonSelPointSizeValue) {
            nonSelPointSizeSlider.addEventListener('input', () => {
                const v = parseFloat(nonSelPointSizeSlider.value);
                nonSelPointSizeValue.textContent = v.toFixed(2);
                this.pointCloud.state.setVisParam('nonSelSize', v);
            });
        }

        // Selected opacity
        if (opacitySlider && opacityValue) {
            opacitySlider.addEventListener('input', () => {
                const v = parseFloat(opacitySlider.value);
                opacityValue.textContent = v.toFixed(2);
                this.pointCloud.state.setVisParam('selOp', v);
            });
        }

        // Non-selected opacity
        if (nonSelOpacitySlider && nonSelOpacityValue) {
            nonSelOpacitySlider.addEventListener('input', () => {
                const v = parseFloat(nonSelOpacitySlider.value);
                nonSelOpacityValue.textContent = v.toFixed(2);
                this.pointCloud.state.setVisParam('nonSelOp', v);
            });
        }

        // Speed
        if (speedSlider && speedValue) {
            speedSlider.addEventListener('input', () => {
                const v = parseFloat(speedSlider.value);
                speedValue.textContent = v;
                this.pointCloud.settings.speed = v;
            });
        }

        // Non-selected color
        if (nonSelColorPicker) {
            nonSelColorPicker.addEventListener('input', () => {
                this.pointCloud.state.setVisParam('nonSelColor', nonSelColorPicker.value);
            });
        }

        // Randomize colors button
        if (randomizeColorsBtn) {
            randomizeColorsBtn.addEventListener('click', () => {
                this.pointCloud.selMgr.randomizeColors();
                this.pointCloud.state._fire('vis');
            });
        }

        // Clear selection button
        const clearSelectionBtn = document.getElementById('clearSelectionBtn');
        if (clearSelectionBtn) {
            clearSelectionBtn.addEventListener('click', () => {
                this.pointCloud.state.clearSel();
            });
        }

        // Setup dropdowns
        this._setupDropdowns();

        /* keep renderer sized */
        window.addEventListener('resize', () => {
            this.pointCloud.camera.aspect = window.innerWidth / window.innerHeight;
            this.pointCloud.camera.updateProjectionMatrix();
            this.pointCloud.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    _setupDropdowns() {
        const colorBySelect = document.getElementById('colorBySelect');
        const selectBySelect = document.getElementById('selectBySelect');

        if (colorBySelect) {
            // Clear existing options
            colorBySelect.innerHTML = '';

            // Populate color by dropdown
            this.cats.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                colorBySelect.appendChild(option);
            });

            colorBySelect.value = this.pointCloud.state.colorBy;
            colorBySelect.addEventListener('change', () => {
                this.pointCloud.selMgr.clearCaches();
                this.pointCloud.state.setColorBy(colorBySelect.value);
                this.colorIdx = Math.max(0, this.cats.indexOf(colorBySelect.value));
            });
        }

        if (selectBySelect) {
            // Clear existing options
            selectBySelect.innerHTML = '';

            // Populate select by dropdown
            this.cats.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                selectBySelect.appendChild(option);
            });

            selectBySelect.value = this.pointCloud.state.selectBy;
            selectBySelect.addEventListener('change', () => {
                this.pointCloud.selMgr.clearCaches();
                this.pointCloud.state.setSelectBy(selectBySelect.value);
                this.selectIdx = Math.max(0, this.cats.indexOf(selectBySelect.value));
            });
        }
    }

    _updateSelectedValuesDisplay() {
        const container = document.getElementById('selectedValuesContainer');
        if (!container) return;

        container.innerHTML = '';

        if (this.pointCloud.state.selection.size === 0) {
            container.innerHTML = '<div style="color: #888; font-style: italic;">None selected</div>';
            return;
        }

        Array.from(this.pointCloud.state.selection).forEach((value, index) => {
            const div = document.createElement('div');
            div.className = 'selected-value-item';
            div.textContent = value;
            div.style.backgroundColor = this.pointCloud.selMgr.palette[index % this.pointCloud.selMgr.palette.length];
            div.addEventListener('click', () => {
                this.pointCloud.state.toggleValue(value);
            });
            container.appendChild(div);
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
                // Update dropdown
                const colorBySelect = document.getElementById('colorBySelect');
                if (colorBySelect) colorBySelect.value = this.cats[this.colorIdx];
            }
            if (e.code === 'KeyV') {
                this.selectIdx = (this.selectIdx + 1) % this.cats.length;
                this.pointCloud.state.setSelectBy(this.cats[this.selectIdx]);
                // Update dropdown
                const selectBySelect = document.getElementById('selectBySelect');
                if (selectBySelect) selectBySelect.value = this.cats[this.selectIdx];
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

        /* update selected values display */
        this._updateSelectedValuesDisplay();
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