/* ui.js */
class UIManager {
    constructor(pointCloud) {
        this.pointCloud = pointCloud;

        /* panel metadata ------------------------------------------------ */
        this.uiConfig = {
            help: { key: 'KeyH', elementId: 'helpMenu', shortcutText: 'h – help', visible: false },
            menu: { key: 'KeyM', elementId: 'controlsMenu', shortcutText: 'm – menu', visible: false },
            info: { key: 'KeyI', elementId: 'infoMenu', shortcutText: 'i – info', visible: false },
            legend: { key: 'KeyL', elementId: 'legendMenu', shortcutText: 'l – legend', visible: true },
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

        /* Performance safeguard ----------------------------------------- */
        this.performanceCheckInterval = 2000; // Check every 2 seconds
        this.lastPerformanceCheck = performance.now();
        this.fpsThreshold = 15;
        this.performanceWarningShown = false;

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

        /* attach once‑only grid listeners for selected values & legend */
        this._attachGridListeners();

        /* performance warning banner */
        this._createPerformanceWarning();

        // for changes in columns, color, or selection
        this._updateLegendDisplay()
        this.pointCloud.state.addEventListener('vis', () => this._updateLegendDisplay());
        this.pointCloud.state.addEventListener('selection', () => this._updateLegendDisplay());

    }

    /* ------------------------------------------------------------------ */
    /* This is where we listen for clicks on the "X" and the legend items */
    _attachGridListeners() {
        /* Selected-values grid (the info panel) */
        const selGrid = document.getElementById('selectedValuesGrid');
        if (selGrid) {
            selGrid.addEventListener('click', (e) => {
                console.log('[UIManager] selected-values grid clicked');
                console.log(e)
                // We look for a .remove-btn in the event’s ancestry
                const btn = e.target.closest('.remove-btn');
                if (!btn) return;
                e.stopPropagation();

                const valueToRemove = btn.getAttribute('data-value');
                console.log('[UIManager] remove-btn clicked ->', valueToRemove);
                this.pointCloud.state.toggleValue(valueToRemove);
            });
        }

        /* Legend grid */
        const legendGrid = document.getElementById('legendGrid');
        if (legendGrid) {
            legendGrid.addEventListener('click', (e) => {
                console.log('[UIManager] legend grid clicked'); 
                console.log(e)
                // We look for a .legend-item-clickable
                const item = e.target.closest('.legend-item-clickable');
                if (!item) return;
                e.stopPropagation();

                const value = item.getAttribute('data-value');
                console.log('[UIManager] legend item clicked ->', value);
                this.pointCloud.state.toggleValue(value);
            });
        }
    }

    _setupNavball() {
        this.navball = new Navball('navball-container');
        // Show navball and legend by default
        document.getElementById('navbar').style.display = 'block';
        document.getElementById('legendMenu').style.display = 'block';
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
                pointSizeValue.textContent = v.toFixed(0);
                this.pointCloud.state.setVisParam('selSize', v);
            });
        }

        // Non-selected point size
        if (nonSelPointSizeSlider && nonSelPointSizeValue) {
            nonSelPointSizeSlider.addEventListener('input', () => {
                const v = parseFloat(nonSelPointSizeSlider.value);
                nonSelPointSizeValue.textContent = v.toFixed(0);
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
        const applyColorBy = document.getElementById('applyColorBy');
        const applySelectBy = document.getElementById('applySelectBy');

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
        }

        // Apply button handlers
        if (applyColorBy) {
            applyColorBy.addEventListener('click', () => {
                this.pointCloud.selMgr.clearCaches();
                this.pointCloud.state.setColorBy(colorBySelect.value);
                this.colorIdx = Math.max(0, this.cats.indexOf(colorBySelect.value));
            });
        }

        if (applySelectBy) {
            applySelectBy.addEventListener('click', () => {
                this.pointCloud.selMgr.clearCaches();
                this.pointCloud.state.setSelectBy(selectBySelect.value);
                this.selectIdx = Math.max(0, this.cats.indexOf(selectBySelect.value));
            });
        }
    }

    _updateColumnInfo() {
        const colorByEl = document.getElementById('currentColorBy');
        const selectByEl = document.getElementById('currentSelectBy');

        if (colorByEl) colorByEl.textContent = this.pointCloud.state.colorBy;
        if (selectByEl) selectByEl.textContent = this.pointCloud.state.selectBy;
    }

    _updateSelectedValuesDisplay() {
        const container = document.getElementById('selectedValuesGrid');
        const header = document.getElementById('selectedValuesHeader');
        const metadata = document.getElementById('selectedValuesMetadata');

        if (!container) return;

        container.innerHTML = '';

        const hasSelection = this.pointCloud.state.selection.size > 0;

        if (hasSelection) {
            // Show selected values
            header.textContent = 'Selected Values:';

            const selectedValues = Array.from(this.pointCloud.state.selection);

            if (selectedValues.length <= 20) {
                selectedValues.forEach((value, index) => {
                    const div = document.createElement('div');
                    div.className = 'value-grid-item selected';

                    const color = this.pointCloud.selMgr._getCategoricalColor(value);
                    div.innerHTML = `
                        <div class="value-grid-color" style="background-color: rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})"></div>
                        <span style="flex: 1;">${value}</span>
                        <span class="remove-btn" data-value="${value}">×</span>
                    `;

                    container.appendChild(div);
                });

                metadata.textContent = `${selectedValues.length} selected`;
            } else {
                metadata.innerHTML = `${selectedValues.length} values selected<br>Too many to display individually`;
            }
        } else {
            // Show message when no selection
            header.textContent = 'No Selection';
            metadata.textContent = 'Click items in the legend to select categories';
        }
    }

    _updateLegendDisplay() {
        const container = document.getElementById('legendGrid');
        const header = document.getElementById('legendHeader');
        const metadata = document.getElementById('legendMetadata');

        if (!container) return;

        container.innerHTML = '';

        // Show legend
        header.textContent = 'Legend:';

        const colorColumn = this.pointCloud.state.colorBy;

        if (this.pointCloud.state.isNumericColumn(colorColumn)) {
            // Show colorbar info
            const values = this.pointCloud.model.df.col(colorColumn).filter(v => typeof v === 'number' && !isNaN(v));
            const min = Math.min(...values);
            const max = Math.max(...values);

            container.innerHTML = `
                <div style="grid-column: 1 / -1;">
                    <div style="font-size: 10px; margin-bottom: 4px;">${colorColumn}</div>
                    <div class="colorbar"></div>
                    <div class="colorbar-labels">
                        <span>${min.toFixed(2)}</span>
                        <span>${max.toFixed(2)}</span>
                    </div>
                </div>
            `;
            metadata.textContent = `Continuous scale: ${min.toFixed(2)} to ${max.toFixed(2)}`;
        } else {
            // Show categorical legend
            const uniqueValues = [...this.pointCloud.model.df.col_unique(colorColumn)]
                .filter(v => v !== null && v !== 'null' && v !== 'unknown')
                .sort();

            if (uniqueValues.length <= 15) {
                uniqueValues.forEach(value => {
                    const color = this.pointCloud.selMgr._getCategoricalColor(value);
                    const div = document.createElement('div');
                    div.className = 'value-grid-item legend-item-clickable';
                    div.style.backgroundColor = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, 0.3)`;
                    div.dataset.value = value;

                    // Check if this value is currently selected
                    const isSelected = this.pointCloud.state.selection.has(value);
                    if (isSelected) {
                        div.classList.add('selected');
                    }

                    div.innerHTML = `
                        <div class="value-grid-color" style="background-color: rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})"></div>
                        <span>${value}</span>
                    `;

                    container.appendChild(div);
                });

                metadata.textContent = `${uniqueValues.length} categories (click to select/deselect)`;
            } else {
                metadata.innerHTML = `${uniqueValues.length} categories<br>Too many to display individually`;
            }
        }
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

        /* update column info */
        this._updateColumnInfo();

        /* update displays */
        this._updateSelectedValuesDisplay();
        // if (this.uiConfig.legend.visible) {
        //     this._updateLegendDisplay();
        // }

        /* performance monitoring */
        this._checkPerformance();
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

    /* ---------- performance warning --------------------------- */
    _createPerformanceWarning() {
        this.performanceWarning = document.createElement('div');
        this.performanceWarning.style.cssText = `   
            position: fixed;
            top: 60px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255, 68, 68, 0.9);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            font-weight: bold;
            z-index: 1000;
            display: none;
            max-width: 400px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;
        document.body.appendChild(this.performanceWarning);
    }

    _showPerformanceWarning(message) {
        this.performanceWarning.textContent = message;
        this.performanceWarning.style.display = 'block';

        // Hide after 4 seconds
        setTimeout(() => {
            this.performanceWarning.style.display = 'none';
        }, 4000);
    }

    _checkPerformance() {
        const now = performance.now();

        // Only check every 2 seconds
        if (now - this.lastPerformanceCheck < this.performanceCheckInterval) {
            return;
        }

        this.lastPerformanceCheck = now;

        // Check if FPS is consistently low
        if (this.fps < this.fpsThreshold && !this.performanceWarningShown) {
            this.performanceWarningShown = true;
            this._optimizeForPerformance();

            // Reset the flag after some time
            setTimeout(() => {
                this.performanceWarningShown = false;
            }, 30000);
        }
    }

    _optimizeForPerformance() {
        let changesApplied = [];

        // Force full opacity
        const opacitySlider = document.getElementById('opacity');
        const nonSelOpacitySlider = document.getElementById('nonSelOpacity');
        const opacityValue = document.getElementById('opacityValue');
        const nonSelOpacityValue = document.getElementById('nonSelOpacityValue');

        if (opacitySlider && parseFloat(opacitySlider.value) < 1.0) {
            opacitySlider.value = '1.0';
            opacityValue.textContent = '1.00';
            this.pointCloud.state.setVisParam('selOp', 1.0);
            changesApplied.push('selected opacity to 100%');
        }

        if (nonSelOpacitySlider && parseFloat(nonSelOpacitySlider.value) < 1.0) {
            nonSelOpacitySlider.value = '1.0';
            nonSelOpacityValue.textContent = '1.00';
            this.pointCloud.state.setVisParam('nonSelOp', 1.0);
            changesApplied.push('non-selected opacity to 100%');
        }

        // Limit point sizes to max 5
        const pointSizeSlider = document.getElementById('pointSize');
        const nonSelPointSizeSlider = document.getElementById('nonSelPointSize');
        const pointSizeValue = document.getElementById('pointSizeValue');
        const nonSelPointSizeValue = document.getElementById('nonSelPointSizeValue');

        if (pointSizeSlider && parseFloat(pointSizeSlider.value) > 5) {
            pointSizeSlider.value = '5';
            pointSizeValue.textContent = '5';
            this.pointCloud.state.setVisParam('selSize', 5);
            changesApplied.push('selected point size to 5');
        }

        if (nonSelPointSizeSlider && parseFloat(nonSelPointSizeSlider.value) > 5) {
            nonSelPointSizeSlider.value = '5';
            nonSelPointSizeValue.textContent = '5';
            this.pointCloud.state.setVisParam('nonSelSize', 5);
            changesApplied.push('non-selected point size to 5');
        }

        // Show warning message
        if (changesApplied.length > 0) {
            const message = `Performance warning: Low FPS detected (${this.fps}). Adjusted: ${changesApplied.join(', ')}.`;
            this._showPerformanceWarning(message);
        }
    }
}
    