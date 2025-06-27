/* ui.js - Updated to use CONFIG values and add export button */
class UIManager {
    constructor(pointCloud) {
        this.pointCloud = pointCloud;

        /* panel metadata - now uses CONFIG -------------------------------- */
        this.uiConfig = {
            help: { key: 'KeyH', elementId: 'helpMenu', shortcutText: 'h – help', visible: CONFIG.panels.help },
            menu: { key: 'KeyM', elementId: 'controlsMenu', shortcutText: 'm – menu', visible: CONFIG.panels.menu },
            info: { key: 'KeyI', elementId: 'infoMenu', shortcutText: 'i – info', visible: CONFIG.panels.info },
            legend: { key: 'KeyL', elementId: 'legendMenu', shortcutText: 'l – legend', visible: CONFIG.panels.legend },
            navbar: { key: 'KeyN', elementId: 'navbar', shortcutText: 'n – navball', visible: CONFIG.panels.navbar },
            stats: { key: 'KeyJ', elementId: 'statsMenu', shortcutText: 'j – stats', visible: CONFIG.panels.stats }
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

        /* Performance safeguard - now uses CONFIG ---------------------- */
        this.performanceCheckInterval = CONFIG.performance.performanceCheckInterval;
        this.lastPerformanceCheck = performance.now();
        this.fpsThreshold = CONFIG.performance.fpsThreshold;
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


        // Apply initial panel visibility from CONFIG
        this._applyInitialPanelVisibility();

        // for changes in columns, color, or selection
        this._updateLegendDisplay()
        this._updateSelectedValuesDisplay();
        this.pointCloud.state.addEventListener('vis', () => {
            this._updateLegendDisplay();
            this._updateSelectedValuesDisplay();
        });
        this.pointCloud.state.addEventListener('selection', () => {
            this._updateLegendDisplay();
            this._updateSelectedValuesDisplay();
        });
    }

    _applyInitialPanelVisibility() {
        for (const [name, cfg] of Object.entries(this.uiConfig)) {
            document.getElementById(cfg.elementId).style.display = cfg.visible ? 'block' : 'none';
        }
    }

    /* ... (keeping _attachGridListeners unchanged) ... */
    _attachGridListeners() {
        /* Selected-values grid (the info panel) */
        const selGrid = document.getElementById('selectedValuesGrid');
        if (selGrid) {
            selGrid.addEventListener('click', (e) => {
                console.log('[UIManager] selected-values grid clicked');
                console.log(e)
                // We look for a .remove-btn in the event's ancestry
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
                // We look for a .legend-item-clickable
                const item = e.target.closest('.legend-item-clickable');
                if (!item) return;
                e.stopPropagation();

                const value = item.getAttribute('data-value');
                this.pointCloud.state.toggleValue(value);
            });
        }
    }

    _setupNavball() {
        this.navball = new Navball('navball-container', CONFIG.navball.size);
        // Apply initial visibility from CONFIG
        document.getElementById('navbar').style.display = CONFIG.panels.navbar ? 'block' : 'none';
        document.getElementById('legendMenu').style.display = CONFIG.panels.legend ? 'block' : 'none';
    }

    /* ---------- shortcuts legend (top-right) - using CONFIG ---------- */
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

        // Add hover, click-select, and right-click shortcuts with status indicators from CONFIG
        sc.insertAdjacentHTML('beforeend', `
        <div class="shortcut-link" data-action="hover-toggle">k – hover UI <span class="status-indicator ${CONFIG.interaction.hoverActive ? 'status-enabled' : 'status-disabled'}" id="hover-status">(${CONFIG.interaction.hoverActive ? 'enabled' : 'disabled'})</span></div>
        <div class="shortcut-link" data-action="click-select-toggle">b – click-select <span class="status-indicator ${CONFIG.interaction.selectOnClick ? 'status-enabled' : 'status-disabled'}" id="click-select-status">(${CONFIG.interaction.selectOnClick ? 'enabled' : 'disabled'})</span></div>
        <div class="shortcut-link" data-action="right-click-toggle">o – right-click <span class="status-indicator ${CONFIG.interaction.rightClickActive ? 'status-enabled' : 'status-disabled'}" id="right-click-status">(${CONFIG.interaction.rightClickActive ? 'enabled' : 'disabled'})</span></div>`);

        sc.addEventListener('click', e => {
            const target = e.target.closest('[data-action]');
            if (!target) return;

            const action = target.dataset.action;

            if (action === 'hover-toggle') {
                CONFIG.interaction.hoverActive = !CONFIG.interaction.hoverActive;
                this.pointCloud.hoverActive = CONFIG.interaction.hoverActive;
                this._updateStatusIndicator('hover-status', CONFIG.interaction.hoverActive);
                if (!CONFIG.interaction.hoverActive) this.hoverPanel.style.display = 'none';
                updateURL(); // Sync to URL
            } else if (action === 'click-select-toggle') {
                CONFIG.interaction.selectOnClick = !CONFIG.interaction.selectOnClick;
                this.pointCloud.selectOnClick = CONFIG.interaction.selectOnClick;
                this._updateStatusIndicator('click-select-status', CONFIG.interaction.selectOnClick);
                updateURL(); // Sync to URL
            } else if (action === 'right-click-toggle') {
                CONFIG.interaction.rightClickActive = !CONFIG.interaction.rightClickActive;
                this.pointCloud.rightClickActive = CONFIG.interaction.rightClickActive;
                this._updateStatusIndicator('right-click-status', CONFIG.interaction.rightClickActive);
                updateURL(); // Sync to URL
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

    /* ---------- sliders for size / opacity / speed - using CONFIG ---------- */
    _setupControlSliders() {
        // Point size controls - using CONFIG bounds
        const pointSizeSlider = document.getElementById('pointSize');
        const pointSizeValue = document.getElementById('pointSizeValue');
        const nonSelPointSizeSlider = document.getElementById('nonSelPointSize');
        const nonSelPointSizeValue = document.getElementById('nonSelPointSizeValue');

        // Opacity controls - using CONFIG bounds
        const opacitySlider = document.getElementById('opacity');
        const opacityValue = document.getElementById('opacityValue');
        const nonSelOpacitySlider = document.getElementById('nonSelOpacity');
        const nonSelOpacityValue = document.getElementById('nonSelOpacityValue');

        // Speed control - using CONFIG bounds
        const speedSlider = document.getElementById('speed');
        const speedValue = document.getElementById('speedValue');

        // Color controls
        const nonSelColorPicker = document.getElementById('nonSelColor');
        const randomizeColorsBtn = document.getElementById('randomizeColors');

        // Set up slider attributes from CONFIG
        if (pointSizeSlider) {
            pointSizeSlider.min = CONFIG.selectedPoints.sizeMin;
            pointSizeSlider.max = CONFIG.selectedPoints.sizeMax;
            pointSizeSlider.step = CONFIG.selectedPoints.sizeStep;
            pointSizeSlider.value = CONFIG.selectedPoints.size;
            if (pointSizeValue) pointSizeValue.textContent = CONFIG.selectedPoints.size;
        }

        if (nonSelPointSizeSlider) {
            nonSelPointSizeSlider.min = CONFIG.nonSelectedPoints.sizeMin;
            nonSelPointSizeSlider.max = CONFIG.nonSelectedPoints.sizeMax;
            nonSelPointSizeSlider.step = CONFIG.nonSelectedPoints.sizeStep;
            nonSelPointSizeSlider.value = CONFIG.nonSelectedPoints.size;
            if (nonSelPointSizeValue) nonSelPointSizeValue.textContent = CONFIG.nonSelectedPoints.size;
        }

        if (opacitySlider) {
            opacitySlider.min = CONFIG.selectedPoints.opacityMin;
            opacitySlider.max = CONFIG.selectedPoints.opacityMax;
            opacitySlider.step = CONFIG.selectedPoints.opacityStep;
            opacitySlider.value = CONFIG.selectedPoints.opacity;
            if (opacityValue) opacityValue.textContent = CONFIG.selectedPoints.opacity.toFixed(2);
        }

        if (nonSelOpacitySlider) {
            nonSelOpacitySlider.min = CONFIG.nonSelectedPoints.opacityMin;
            nonSelOpacitySlider.max = CONFIG.nonSelectedPoints.opacityMax;
            nonSelOpacitySlider.step = CONFIG.nonSelectedPoints.opacityStep;
            nonSelOpacitySlider.value = CONFIG.nonSelectedPoints.opacity;
            if (nonSelOpacityValue) nonSelOpacityValue.textContent = CONFIG.nonSelectedPoints.opacity.toFixed(2);
        }

        if (speedSlider) {
            speedSlider.min = CONFIG.movement.speedMin;
            speedSlider.max = CONFIG.movement.speedMax;
            speedSlider.step = CONFIG.movement.speedStep;
            speedSlider.value = CONFIG.movement.speed;
            if (speedValue) speedValue.textContent = CONFIG.movement.speed;
        }

        if (nonSelColorPicker) {
            nonSelColorPicker.value = CONFIG.nonSelectedPoints.color;
        }

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
                CONFIG.movement.speed = v;
                this.pointCloud.settings.speed = v;
                updateURL(); // Sync to URL
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
        // Apply menu width from CONFIG
        const controlsMenu = document.getElementById('controlsMenu');
        if (controlsMenu) {
            controlsMenu.style.minWidth = `${CONFIG.ui.menuWidth}px`;
            controlsMenu.style.width = `${CONFIG.ui.menuWidth}px`;
        }

        const colorBySelect = document.getElementById('colorBySelect');
        const selectBySelect = document.getElementById('selectBySelect');
        const applyColumns = document.getElementById('applyColumns');

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

        // Single Apply button for columns with loading notifications
        if (applyColumns) {
            applyColumns.addEventListener('click', async () => {
                const sp = NOTIF.spinner('Updating columns...');

                const newColorBy = colorBySelect.value;
                const newSelectBy = selectBySelect.value;

                // Check if changes are needed
                const colorChanged = newColorBy !== this.pointCloud.state.colorBy;
                const selectChanged = newSelectBy !== this.pointCloud.state.selectBy;

                if (!colorChanged && !selectChanged) {
                    sp.complete();
                    NOTIF.show('No changes to apply', 2000);
                    return;
                }

                try {
                    // Give the UI time to render the spinner
                    await new Promise(resolve => setTimeout(resolve, 50));

                    // Clear caches first
                    this.pointCloud.selMgr.clearCaches();

                    // Apply color column change
                    if (colorChanged) {
                        this.pointCloud.state.setColorBy(newColorBy);
                        this.colorIdx = Math.max(0, this.cats.indexOf(newColorBy));
                    }

                    // Apply selection column change
                    if (selectChanged) {
                        this.pointCloud.state.setSelectBy(newSelectBy);
                        this.selectIdx = Math.max(0, this.cats.indexOf(newSelectBy));
                    }

                    // Give the UI another frame to update if needed
                    await new Promise(resolve => setTimeout(resolve, 100));

                    sp.complete();
                    const changes = [];
                    if (colorChanged) changes.push(`color: ${newColorBy}`);
                    if (selectChanged) changes.push(`selection: ${newSelectBy}`);
                    NOTIF.success(`Updated ${changes.join(', ')}`);

                } catch (error) {
                    sp.complete();
                    NOTIF.error('Failed to update columns', error);
                }
            });
        }

        // Setup axis dropdowns
        const xAxisSelect = document.getElementById('xAxisSelect');
        const yAxisSelect = document.getElementById('yAxisSelect');
        const zAxisSelect = document.getElementById('zAxisSelect');
        const applyAxes = document.getElementById('applyAxes');

        if (xAxisSelect && yAxisSelect && zAxisSelect) {
            // Populate axis dropdowns with available PCA components
            const numComponents = this.pointCloud.model.numericCols.length;
            [xAxisSelect, yAxisSelect, zAxisSelect].forEach(select => {
                select.innerHTML = '';
                for (let i = 0; i < numComponents; i++) {
                    const option = document.createElement('option');
                    option.value = i;
                    option.textContent = this.pointCloud.model.numericCols[i] || `PC${i}`;
                    select.appendChild(option);
                }
            });

            // Set current values from CONFIG
            xAxisSelect.value = CONFIG.axes.x;
            yAxisSelect.value = CONFIG.axes.y;
            zAxisSelect.value = CONFIG.axes.z;
        }

        // Single Apply button for axes with notifications
        if (applyAxes) {
            applyAxes.addEventListener('click', async () => {
                const newX = parseInt(xAxisSelect.value);
                const newY = parseInt(yAxisSelect.value);
                const newZ = parseInt(zAxisSelect.value);

                // Check if changes are needed
                const xChanged = newX !== CONFIG.axes.x;
                const yChanged = newY !== CONFIG.axes.y;
                const zChanged = newZ !== CONFIG.axes.z;

                if (!xChanged && !yChanged && !zChanged) {
                    NOTIF.show('No axis changes to apply', 2000);
                    return;
                }

                const spinner = NOTIF.spinner('Updating visualization axes...');

                try {
                    // Give the UI time to render the spinner
                    await new Promise(resolve => setTimeout(resolve, 50));

                    // Apply axis changes
                    this.pointCloud.state.setAxis('x', newX);
                    this.pointCloud.state.setAxis('y', newY);
                    this.pointCloud.state.setAxis('z', newZ);

                    // Give UI another frame before the heavy _buildGeometry operation
                    await new Promise(resolve => setTimeout(resolve, 10));

                    this.pointCloud._buildGeometry();

                    spinner.complete();

                    const changes = [];
                    if (xChanged) changes.push(`X: ${this.pointCloud.model.numericCols[newX]}`);
                    if (yChanged) changes.push(`Y: ${this.pointCloud.model.numericCols[newY]}`);
                    if (zChanged) changes.push(`Z: ${this.pointCloud.model.numericCols[newZ]}`);

                    NOTIF.success(`Updated axes - ${changes.join(', ')}`);

                } catch (error) {
                    spinner.complete();
                    NOTIF.error('Failed to update axes', error);
                }
            });
        }

        // Setup export config button
        const exportBtn = document.getElementById('exportConfigBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                exportConfigToNewTab();
                NOTIF.success('Configuration exported to new tab');
            });
        }

        // Setup reset config button
        const resetBtn = document.getElementById('resetConfigBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', async () => {
                const spinner = NOTIF.spinner('Resetting configuration...');

                try {
                    // Give UI time to show spinner
                    await new Promise(resolve => setTimeout(resolve, 50));

                    // Reset config to loaded state and clear URL
                    resetConfigToLoaded();

                    // Trigger a page reload to apply the reset config
                    // This is the cleanest way to ensure all state is properly reset
                    window.location.reload();

                } catch (error) {
                    spinner.complete();
                    NOTIF.error('Failed to reset configuration', error);
                }
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

            if (selectedValues.length <= CONFIG.ui.maxSelectedDisplay) {
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

            if (uniqueValues.length <= CONFIG.ui.maxCategoricalDisplay) {
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
                CONFIG.interaction.hoverActive = !CONFIG.interaction.hoverActive;
                this.pointCloud.hoverActive = CONFIG.interaction.hoverActive;
                this._updateStatusIndicator('hover-status', CONFIG.interaction.hoverActive);
                if (!CONFIG.interaction.hoverActive) this.hoverPanel.style.display = 'none';
                updateURL(); // Sync to URL
            }

            /* click-select toggle */
            if (e.code === 'KeyB') {
                CONFIG.interaction.selectOnClick = !CONFIG.interaction.selectOnClick;
                this.pointCloud.selectOnClick = CONFIG.interaction.selectOnClick;
                this._updateStatusIndicator('click-select-status', CONFIG.interaction.selectOnClick);
                updateURL(); // Sync to URL
            }

            /* right-click toggle */
            if (e.code === 'KeyO') {
                CONFIG.interaction.rightClickActive = !CONFIG.interaction.rightClickActive;
                this.pointCloud.rightClickActive = CONFIG.interaction.rightClickActive;
                this._updateStatusIndicator('right-click-status', CONFIG.interaction.rightClickActive);
                updateURL(); // Sync to URL
            }
        });
    }

    _togglePanel(name) {
        const cfg = this.uiConfig[name];
        cfg.visible = !cfg.visible;

        // Update CONFIG to keep it in sync
        CONFIG.panels[name] = cfg.visible;
        updateURL(); // Sync to URL

        document.getElementById(cfg.elementId).style.display = cfg.visible ? 'block' : 'none';
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

    /* ---------- per-frame UI refresh ------------------------------ */
    updateUI() {
        /* navbar + navball */
        if (this.uiConfig.navbar.visible) {
            const p = this.pointCloud.camera.position;
            ['posX', 'posY', 'posZ'].forEach((id, i) =>
                document.getElementById(id).textContent =
                p[['x', 'y', 'z'][i]].toFixed(1));
            this.navball.syncWithCameraQuaternion(
                this.pointCloud.camera.quaternion);
        }

        /* stats */
        if (this.uiConfig.stats.visible) this._updateStats();

        /* hover tooltip */
        this._showHover(this.pointCloud.hoverId);

        /* current "Color by / Select by" labels */
        this._updateColumnInfo();

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
        if (!CONFIG.interaction.hoverActive || id == null) {
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
        this.hoverPanel.style.left = (x + CONFIG.ui.hoverOffset.x) + 'px';
        this.hoverPanel.style.top = (y + CONFIG.ui.hoverOffset.y) + 'px';
        this.hoverPanel.style.display = 'block';
    }

    _checkPerformance() {
        const now = performance.now();

        // Only check every interval from CONFIG
        if (now - this.lastPerformanceCheck < this.performanceCheckInterval) {
            return;
        }

        this.lastPerformanceCheck = now;

        // Check if FPS is consistently low
        if (this.fps < this.fpsThreshold && !this.performanceWarningShown) {
            this.performanceWarningShown = true;
            this._optimizeForPerformance();

            // Reset the flag after cooldown from CONFIG
            setTimeout(() => {
                this.performanceWarningShown = false;
            }, CONFIG.performance.performanceOptimizationCooldown);
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

        // Show error notification using NOTIF system
        if (changesApplied.length > 0) {
            const message = `Performance warning: Low FPS detected (${this.fps}). Adjusted: ${changesApplied.join(', ')}.`;
            NOTIF.error(message, null, CONFIG.performance.performanceWarningDuration);
        }
    }
}