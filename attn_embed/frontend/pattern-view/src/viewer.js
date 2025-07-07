/**
 * Attention Pattern Viewer Module
 * Handles visualization of attention patterns
 */

class AttentionPatternViewer {
    constructor(containerId) {
        // Constants
        this.SIZE = 500;
        this.HM_grid_strokeStyle = '#ddd';
        this.HM_grid_lineWidth = 0.5;
        this.HM_highlight_strokeStyle = '#ffaaaa';
        this.HM_highlight_lineWidth = 2;
        this.THROTTLE_DELAY = 16; // ~60fps
        
        // State
        this.n = 0;
        this.tokens = [];
        this.matrix = null;
        this.pixelSize = 0;
        this.cellBoundaries = [];
        this.baseCanvas = null;
        this.baseCtx = null;
        this.lastMouseTime = 0;
        this.animationFrame = null;
        this.labelElements = { x: [], y: [] };
        
        // DOM elements
        this.container = document.getElementById(containerId);
        this.canvas = document.getElementById('heatmapCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.tooltip = document.getElementById('tooltip');
        this.xLabelsContainer = document.getElementById('xLabels');
        this.yLabelsContainer = document.getElementById('yLabels');
        
        this.ctx.imageSmoothingEnabled = false;
        
        // Initialize offscreen canvas for caching
        this.initBaseCanvas();
        
        // Set up event listeners
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => this.handleMouseLeave());
    }
    
    initBaseCanvas() {
        this.baseCanvas = document.createElement('canvas');
        this.baseCanvas.width = this.SIZE;
        this.baseCanvas.height = this.SIZE;
        this.baseCtx = this.baseCanvas.getContext('2d');
        this.baseCtx.imageSmoothingEnabled = false;
    }
    
    cmap(value) {
        // Blues colormap: white to blue
        const intensity = Math.floor((1 - value) * 255);
        return `rgb(${intensity}, ${intensity}, 255)`;
    }
    
    precalculateBoundaries() {
        this.cellBoundaries = [];
        for (let i = 0; i <= this.n; i++) {
            this.cellBoundaries.push(i * this.pixelSize);
        }
    }
    
    renderBaseHeatmap() {
        // Clear the base canvas
        this.baseCtx.clearRect(0, 0, this.SIZE, this.SIZE);
        
        // Draw heatmap cells
        for (let i = 0; i < this.n; i++) {
            for (let j = 0; j < this.n; j++) {
                const value = this.matrix[i * this.n + j];
                this.baseCtx.fillStyle = this.cmap(value);
                this.baseCtx.fillRect(j * this.pixelSize, i * this.pixelSize, this.pixelSize, this.pixelSize);
            }
        }
        
        // Draw grid lines in one batch
        this.baseCtx.strokeStyle = this.HM_grid_strokeStyle;
        this.baseCtx.lineWidth = this.HM_grid_lineWidth;
        this.baseCtx.beginPath();
        
        for (let i = 0; i <= this.n; i++) {
            const pos = this.cellBoundaries[i];
            // Horizontal line
            this.baseCtx.moveTo(0, pos);
            this.baseCtx.lineTo(this.SIZE, pos);
            // Vertical line
            this.baseCtx.moveTo(pos, 0);
            this.baseCtx.lineTo(pos, this.SIZE);
        }
        
        this.baseCtx.stroke();
    }
    
    renderHighlights(hoverX, hoverY) {
        // Copy base heatmap
        this.ctx.clearRect(0, 0, this.SIZE, this.SIZE);
        this.ctx.drawImage(this.baseCanvas, 0, 0);
        
        // Draw highlights if hovering
        if (hoverX >= 0 && hoverY >= 0 && hoverX < this.n && hoverY < this.n) {
            this.ctx.strokeStyle = this.HM_highlight_strokeStyle;
            this.ctx.lineWidth = this.HM_highlight_lineWidth;
            
            const x1 = this.cellBoundaries[hoverX];
            const y1 = this.cellBoundaries[hoverY];
            const x2 = this.cellBoundaries[hoverX + 1];
            const y2 = this.cellBoundaries[hoverY + 1];
            
            // Highlight the cell's own borders
            this.ctx.strokeRect(x1, y1, this.pixelSize, this.pixelSize);
            
            this.ctx.beginPath();
            
            // Highlight row (only to the left of hovered cell)
            if (hoverX > 0) {
                this.ctx.moveTo(0, y1);
                this.ctx.lineTo(x1, y1);
                this.ctx.moveTo(0, y2);
                this.ctx.lineTo(x1, y2);
            }
            
            // Highlight column (only below hovered cell)
            if (hoverY < this.n - 1) {
                this.ctx.moveTo(x1, y2);
                this.ctx.lineTo(x1, this.SIZE);
                this.ctx.moveTo(x2, y2);
                this.ctx.lineTo(x2, this.SIZE);
            }
            
            this.ctx.stroke();
        }
    }
    
    createAxisLabels() {
        // Clear existing labels
        this.xLabelsContainer.innerHTML = '';
        this.yLabelsContainer.innerHTML = '';
        this.labelElements.x = [];
        this.labelElements.y = [];
        
        this.tokens.forEach((token, i) => {
            const xLabel = document.createElement('div');
            xLabel.className = 'label x-label';
            xLabel.textContent = token;
            xLabel.style.width = this.pixelSize + 'px';
            this.xLabelsContainer.appendChild(xLabel);
            this.labelElements.x.push(xLabel);
            
            const yLabel = document.createElement('div');
            yLabel.className = 'label y-label';
            yLabel.textContent = token;
            yLabel.style.height = this.pixelSize + 'px';
            yLabel.style.lineHeight = this.pixelSize + 'px';
            this.yLabelsContainer.appendChild(yLabel);
            this.labelElements.y.push(yLabel);
        });
    }
    
    updateHighlights(x, y) {
        // Update label highlights
        this.labelElements.x.forEach(label => label.classList.remove('highlight'));
        this.labelElements.y.forEach(label => label.classList.remove('highlight'));
        
        if (x >= 0 && x < this.n && y >= 0 && y < this.n) {
            this.labelElements.x[x].classList.add('highlight');
            this.labelElements.y[y].classList.add('highlight');
        }
        
        // Render highlights
        this.renderHighlights(x, y);
    }
    
    handleMouseMove(e) {
        const now = Date.now();
        if (now - this.lastMouseTime < this.THROTTLE_DELAY) {
            return;
        }
        this.lastMouseTime = now;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / rect.width * this.n);
        const y = Math.floor((e.clientY - rect.top) / rect.height * this.n);
        
        if (x >= 0 && x < this.n && y >= 0 && y < this.n) {
            const xToken = this.tokens[x];
            const yToken = this.tokens[y];
            const value = this.matrix[y * this.n + x].toFixed(3);
            
            this.tooltip.innerHTML = `
                <div class="tooltip-row"><span>X:</span><span>${xToken}</span></div>
                <div class="tooltip-row"><span>Y:</span><span>${yToken}</span></div>
                <div class="tooltip-row"><span>Value:</span><span>${value}</span></div>
            `;
            this.tooltip.style.left = (e.clientX + 10) + 'px';
            this.tooltip.style.top = (e.clientY - 10) + 'px';
            this.tooltip.style.display = 'block';
            
            // Cancel any pending animation frame
            if (this.animationFrame) {
                cancelAnimationFrame(this.animationFrame);
            }
            
            // Schedule highlight update
            this.animationFrame = requestAnimationFrame(() => {
                this.updateHighlights(x, y);
                this.animationFrame = null;
            });
        } else {
            this.handleMouseLeave();
        }
    }
    
    handleMouseLeave() {
        this.tooltip.style.display = 'none';
        
        // Cancel any pending animation frame
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        // Remove highlights
        this.labelElements.x.forEach(label => label.classList.remove('highlight'));
        this.labelElements.y.forEach(label => label.classList.remove('highlight'));
        
        // Render without highlights
        this.renderHighlights(-1, -1);
    }
    
    async displayPattern(dataLoader, model, promptHash, layerIdx, headIdx) {
        // Load prompt metadata
        const metadata = await dataLoader.loadPromptMetadata(model, promptHash);
        this.tokens = metadata.tokens;
        this.n = this.tokens.length;
        this.pixelSize = this.SIZE / this.n;
        
        // Load attention pattern
        const patternData = await dataLoader.loadAttentionPattern(model, promptHash, layerIdx, headIdx);
        this.matrix = patternData.data;
        
        // Verify dimensions match
        if (patternData.width !== this.n || patternData.height !== this.n) {
            throw new Error(`Dimension mismatch: expected ${this.n}x${this.n}, got ${patternData.width}x${patternData.height}`);
        }
        
        // Precalculate boundaries
        this.precalculateBoundaries();
        
        // Set canvas dimensions
        this.canvas.width = this.SIZE;
        this.canvas.height = this.SIZE;
        
        // Render base heatmap to cache
        this.renderBaseHeatmap();
        
        // Initial render
        this.renderHighlights(-1, -1);
        
        // Create labels
        this.createAxisLabels();
        
        // Update page title
        document.title = `${model} L${layerIdx}H${headIdx} - ${promptHash.substring(0, 8)}`;
    }
}