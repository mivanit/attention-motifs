/**
 * Attention Pattern Viewer Module - Simplified version
 * Displays PNG directly with overlay for highlights
 */

class AttentionPatternViewer {
    constructor(containerId) {
        // Constants
        this.SIZE = 500;
        this.HM_highlight_strokeStyle = '#ff0000';
        this.HM_highlight_lineWidth = 2;
        this.THROTTLE_DELAY = 16; // ~60fps

        // State
        this.n = 0;
        this.tokens = [];
        this.pixelSize = 0;
        this.cellBoundaries = [];
        this.lastMouseTime = 0;
        this.animationFrame = null;
        this.labelElements = { x: [], y: [] };
        this.pngImage = null;

        // DOM elements
        this.container = document.getElementById(containerId);

        // Create main canvas for PNG display
        this.canvas = document.getElementById('heatmapCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.webkitImageSmoothingEnabled = false;
        this.ctx.mozImageSmoothingEnabled = false;
        this.ctx.msImageSmoothingEnabled = false;
        this.ctx.oImageSmoothingEnabled = false;

        // Create overlay canvas for highlights
        this.overlayCanvas = document.createElement('canvas');
        this.overlayCanvas.style.position = 'absolute';
        this.overlayCanvas.style.left = '0';
        this.overlayCanvas.style.top = '0';
        this.overlayCanvas.style.pointerEvents = 'none';
        this.overlayCtx = this.overlayCanvas.getContext('2d');

        // Add overlay to container
        this.canvas.parentElement.style.position = 'relative';
        this.canvas.parentElement.appendChild(this.overlayCanvas);

        this.tooltip = document.getElementById('tooltip');
        this.xLabelsContainer = document.getElementById('xLabels');
        this.yLabelsContainer = document.getElementById('yLabels');

        // Set up event listeners
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => this.handleMouseLeave());
    }

    precalculateBoundaries() {
        this.cellBoundaries = [];
        for (let i = 0; i <= this.n; i++) {
            this.cellBoundaries.push(i * this.pixelSize);
        }
    }


    renderHighlights(hoverX, hoverY) {
        // Clear overlay
        this.overlayCtx.clearRect(0, 0, this.SIZE, this.SIZE);

        // Draw grid lines
        this.overlayCtx.strokeStyle = '#ddd';
        this.overlayCtx.lineWidth = 0.5;
        this.overlayCtx.beginPath();

        for (let i = 0; i <= this.n; i++) {
            const pos = this.cellBoundaries[i];
            // Horizontal line
            this.overlayCtx.moveTo(0, pos);
            this.overlayCtx.lineTo(this.SIZE, pos);
            // Vertical line
            this.overlayCtx.moveTo(pos, 0);
            this.overlayCtx.lineTo(pos, this.SIZE);
        }

        this.overlayCtx.stroke();

        // Draw highlights if hovering
        if (hoverX >= 0 && hoverY >= 0 && hoverX < this.n && hoverY < this.n) {
            this.overlayCtx.strokeStyle = this.HM_highlight_strokeStyle;
            this.overlayCtx.lineWidth = this.HM_highlight_lineWidth;

            const x1 = this.cellBoundaries[hoverX];
            const y1 = this.cellBoundaries[hoverY];

            // Highlight the cell's own borders
            this.overlayCtx.strokeRect(x1, y1, this.pixelSize, this.pixelSize);

            this.overlayCtx.beginPath();

            // Highlight row (only to the left of hovered cell)
            if (hoverX > 0) {
                this.overlayCtx.moveTo(0, y1);
                this.overlayCtx.lineTo(x1, y1);
                this.overlayCtx.moveTo(0, y1 + this.pixelSize);
                this.overlayCtx.lineTo(x1, y1 + this.pixelSize);
            }

            // Highlight column (only below hovered cell)
            if (hoverY < this.n - 1) {
                this.overlayCtx.moveTo(x1, y1 + this.pixelSize);
                this.overlayCtx.lineTo(x1, this.SIZE);
                this.overlayCtx.moveTo(x1 + this.pixelSize, y1 + this.pixelSize);
                this.overlayCtx.lineTo(x1 + this.pixelSize, this.SIZE);
            }

            this.overlayCtx.stroke();
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

    getPixelValue(x, y) {
        // Get pixel value from the displayed PNG
        const imageData = this.ctx.getImageData(
            x * this.pixelSize + this.pixelSize / 2,
            y * this.pixelSize + this.pixelSize / 2,
            1, 1
        );
        // Convert from 0-255 to 0-1
        return imageData.data[0] / 255.0;
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
            const value = this.getPixelValue(x, y).toFixed(3);

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

        // Load PNG directly
        const pngPath = `${dataLoader.basePath}${model}/prompts/${promptHash}/L${layerIdx}/H${headIdx}/attn.png`;

        return new Promise((resolve, reject) => {
            this.pngImage = new Image();
            this.pngImage.crossOrigin = 'anonymous';

            this.pngImage.onload = () => {
                // Precalculate boundaries
                this.precalculateBoundaries();

                // Set canvas dimensions to fixed size
                this.canvas.width = this.SIZE;
                this.canvas.height = this.SIZE;
                this.overlayCanvas.width = this.SIZE;
                this.overlayCanvas.height = this.SIZE;

                // Calculate pixel size based on fixed canvas size
                this.pixelSize = this.SIZE / this.n;

                // Render PNG scaled to canvas size
                this.ctx.drawImage(this.pngImage, 0, 0, this.SIZE, this.SIZE);

                // Initial render of grid
                this.renderHighlights(-1, -1);

                // Create labels
                this.createAxisLabels();

                // Update page title
                document.title = `${model} L${layerIdx}H${headIdx} - ${promptHash.substring(0, 8)}`;

                resolve();
            };

            this.pngImage.onerror = () => {
                reject(new Error(`Failed to load image: ${pngPath}`));
            };

            this.pngImage.src = pngPath;
        });
    }
}