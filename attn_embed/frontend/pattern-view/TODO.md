# Pattern View Integration Plan

## Overview
Integrate the attention pattern viewer with the existing data storage system to display attention patterns for specific prompts and attention heads.

## Current State
- **Pattern Viewer**: Basic heatmap visualization in `index.html` with randomly generated data
- **Data Storage**: 
  - Attention patterns stored in `data/patterns/{model}/prompts/{prompt_hash}/`
  - NPZ files contain all layers: `blocks.{layer_idx}.attn.hook_pattern` with shape `(1, n_heads, seq_len, seq_len)`
  - PNG files exist in `L{layer_idx}/H{head_idx}/attn.png` directories
  - Prompt metadata in `prompt.json` with tokens and text

## Implementation Plan

### 1. Data Access Strategy
**Option A: Direct NPZ Loading**
- Load NPZ file and extract specific layer/head on demand
- Pros: Simple, no preprocessing needed
- Cons: Loading ~3MB file for each request, inefficient for single head access

**Option B: Pre-extract to NPY Files**
- Extract NPZ into individual NPY files per layer
- Store as `npy/layer_{0-17}.npy` containing all heads for that layer
- Pros: Efficient loading of specific layers, smaller file sizes
- Cons: Requires preprocessing step

**Option C: Use Existing PNG Files** ✓ (Selected)
- Load pre-rendered PNG images from `L{layer}/H{head}/attn.png`
- Extract pixel values to reconstruct attention matrix
- PNG encoding: linear scale from 0 (black) to 1 (white)
- Matrix properties: row-stochastic, lower triangular
- Top-left pixel is always 1.0, pixel to its right is always 0.0
- Pros: Fast loading, no data processing needed, consistent format

### 2. URL Parameter Handling
- Accept parameters: `?prompt={prompt_hash}&head={model}.L{layer_idx}.H{head_idx}`
- Example: `index.html?prompt=MjWkadUxH4Z5nfLEFtUc1w&head=gemma-2b.L5.H3`
- Parse URL parameters on page load
- Minimal error handling - fail fast during development

### 3. Implementation Plan

#### Step 1: Create PNG Loader
```javascript
// pngLoader.js
async function loadPNGAsMatrix(url) {
  // Load PNG image
  // Extract pixel values from canvas
  // Convert grayscale values (0-255) to float values (0-1)
  // Return as Float32Array with dimensions
}
```

#### Step 2: Update Data Loader
```javascript
// dataLoader.js
const DATA_BASE_PATH = '../../data/patterns/'; // Global variable for easy configuration

class AttentionDataLoader {
  async loadAttentionPattern(model, promptHash, layerIdx, headIdx) {
    const pngPath = `${DATA_BASE_PATH}${model}/prompts/${promptHash}/L${layerIdx}/H${headIdx}/attn.png`;
    return await loadPNGAsMatrix(pngPath);
  }
  
  async loadPromptMetadata(model, promptHash) {
    const jsonPath = `${DATA_BASE_PATH}${model}/prompts/${promptHash}/prompt.json`;
    return await fetch(jsonPath).then(r => r.json());
  }
}
```

#### Step 3: Refactor Visualization
- Extract heatmap rendering logic into reusable module
- Update to use real attention data instead of random generation
- Use prompt tokens for axis labels

#### Step 4: Wire Everything Together
- Parse URL parameters
- Load prompt metadata
- Load attention pattern PNG
- Display with proper labels

### 4. Key Technical Details

#### PNG to Matrix Conversion
1. Load image into canvas
2. Get image data using `getImageData()`
3. Extract grayscale values (R channel since image is grayscale)
4. Convert from 0-255 to 0-1 range: `value / 255`
5. Reshape into 2D matrix based on image dimensions

#### Data Path Configuration
- Global `DATA_BASE_PATH` variable for easy reconfiguration
- Default: `'../../data/patterns/'` when serving from `pattern-view/src/`
- Can be changed for different deployment scenarios

### 5. Minimal Example Flow
1. User visits: `index.html?prompt=MjWkadUxH4Z5nfLEFtUc1w&head=gemma-2b.L5.H3`
2. Parse: model="gemma-2b", layer=5, head=3
3. Load: `${DATA_BASE_PATH}gemma-2b/prompts/MjWkadUxH4Z5nfLEFtUc1w/prompt.json`
4. Load: `${DATA_BASE_PATH}gemma-2b/prompts/MjWkadUxH4Z5nfLEFtUc1w/L5/H3/attn.png`
5. Display heatmap with token labels

