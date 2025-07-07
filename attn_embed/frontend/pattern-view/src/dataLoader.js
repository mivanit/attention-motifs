/**
 * Data Loader Module
 * Handles fetching attention patterns and prompt metadata
 */

const DATA_BASE_PATH = '../../../../../data/patterns/';

class AttentionDataLoader {
    constructor(basePath = DATA_BASE_PATH) {
        this.basePath = basePath;
    }

    async loadAttentionPattern(model, promptHash, layerIdx, headIdx) {
        const pngPath = `${this.basePath}${model}/prompts/${promptHash}/L${layerIdx}/H${headIdx}/attn.png`;
        return await pngToMatrix(pngPath);
    }

    async loadPromptMetadata(model, promptHash) {
        const jsonPath = `${this.basePath}${model}/prompts/${promptHash}/prompt.json`;
        const response = await fetch(jsonPath);
        if (!response.ok) {
            throw new Error(`Failed to load prompt metadata: ${response.statusText}`);
        }
        return await response.json();
    }
}