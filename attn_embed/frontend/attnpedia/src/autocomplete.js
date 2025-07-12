/**
 * Autocomplete functionality for head search input
 * Provides smart suggestions based on available models, layers, and heads
 */

class HeadAutocomplete {
    constructor(modelsData, headDistances) {
        this.modelsData = modelsData;
        this.headDistances = headDistances;
        this.modelMap = new Map();
        this.allHeads = [];
        this.init();
    }

    init() {
        // Create a map for quick model lookups
        this.modelsData.forEach(model => {
            this.modelMap.set(model.model_name, model);
        });
    }

    async ensureHeadsLoaded() {
        if (this.allHeads.length === 0) {
            await this.headDistances._ensureLoaded();
            this.allHeads = this.headDistances.head_dists_meta.cls_values;
        }
    }

    /**
     * Get autocomplete suggestions based on current input
     * @param {string} input - Current input text
     * @returns {Array} Array of suggestion objects
     */
    async getSuggestions(input) {
        if (!input.trim()) {
            return [];
        }

        await this.ensureHeadsLoaded();

        const colonCount = (input.match(/:/g) || []).length;
        
        if (colonCount === 0) {
            return this.getModelSuggestions(input);
        } else if (colonCount === 1) {
            return this.getLayerSuggestions(input);
        } else if (colonCount === 2) {
            return this.getHeadSuggestions(input);
        }
        
        return [];
    }

    /**
     * Get model name suggestions
     * @param {string} input - Current input
     * @returns {Array} Array of suggestion objects
     */
    getModelSuggestions(input) {
        const modelNames = Array.from(this.modelMap.keys());
        const filtered = modelNames.filter(name => 
            name.toLowerCase().includes(input.toLowerCase())
        ).slice(0, 8);

        return filtered.map(modelName => {
            const model = this.modelMap.get(modelName);
            const dimensions = model ? `${model.n_layers} layers × ${model.n_heads} heads` : '';
            return {
                type: 'model',
                text: modelName,
                display: dimensions,
                completion: modelName + ':L',
                valid: true
            };
        });
    }

    /**
     * Get layer suggestions for a specific model
     * @param {string} input - Current input (e.g., "gpt2-small:L")
     * @returns {Array} Array of suggestion objects
     */
    getLayerSuggestions(input) {
        const parts = input.split(':');
        const modelName = parts[0];
        const layerPrefix = parts[1] || '';

        const model = this.modelMap.get(modelName);
        if (!model) {
            return [{
                type: 'error',
                text: input,
                display: `Unknown model: ${modelName}`,
                completion: input,
                valid: false
            }];
        }

        // Generate layer suggestions
        const suggestions = [];
        for (let i = 0; i < model.n_layers; i++) {
            const layerStr = `L${i}`;
            if (layerStr.toLowerCase().startsWith(layerPrefix.toLowerCase())) {
                suggestions.push({
                    type: 'layer',
                    text: `${modelName}:${layerStr}`,
                    display: `Layer ${i}`,
                    completion: `${modelName}:${layerStr}:H`,
                    valid: true
                });
            }
        }

        return suggestions.slice(0, 8);
    }

    /**
     * Get head suggestions for a specific model and layer
     * @param {string} input - Current input (e.g., "gpt2-small:L5:H")
     * @returns {Array} Array of suggestion objects
     */
    getHeadSuggestions(input) {
        const parts = input.split(':');
        const modelName = parts[0];
        const layerPart = parts[1] || '';
        const headPrefix = parts[2] || '';

        const model = this.modelMap.get(modelName);
        if (!model) {
            return [{
                type: 'error',
                text: input,
                display: `Unknown model: ${modelName}`,
                completion: input,
                valid: false
            }];
        }

        // Parse layer number
        const layerMatch = layerPart.match(/^L(\d+)$/);
        if (!layerMatch) {
            return [{
                type: 'error',
                text: input,
                display: `Invalid layer format: ${layerPart}`,
                completion: input,
                valid: false
            }];
        }

        const layerNum = parseInt(layerMatch[1]);
        if (layerNum >= model.n_layers) {
            return [{
                type: 'error',
                text: input,
                display: `Layer ${layerNum} doesn't exist (max: ${model.n_layers - 1})`,
                completion: input,
                valid: false
            }];
        }

        // Generate head suggestions
        const suggestions = [];
        for (let i = 0; i < model.n_heads; i++) {
            const headStr = `H${i}`;
            if (headStr.toLowerCase().startsWith(headPrefix.toLowerCase())) {
                const fullHeadId = `${modelName}:L${layerNum}:H${i}`;
                suggestions.push({
                    type: 'head',
                    text: fullHeadId,
                    display: `Head ${i}`,
                    completion: fullHeadId,
                    valid: true
                });
            }
        }

        return suggestions.slice(0, 8);
    }

    /**
     * Check if a complete head ID exists in the dataset
     * @param {string} headId - Complete head ID to check
     * @returns {boolean} True if head exists
     */
    async headExists(headId) {
        await this.ensureHeadsLoaded();
        return this.allHeads.includes(headId);
    }
}