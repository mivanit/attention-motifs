class HeadInfo {
    constructor(id, model, layer, head) {
        this.id = id;
        this.model = model;
        this.layer = layer;
        this.head = head;
        this.classifications = [];
    }

    static from_id(id) {
        // split by ":", should have 3 components
        // `{model}:L{layer}:H{head}`
        const parts = id.split(':');
        if (parts.length !== 3) {
            throw new Error(`Invalid id format: ${id}. Expected format: {model}:L{layer}:H{head}`);
        }
        const [model, layerPart, headPart] = parts;
        if (!layerPart.startsWith('L')) {
            throw new Error(`Invalid layer format: ${layerPart}. Expected format: L{number}`);
        }
        if (!headPart.startsWith('H')) {
            throw new Error(`Invalid head format: ${headPart}. Expected format: H{number}`);
        }
        const layer = parseInt(layerPart.slice(1), 10);
        const head = parseInt(headPart.slice(1), 10);
        if (isNaN(layer) || isNaN(head)) {
            throw new Error(`Invalid numeric values in id: ${id}`);
        }
        return new HeadInfo(id, model, layer, head);
    }

    async get_pattern_url(promptHash) {
        return `${this.model}/prompts/${promptHash}/L${this.layer}/H${this.head}/raw.png`;
    }
}