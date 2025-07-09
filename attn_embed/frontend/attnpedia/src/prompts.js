// TODO: fallback to getting from github?
// || "https://raw.githubusercontent.com/<TODO>";

async function load_prompts() {
	const r = await fetch(CONFIG.prompts_url);
	if (r.ok) {
		const text = await r.text();
		const lines = text.trim().split('\n');
		const prompts = [];
		
		for (const line of lines) {
			if (line.trim()) {
				try {
					const prompt = JSON.parse(line);
					prompts.push(prompt);
				} catch (e) {
					console.warn('Failed to parse prompt line:', line, e);
				}
			}
		}
		
		return prompts;
	} else {
		throw new Error(`Failed to load prompts data from ${CONFIG.prompts_url}`);
	}
}

class PromptsLoader {
	constructor() {
		this._data = null;
		this._hashToIndex = null;
		this._loaded = false;
	}

	async _ensureLoaded() {
		if (!this._loaded) {
			this._data = await load_prompts();
			// Build hash to index map for efficient lookups
			this._hashToIndex = {};
			for (let i = 0; i < this._data.length; i++) {
				const prompt = this._data[i];
				if (prompt.hash) {
					this._hashToIndex[prompt.hash] = i;
				}
			}
			this._loaded = true;
		}
	}

	async get_all() {
		await this._ensureLoaded();
		return this._data || [];
	}

	async getPromptByHash(hash) {
		await this._ensureLoaded();
		if (!this._hashToIndex || !this._data) return null;
		const index = this._hashToIndex[hash];
		return index !== undefined ? this._data[index] : null;
	}

	async getPromptHashes() {
		await this._ensureLoaded();
		return this._hashToIndex ? Object.keys(this._hashToIndex) : [];
	}
}