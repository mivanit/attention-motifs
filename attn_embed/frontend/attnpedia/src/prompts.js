const PROMPTS_URL = CONFIG.prompts_url;

async function load_prompts(path = PROMPTS_URL) {
	const r = await fetch(path);
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
		throw new Error(`Failed to load prompts data from ${path}`);
	}
}

class PromptsLoader {
	constructor() {
		this._data = null;
		this._loaded = false;
	}

	async _ensureLoaded() {
		if (!this._loaded) {
			this._data = await load_prompts();
			this._loaded = true;
		}
	}

	async getPrompts() {
		await this._ensureLoaded();
		return this._data || [];
	}

	async getPromptByHash(hash) {
		await this._ensureLoaded();
		return this._data?.find(prompt => prompt.hash === hash) || null;
	}

	async getPromptHashes() {
		await this._ensureLoaded();
		return this._data?.map(prompt => prompt.hash) || [];
	}
}