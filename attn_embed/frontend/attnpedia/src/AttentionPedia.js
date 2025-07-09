// TODO: fallback to getting from github?
// || "https://raw.githubusercontent.com/<TODO>";

async function load_attnpedia() {
	const r = await fetch(CONFIG.attnpedia_url);
	if (r.ok) {
		const loaded = await r.json();
		return loaded;
	} else {
		throw new Error(`Failed to load AttentionPedia data from ${path}`);
	}
}

class AttentionPedia {
	constructor() {
		this._data = null;
		this._loaded = false;
	}

	async _ensureLoaded() {
		if (!this._loaded) {
			this._data = await load_attnpedia();
			this._loaded = true;
		}
	}

	async get_head_types(head_id) {
		await this._ensureLoaded();
		return this._data?.head_to_types?.[head_id] || [];
	}

	async get_type_heads(type) {
		await this._ensureLoaded();
		return this._data?.type_to_heads?.[type] || [];
	}

	async get_type_meta(type) {
		await this._ensureLoaded();
		return this._data?.type_metadata?.[type] || {};
	}
}