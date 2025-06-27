const ATTNPEDIA_URL = CONFIG.attnpedia_url; 
// TODO: fallback to getting from github?
// || "https://raw.githubusercontent.com/<TODO>";

async function load_attnpedia(path = ATTNPEDIA_URL) {
	const r = await fetch(path);
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
		if (this._data && head_id in this._data.head_to_types) {
			return this._data.head_to_types[head_id];
		}
		else {
			return [];
		}
	}

	async get_type_heads(type) {
		await this._ensureLoaded();
		if (this._data && type in this._data.type_to_heads) {
			return this._data.type_to_heads[type];
		}
		else {
			return [];
		}
	}

	async get_type_meta(type) {
		await this._ensureLoaded();
		if (this._data && type in this._data.type_metadata) {
			return this._data.type_metadata[type];
		}
		else {
			return {};
		}
	}

	async getNearestHeads(headId, n = 5) {
		await this._ensureLoaded();
		const targetHead = this.heads.get(headId);
		if (!targetHead) return [];

		const distances = [];
		this.heads.forEach((head, id) => {
			if (id !== headId && head.model === targetHead.model) {
				distances.push({
					head: head,
					distance: targetHead.distanceTo(head)
				});
			}
		});

		return distances
			.sort((a, b) => a.distance - b.distance)
			.slice(0, n)
			.map(item => ({ ...item.head, distance: item.distance }));
	}
}