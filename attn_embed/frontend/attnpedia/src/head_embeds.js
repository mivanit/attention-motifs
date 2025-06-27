const HEAD_EMBEDS_URL = CONFIG.head_embeds_url;
// TODO: fallback to getting from github?
// || "https://raw.githubusercontent.com/<TODO>";

async function load_head_embeds(path = HEAD_EMBEDS_URL) {
	const r = await fetch(path);
	if (r.ok) {
		const loaded = await r.json();
		return loaded;
	} else {
		throw new Error(`Failed to load head embeddings data from ${path}`);
	}
}

class HeadEmbeds {
	constructor() {
		this._is_loaded = false;
		this.head_embeddings = null;
		this._ensureLoaded();
	}

	async _ensureLoaded() {
		if (!this._is_loaded) {
			this.head_embeddings = await load_head_embeds();
			this._is_loaded = true;
		}
	}

	async get_head_loc() {
		await this._ensureLoaded();

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