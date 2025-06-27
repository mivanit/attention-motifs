const ATTNPEDIA_URL = "ap.json";

async function load_attnpedia(path = ATTNPEDIA_URL) {
	const r = await fetch(path);
	if (r.ok) {
		const loaded = await r.json();
		return this._load(loaded);
	} else {
		throw new Error(`Failed to load AttentionPedia data from ${path}`);
	}
}

class AttentionPedia {
	constructor() {
		this._data = load_attnpedia();
	}

	get_head_types(head_id) {
		if (head_id in this._data.head_to_types) {
			return this._data.head_to_types[head_id];
		}
		else {
			return [];
		}
	}

	get_type_heads(type) {
		if (type in this._data.type_to_heads) {
			return this._data.type_to_heads[type];
		}
		else {
			return [];
		}
	}

	get_type_meta(type) {
		if (type in this._data.type_metadata) {
			return this._data.type_metadata[type];
		}
		else {
			return {};
		}
	}

	getNearestHeads(headId, n = 5) {
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

let ATTENTION_PEDIA = new AttentionPedia();