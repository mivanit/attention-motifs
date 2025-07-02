const HEADDISTSNPY_URL = CONFIG.headDistsnpy_url;
const HEADDISTSMETA_URL = CONFIG.headDistsmeta_url;
// TODO: fallback to getting from github?
// || "https://raw.githubusercontent.com/<TODO>";

async function _load_dists_meta(path_meta = HEADDISTSMETA_URL) {
	const r = await fetch(path_meta);
	if (r.ok) {
		const loaded = await r.json();
		return loaded;
	} else {
		throw new Error(`Failed to load head embeddings data from ${path_meta}`);
	}
}

async function _load_dists_npy(path_npy = HEADDISTSNPY_URL) {
	return NDArray.load(path_npy);
}


class HeadDistances {
	constructor() {
		this._is_loaded = false;
		this.head_dists_meta = null;
		this.head_dists_arr = null;
		this._ensureLoaded();
	}

	async _ensureLoaded() {
		if (!this._is_loaded) {
			this.head_dists_meta = await _load_dists_meta();
			this.head_dists_arr = await _load_dists_npy();
			this._is_loaded = true;
		}
	}

	async get_head_loc() {
		await this._ensureLoaded();
	}

	get_head_idx(head_name) {
		// {"cls_values": ["pythia-1b:L0:H0", "pythia-1b:L0:H1", "pythia-1b:L0:H2", ...]}
		return this.head_dists_meta.cls_values.indexOf(head_name);
	}

	get_head_name(head_idx) {
		return this.head_dists_meta.cls_values[head_idx];
	}

	async getNearestHeads(head_name, n = 5) {
		await this._ensureLoaded();
		// get the index of this head
		const head_idx = this.get_head_idx(head_name);
		// get the distances for this head
		const dists_to_this_head = this.head_dists_arr.get(head_idx);
		// sort the distances and get the indices of the n nearest
		// excluding the closest head, which will be itself
		const indexed_dists = Array.from(dists_to_this_head, (dist, idx) => ({ dist, idx }));
		indexed_dists.sort((a, b) => a.dist - b.dist);
		// Skip the first element (itself) and take the next n
		const best_n_items = indexed_dists.slice(1, n + 1);
		// return dict with head names and distances
		return {
			head_names: best_n_items.map(item => this.get_head_name(item.idx)),
			distances: best_n_items.map(item => item.dist)
		};
	}

	async getHeadDistance(head_name1, head_name2) {
		await this._ensureLoaded();
		const head_idx1 = this.get_head_idx(head_name1);
		const head_idx2 = this.get_head_idx(head_name2);
		const dists_from_head1 = this.head_dists_arr.get(head_idx1);
		return dists_from_head1[head_idx2];
	}

	async getHeadDistances(head_name, head_names_list) {
		await this._ensureLoaded();
		const head_idx = this.get_head_idx(head_name);
		if (head_idx === -1) {
			console.warn(`Head not found: ${head_name}`);
			return head_names_list.map(target_head => ({
				head_name: target_head,
				distance: 0
			}));
		}
		
		const dists_from_head = this.head_dists_arr.get(head_idx);
		if (!dists_from_head) {
			console.warn(`No distances found for head: ${head_name}`);
			return head_names_list.map(target_head => ({
				head_name: target_head,
				distance: 0
			}));
		}
		
		return head_names_list.map(target_head => {
			const target_idx = this.get_head_idx(target_head);
			if (target_idx === -1) {
				console.warn(`Target head not found: ${target_head}`);
				return {
					head_name: target_head,
					distance: 0
				};
			}
			
			const distance = dists_from_head[target_idx];
			return {
				head_name: target_head,
				distance: distance !== undefined ? distance : 0
			};
		});
	}
}