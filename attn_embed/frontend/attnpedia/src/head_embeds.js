// TODO: fallback to getting from github?
// || "https://raw.githubusercontent.com/<TODO>";

async function _load_dists_meta() {
  const r = await fetch(CONFIG.headDistsmeta_url);
  if (r.ok) {
    const loaded = await r.json();
    return loaded;
  } else {
    throw new Error(
      `Failed to load head embeddings data from ${CONFIG.headDistsmeta_url}`,
    );
  }
}

async function _load_dists_npy() {
  return NDArray.load(CONFIG.headDistsnpy_url);
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

  get_head_idx(head_name) {
    const idx = this.head_dists_meta.cls_values.indexOf(head_name);
    if (idx === -1) {
      throw new Error(`Head not found: ${head_name}`);
    }
    return idx;
  }

  get_head_name(head_idx) {
    if (head_idx < 0 || head_idx >= this.head_dists_meta.cls_values.length) {
      throw new Error(`Invalid head index: ${head_idx}`);
    }
    return this.head_dists_meta.cls_values[head_idx];
  }

  async getNearestHeads(head_name, n = 5) {
    await this._ensureLoaded();

    const head_idx = this.get_head_idx(head_name);
    const distanceRow = this.head_dists_arr.get(head_idx);
    const distanceArray = Array.from(distanceRow.data);

    // Sort distances and get indices of n nearest (excluding self)
    const indexed_dists = Array.from(distanceArray, (dist, idx) => ({
      dist,
      idx,
    }));
    indexed_dists.sort((a, b) => a.dist - b.dist);
    const best_n_items = indexed_dists.slice(1, n + 1);

    return {
      head_names: best_n_items.map((item) => this.get_head_name(item.idx)),
      distances: best_n_items.map((item) => item.dist),
    };
  }

  async getHeadDistance(head_name1, head_name2) {
    await this._ensureLoaded();
    const head_idx1 = this.get_head_idx(head_name1);
    const head_idx2 = this.get_head_idx(head_name2);

    const distanceRow = this.head_dists_arr.get(head_idx1);
    return distanceRow.data[head_idx2];
  }

  async getHeadDistances(head_name, head_names_list) {
    await this._ensureLoaded();
    const head_idx = this.get_head_idx(head_name);
    const distanceRow = this.head_dists_arr.get(head_idx);
    const distanceArray = Array.from(distanceRow.data);

    return head_names_list.map((target_head) => {
      const target_idx = this.get_head_idx(target_head);
      const distance = distanceArray[target_idx];
      return {
        head_name: target_head,
        distance: distance,
      };
    });
  }
}
