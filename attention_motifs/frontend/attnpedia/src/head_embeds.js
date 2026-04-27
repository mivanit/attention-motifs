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

class HeadDistances {
  constructor() {
    this._meta_loaded = false;
    this._meta_promise = null;
    this.head_dists_meta = null;
    this._rowCache = new Map();
  }

  async _ensureMetaLoaded() {
    if (this._meta_loaded) {
      return;
    }

    if (this._meta_promise) {
      return this._meta_promise;
    }

    this._meta_promise = (async () => {
      try {
        this.head_dists_meta = await _load_dists_meta();
        this._meta_loaded = true;
      } catch (error) {
        NOTIF.error("Failed to load head distances metadata", error);
        throw error;
      }
    })();

    return this._meta_promise;
  }

  /** Fetch a single row from the distances matrix via HTTP range request */
  async _loadRow(head_idx) {
    if (this._rowCache.has(head_idx)) {
      return this._rowCache.get(head_idx);
    }
    const promise = NDArray.loadSlice(CONFIG.headDistsnpy_url, head_idx);
    this._rowCache.set(head_idx, promise);
    return promise;
  }

  isMetadataLoaded() {
    return this.head_dists_meta !== null;
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
    await this._ensureMetaLoaded();

    const head_idx = this.get_head_idx(head_name);
    const distanceRow = await this._loadRow(head_idx);
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
    await this._ensureMetaLoaded();
    const head_idx1 = this.get_head_idx(head_name1);
    const head_idx2 = this.get_head_idx(head_name2);

    const distanceRow = await this._loadRow(head_idx1);
    return distanceRow.data[head_idx2];
  }

  async getHeadDistances(head_name, head_names_list) {
    await this._ensureMetaLoaded();
    const head_idx = this.get_head_idx(head_name);
    const distanceRow = await this._loadRow(head_idx);
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
