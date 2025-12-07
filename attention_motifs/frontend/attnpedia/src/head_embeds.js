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
    this._load_promise = null;
    this.head_dists_meta = null;
    this.head_dists_arr = null;
    // Don't start loading in constructor - let methods trigger it when needed
  }

  async _ensureLoaded() {
    // If already loaded, return immediately
    if (this._is_loaded) {
      return;
    }

    // If loading is in progress, wait for the existing promise
    if (this._load_promise) {
      return this._load_promise;
    }

    // Start loading and store the promise to prevent duplicate loads
    this._load_promise = (async () => {
      try {
        const notif = NOTIF.pbar("Loading head distances data...");

        // Load metadata first
        this.head_dists_meta = await _load_dists_meta();
        notif.progress(0.2);

        // Load the large distances array
        this.head_dists_arr = await _load_dists_npy();
        notif.progress(1.0);

        this._is_loaded = true;
        notif.complete();
        NOTIF.success("Head distances loaded successfully");
      } catch (error) {
        NOTIF.error("Failed to load head distances", error);
        throw error;
      }
    })();

    return this._load_promise;
  }

  isMetadataLoaded() {
    return this.head_dists_meta !== null;
  }

  isFullyLoaded() {
    return this._is_loaded;
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
