/**
 * ClusteringLoader for 3D Embeddings
 *
 * Loads hierarchical clustering data and computes cluster assignments
 * for use in the 3D embedding visualization.
 */
class ClusteringLoader {
  constructor() {
    this._meta = null;
    this._linkage = null;
    this._is_loaded = false;
    this._load_promise = null;
    this._assignments = {};
    this._nClusters = 10;

    // Generate distinct colors using golden angle
    this._colors = this._generateColors(50);
  }

  /**
   * Generate n distinct colors as THREE.Color objects
   */
  _generateColors(n) {
    const colors = [];
    const goldenAngle = 137.508;
    for (let i = 0; i < n; i++) {
      const hue = (i * goldenAngle) % 360;
      const saturation = 0.65 + (i % 3) * 0.1;
      const lightness = 0.45 + (i % 2) * 0.1;
      // Convert HSL to RGB for THREE.js
      const color = new THREE.Color();
      color.setHSL(hue / 360, saturation, lightness);
      colors.push(color);
    }
    return colors;
  }

  /**
   * Ensure clustering data is loaded
   */
  async _ensureLoaded() {
    if (this._is_loaded) return;
    if (this._load_promise) {
      await this._load_promise;
      return;
    }

    this._load_promise = this._load();
    await this._load_promise;
  }

  async _load() {
    try {
      // Try to load from CONFIG or default paths
      const metaUrl =
        CONFIG.clustering?.metaFile ||
        "../../../features/clustering/clustering_meta.json";
      const linkageUrl =
        CONFIG.clustering?.linkageFile ||
        "../../../features/clustering/linkage.json";

      const [metaResp, linkageResp] = await Promise.all([
        fetch(metaUrl),
        fetch(linkageUrl),
      ]);

      if (!metaResp.ok || !linkageResp.ok) {
        console.warn("Clustering data not available");
        this._is_loaded = true;
        return;
      }

      this._meta = await metaResp.json();
      this._linkage = await linkageResp.json();
      this._is_loaded = true;

      // Compute initial assignments
      this._computeAssignments(this._nClusters);
    } catch (e) {
      console.warn("Failed to load clustering data:", e);
      this._is_loaded = true;
    }
  }

  /**
   * Check if clustering data is available
   */
  async isAvailable() {
    await this._ensureLoaded();
    return this._meta !== null && this._linkage !== null;
  }

  /**
   * Compute cluster assignments for n clusters
   */
  _computeAssignments(nClusters) {
    if (!this._linkage || !this._meta) return;

    const linkage = this._linkage;
    const clsValues = this._meta.cls_values;
    const n = clsValues.length;

    if (nClusters >= n) {
      clsValues.forEach((cls, i) => {
        this._assignments[cls] = i;
      });
      return;
    }

    const heights = linkage.map((row) => row[2]).sort((a, b) => b - a);
    const cutHeight = heights[n - nClusters - 1] + 1e-10;

    const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i);

    const find = (x) => {
      if (parent[x] !== x) parent[x] = find(parent[x]);
      return parent[x];
    };

    const union = (x, y, newParent) => {
      parent[find(x)] = newParent;
      parent[find(y)] = newParent;
    };

    for (let i = 0; i < linkage.length; i++) {
      const [idx1, idx2, distance] = linkage[i];
      if (distance <= cutHeight) {
        union(Math.floor(idx1), Math.floor(idx2), n + i);
      }
    }

    const rootToCluster = {};
    let nextCluster = 0;

    this._assignments = {};
    for (let i = 0; i < n; i++) {
      const root = find(i);
      if (!(root in rootToCluster)) {
        rootToCluster[root] = nextCluster++;
      }
      this._assignments[clsValues[i]] = rootToCluster[root];
    }

    this._nClusters = nClusters;
  }

  /**
   * Set number of clusters and recompute assignments
   */
  async setNClusters(n) {
    await this._ensureLoaded();
    this._computeAssignments(n);
  }

  /**
   * Get cluster color for a head as THREE.Color
   */
  async getColor(headId) {
    await this._ensureLoaded();
    const clusterId = this._assignments[headId];
    if (clusterId === undefined) {
      return new THREE.Color(0x888888);
    }
    return this._colors[clusterId % this._colors.length];
  }

  /**
   * Get cluster color synchronously (after loading)
   */
  getColorSync(headId) {
    const clusterId = this._assignments[headId];
    if (clusterId === undefined) {
      return new THREE.Color(0x888888);
    }
    return this._colors[clusterId % this._colors.length];
  }

  /**
   * Get cluster ID for a head
   */
  getClusterId(headId) {
    return this._assignments[headId];
  }

  /**
   * Get current number of clusters
   */
  getNClusters() {
    return this._nClusters;
  }
}

// Global singleton
window.CLUSTERING = new ClusteringLoader();
