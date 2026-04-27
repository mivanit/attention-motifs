/**
 * Shared clustering configuration state.
 *
 * Persists clustering UI state (cut height, highlighted cluster, etc.)
 * to localStorage so it can be shared across cluster_trends and clustering
 * pages without URL parameters.
 */

const CLUSTERING_CONFIG_KEY = "clustering_config";

const ClusteringConfig = {
  /**
   * Load full state from localStorage.
   * @returns {Object} parsed state (empty object if nothing stored)
   */
  load() {
    try {
      return JSON.parse(localStorage.getItem(CLUSTERING_CONFIG_KEY)) || {};
    } catch {
      return {};
    }
  },

  /**
   * Merge partial state into localStorage.
   * @param {Object} partial - key/value pairs to merge
   */
  save(partial) {
    const current = this.load();
    localStorage.setItem(
      CLUSTERING_CONFIG_KEY,
      JSON.stringify({ ...current, ...partial }),
    );
  },

  /** @returns {number|null} */
  getCutHeight() {
    return this.load().cutHeight ?? null;
  },

  /** @param {number} h */
  setCutHeight(h) {
    this.save({ cutHeight: h });
  },

  /** @returns {number|null} */
  getHighlightCluster() {
    return this.load().highlightCluster ?? null;
  },

  /** @param {number} id */
  setHighlightCluster(id) {
    this.save({ highlightCluster: id });
  },

  clearHighlightCluster() {
    const s = this.load();
    delete s.highlightCluster;
    localStorage.setItem(CLUSTERING_CONFIG_KEY, JSON.stringify(s));
  },

  /** @returns {string|null} clustering method name ("hierarchical", "hdbscan", "leiden") */
  getMethod() {
    return this.load().method ?? null;
  },

  /** @param {string} method */
  setMethod(method) {
    this.save({ method });
  },

  /** @returns {string|null} current parameter key for flat methods */
  getParamKey() {
    return this.load().paramKey ?? null;
  },

  /** @param {string} key */
  setParamKey(key) {
    this.save({ paramKey: key });
  },
};
