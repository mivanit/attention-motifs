/**
 * ClusterState - Manages cluster assignments and provides colors
 *
 * This is a singleton that can be imported by other frontend components
 * to get consistent cluster coloring across views.
 */
class ClusterState {
  constructor() {
    this.assignments = {}; // headId -> clusterId (0-indexed)
    this.nClusters = 10;
    this.listeners = [];
  }

  /**
   * Set cluster assignments from a map
   * @param {Object.<string, number>} assignments - Map of headId to clusterId
   * @param {number} nClusters - Number of clusters
   */
  setAssignments(assignments, nClusters) {
    this.assignments = assignments;
    this.nClusters = nClusters;
    this._notify();
  }

  /**
   * Get the color for a given head ID
   * @param {string} headId - The head identifier (e.g., "gpt2-small:L5:H3")
   * @returns {string} CSS color string
   */
  getColor(headId) {
    const clusterId = this.assignments[headId];
    if (clusterId === undefined) {
      return "#888888"; // Gray for unknown heads
    }
    if (clusterId === -1) {
      return "#666666"; // Dark gray for misc cluster
    }
    return clusterColor(clusterId);
  }

  /**
   * Get all assignments
   * @returns {Object.<string, number>} Map of headId to clusterId
   */
  getAssignments() {
    return { ...this.assignments };
  }

  /**
   * Get number of clusters
   * @returns {number} Number of clusters
   */
  getNumClusters() {
    return this.nClusters;
  }

  /**
   * Get the cluster ID for a given head
   * @param {string} headId - The head identifier
   * @returns {number|undefined} Cluster ID or undefined if not found
   */
  getClusterId(headId) {
    return this.assignments[headId];
  }

  /**
   * Get all heads in a specific cluster
   * @param {number} clusterId - The cluster ID
   * @returns {string[]} Array of head IDs in the cluster
   */
  getHeadsInCluster(clusterId) {
    return Object.entries(this.assignments)
      .filter(([_, cid]) => cid === clusterId)
      .map(([headId, _]) => headId);
  }

  /**
   * Get cluster sizes
   * @returns {Object.<number, number>} Map of clusterId to count
   */
  getClusterSizes() {
    const sizes = {};
    for (const clusterId of Object.values(this.assignments)) {
      sizes[clusterId] = (sizes[clusterId] || 0) + 1;
    }
    return sizes;
  }

  /**
   * Add a listener for cluster changes
   * @param {function} callback - Called with (assignments, nClusters) when clusters change
   */
  addListener(callback) {
    this.listeners.push(callback);
  }

  /**
   * Remove a listener
   * @param {function} callback - The callback to remove
   */
  removeListener(callback) {
    this.listeners = this.listeners.filter((cb) => cb !== callback);
  }

  /**
   * Notify all listeners of changes
   */
  _notify() {
    for (const callback of this.listeners) {
      try {
        callback(this.assignments, this.nClusters);
      } catch (e) {
        console.error("ClusterState listener error:", e);
      }
    }
  }
}

// Global singleton instance
window.CLUSTER_STATE = new ClusterState();
