/**
 * Shared clustering computation engine.
 *
 * Pure functions for computing cluster assignments from a scipy-format
 * linkage matrix. Used by clustering, cluster_trends, and attnpedia views.
 */

/**
 * Compute cluster assignments by cutting the dendrogram at a given height.
 * Uses union-find with path compression.
 *
 * @param {number[][]} linkage - Scipy-format linkage matrix [[idx1, idx2, distance, count], ...]
 * @param {string[]} clsValues - Head IDs in clustering order
 * @param {number} cutHeight - Height at which to cut the dendrogram
 * @returns {Object<string, number>} headId -> clusterId (0-indexed)
 */
function computeClustersByHeight(linkage, clsValues, cutHeight) {
  if (!linkage || !clsValues) return {};

  const n = clsValues.length;
  const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i);

  function find(x) {
    if (parent[x] !== x) parent[x] = find(parent[x]);
    return parent[x];
  }

  function union(x, y, newParent) {
    parent[find(x)] = newParent;
    parent[find(y)] = newParent;
  }

  for (let i = 0; i < linkage.length; i++) {
    const [idx1, idx2, distance] = linkage[i];
    if (distance <= cutHeight) {
      union(Math.floor(idx1), Math.floor(idx2), n + i);
    }
  }

  const rootToCluster = {};
  let nextCluster = 0;
  const assignments = {};

  for (let i = 0; i < n; i++) {
    const root = find(i);
    if (!(root in rootToCluster)) {
      rootToCluster[root] = nextCluster++;
    }
    assignments[clsValues[i]] = rootToCluster[root];
  }

  return assignments;
}

/**
 * Apply minimum-size filtering to cluster assignments.
 * Clusters smaller than minSize are reassigned to -1 (misc).
 *
 * @param {Object<string, number>} assignments - Raw headId -> clusterId
 * @param {number} minSize - Minimum cluster size (0 = no filtering)
 * @returns {{ assignments: Object<string, number>, smallClusters: Set<number>, nClusters: number }}
 */
function applyMinSizeFilter(assignments, minSize) {
  if (minSize <= 0) {
    return {
      assignments: assignments,
      smallClusters: new Set(),
      nClusters: new Set(Object.values(assignments)).size,
    };
  }

  // Count cluster sizes
  const clusterCounts = {};
  for (const cid of Object.values(assignments)) {
    clusterCounts[cid] = (clusterCounts[cid] || 0) + 1;
  }

  // Find small clusters
  const smallClusters = new Set();
  for (const [cid, count] of Object.entries(clusterCounts)) {
    if (count < minSize) {
      smallClusters.add(parseInt(cid));
    }
  }

  // Reassign small cluster heads to misc (-1)
  if (smallClusters.size > 0) {
    const merged = {};
    for (const [headId, cid] of Object.entries(assignments)) {
      merged[headId] = smallClusters.has(cid) ? -1 : cid;
    }
    return {
      assignments: merged,
      smallClusters: smallClusters,
      nClusters: new Set(Object.values(merged)).size,
    };
  }

  return {
    assignments: assignments,
    smallClusters: smallClusters,
    nClusters: new Set(Object.values(assignments)).size,
  };
}

/**
 * Find the cut height that produces a target number of clusters.
 *
 * @param {number[][]} linkage - Scipy-format linkage matrix
 * @param {number} nHeads - Number of leaf nodes (heads)
 * @param {number} nClusters - Desired number of clusters
 * @returns {number} Cut height
 */
function cutHeightForNClusters(linkage, nHeads, nClusters) {
  if (nClusters >= nHeads) {
    return Math.max(...linkage.map((row) => row[2]));
  }
  const heights = linkage.map((row) => row[2]).sort((a, b) => b - a);
  return heights[nHeads - nClusters - 1] + 1e-10;
}

/**
 * Compute cluster assignments for a target number of clusters.
 *
 * @param {number[][]} linkage - Scipy-format linkage matrix
 * @param {string[]} clsValues - Head IDs in clustering order
 * @param {number} nClusters - Desired number of clusters
 * @returns {{ assignments: Object<string, number>, cutHeight: number }}
 */
function computeClustersByNClusters(linkage, clsValues, nClusters) {
  const n = clsValues.length;

  if (nClusters >= n) {
    const assignments = {};
    clsValues.forEach((cls, i) => {
      assignments[cls] = i;
    });
    return {
      assignments,
      cutHeight: Math.max(...linkage.map((row) => row[2])),
    };
  }

  const cutHeight = cutHeightForNClusters(linkage, n, nClusters);
  return {
    assignments: computeClustersByHeight(linkage, clsValues, cutHeight),
    cutHeight,
  };
}

/**
 * Look up precomputed flat partition assignments by parameter key.
 * Used for HDBSCAN and Leiden methods where assignments are precomputed
 * at discrete parameter values (not computed client-side like hierarchical).
 *
 * @param {Object<string, Object<string, number>>} partitions - paramKey -> { headId: clusterId }
 * @param {string} paramKey - The parameter key to look up
 * @returns {Object<string, number>} headId -> clusterId (empty object if key not found)
 */
function getFlatPartitionAssignments(partitions, paramKey) {
  return partitions[paramKey] || {};
}
