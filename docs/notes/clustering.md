# Hierarchical Clustering Notes

## Overview

Hierarchical (agglomerative) clustering groups all attention heads across models based on pairwise distances computed from their feature vectors. The system supports interactive exploration via frontend visualizations, and static export via the pattern types module.

## Pipeline

### Data flow

```
s3: Head distance matrix (mean_dists from DistanceTensorResult)
  |
  v
s4c: HierarchicalClusteringResult.from_distance_matrix()
  |  -- uses scipy linkage() with "average" method
  |  -- converts square distance matrix to condensed form via squareform()
  v
Save to data/features/clustering/
  |  clustering_meta.json   (cls_values, linkage_method)
  |  linkage.npy            (numpy, for Python)
  |  linkage.json           (JSON array, for browser)
  v
Frontend loads linkage.json + clustering_meta.json
  |  -- computes cluster assignments client-side
  |  -- user adjusts n_clusters / cut_height / min_cluster_size interactively
  v
Export as pattern_types.json (browser download or CLI)
```

### Key config

- Linkage method is hardcoded to `"average"` in `s4c_clustering.py`
- Supported methods: `"ward"`, `"average"`, `"complete"`, `"single"` (the `LinkageMethod` type alias)
- `TODO` in s4c to make method configurable via pipeline config

## Core module: `attention_motifs/features/clustering.py`

### `HierarchicalClusteringResult`

Dataclass wrapping scipy's linkage output:

| Field | Type | Description |
|-------|------|-------------|
| `cls_values` | `list[str]` | Head IDs, e.g. `"gpt2-small:L5:H3"` |
| `linkage_matrix` | `Float[np.ndarray, "n_merges 4"]` | Scipy linkage format: `[i, j, distance, count]` per row |
| `linkage_method` | `LinkageMethod` | One of `"ward"`, `"average"`, `"complete"`, `"single"` |

Key methods:

- `from_distance_matrix(distances, cls_values, method)` -- factory from square distance matrix
- `get_clusters(n_clusters=..., cut_height=...)` -- returns `dict[str, int]` (head_id -> 0-indexed cluster)
  - Uses `scipy.cluster.hierarchy.fcluster` with `criterion="maxclust"` or `"distance"`
- `get_max_height()` -- max dendrogram height (root merge distance)
- `save(path)` / `read(path)` -- directory-based I/O (meta JSON + npy + browser JSON)

## Pattern Types module: `attention_motifs/pattern_types/`

Separate system from AttentionPedia. Exports a clustering snapshot with human-editable labels.

### Data model

```
PatternTypes
├── meta: PatternTypesMeta
│   ├── cut_height: float | None
│   ├── n_clusters: int
│   ├── linkage_method: str
│   ├── clustering_path: str
│   ├── created_at: str (ISO timestamp)
│   └── min_cluster_size: int | None
├── stats: PatternTypesStats
│   ├── n_heads: int
│   └── cluster_sizes: dict[str, int]  (cluster_id_str -> count)
├── types: list[PatternType]
│   └── PatternType { id: int, name: str = "none", description: str = "none" }
└── assignments: dict[str, int]  (head_id -> cluster_id)
```

### Misc cluster merging

When `min_cluster_size > 0`, clusters smaller than the threshold get merged into a special "misc" cluster with `id = -1` (constant `PatternTypes.MISC_CLUSTER_ID`). The misc type's description records which original cluster IDs were merged: `"Merged from clusters: 0, 1, 9"`.

### Labeling workflow

Pattern types export with `name = "none"` and `description = "none"` by default. The user edits the JSON file directly to fill in human-readable labels for each cluster.

### CLI

```bash
python -m attention_motifs.pattern_types export \
    --clustering-path data/features/clustering \
    --cut-height 0.35 \       # OR --n-clusters 10 (not both)
    --output pattern_types.json \
    --min-cluster-size 10      # optional, default 0 (disabled)
```

If both `--cut-height` and `--n-clusters` are given, `--n-clusters` takes precedence (with a warning).

## Frontend: `attention_motifs/frontend/clustering/`

### Files

| File | Purpose |
|------|---------|
| `src/index.html` | Page structure, controls layout, script loading |
| `src/clusterState.js` | `ClusterState` singleton (`window.CLUSTER_STATE`) -- manages assignments and colors |
| `src/gridView.js` | Grid visualization, clustering computation, controls, export, mini-dendrogram |
| `src/style.css` | All styling |

### Layout

```
┌─────────────────────────────────────────────────┐
│ Nav bar                                         │
├────────────────────────┬────────────────────────┤
│ Controls (left)        │ Mini dendrogram (right) │
│  - n-clusters          │  - collapsed tree view  │
│  - cut-height          │  - leaves = clusters    │
│  - min-cluster-size    │  - click to select      │
│  - export button       │                         │
│  - scale / sort / order│                         │
├────────────────────────┴────────────────────────┤
│ Stats bar + sparkline                           │
│ Top clusters (clickable chips)                  │
├──────────────────────────┬──────────────────────┤
│ Model grids (left pane)  │ Selected heads (right)│
│  - layers × heads grid   │  - attention patterns │
│  - colored by cluster    │  - links to attnpedia │
│  - resizable divider     │  - selection notes    │
└──────────────────────────┴──────────────────────┘
```

### Clustering computation (in-browser)

The browser recomputes cluster assignments dynamically rather than loading pre-computed assignments:

- **`computeClusters(nClusters)`** -- finds the cut height that gives `nClusters` clusters, then delegates to `computeClustersByHeightInternal`
- **`computeClustersByHeightInternal(cutHeight)`** -- union-find on the linkage matrix: merges clusters at merge distances <= cutHeight
- **`applyMinClusterSize(assignments)`** -- post-processing step that reassigns heads in clusters smaller than `gridState.minClusterSize` to cluster `-1` (misc)

### ClusterState singleton

Manages assignments and color palette:

- **Colors**: 50-color palette using golden angle (137.508 deg) hue spacing with varied saturation/lightness
- **Misc cluster** (`-1`): dark gray `#666666`
- **Unknown heads**: medium gray `#888888`
- **Observer pattern**: `addListener(callback)` for reacting to assignment changes

### Controls (all bidirectional slider + text input)

| Control | Slider range | Effect |
|---------|-------------|--------|
| Number of Clusters | 2 -- min(nHeads, 100) | Recomputes via `computeClusters()` |
| Cut Height | 0 -- min(maxHeight, 20) | Recomputes via `computeClustersByHeightInternal()` |
| Min Cluster Size | 0 -- 100 | Post-processes via `applyMinClusterSize()` |

Changing cut height updates the n-clusters display (and vice versa conceptually). Changing min-cluster-size calls `reapplyCurrentClustering()` which re-runs whichever of cut-height or n-clusters was last used.

### Mini dendrogram

SVG tree rendered in the controls header (right side). Built by:

1. `buildClusterTree()` -- constructs the full linkage tree, then collapses subtrees where all leaves belong to the same cluster into a single leaf node
2. `renderMiniDendrogram()` -- lays out the collapsed tree as a horizontal dendrogram:
   - Leaves on the right with colored circles (click to select cluster)
   - Internal nodes with elbow paths
   - Size labels next to each leaf
   - X-axis scaled to avoid excessively long root line (0.15 offset + 0.85 * normalized height)

### Export (browser)

The "Export Pattern Types" button downloads a `pattern_types.json` with the same schema as the CLI export, plus an extra `selection` field:

```json
{
  "meta": { "cut_height": ..., "n_clusters": ..., ... },
  "stats": { "n_heads": ..., "cluster_sizes": { ... } },
  "types": [ { "id": 0, "name": "none", "description": "none" }, ... ],
  "assignments": { "gpt2-small:L0:H0": 0, ... },
  "selection": { "heads": [...], "note": "..." }
}
```

### Interaction patterns

- **Click cell** -> toggle head selection
- **Shift-click cell** -> select/deselect all heads in that cluster
- **Click cluster chip** -> select/deselect all heads in that cluster
- **Click mini-dendrogram leaf** -> same as clicking chip
- **Randomize button** -> reshuffle which prompts are shown for pattern images

## Relationship to AttentionPedia

| | Pattern Types | AttentionPedia |
|-|---------------|----------------|
| **Purpose** | Labeled clustering snapshots | Metadata/annotation database |
| **Granularity** | Cluster-level labels | Per-head annotations |
| **Source** | Hierarchical clustering cuts | Manual annotation + automated classification |
| **Format** | Single JSON file | Structured frontend app |
| **Depends on clustering?** | Yes (primary input) | Optional (can use ClusteringLoader) |

Pattern types can inform AttentionPedia labels, but the two systems are independent.

## Future work

- [ ] Plots of pattern type fractions across model scales
- [ ] Make linkage method configurable in pipeline config
- [ ] Pre-label clusters using AttentionPedia features
