# Interface Relations and TODOs

> Note: we should attempt to make the needed connections with minimal changes to the code, instead working with the configuration files.


misc todos:

- **IMPORTANT**: in attentionpedia, specifying a list of heads to view doesnt work because we use the same separator between heads and between model/lyr_idx/head_idx!
- [x] create a classifications page that lists all head classifications with links to their respective AttentionPedia pages
- [ ] embedding visualizations: 
  - [ ] hovering over point sometimes doesnt work, the "hover radius" is much smaller than the point size. make the hover radius scale with the point size by some constant factor which we can configure in config.js
  - [ ] make sure default configs are reasonable
- [x] filter by model, layer in attentionpedia and classifications page?
- [ ] for each model, a nice display of all the heads (colored by classification, if any?) where you can click on a head to go to attentionpedia for that head
- [x] for pattern embeddings save and load from a reduced-precision csv file to save space
  - [x] save `pca.csv`
  - [x] load `pca.csv` in the pattern embedding vis (we might still use jsonl for head embedding -- need to detect which loading func to use based on file extension)

## Pattern Lens (`data/patterns/index.html`)

Allows comparing lots of patterns across many heads, models, etc. Uses tilde-separated URLs for compatibility and has model/head selection grids and prompt tables.

Outgoing connections:
- [x] **Single Pattern View** (pattern images): Click on pattern images to view detailed single pattern

## Single Pattern View (`data/patterns/single.html`)

Shows a single pattern and its prompt in detail with interactive heatmap and token highlighting.

Outgoing connections:
- [x] **Pattern Lens** (prompt hash): Click prompt hash to go to pattern lens with that prompt selected
- [x] **AttentionPedia** (head ID): Click head ID to go to attentionpedia for that head
- [ ] **Pattern Embedding Vis** just this pattern selected

## AttentionPedia (`data/vis/attnpedia/index.html`)

Shows a given head and then a selection of nearby/random/distant (in embedding space) heads, as well as heads with the same classification. These heads are the rows, columns are different prompts.

Outgoing connections:
- [x] **Single Pattern View** (patterns in each cell): Click on patterns in table cells to view detailed single pattern
- [x] **Pattern Lens** (current head or all displayed heads): Use links at top to go to pattern lens for current head or all displayed heads
- [x] **Classifications Page** (classification links): view classifications page
- [ ] **Pattern Embedding Vis** (current/all heads selected): Add way to go to pattern embedding vis with current head/context selected
- [ ] **Head Embedding Vis** (current/all heads selected): Add way to go to head embedding vis with current head/context selected

## Pattern Embedding Vis (`data/vis/embeds/patterns/index.html`)

3D visualization of all attention patterns where each point represents a pattern.

Outgoing connections:
- [x] **Single Pattern View** (each point - right click): Right-click on pattern points to go to single pattern view for that pattern

## Head Embedding Vis (`data/vis/embeds/heads/index.html`)

3D visualization where each point is a head in embedding space.

Outgoing connections:
- [x] **AttentionPedia** (each point - right click): Right-click on head points to go to attentionpedia for that head

## Head Embedding Table (`data/figures/head_embed_table.html`)

Table view of all head embedding plots with SVG previews and links to 3D visualization.

Outgoing connections:
- [x] **Head Embedding Vis** (embed links): Click embedding links to view in 3D visualization with specific parameters

## Classifications Page (`data/figures/classifications.html`)

Lists all head classifications with links to view heads of each classification in AttentionPedia.

Outgoing connections:
- [x] **AttentionPedia** (classification links): Click on classification to view AttentionPedia page filtered by that classification

## Model View (`data/figures/model_view.html`)

Grid display of all heads for each model, colored by classification if any, allowing direct navigation to AttentionPedia.

Outgoing connections:
- [ ] **AttentionPedia** (click on head): Click on any head to go to AttentionPedia for that head


# interface relations diagram

- solid lines / blue label: existing connections
- dashed lines / red label: missing connections that should be added



```mermaid
---
config:
  themeVariables:
    clusterBkg: transparent
    clusterBorder: transparent
  
  flowchart:
    defaultRenderer: elk
  elk:
    mergeEdges: false
    nodePlacementStrategy: LINEAR_SEGMENTS
    hierarchyHandling: INCLUDE_CHILDREN
    direction: RIGHT
    spacing:
      nodeNode: 50
    layered:
      considerModelOrder: true
---
graph TD
    %% heads
    CL@{img: "docs/resources/assets/diagram/head_classes.png", label: "<a href='data/figures/classifications.html'>Classifications Page</a>", pos: "t", w: 60, h: 120}
    HET@{img: "docs/resources/assets/diagram/head_embed_table.png", label: "<a href='data/figures/head_embed_table.html'>Head Embedding Table</a>", pos: "t", w: 60, h: 120}
    HEV@{img: "docs/resources/assets/diagram/embed_heads.png", label: "<a href='data/vis/embeds/heads/index.html'>Head Embedding Vis</a>", pos: "t", w: 60, h: 120}


    %% attnpedia
    AP@{img: "docs/resources/assets/diagram/attnpedia.png", label: "<a href='data/vis/attnpedia/index.html'>AttentionPedia</a>", pos: "t", w: 60, h: 120}
    MV["Model View"]

    %% patterns
    PL@{img: "docs/resources/assets/diagram/pattern_lens.png", label: "<a href='data/patterns/index.html'>Pattern Lens</a>", pos: "t", w: 60, h: 160}
    SPV@{img: "docs/resources/assets/diagram/single_pattern.png", label: "<a href='data/patterns/single.html'>Single Pattern View</a>", pos: "t", w: 60, h: 120}
    PEV@{img: "docs/resources/assets/diagram/embed_patterns.png", label: "<a href='data/vis/embeds/patterns/index.html'>Pattern Embedding Vis</a>", pos: "t", w: 60, h: 120}

    %% within heads
    HET -->|links to each variant| HEV



    %% within patterns
    PL -->|pattern images| SPV
    SPV -->|prompt hash| PL
    SPV -.->|<span style='background-color: red'>just this pattern selected</span>| PEV
    PEV -->|each point - right click| SPV

    %% attnpedia
    MV -.->|<span style='background-color: red'>click on head</span>| AP

    %% patterns and attnpedia
    SPV -->|head ID| AP
    AP -->|patterns in each cell| SPV
    AP -->|current head <br/>or<br/> all displayed heads| PL
    AP -.->|<span style='background-color: red'>current/all heads selected</span>| PEV

    %% attnpedia and heads
    CL -->|classification links| AP
    AP -->|link| CL
    HEV -->|each point - right click| AP
    AP -.->|<span style='background-color: red'>current/all heads selected</span>| HEV
```



