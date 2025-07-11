# Interface Relations and TODOs

> Note: we should attempt to make the needed connections with minimal changes to the code, instead working with the configuration files.


misc todos:

- pages:
    - [x] create a classifications page that lists all head classifications with links to their respective AttentionPedia pages
    - [ ] embedding visualizations: hovering over point sometimes doesnt work, the "hover radius" is much smaller than the point size. make the hover radius scale with the point size by some constant factor which we can configure in config.js
    - [ ] filter by model, layer in attentionpedia and classifications page?
    - [ ] for each model, a nice display of all the heads (colored by classification, if any?) where you can click on a head to go to attentionpedia for that head
- python pipeline:
    - [x] move head embedding table generation from s5b into s4b_write_frontend.py for consistency
    - [ ] default pipeline config should more accurately reflect all options
    - [ ] integrate s4b_write_frontend into full.py pipeline (currently manual via Makefile)
    - [ ] add configuration validation for vis_configs and data file dependencies
    - [ ] clean up s5 scripts -- type hints, config stuff, etc

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
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
flowchart LR
    %% MV["Model View"]
    CL["<a href='data/figures/classifications.html'>Classifications Page</a>"]
    AP["<a href='data/vis/attnpedia/index.html'>AttentionPedia</a>"]

    CL ~~~ AP
    
    %% PL["<a href='data/patterns/index.html'>Pattern Lens</a>"]
    PL@{img: "docs/resources/assets/diagram/pattern_lens.png", label: "<a href='data/patterns/index.html'>Pattern Lens</a>", pos: "t", w: 60, h: 160}
    %% SPV["<a href='data/patterns/single.html'>Single Pattern View</a>"]
    SPV@{img: "docs/resources/assets/diagram/single_pattern.png", label: "<a href='data/patterns/single.html'>Single Pattern View</a>", pos: "t", w: 60, h: 120}
    
    %% PEV["<a href='data/vis/embeds/patterns/index.html'>Pattern Embedding Vis</a>"]
    PEV@{img: "docs/resources/assets/diagram/embed_patterns.png", label: "<a href='data/vis/embeds/patterns/index.html'>Pattern Embedding Vis</a>", pos: "t", w: 60, h: 120}
    HEV["<a href='data/vis/embeds/heads/index.html'>Head Embedding Vis</a>"]
    HET["<a href='data/figures/head_embed_table.html'>Head Embedding Table</a>"]

    %% Existing connections (solid lines)
    SPV -->|head ID| AP
    SPV -->|prompt hash| PL
    PL -->|pattern images| SPV
    AP -->|patterns in each cell| SPV
    AP -->|current head <br/>or<br/> all displayed heads| PL

    PEV -->|each point - right click| SPV
    HET -->|links to each variant| HEV

    CL -->|classification links| AP
    AP -->|link| CL
    HEV -->|each point - right click| AP

    %% Missing connections (dashed lines)
    SPV -.->|<span style='background-color: red'>just this pattern selected</span>| PEV
    AP -.->|<span style='background-color: red'>current/all heads selected</span>| PEV
    AP -.->|<span style='background-color: red'>current/all heads selected</span>| HEV
    %% MV -.->|<span style='background-color: red'>click on head</span>| AP
```



```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
flowchart LR
    subgraph col1[" "]
        direction TB
        CL["<a href='data/figures/classifications.html'>Classifications Page</a>"]
        MV["Model View"]
    end
    
    subgraph AP[" "]
        direction LR
        APH["<a href='data/vis/attnpedia/index.html'>AttentionPedia Heads</a>"]
        APC["<a href='data/vis/attnpedia/index.html'>AttentionPedia Classes</a>"]
        APH <--> APC
    end
    
    subgraph col3[" "]
        direction TB
        SPV["<a href='data/patterns/single.html'>Single Pattern View</a>"]
        PL["<a href='data/patterns/index.html'>Pattern Lens</a>"]
        HEV["<a href='data/vis/embeds/heads/index.html'>Head Embedding Vis</a>"]
        PEV["<a href='data/vis/embeds/patterns/index.html'>Pattern Embedding Vis</a>"]
        HET["<a href='data/figures/head_embed_table.html'>Head Embedding Table</a>"]
    end

    %% col1 ~~~ col2 ~~~ col3
    
    %% Style to hide subgraph borders
    %% style col1 fill:none,stroke:none
    %% style col2 fill:none,stroke:none
    %% style col3 fill:none,stroke:none
    %% style col4 fill:none,stroke:none
    
    %% Existing connections (solid lines)
    SPV -->|head ID| APH
    SPV -->|prompt hash| PL
    PL -->|pattern images| SPV
    AP -->|patterns in each cell| SPV
    AP -->|current head <br/>or<br/> all displayed heads| PL

    PEV -->|each point - right click| SPV
    HET -->|links to each variant| HEV

    CL -->|classification links| APC
    APC -->|link| CL
    HEV -->|each point - right click| AP

    %% Missing connections (dashed lines)
    SPV -.->|<span style='background-color: red'>just this pattern selected</span>| PEV
    AP -.->|<span style='background-color: red'>current/all heads selected</span>| PEV
    AP -.->|<span style='background-color: red'>current/all heads selected</span>| HEV
    MV -.->|<span style='background-color: red'>click on head</span>| AP
```