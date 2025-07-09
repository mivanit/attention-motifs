# Interface Relations and TODOs

> Note: we should attempt to make the needed connections with minimal changes to the code, instead working with the configuration files.


misc todos:

- [ ] default pipeline config should more accurately reflect all options
- [ ] integrate s4b_write_frontend into full.py pipeline (currently manual via Makefile)
- [ ] move head embedding table generation from s5b into s4b_write_frontend.py for consistency
- [ ] add configuration validation for vis_configs and data file dependencies


## Pattern Lens (`data/patterns/index.html`)

Allows comparing lots of patterns across many heads, models, etc. Uses tilde-separated URLs for compatibility and has model/head selection grids and prompt tables.

Outgoing connections:
- [x] **→ Single Pattern View** (pattern images): Click on pattern images to view detailed single pattern

## Single Pattern View (`data/patterns/single.html`)

Shows a single pattern and its prompt in detail with interactive heatmap and token highlighting.

Outgoing connections:
- [x] **→ Pattern Lens** (prompt hash): Click prompt hash to go to pattern lens with that prompt selected
- [x] **→ AttentionPedia** (head ID): Click head ID to go to attentionpedia for that head

## AttentionPedia (`data/vis/attnpedia/index.html`)

Shows a given head and then a selection of nearby/random/distant (in embedding space) heads, as well as heads with the same classification. These heads are the rows, columns are different prompts.

Outgoing connections:
- [x] **→ Single Pattern View** (patterns in each cell): Click on patterns in table cells to view detailed single pattern
- [x] **→ Pattern Lens** (current head or all displayed heads): Use links at top to go to pattern lens for current head or all displayed heads
- [ ] **→ Pattern Embedding Vis** (current/all heads selected): Add way to go to pattern embedding vis with current head/context selected
- [ ] **→ Head Embedding Vis** (current/all heads selected): Add way to go to head embedding vis with current head/context selected

## Pattern Embedding Vis (`data/vis/embeds/patterns/index.html`)

3D visualization of all attention patterns where each point represents a pattern.

Outgoing connections:
- [x] **→ Single Pattern View** (each point - right click): Right-click on pattern points to go to single pattern view for that pattern

## Head Embedding Vis (`data/vis/embeds/heads/index.html`)

3D visualization where each point is a head in embedding space.

Outgoing connections:
- [ ] **→ AttentionPedia** (each point - right click): Right-click on head points to go to attentionpedia for that head

## Head Embedding Table (`head_embed_table.html`)

Table view of all head embedding plots with SVG previews and links to 3D visualization.

Other todos:
- [ ] checkboxes/sliders for selecting methods, dims, neighbor counts, etc
- [ ] no "see 3d" button on 2d stuff
- [ ] clean up s5 scripts -- type hints, config stuff, etc

Outgoing connections:
- [x] **→ Head Embedding Vis** (embed links): Click embedding links to view in 3D visualization with specific parameters


# interface relations diagram

- solid lines / blue label: existing connections
- dashed lines / red label: missing connections that should be added

```mermaid
flowchart TD
    PL[Pattern Lens<br/>patterns/index.html] 
    SPV[Single Pattern View<br/>patterns/single.html]
    AP[AttentionPedia<br/>vis/attnpedia/index.html]
    PEV[Pattern Embedding Vis<br/>vis/embeds/patterns/index.html]
    HEV[Head Embedding Vis<br/>vis/embeds/heads/index.html]
    HET[Head Embedding Table<br/>head_embed_table.html]

    %% Existing connections (solid lines)
    PL -->|<span style='background-color: blue'>pattern images</span>| SPV
    AP -->|<span style='background-color: blue'>patterns in each cell</span>| SPV
    AP -->|<span style='background-color: blue'>current head <br/>or<br/> all displayed heads</span>| PL
    SPV -->|<span style='background-color: blue'>prompt hash</span>| PL
    SPV -->|<span style='background-color: blue'>head ID</span>| AP

    PEV -->|<span style='background-color: blue'>each point - right click</span>| SPV
    HET -->|<span style='background-color: blue'>embed links</span>| HEV

    %% Missing connections (dashed lines)
    AP -.->|<span style='background-color: red'>current/all heads selected</span>| PEV
    AP -.->|<span style='background-color: red'>current/all heads selected</span>| HEV
    HEV -.->|<span style='background-color: red'>each point - right click</span>| AP
```