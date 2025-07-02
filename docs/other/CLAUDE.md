# Attention Motifs Repository - Claude Notes

This repository analyzes attention patterns in transformer models and creates interactive visualizations.

## Repository Structure

### Main Components
- `attn_embed/` - Core analysis library
- `data/` - Generated data files (features, patterns, figures)
- `notebooks/` - Jupyter notebooks for analysis
- `tests/` - Test suite

### Key Directories

#### attn_embed/
- `pipeline/` - Data processing pipeline scripts
  - `s0_download_models.py` - Download models
  - `s1_activations.py` - Extract activations
  - `s2_features.py` - Generate features
  - `s3_feat_proc.py` - Process features
  - `s4_head_dist.py` - Compute head distances
- `features/` - Feature extraction and analysis
- `math/` - Mathematical utilities (cosine similarity, matrix operations)
- `frontend/` - Web visualizations
  - `attnpedia/` - Main attention analysis interface
  - `displayV2/` - Updated visualization display
  - `embeds/` - Embedding visualizations

#### Frontend Structure
- `attnpedia/src/`
  - `head_embeds.js` - Head similarity analysis (HeadDistances class)
  - `AttentionPedia.js` - Main app component
  - `headInfo.js` - Head information utilities
  - `config.js` - Configuration settings

### Data Flow
1. Models downloaded via `s0_download_models.py`
2. Activations extracted via `s1_activations.py` 
3. Features computed via `s2_features.py`
4. Head distances calculated via `s4_head_dist.py`
5. Frontend loads data from `data/features/` for visualization

### Key Files
- `pipeline_cfg.toml` - Pipeline configuration
- `data/features/head_dists.zanj` - Precomputed head distances
- `data/features/pca.jsonl` - PCA embeddings for visualization

### Testing
- Run tests with standard Python test tools
- Test files in `tests/` directory
- Coverage reports generated in `docs/coverage/`

### Development Notes
- Uses numpy arrays for efficient computation
- NDArray.load() used for loading .npy files in frontend
- HeadDistances class provides similarity search functionality
- Configuration URLs defined in config.js for data loading