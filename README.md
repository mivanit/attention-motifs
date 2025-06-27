# attention-motifs

Supplement for the paper "Motifs in Attention Patterns of Large Language Models"

Code and interactive figures also available at https://attention-motifs.github.io

# Structure:

- `attn_embed/`: utilities for processing, computing features about, embedding, and doing other things to attention patterns
- `notebooks/`: Jupyter notebooks for generating and displaying attention patterns. most of the interesting stuff happens here.
- `tests/`: run these to check that everything is working
- `data/`: some data from pile-10k, code will add stuff here as well

# Setup

You will need:

- Python
- [`uv`](https://docs.astral.sh/uv/) for package/environment management (or use the `.meta/requirements/requirements.txt` file)
- `make` for running the `Makefile` (also, a posix shell)
- a way to run Jupyter notebooks (preferably VSCode)

simply run `uv sync` to install all dependencies and create a new environment.

# Usage

## First, we test things are working:

- run `make am-help` to see some commands particular to this repo
- run `make test` to run tests. if something fails here, it is likely that dependencies were not installed correctly. try using `uv`.


## Then, we need some attention patterns to work with:

- run `make am-activations` to generate activations for selected models.
- (optional) run `make am-figures` and then `am-server-patternlens` to view the attention patterns in a web interface. Identical to https://attention-motifs.github.io/patterns and was used for figures 2, 7, and 8.


## Analyzing the attention patterns:

in the `notebooks` directory, run the notebooks in order: `01` to `04`.

- `01` loads the attention patterns from the previous step, and computes a table of features about them
- `02` reads the big table of raw features, does some filtering, covariance analysis, and dimensionality reduction. This was used for figures 3, 9, 10, and 12.
- (optional) run `make am-server-embed` to view the embeddings in a web interface. Identical to https://attention-motifs.github.io/embed and was used for figure 4.
- `03` computes and saves distances between heads according to equation 5, and was used for figure 5.
- `04` loads the distances between heads, labelling some and projecting them to a 2D space. This was used for figures 6, 12, and 13.

Other notebooks include `A0` for a basic example of getting attention patterns, `A1` for the synthetic example used in figure 1, and `A2` for working with the `pattern_lens` interface.

# Configuration

By default, the code and figures use a reduced set of models and reduced sample size to make things easier to work with. In the `Makefile`, towards the end (do a `Ctrl+F` for `# CONFIGURE DEMO`), you will find:

```makefile
# for a small run
DEMO_MODELS ?= gpt2-small,pythia-1b
DEMO_N_SAMPLES ?= 16

# for reproducing the paper
# DEMO_N_SAMPLES ?= 128
# DEMO_MODELS ?= pythia-1b,gpt2-small,gpt2-medium,Llama-3.2-1B,gemma-2b,gemma-2-2b
```

Comment out the first pair, and uncomment the second pair to reproduce exactly the figures in the paper. This will take somewhat longer, but is still practical on consumer grade hardware (albeit it may take quite a bit of disk space).


# TODO:

- computing features:
	- [x] add `activation.n_ctx` to the feature set
	- [ ] speed up via torch computing of the features
	- [ ] once it's faster, add in all the other features

- analyzing features:
	- [x] figure out the most important features
	- [x] distance matrix between heads via
		- [x] naive point to point dist between each prompt (look at distributions)
		- [ ] wasserstein or similar matching over all prompts
	- [x] tsne/UMAP over that distance matrix to find clusters

- embeddings display:
	- [x] fix colors not matching between bottom bar and actual plot
	- [ ] option to show attention pattern on hover in the 3D embeddings
	- [x] when default coloring by a scalar column, use a colormap to show the scalar value
		- [ ] maybe up to 3 scalar column via RGB?
	- [x] saving selections/config/camera position in URL?
		- no camera pos, whatever
	- [ ] have a column "layer_depth" which is a float 0 to 1, since different models have different number of layers

- attentionpedia:
	- [~] settle on a json format
		- settled on one for now
	- [x] throw in a few papers: ioi, induction heads, nanda all gpt2-small heads
		- idk what model induction heads paper used


