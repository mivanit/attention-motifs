# attention-motifs

Supplement for the paper "Motifs in Attention Patterns of Large Language Models"

Code and interactive figures also available at https://attention-motifs.github.io

# Structure:

- `attention_motifs/`: core code. for processing, computing features about, embedding, and doing other things to attention patterns
- `notebooks/`: Jupyter notebooks for running the pipeline. most of the interesting stuff happens here.
- `tests/`: run these with `make test` to check that everything is working
- `data/`: some data from pile-10k, code will write files here as well


# Usage

## Setup:

You will need:

- Python
- [`uv`](https://docs.astral.sh/uv/) for package/environment management (or use the `.meta/requirements/requirements.txt` file)
- `make` for running the `Makefile` (also, a posix shell)
- a way to run Jupyter notebooks (preferably VSCode)

simply run `uv sync` or `make dep` to install all dependencies and create a new environment.

Optionally, you can:

- run `make am-help` to see some commands particular to this repo
- run `make test` to run tests. if something fails here, it is likely that dependencies were not installed correctly. try using `uv`.


## Pipeline:

edit config values in `pipeline_cfg.toml`, and then run `make am-pipeline` to run the full pipeline.

## Notebooks:

The functionality of the pipeline, along with some extras, is also found in the `notebooks/` folder. The main notebooks to run the pipeline are:

- `01` loads the attention patterns from the previous step, and computes a table of features about them
- `02` reads the big table of raw features, does some filtering, covariance analysis, and dimensionality reduction. This was used for figures 3, 9, 10, and 12.
- (optional) run `make am-serve` to view the embeddings in a web interface. Identical to https://attention-motifs.github.io/embed and was used for figure 4.
- `03` computes and saves distances between heads according to equation 5, and was used for figure 5.
- `04` loads the distances between heads, labelling some and projecting them to a 2D space. This was used for figures 6, 12, and 13.

Other notebooks include:

- `A0` for a basic example of getting attention patterns
- `A1` for the synthetic example patterns
- `A2` for working with the `pattern_lens` interface.

