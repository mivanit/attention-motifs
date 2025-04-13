# attention-motifs

Trying to find structure in attention patterns of autoregressive LLMs in order to classify heads by their function


# Notebooks:

in the `notebooks/` directory:

- `demo_dataset.ipynb` how to generate and load datasets of attention patterns




# TODO:

- computing features:
	- [x] add `activation.n_ctx` to the feature set
	- [ ] speed up via torch computing of the features
	- [ ] once it's faster, add in all the other features

- analyzing features:
	- [ ] figure out the most important features
	- [ ] distance matrix between heads via
		- [ ] naive point to point dist between each prompt (look at distributions)
		- [ ] wasserstein or similar matching over all prompts
	- [ ] tsne/UMAP over that distance matrix to find clusters

- embeddings display:
	- [ ] FIX COLORS NOT MATCHING BETWEEN BOTTOM BAR AND ACTUAL PLOT
	- [ ] option to show attention pattern on hover in the 3D embeddings
	- [ ] when default coloring by a scalar column, use a colormap to show the scalar value
		- [ ] maybe up to 3 scalar column via RGB?
	- [ ] saving selections/config/camera position in URL?


- attentionpedia:
	- [ ] settle on a json format
	- [ ] throw in a few papers: ioi, induction heads, nanda all gpt2-small heads


