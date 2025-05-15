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
	- [x] figure out the most important features
	- [x] distance matrix between heads via
		- [x] naive point to point dist between each prompt (look at distributions)
		- [ ] wasserstein or similar matching over all prompts
	- [x] tsne/UMAP over that distance matrix to find clusters

- embeddings display:
	- [x] FIX COLORS NOT MATCHING BETWEEN BOTTOM BAR AND ACTUAL PLOT
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


